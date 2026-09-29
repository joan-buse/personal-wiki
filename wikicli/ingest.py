"""`wiki ingest`: originals in vault/raw/ -> passages index + Gemma-written wiki notes.

Steps per run:
1. Read every source, split it into passages, and rebuild data/passages.jsonl (no model needed).
2. For each new or changed source (by SHA-256), send its text + instructions to local Gemma,
   which returns topics as JSON. Unchanged sources are skipped unless --force.
3. Merge topics into notes (same title => same note, so re-ingest never duplicates),
   render vault/wiki/<Folder>/<Title>.md and vault/index.md, and save the source catalog.
"""
from __future__ import annotations

import json
import re
import time
from datetime import datetime
from pathlib import Path

from .config import Config
from .llm import LLMError, OllamaClient, internet_reachable
from .notes import NoteStore, clean_title
from .retrieval import save_passages
from .sources import chunk, list_sources, read_pages, sha256, source_id, source_title


def load_prompt(cfg: Config, name: str) -> str:
    path = cfg.prompts_dir / name
    if not path.exists():
        raise FileNotFoundError(f"Missing instructions file: {cfg.rel(path)}")
    return path.read_text(encoding="utf-8").strip()


def load_catalog(cfg: Config) -> dict:
    if cfg.catalog_file.exists():
        return json.loads(cfg.catalog_file.read_text(encoding="utf-8"))
    return {}


def _segments(pages, limit: int) -> list[str]:
    """Group source text into pieces of at most `limit` characters, labelled by page."""
    segs, cur = [], ""
    for page, text in pages:
        block = (f"[page {page}]\n" if page else "") + text.strip() + "\n"
        while len(block) > limit:
            cut = block.rfind("\n\n", 0, limit)
            cut = cut if cut > limit // 2 else limit
            if cur:
                segs.append(cur)
                cur = ""
            segs.append(block[:cut])
            block = block[cut:]
        if len(cur) + len(block) > limit and cur:
            segs.append(cur)
            cur = ""
        cur += block
    if cur.strip():
        segs.append(cur)
    return segs


def _old_passages(cfg: Config) -> list:
    from .retrieval import Index
    try:
        return Index.load(cfg.passages_file).passages
    except FileNotFoundError:
        return []


def _blank_frontmatter(text: str) -> str:
    """Replace YAML frontmatter with blank lines so passage line numbers still match the file."""
    lines = text.split("\n")
    if lines and lines[0] == "---" and "---" in lines[1:]:
        end = lines.index("---", 1)
        lines[: end + 1] = [""] * (end + 1)
    return "\n".join(lines)


def _parse_topics(text: str, folders: list[str]) -> list[dict]:
    try:
        data = json.loads(text)
    except json.JSONDecodeError:
        m = re.search(r"\{.*\}", text, re.S)
        if not m:
            raise ValueError("model did not return JSON")
        data = json.loads(m.group(0))
    topics = data.get("topics") if isinstance(data, dict) else data
    if not isinstance(topics, list):
        raise ValueError("JSON has no 'topics' list")
    clean = []
    for t in topics:
        if not isinstance(t, dict):
            continue
        title = clean_title(str(t.get("title", "")))
        summary = str(t.get("summary", "")).strip()
        if not title or not summary:
            continue
        folder = str(t.get("folder", "")).strip().title()
        details = [str(d).strip() for d in t.get("details", []) if str(d).strip()]
        related = []
        for r in t.get("related", []) or []:
            if isinstance(r, dict) and r.get("title"):
                related.append({"title": str(r["title"]), "reason": str(r.get("reason", ""))})
            elif isinstance(r, str) and r.strip():
                related.append({"title": r, "reason": ""})
        clean.append({"title": title, "folder": folder if folder in folders else folders[0],
                      "summary": summary, "details": details[:8], "related": related[:5]})
    if not clean:
        raise ValueError("no usable topics in model output")
    return clean


def generate_topics(cfg: Config, llm: OllamaClient, title: str, vault_path: str, segment: str,
                    existing: dict, previous_titles: list[str], part: str) -> tuple[list[dict], dict]:
    instructions = load_prompt(cfg, "ingest-instructions.md")
    user = (
        f"Allowed folders: {', '.join(cfg.folders)}\n"
        "Existing wiki notes (title: summary). If this source covers the same subject as one of them, "
        "reuse that exact title instead of inventing a similar one:\n"
        + ("\n".join(f"- {t}: {s}" for t, s in existing.items()) or "- (none yet)") + "\n"
        f"Titles previously generated from this source (keep them if still accurate): {json.dumps(previous_titles)}\n\n"
        f"SOURCE FILE: {vault_path}\nSOURCE TITLE: {title}\nPART: {part}\n"
        f"<<<SOURCE TEXT\n{segment}\nSOURCE TEXT>>>\n\n"
        'Reply with JSON only: {"topics": [{"title": "...", "folder": "...", "summary": "...", '
        '"details": ["..."], "related": [{"title": "...", "reason": "..."}]}]}'
    )
    messages = [{"role": "system", "content": instructions}, {"role": "user", "content": user}]
    last_error = None
    for attempt in range(2):
        result = llm.chat(messages, temperature=0.2, json_mode=True)
        try:
            return _parse_topics(result.text, cfg.folders), {
                "seconds": result.seconds, "prompt_tokens": result.prompt_tokens,
                "output_tokens": result.output_tokens, "attempts": attempt + 1}
        except (ValueError, json.JSONDecodeError) as e:
            last_error = e
    raise ValueError(f"could not parse topics after 2 attempts: {last_error}")


def link_notes(cfg: Config, llm: OllamaClient, store: NoteStore, titles: set, log=print) -> list:
    """Ask Gemma which other notes are genuinely related to each touched note (with a reason)."""
    instructions = load_prompt(cfg, "link-instructions.md")
    summaries = store.summaries()
    calls = []
    for title in sorted(titles):
        note = store.find_similar(title)
        if note is None:
            continue
        others = "\n".join(f"- {t}: {s}" for t, s in summaries.items() if t != note["title"])
        user = (f"NOTE: {note['title']}: {summaries.get(note['title'], '')}\n\nOTHER NOTES:\n{others}\n\n"
                'Reply with JSON only: {"related": [{"title": "...", "reason": "..."}]}')
        try:
            r = llm.chat([{"role": "system", "content": instructions}, {"role": "user", "content": user}],
                         temperature=0.1, json_mode=True)
            data = json.loads(r.text)
            related = data.get("related", []) if isinstance(data, dict) else []
            note["links"] = [{"title": str(x["title"]), "reason": str(x.get("reason", ""))}
                             for x in related if isinstance(x, dict) and x.get("title")][:3]
            calls.append({"note": note["title"], "seconds": r.seconds, "links": [x["title"] for x in note["links"]]})
        except (LLMError, ValueError, json.JSONDecodeError) as e:
            log(f"  ! linking {title}: {e}")
            calls.append({"note": title, "error": str(e)})
    return calls


def ingest(cfg: Config, llm: OllamaClient | None, only: Path | None = None, force: bool = False,
           log=print) -> dict:
    cfg.raw_dir.mkdir(parents=True, exist_ok=True)
    cfg.wiki_dir.mkdir(parents=True, exist_ok=True)
    files = list_sources(cfg.raw_dir)
    if not files:
        raise FileNotFoundError(f"No .md, .txt or .pdf sources in {cfg.rel(cfg.raw_dir)}")
    if only is not None:
        only = only.resolve()
        if only.is_file():
            targets = {only}
        else:
            targets = {f for f in files if only in f.resolve().parents or f.resolve() == only}
        if not targets:
            raise FileNotFoundError(f"{only} is not a supported source inside {cfg.rel(cfg.raw_dir)}")
    else:
        targets = set(files)

    catalog = load_catalog(cfg)
    store = NoteStore(cfg)
    run = {"started": datetime.now().isoformat(timespec="seconds"), "model": cfg.model,
           "internet_reachable": internet_reachable(), "sources": [], "errors": []}
    t0 = time.perf_counter()

    # 1. Passages for retrieval (all sources, every run: cheap and keeps paths in sync).
    all_passages, texts = [], {}
    live_ids = set()
    for f in files:
        sid = source_id(cfg.raw_dir, f)
        live_ids.add(sid)
        try:
            pages = read_pages(f)
        except ValueError as e:
            run["errors"].append({"source": cfg.rel(f), "error": str(e)})
            log(f"  ! {e}")
            continue
        texts[f] = pages
        all_passages += chunk(pages, sid, cfg.rel(f), cfg.passage_words)

    # Sources deleted from raw/: drop their notes.
    for sid in [s for s in catalog if s not in live_ids]:
        store.remove_source(sid)
        del catalog[sid]
        log(f"  - removed missing source {sid}")

    # 2. Gemma writes notes for new/changed sources.
    need_model = [f for f in sorted(targets) if f in texts and
                  (force or catalog.get(source_id(cfg.raw_dir, f), {}).get("sha256") != sha256(f))]
    if need_model:
        # Save the source passages first so `wiki search` works even if the model is unavailable.
        save_passages(cfg.passages_file, all_passages + [p for p in _old_passages(cfg) if p.kind == "wiki"])
        try:
            if llm is None:
                raise LLMError("A local model is required to generate wiki notes.")
            llm.check()
        except LLMError as e:
            raise LLMError(f"{e}\n(Source passages were indexed, so `wiki search` already works; "
                           "wiki notes need the model.)") from e
    for f in sorted(targets):
        if f not in texts:
            continue
        sid, digest = source_id(cfg.raw_dir, f), sha256(f)
        title = source_title(f, texts[f])
        entry = {"title": title, "raw_path": cfg.rel(f), "vault_path": cfg.vault_rel(f),
                 "sha256": digest, "notes": catalog.get(sid, {}).get("notes", []),
                 "ingested": catalog.get(sid, {}).get("ingested"), "model": catalog.get(sid, {}).get("model")}
        if f not in need_model:
            log(f"  = {cfg.rel(f)} unchanged (use --force to regenerate)")
            catalog[sid] = entry
            run["sources"].append({"source": cfg.rel(f), "status": "unchanged"})
            continue
        log(f"  > {cfg.rel(f)}: asking {cfg.model} for wiki notes ...")
        previous = store.titles_for_source(sid)
        segs = _segments(texts[f], cfg.ingest_chars)
        topics, calls = [], []
        try:
            for i, seg in enumerate(segs, 1):
                existing = {t: s for t, s in store.summaries().items() if t not in previous}
                got, meta = generate_topics(cfg, llm, title, entry["vault_path"], seg, existing, previous,
                                            f"{i} of {len(segs)}")
                topics += got
                calls.append(meta)
        except (ValueError, LLMError) as e:
            run["errors"].append({"source": cfg.rel(f), "error": str(e)})
            log(f"  ! {cfg.rel(f)}: {e} (existing notes for this source kept)")
            if sid in catalog:
                run["sources"].append({"source": cfg.rel(f), "status": "failed"})
                continue
            catalog.pop(sid, None)
            continue
        store.remove_source(sid)
        made = sorted({store.add_topic(sid, t, cfg.model) for t in topics})
        entry.update(notes=made, ingested=datetime.now().isoformat(timespec="seconds"), model=cfg.model)
        catalog[sid] = entry
        run["sources"].append({"source": cfg.rel(f), "status": "generated", "notes": made,
                               "segments": len(segs), "chars_sent": sum(len(s) for s in segs), "calls": calls})
        log(f"    notes: {', '.join(made)}")

    # 3. Linking pass: for notes touched in this run, Gemma picks related notes with reasons.
    touched = {t for s in run["sources"] if s.get("status") == "generated" for t in s["notes"]}
    if touched:
        log(f"  > linking {len(touched)} notes ...")
        run["link_calls"] = link_notes(cfg, llm, store, touched, log)

    # 4. Render notes + index, then index the rendered notes too (for `wiki search --scope all`).
    for sid, entry in catalog.items():
        entry["notes"] = store.titles_for_source(sid)
    report = store.render_all(catalog)
    store.write_index(catalog)
    store.save()
    for n in store.notes.values():
        p = store.path_for(n)
        if p.exists():
            all_passages += chunk([(None, _blank_frontmatter(p.read_text(encoding="utf-8")))],
                                  "note:" + n["title"], cfg.rel(p),
                                  cfg.passage_words, kind="wiki")
    save_passages(cfg.passages_file, all_passages)
    cfg.catalog_file.write_text(json.dumps(catalog, indent=2, ensure_ascii=False), encoding="utf-8")

    run.update(render=report, passages=len(all_passages), seconds=round(time.perf_counter() - t0, 2))
    if llm is not None and need_model:
        run["memory"] = llm.memory_snapshot()
    out = cfg.evidence_dir / "ingest" / f"ingest-{datetime.now():%Y%m%d-%H%M%S}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(run, indent=2), encoding="utf-8")
    run["log_file"] = cfg.rel(out)
    return run
