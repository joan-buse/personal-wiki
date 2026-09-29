"""Curated wiki notes: readable filenames, links, source references, and index.md.

Structured note data lives in data/notes.json (outside the vault); Markdown in
vault/wiki/ is rendered from it. A note that was edited by hand since it was
last rendered is never overwritten: the new version goes to data/pending/.
"""
from __future__ import annotations

import hashlib
import json
import re
from datetime import date
from pathlib import Path

from .config import Config

SMALL_WORDS = {"a", "an", "and", "as", "at", "but", "by", "for", "in", "of", "on", "or", "the", "to", "vs", "with"}
MAX_TITLE_WORDS = 6


def clean_title(raw: str) -> str | None:
    """Short, filename-safe, human title such as 'GPU Parallel Training'."""
    t = re.sub(r"[\\/:*?\"<>|#^\[\]{}`_.,;!()]|(?<=\w)-(?=\w)|\s-\s", " ", raw or "")
    # Drop machine-style tokens: years/timestamps, long numbers, hex hashes.
    words = [w for w in t.split() if not re.fullmatch(r"\d{4,}|[0-9a-f]{8,}|-+", w.lower())]
    words = words[:MAX_TITLE_WORDS]
    while words and words[-1].lower() in SMALL_WORDS:
        words.pop()
    if not words:
        return None
    out = []
    for i, w in enumerate(words):
        if len(w) > 1 and (w.isupper() or any(c.isupper() for c in w[1:])):   # keep acronyms / camelCase (GPU, PyTorch)
            out.append(w)
        elif i > 0 and w.lower() in SMALL_WORDS:
            out.append(w.lower())
        else:
            out.append(w[:1].upper() + w[1:])
    return " ".join(out)


def key(title: str) -> str:
    return title.lower()


def _title_words(title: str) -> set:
    out = set()
    for w in title.lower().split():
        if w in SMALL_WORDS:
            continue
        for suffix in ("ting", "ing", "s"):   # formatting ~ format, bullets ~ bullet
            if w.endswith(suffix) and len(w) - len(suffix) >= 4:
                w = w[: -len(suffix)]
                break
        out.add(w)
    return out


def _sha(text: str) -> str:
    return hashlib.sha256(text.encode()).hexdigest()


class NoteStore:
    def __init__(self, cfg: Config):
        self.cfg = cfg
        self.notes: dict = {}
        if cfg.notes_file.exists():
            self.notes = json.loads(cfg.notes_file.read_text(encoding="utf-8"))

    def save(self) -> None:
        self.cfg.notes_file.parent.mkdir(parents=True, exist_ok=True)
        self.cfg.notes_file.write_text(json.dumps(self.notes, indent=2, ensure_ascii=False), encoding="utf-8")

    def titles(self) -> list[str]:
        return sorted(n["title"] for n in self.notes.values())

    def titles_for_source(self, sid: str) -> list[str]:
        return sorted(n["title"] for n in self.notes.values() if sid in n["contributions"])

    def path_for(self, note: dict) -> Path:
        return self.cfg.wiki_dir / note["folder"] / f"{note['title']}.md"

    def remove_source(self, sid: str) -> None:
        """Drop one source's contributions before regenerating them (re-ingest)."""
        for k in list(self.notes):
            self.notes[k]["contributions"].pop(sid, None)

    def summaries(self) -> dict:
        """title -> first summary, for notes that currently have sources."""
        return {n["title"]: next(iter(sorted(n["contributions"].items())))[1]["summary"]
                for n in self.notes.values() if n["contributions"]}

    def find_similar(self, title: str, also: tuple = ()) -> dict | None:
        """An existing note (with sources) on the same subject: exact title, or titles sharing
        at least two content words and two-thirds of the shorter title's words
        ('Resume Bullet Components' ~ 'Resume Bullet Structure Elements')."""
        also_keys = {key(x) for x in also}   # e.g. this source's previous titles during a re-ingest
        live = [n for n in self.notes.values() if n["contributions"] or key(n["title"]) in also_keys]
        for n in live:
            if key(n["title"]) == key(title):
                return n
        words = _title_words(title)
        best, best_score = None, 0.0
        for n in live:
            other = _title_words(n["title"])
            shared = len(words & other)
            score = shared / min(len(words), len(other)) if words and other else 0
            if shared >= 2 and score >= 0.66 and score > best_score:
                best, best_score = n, score
        return best

    def add_topic(self, sid: str, topic: dict, model: str, previous: tuple = ()) -> str:
        title = topic["title"]
        note = self.find_similar(title, also=previous) or self.notes.get(key(title))
        if note is None:
            note = {"title": title, "folder": topic["folder"], "contributions": {}, "rendered_sha": None}
            self.notes[key(title)] = note
        prev = note["contributions"].get(sid)
        if prev:  # same source produced the same topic twice (long source split into segments)
            prev["details"] += [d for d in topic["details"] if d not in prev["details"]]
            prev["related"] += [r for r in topic["related"] if r not in prev["related"]]
        else:
            note["contributions"][sid] = {
                "summary": topic["summary"], "details": topic["details"],
                "related": topic["related"], "model": model, "date": date.today().isoformat(),
            }
        return note["title"]

    # --- rendering -------------------------------------------------------------
    def render_all(self, catalog: dict) -> dict:
        """Write every note; delete notes with no remaining sources. Returns a report."""
        report = {"written": [], "unchanged": [], "kept_manual_edits": [], "removed": []}
        existing = {key(n["title"]) for n in self.notes.values() if n["contributions"]}
        for k in list(self.notes):
            note = self.notes[k]
            path = self.path_for(note)
            edited = False
            if path.exists():
                current = path.read_text(encoding="utf-8")
                if note.get("rendered_sha"):
                    edited = _sha(current) != note["rendered_sha"]
                else:  # a file we never rendered (e.g. written by hand): don't clobber it
                    edited = "type: wiki-note" not in current
            if not note["contributions"]:
                if path.exists() and not edited:
                    path.unlink()
                    report["removed"].append(note["title"])
                elif edited:
                    report["kept_manual_edits"].append(note["title"])
                del self.notes[k]
                continue
            text = self.render(note, catalog, existing)
            if edited:
                pending = self.cfg.data_dir / "pending" / f"{note['title']}.md"
                pending.parent.mkdir(parents=True, exist_ok=True)
                pending.write_text(text, encoding="utf-8")
                report["kept_manual_edits"].append(note["title"])
                continue
            if path.exists() and path.read_text(encoding="utf-8") == text:
                report["unchanged"].append(note["title"])
            else:
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_text(text, encoding="utf-8")
                report["written"].append(note["title"])
            note["rendered_sha"] = _sha(text)
        self._remove_empty_folders()
        return report

    def render(self, note: dict, catalog: dict, existing: set) -> str:
        contribs = note["contributions"]
        sids = sorted(contribs)
        src_files = [catalog[s]["vault_path"] for s in sids if s in catalog]
        fm = ["---", f"title: {note['title']}", "type: wiki-note", "sources:"]
        fm += [f"  - {s}" for s in sids]
        fm += ["source_files:"] + [f"  - \"{p}\"" for p in src_files]
        fm += [f"generated_by: {', '.join(sorted({c['model'] for c in contribs.values()}))}",
               f"updated: {max(c['date'] for c in contribs.values())}", "---", ""]
        lines = fm + [f"# {note['title']}", ""]

        for i, s in enumerate(sids):
            summary = contribs[s]["summary"].strip()
            if i == 0 or len(sids) == 1:
                lines.append(summary)
            else:
                lines.append(f"From {self._src_link(catalog, s)}: {summary}")
            lines.append("")

        lines += ["## Details", ""]
        for s in sids:
            for d in contribs[s]["details"]:
                lines.append(f"- {d.strip()} *(source: {self._src_link(catalog, s)})*")
        lines.append("")

        related, seen = [], {key(note["title"])}
        candidates = [r for s in sids for r in contribs[s]["related"]] + note.get("links", [])
        for r in candidates:
            t = clean_title(r.get("title", ""))
            target = self.find_similar(t) if t else None
            if target and key(target["title"]) in existing and key(target["title"]) not in seen:
                seen.add(key(target["title"]))
                reason = r.get("reason", "").strip()
                related.append(f"- [[{target['title']}]]" + (f" — {reason}" if reason else ""))
        if related:
            lines += ["## Related notes", ""] + related + [""]

        lines += ["## Sources", ""]
        for s in sids:
            lines.append(f"- {self._src_link(catalog, s)} — original, unchanged ({s})")
        lines += ["", "[[index|Back to index]]", ""]
        return "\n".join(lines)

    def _src_link(self, catalog: dict, sid: str) -> str:
        entry = catalog.get(sid)
        if not entry:
            return sid
        return f"[[{entry['vault_path']}|{entry['title']}]]"

    def _remove_empty_folders(self) -> None:
        if not self.cfg.wiki_dir.exists():
            return
        for d in sorted(self.cfg.wiki_dir.rglob("*"), reverse=True):
            if d.is_dir() and not any(d.iterdir()):
                d.rmdir()

    # --- index.md --------------------------------------------------------------
    def write_index(self, catalog: dict) -> None:
        by_folder: dict[str, list[dict]] = {}
        for n in self.notes.values():
            by_folder.setdefault(n["folder"], []).append(n)
        lines = ["# Personal Wiki", "",
                 "Start here. Notes are grouped by topic; each links back to its original source in `raw/`.", ""]
        for folder in sorted(by_folder):
            lines += [f"## {folder}", ""]
            for n in sorted(by_folder[folder], key=lambda n: n["title"].lower()):
                first = next(iter(sorted(n["contributions"].items())))[1]["summary"]
                blurb = re.split(r"(?<=[.!?])\s", first.strip(), maxsplit=1)[0]
                lines.append(f"- [[{n['title']}]] — {blurb}")
            lines.append("")
        lines += ["## Original sources", ""]
        for sid, entry in sorted(catalog.items(), key=lambda kv: kv[1]["title"].lower()):
            notes = ", ".join(f"[[{t}]]" for t in entry.get("notes", []))
            lines.append(f"- [[{entry['vault_path']}|{entry['title']}]]" + (f" → {notes}" if notes else ""))
        lines.append("")
        self.cfg.index_md.write_text("\n".join(lines), encoding="utf-8")
