"""`wiki` command line: parses the command, picks the mode, and calls the harness."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from . import evidence
from .config import load_config
from .harness import CHAT_HELP, COMMANDS, ChatSession, ask
from .ingest import ingest
from .llm import LLMError, OllamaClient
from .retrieval import Index

DESCRIPTION = """Personal wiki CLI: your own notes + a local Gemma model, fully offline.

commands:
""" + COMMANDS

EPILOG = """setup (once, while online):
  1. install Ollama and pull the model:   ollama pull gemma4:e2b-it-qat
  2. put 3+ original sources (.md/.txt/.pdf) in vault/raw/
  3. wiki ingest

configuration: wiki.json in the project root (model, num_ctx, top_k, ...),
or env WIKI_MODEL / OLLAMA_HOST_URL, or --model / --host flags.
Mode: local is the default and only implemented execution mode."""


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="wiki", description=DESCRIPTION, epilog=EPILOG,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--model", help="Ollama model tag (default from wiki.json, else gemma4:e2b-it-qat)")
    p.add_argument("--host", help="Ollama URL (default http://127.0.0.1:11434)")
    p.add_argument("--root", help=argparse.SUPPRESS)
    sub = p.add_subparsers(dest="command", metavar="COMMAND")

    s = sub.add_parser("ingest", help="generate wiki notes from vault/raw/ and rebuild the index")
    s.add_argument("path", nargs="?", help="a file or folder inside vault/raw/ (default: all)")
    s.add_argument("--force", action="store_true", help="regenerate notes even if the source is unchanged")

    s = sub.add_parser("search", help="show matching original passages (no model)")
    s.add_argument("query")
    s.add_argument("-k", type=int, default=None, help="number of passages (default top_k)")
    s.add_argument("--scope", choices=["raw", "wiki", "all"], default="raw",
                   help="raw = original sources (default), wiki = generated notes, all = both")
    s.add_argument("--save", action="store_true", help="save results to evidence/search/")

    s = sub.add_parser("ask", help="grounded answer with citations, independent of chat")
    s.add_argument("question")
    s.add_argument("--mode", choices=["local", "online"], default="local")
    s.add_argument("--save", action="store_true", help="save an evidence card to evidence/ask/")

    s = sub.add_parser("chat", help="personal assistant session")
    s.add_argument("--mode", choices=["local", "online"], default="local")

    s = sub.add_parser("test", help="run eval/questions.json in ask mode and save evidence cards")
    s.add_argument("--file", default=None, help="question file (default eval/questions.json)")

    sub.add_parser("modecheck", help="run eval/mode_checks.json (chat/search/ask boundaries)")
    sub.add_parser("status", help="model, runtime, memory and network state")
    return p


def require_local(mode: str) -> None:
    if mode != "local":
        raise LLMError("Online mode is not implemented. Local mode is the default: omit --mode or use --mode local.")


def print_hits(hits, verbose=True) -> None:
    if not hits:
        print("No matching passages.")
    for i, h in enumerate(hits, 1):
        print(f"\n[{i}] {h.passage.location()}  score={h.score}  terms={','.join(h.matched)}")
        if verbose:
            for line in h.passage.text.splitlines():
                print("    " + line)


def cmd_ingest(cfg, llm, args) -> int:
    print(f"Ingesting {cfg.rel(cfg.raw_dir)} with {cfg.model} (local)")
    run = ingest(cfg, llm, Path(args.path) if args.path else None, force=args.force)
    r = run["render"]
    print(f"\nNotes written: {len(r['written'])}, unchanged: {len(r['unchanged'])}, removed: {len(r['removed'])}")
    if r["kept_manual_edits"]:
        print(f"Kept your manual edits (new versions in data/pending/): {', '.join(r['kept_manual_edits'])}")
    print(f"Passages indexed: {run['passages']}  |  {run['seconds']}s  |  log: {run['log_file']}")
    if run.get("memory"):
        print(f"Memory: {fmt_mem(run['memory'])}")
    for e in run["errors"]:
        print(f"ERROR {e['source']}: {e['error']}", file=sys.stderr)
    return 1 if run["errors"] else 0


def cmd_search(cfg, args) -> int:
    index = Index.load(cfg.passages_file)
    kinds = {"raw": ("raw",), "wiki": ("wiki",), "all": ("raw", "wiki")}[args.scope]
    hits = index.search(args.query, k=args.k or cfg.top_k, kinds=kinds)
    print(f"search (no model) · scope={args.scope} · \"{args.query}\"")
    print_hits(hits)
    if args.save:
        path = evidence.save_search(cfg, evidence.run_info(cfg, None), args.query, hits, args.scope)
        print(f"\nsaved: {cfg.rel(path)}")
    return 0


def cmd_ask(cfg, llm, args) -> int:
    require_local(args.mode)
    index = Index.load(cfg.passages_file)
    llm.check()
    print(f"ask · model {cfg.model} · mode local · no chat history\n")
    res = ask(cfg, llm, index, args.question, on_token=lambda t: print(t, end="", flush=True))
    if not res.model_called:
        print(res.answer)
    print("\n\nSources:")
    for i, h in enumerate(res.hits, 1):
        print(f"  [S{i}] {h.passage.location()}")
    if not res.hits:
        print("  (none retrieved)")
    print(f"\ncitation check: {res.check['status']}  |  {res.seconds}s")
    if args.save:
        path = evidence.save_ask(cfg, evidence.run_info(cfg, llm), res, llm.memory_snapshot())
        print(f"saved: {cfg.rel(path)}")
    return 0


def cmd_chat(cfg, llm, args) -> int:
    require_local(args.mode)
    llm.check()
    try:
        index = Index.load(cfg.passages_file)
    except FileNotFoundError:
        index = None
        print("(No wiki index yet — chat works, but cannot look up notes. Run `wiki ingest`.)")
    session = ChatSession(cfg, llm, index)
    info = evidence.run_info(cfg, llm)
    print(f"chat · {cfg.model} · local · conversation kept for this session only "
          f"(last {cfg.chat_history_turns} exchanges)\n{CHAT_HELP}\n")
    try:
        while True:
            try:
                msg = input("you › ").strip()
            except EOFError:
                break
            if not msg:
                continue
            if msg in ("/exit", "/quit"):
                break
            if msg == "/reset":
                session.reset()
                print("(conversation cleared)")
                continue
            if msg == "/sources":
                last = session.turns[-1] if session.turns else None
                print_hits(last.hits if last else [], verbose=True)
                continue
            if msg.startswith("/") and not msg.startswith("/notes "):
                print(CHAT_HELP)
                continue
            print("assistant › ", end="", flush=True)
            turn = session.send(msg, on_token=lambda t: print(t, end="", flush=True))
            note = f"notes used: {len(turn.hits)}" if turn.retrieved else "no notes lookup"
            print(f"\n  ({note} — {turn.reason}; {turn.seconds}s)")
            for i, h in enumerate(turn.hits, 1):
                print(f"  [S{i}] {h.passage.location()}")
            print()
    except KeyboardInterrupt:
        print()
    if session.turns:
        path = evidence.save_chat(cfg, info, session.turns)
        print(f"transcript saved: {cfg.rel(path)}")
    return 0


def cmd_test(cfg, llm, args) -> int:
    qfile = Path(args.file) if args.file else cfg.eval_dir / "questions.json"
    tests = json.loads(qfile.read_text(encoding="utf-8"))["questions"]
    index = Index.load(cfg.passages_file)
    llm.check()
    info = evidence.run_info(cfg, llm)
    rows = []
    for t in tests:
        print(f"\n=== {t['id']}: {t['question']}")
        res = ask(cfg, llm, index, t["question"])
        path = evidence.save_ask(cfg, info, res, llm.memory_snapshot(), test=t,
                                 name=f"{t['id']}-{evidence.stamp()}")
        data = json.loads(path.with_suffix(".json").read_text(encoding="utf-8"))
        ac = data["auto_checks"]
        print(res.answer)
        print(f"-> retrieved expected source: {ac['expected_source_retrieved']} | behavior ok: "
              f"{ac['behavior_matches']} | citations: {res.check['status']} | {res.seconds}s | {cfg.rel(path)}")
        rows.append((t["id"], t["question"], ac["expected_source_retrieved"], ac["behavior_matches"],
                     res.check["status"], res.seconds, path.name))
    md = ["# Ask-mode test summary", "", f"- {info['time']} · model `{cfg.model}` · local · internet reachable: "
          f"{info['internet_reachable']}", "",
          "| Test | Question | Expected source retrieved | Behavior as expected | Citations | Seconds | Card |",
          "|---|---|---|---|---|---|---|"]
    md += [f"| {a} | {b} | {c} | {d} | {e} | {f} | [{g}]({g}) |" for a, b, c, d, e, f, g in rows]
    out = cfg.evidence_dir / "ask" / f"summary-{evidence.stamp()}.md"
    out.write_text("\n".join(md) + "\n", encoding="utf-8")
    print(f"\nsummary: {cfg.rel(out)}\nAutomatic checks are not the grade: open each card and write the human assessment.")
    return 0


def cmd_modecheck(cfg, llm) -> int:
    spec = json.loads((cfg.eval_dir / "mode_checks.json").read_text(encoding="utf-8"))
    index = Index.load(cfg.passages_file)
    llm.check()
    info = evidence.run_info(cfg, llm)
    session = ChatSession(cfg, llm, index)
    md = ["# Mode boundary checks", "", f"- {info['time']} · model `{cfg.model}` · local · internet reachable: "
          f"{info['internet_reachable']}", "", "## 1. Chat (capabilities, draft, follow-up, chat-only claim)", ""]
    for msg in spec["chat"] + [spec["chat_only_claim"]]:
        print(f"\nyou › {msg}")
        t = session.send(msg)
        print(f"assistant › {t.reply}\n  (retrieval {'ON' if t.retrieved else 'off'}: {t.reason})")
    md += evidence.chat_markdown(session.turns)

    hits = index.search(spec["search"], k=cfg.top_k)
    md += ["## 2. Search (original passages, no generated answer)", "", f"Query: `{spec['search']}`", ""]
    for i, h in enumerate(hits, 1):
        md += [f"**{i}. {h.passage.location()}** (score {h.score})", "", "> " + h.passage.text.replace("\n", "\n> "), ""]
    if not hits:
        md.append("_No matching passages._")

    res = ask(cfg, llm, index, spec["ask_after_claim"])
    token = spec.get("claim_token", "")
    md += ["", "## 3. Ask after a chat-only claim (must not use chat as evidence)", "",
           f"Question: `{res.question}`", "", f"Retrieved: {', '.join(h.passage.location() for h in res.hits) or 'none'}",
           "", f"**Answer:** {res.answer}", "",
           f"- Citation check: `{res.check['status']}`",
           f"- Chat-only claim token `{token}` appears in ask answer: {bool(token) and token.lower() in res.answer.lower()}",
           "", "## Human assessment", "", "_Fill in: did each mode behave as specified?_", ""]
    print(f"\nask › {res.answer}")
    out = cfg.evidence_dir / "mode_checks" / f"modecheck-{evidence.stamp()}.md"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("\n".join(md), encoding="utf-8")
    evidence.save_chat(cfg, info, session.turns, subdir="mode_checks")
    print(f"\nsaved: {cfg.rel(out)}")
    return 0


def fmt_mem(m: dict) -> str:
    gb = lambda b: f"{b / 2**30:.2f} GB" if b else "n/a"
    return (f"model loaded {gb(m.get('model_size_bytes'))} (on GPU {gb(m.get('model_size_vram_bytes'))}), "
            f"ollama processes RSS {gb(m.get('ollama_rss_bytes'))}")


def cmd_status(cfg, llm) -> int:
    info = evidence.run_info(cfg, llm)
    print(json.dumps(info, indent=2))
    try:
        print("local models:", ", ".join(llm.local_models()) or "none")
        llm.check()
        print(f"model {cfg.model}: available")
        print("memory:", fmt_mem(llm.memory_snapshot()), "(model loads on first call)")
    except LLMError as e:
        print(f"model check: {e}")
    n = sum(1 for _ in cfg.passages_file.open()) if cfg.passages_file.exists() else 0
    print(f"passages indexed: {n}")
    return 0


def main(argv=None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if not args.command:
        parser.print_help()
        return 0
    cfg = load_config(args.root, args.model, args.host)
    llm = OllamaClient(cfg.host, cfg.model, cfg.num_ctx, think=cfg.think)
    try:
        if args.command == "ingest":
            return cmd_ingest(cfg, llm, args)
        if args.command == "search":
            return cmd_search(cfg, args)
        if args.command == "ask":
            return cmd_ask(cfg, llm, args)
        if args.command == "chat":
            return cmd_chat(cfg, llm, args)
        if args.command == "test":
            return cmd_test(cfg, llm, args)
        if args.command == "modecheck":
            return cmd_modecheck(cfg, llm)
        if args.command == "status":
            return cmd_status(cfg, llm)
    except (LLMError, FileNotFoundError, ValueError) as e:
        print(f"error: {e}", file=sys.stderr)
        return 2
    return 0
