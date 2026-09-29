"""Saved outputs (evidence/): ask cards, search results, chat transcripts, run info.

Everything is written exactly as produced; nothing here edits a result.
"""
from __future__ import annotations

import json
import platform
import subprocess
from datetime import datetime
from pathlib import Path

from .config import Config
from .harness import AskResult, ChatTurn
from .llm import LLMError, OllamaClient, internet_reachable
from .retrieval import Hit


def stamp() -> str:
    return datetime.now().strftime("%Y%m%d-%H%M%S")


def device_info() -> dict:
    def sh(*cmd):
        try:
            return subprocess.run(cmd, capture_output=True, text=True, timeout=5).stdout.strip()
        except (OSError, subprocess.SubprocessError):
            return ""
    info = {"os": f"{platform.system()} {platform.mac_ver()[0] or platform.release()}",
            "machine": platform.machine(), "python": platform.python_version()}
    if platform.system() == "Darwin":
        info["chip"] = sh("sysctl", "-n", "machdep.cpu.brand_string")
        mem = sh("sysctl", "-n", "hw.memsize")
        info["ram_gb"] = round(int(mem) / 2**30, 1) if mem.isdigit() else None
    return info


def run_info(cfg: Config, llm: OllamaClient | None) -> dict:
    info = {"time": datetime.now().isoformat(timespec="seconds"), "mode": "local", "model": cfg.model,
            "runtime": "ollama", "host": cfg.host, "internet_reachable": internet_reachable(),
            "device": device_info()}
    if llm is not None:
        try:
            info["runtime_version"] = llm.version()
        except LLMError:
            info["runtime_version"] = "unreachable"
    return info


def hit_dict(h: Hit, n: int | None = None) -> dict:
    d = {"location": h.passage.location(), "path": h.passage.path, "section": h.passage.section,
         "page": h.passage.page, "lines": [h.passage.start_line, h.passage.end_line],
         "score": h.score, "term_coverage": h.coverage, "matched_terms": h.matched, "text": h.passage.text}
    if n is not None:
        d = {"label": f"S{n}", **d}
    return d


def _write(path: Path, data: dict, markdown: str) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.with_suffix(".json").write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")
    path.with_suffix(".md").write_text(markdown, encoding="utf-8")
    return path.with_suffix(".md")


def save_ask(cfg: Config, info: dict, res: AskResult, memory: dict | None, test: dict | None = None,
             name: str | None = None) -> Path:
    retrieved_paths = [h.passage.path for h in res.hits]
    data = {"kind": "ask", **info, "question": res.question, "model_called": res.model_called,
            "retrieved": [hit_dict(h, i) for i, h in enumerate(res.hits, 1)], "answer": res.answer,
            "citation_check": res.check, "response_seconds": res.seconds,
            "prompt_tokens": res.prompt_tokens, "output_tokens": res.output_tokens, "memory": memory or {}}
    if test:
        expected = test.get("expected_sources", [])
        data["test"] = test
        data["auto_checks"] = {
            "expected_source_retrieved": any(e in retrieved_paths for e in expected) if expected else None,
            "expected_behavior": test.get("expect", "answer"),
            "behavior_matches": (res.check["insufficient"] == (test.get("expect") == "insufficient")),
        }
    md = [f"# Ask evidence card{': ' + test['id'] if test else ''}", "",
          f"- **Question:** {res.question}",
          f"- **Time:** {info['time']}  |  **Mode:** {info['mode']}  |  **Internet reachable:** {info['internet_reachable']}",
          f"- **Model:** `{info['model']}` via {info['runtime']} {info.get('runtime_version', '')}",
          f"- **Response time:** {res.seconds}s  |  tokens in/out: {res.prompt_tokens}/{res.output_tokens}"]
    if memory:
        md.append(f"- **Memory:** {json.dumps(memory)}")
    if test:
        md += ["", "## Expected (written before running)", "",
               f"- Behavior: {test.get('expect', 'answer')}",
               f"- Sources: {', '.join(test.get('expected_sources', [])) or '—'}",
               f"- Passage: {test.get('expected_passage', '—')}"]
    md += ["", "## Retrieved passages", ""]
    if not res.hits:
        md.append("_None above the score threshold; the model was not called._")
    for i, h in enumerate(res.hits, 1):
        md += [f"**[S{i}] {h.passage.location()}** (score {h.score}, term coverage {h.coverage:.0%})", "",
               "> " + h.passage.text.replace("\n", "\n> "), ""]
    md += ["## Gemma answer", "", res.answer, "", "## Automatic checks", "",
           f"- Citation check: `{res.check['status']}` — cited {res.check['cited'] or 'none'}"
           f"{', invalid ' + str(res.check['invalid']) if res.check['invalid'] else ''}"]
    if test:
        ac = data["auto_checks"]
        md += [f"- Expected source retrieved: {ac['expected_source_retrieved']}",
               f"- Behavior matches expectation ({ac['expected_behavior']}): {ac['behavior_matches']}"]
    md += ["", "## Human assessment", "",
           "_To fill in after opening each cited passage: does every material claim follow from it? "
           "Any invented details or wrong citations?_", ""]
    fname = name or f"ask-{stamp()}"
    return _write(cfg.evidence_dir / "ask" / fname, data, "\n".join(md))


def save_search(cfg: Config, info: dict, query: str, hits: list[Hit], scope: str) -> Path:
    data = {"kind": "search", **info, "query": query, "scope": scope, "results": [hit_dict(h) for h in hits]}
    md = [f"# Search: {query}", "", f"- {info['time']} | scope: {scope} | no model used | "
          f"internet reachable: {info['internet_reachable']}", ""]
    for i, h in enumerate(hits, 1):
        md += [f"**{i}. {h.passage.location()}** (score {h.score})", "", "> " + h.passage.text.replace("\n", "\n> "), ""]
    return _write(cfg.evidence_dir / "search" / f"search-{stamp()}", data, "\n".join(md))


def turn_dict(t: ChatTurn) -> dict:
    return {"user": t.user, "retrieved": t.retrieved, "routing_reason": t.reason,
            "passages": [hit_dict(h, i) for i, h in enumerate(t.hits, 1)], "reply": t.reply,
            "citation_check": t.check, "response_seconds": t.seconds}


def chat_markdown(turns: list[ChatTurn]) -> list[str]:
    md = []
    for i, t in enumerate(turns, 1):
        md += [f"### Turn {i}", "", f"**You:** {t.user}", "",
               f"*Harness: retrieval {'ON' if t.retrieved else 'off'} — {t.reason}*", ""]
        for j, h in enumerate(t.hits, 1):
            md.append(f"- [S{j}] {h.passage.location()}")
        if t.hits:
            md.append("")
        md += [f"**Assistant:** {t.reply}", "", f"*({t.seconds}s)*", ""]
    return md


def save_chat(cfg: Config, info: dict, turns: list[ChatTurn], subdir: str = "chat") -> Path:
    data = {"kind": "chat", **info, "turns": [turn_dict(t) for t in turns]}
    md = ["# Chat transcript", "", f"- {info['time']} | model `{info['model']}` | mode {info['mode']} | "
          f"internet reachable: {info['internet_reachable']}", ""] + chat_markdown(turns)
    return _write(cfg.evidence_dir / subdir / f"chat-{stamp()}", data, "\n".join(md))
