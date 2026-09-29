"""Project paths and settings.

Settings come from (lowest to highest priority): defaults below, wiki.json in the
project root, environment variables, then CLI flags.
"""
from __future__ import annotations

import json
import os
from dataclasses import dataclass
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent

DEFAULTS = {
    "model": "gemma4:e2b-it-qat",          # Ollama tag; verify with `ollama list` after pulling
    "host": "http://127.0.0.1:11434",
    "num_ctx": 4096,                # tokens of context; largest prompt is ~2.5k tokens (9k-char ingest piece + rules)
    "top_k": 5,                     # passages retrieved per question
    "passage_words": 150,           # max words per retrieval passage
    "evidence_chars": 6000,         # max characters of passages sent to Gemma per turn
    "ingest_chars": 9000,           # max characters of source text sent per ingest call
    "min_score": 0.5,               # BM25 floor for ask-mode evidence
    "chat_min_coverage": 0.5,       # share of query terms a passage must match for chat to auto-retrieve
    "chat_history_turns": 6,        # user/assistant pairs kept as conversation context
    "think": False,                 # Gemma 4 thinking mode (slower; off by default)
    "folders": ["Concepts", "Projects", "People", "Courses", "Resources"],
}


@dataclass
class Config:
    root: Path
    model: str
    host: str
    num_ctx: int
    top_k: int
    passage_words: int
    evidence_chars: int
    ingest_chars: int
    min_score: float
    chat_min_coverage: float
    chat_history_turns: int
    think: bool
    folders: list

    # Vault (the only folder opened in Obsidian): originals + curated notes.
    @property
    def vault(self) -> Path:
        return self.root / "vault"

    @property
    def raw_dir(self) -> Path:
        return self.vault / "raw"

    @property
    def wiki_dir(self) -> Path:
        return self.vault / "wiki"

    @property
    def index_md(self) -> Path:
        return self.vault / "index.md"

    # Machine files live outside the vault.
    @property
    def data_dir(self) -> Path:
        return self.root / "data"

    @property
    def passages_file(self) -> Path:
        return self.data_dir / "passages.jsonl"

    @property
    def catalog_file(self) -> Path:
        return self.data_dir / "source_catalog.json"

    @property
    def notes_file(self) -> Path:
        return self.data_dir / "notes.json"

    @property
    def prompts_dir(self) -> Path:
        return self.root / "prompts"

    @property
    def evidence_dir(self) -> Path:
        return self.root / "evidence"

    @property
    def eval_dir(self) -> Path:
        return self.root / "eval"

    def rel(self, path: Path) -> str:
        """Path relative to the project root, for display and citations."""
        try:
            return str(Path(path).resolve().relative_to(self.root.resolve()))
        except ValueError:
            return str(path)

    def vault_rel(self, path: Path) -> str:
        """Path relative to the vault, as Obsidian resolves it."""
        return str(Path(path).resolve().relative_to(self.vault.resolve()))


def load_config(root: str | None = None, model: str | None = None, host: str | None = None) -> Config:
    root_path = Path(root or os.environ.get("WIKI_ROOT") or PROJECT_ROOT).resolve()
    values = dict(DEFAULTS)
    settings = root_path / "wiki.json"
    if settings.exists():
        values.update(json.loads(settings.read_text(encoding="utf-8")))
    values["model"] = model or os.environ.get("WIKI_MODEL") or values["model"]
    values["host"] = host or os.environ.get("OLLAMA_HOST_URL") or values["host"]
    return Config(root=root_path, **{k: values[k] for k in DEFAULTS})
