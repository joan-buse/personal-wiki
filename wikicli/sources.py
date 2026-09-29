"""Reading original sources from vault/raw/ and splitting them into passages.

Originals are only ever read, never modified.
"""
from __future__ import annotations

import hashlib
import re
import shutil
import subprocess
from dataclasses import asdict, dataclass
from pathlib import Path

TEXT_EXTS = {".md", ".markdown", ".txt"}
PDF_EXTS = {".pdf"}
SUPPORTED = TEXT_EXTS | PDF_EXTS


@dataclass
class Passage:
    id: str
    kind: str            # "raw" (original source) or "wiki" (generated note)
    source_id: str
    path: str            # relative to project root
    section: str
    page: int | None
    start_line: int
    end_line: int
    text: str

    def location(self) -> str:
        loc = self.path
        if self.page:
            loc += f" p.{self.page}"
        if self.section:
            loc += f" §{self.section}"
        return loc + f" (lines {self.start_line}-{self.end_line})"

    def to_dict(self) -> dict:
        return asdict(self)


def list_sources(raw_dir: Path) -> list[Path]:
    files = []
    for p in sorted(raw_dir.rglob("*")):
        rel_parts = p.relative_to(raw_dir).parts
        if any(part.startswith(".") for part in rel_parts) or p.is_symlink() or not p.is_file():
            continue
        if p.suffix.lower() in SUPPORTED:
            files.append(p)
    return files


def source_id(raw_dir: Path, path: Path) -> str:
    """Stable machine ID derived from the path inside vault/raw/."""
    rel = str(path.relative_to(raw_dir).with_suffix("")).lower()
    return "src-" + re.sub(r"[^a-z0-9]+", "-", rel).strip("-")


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read_pages(path: Path) -> list[tuple[int | None, str]]:
    """Return [(page_number_or_None, text)]."""
    if path.suffix.lower() in TEXT_EXTS:
        try:
            return [(None, path.read_text(encoding="utf-8"))]
        except UnicodeDecodeError as e:
            raise ValueError(f"{path.name} is not UTF-8 text: {e}") from e
    if shutil.which("pdftotext") is None:
        raise ValueError(f"{path.name}: PDF support needs the local `pdftotext` tool (brew install poppler).")
    out = subprocess.run(["pdftotext", "-enc", "UTF-8", "-layout", str(path), "-"],
                         capture_output=True, text=True)
    if out.returncode != 0:
        raise ValueError(f"{path.name}: pdftotext failed: {out.stderr.strip()}")
    pages = out.stdout.split("\f")
    result = [(i + 1, t) for i, t in enumerate(pages) if t.strip()]
    if not result:
        raise ValueError(f"{path.name}: no extractable text (scanned PDF? run OCR first).")
    return result


def source_title(path: Path, pages: list[tuple[int | None, str]]) -> str:
    for _, text in pages[:1]:
        for line in text.splitlines():
            m = re.match(r"^#\s+(.+)", line.strip())
            if m:
                return m.group(1).strip()
    return re.sub(r"\s+", " ", path.stem.replace("_", " ")).strip()


_HEADING = re.compile(r"^(#{1,6})\s+(.+?)\s*#*\s*$")
_SENTENCE = re.compile(r"(?<=[.!?])\s+")


_BULLET = re.compile(r"^[●○■•▪◦\-*–]|^\d+[.)]\s")


def _pdf_heading(line: str) -> bool:
    """Heuristic for PDF text (pdftotext -layout): a short, unindented, non-bullet line
    that starts with a capital and does not end like a sentence, e.g. 'Positioning'."""
    s = line.strip()
    if not s or line[:1].isspace() or _BULLET.match(s):
        return False
    return (len(s.split()) <= 10 and s[0].isupper() and not s.endswith((".", ",", ";", ":", "?", "!", "\""))
            and "  " not in s)


def chunk(pages: list[tuple[int | None, str]], sid: str, rel_path: str, max_words: int,
          kind: str = "raw") -> list[Passage]:
    """Split text into passages of <= max_words, breaking at headings and paragraphs."""
    passages: list[Passage] = []

    def emit(buf: list[tuple[int, str]], section: str, page):
        text = "\n".join(t for _, t in buf).strip()
        if not text:
            return
        passages.append(Passage(
            id=f"{sid}#{len(passages) + 1}", kind=kind, source_id=sid, path=rel_path,
            section=section, page=page, start_line=buf[0][0], end_line=buf[-1][0], text=text,
        ))

    section = ""   # carries across PDF pages until the next heading
    for page, text in pages:
        buf: list[tuple[int, str]] = []
        words = 0
        for lineno, line in enumerate(text.splitlines(), start=1):
            h = _HEADING.match(line.strip())
            heading = h.group(2).strip() if h else (line.strip() if page and _pdf_heading(line) else None)
            if heading:
                emit(buf, section, page)
                buf, words = [], 0
                section = heading
                continue
            n = len(line.split())
            if n == 0:
                # Paragraph break: flush if the passage is already reasonably sized.
                if words >= max_words * 0.6:
                    emit(buf, section, page)
                    buf, words = [], 0
                continue
            if words + n > max_words and buf:
                emit(buf, section, page)
                buf, words = [], 0
            if n > max_words:
                # A single very long line (common in PDFs): split by sentences.
                piece: list[str] = []
                for sent in _SENTENCE.split(line):
                    if piece and len(" ".join(piece + [sent]).split()) > max_words:
                        emit([(lineno, " ".join(piece))], section, page)
                        piece = []
                    piece.append(sent)
                buf, words = [(lineno, " ".join(piece))], len(" ".join(piece).split())
                continue
            buf.append((lineno, line.rstrip()))
            words += n
        emit(buf, section, page)
    return _merge_short_tails(passages, max_words)


def _merge_short_tails(passages: list[Passage], max_words: int) -> list[Passage]:
    """Fold a short leftover passage (< 1/3 of max_words) into the previous one from the same
    page and section, so a list is not cut off one item before its end."""
    out: list[Passage] = []
    for p in passages:
        prev = out[-1] if out else None
        if (prev and len(p.text.split()) < max_words / 3 and prev.page == p.page
                and prev.section == p.section and prev.source_id == p.source_id):
            prev.text += "\n" + p.text
            prev.end_line = p.end_line
            continue
        out.append(p)
    for i, p in enumerate(out, 1):
        p.id = f"{p.source_id}#{i}"
    return out
