"""The harness: mode behavior, prompt assembly, retrieval decisions, citation checks.

  search -> retrieval tool only (no model)
  ask    -> RAG: retrieve -> evidence + research rules -> Gemma -> citation check (no chat history)
  chat   -> persona + recent conversation; retrieves only when the turn needs notes
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field

from .config import Config
from .ingest import load_prompt
from .llm import OllamaClient
from .retrieval import Hit, Index

INSUFFICIENT = "Insufficient evidence"

# Repeated after the question: small models follow the last instruction most reliably.
ASK_REMINDER = ("Answer only from SOURCES and cite each fact like [S1]. If SOURCES do not contain the answer, "
                "reply with one line starting exactly \"Insufficient evidence:\" followed by what is missing, "
                "with no citation.")

COMMANDS = """\
  wiki ingest [PATH] [--force]   read vault/raw/, write linked notes in vault/wiki/, rebuild the index
  wiki search "QUERY"            show original passages + paths (no model, no generated answer)
  wiki ask "QUESTION"            standalone, neutral answer from retrieved passages with citations
  wiki chat                      personal assistant with conversation memory for this session
  wiki test                      run the ask-mode question tests in eval/questions.json
  wiki modecheck                 run the scripted chat/search/ask boundary checks
  wiki status                    show model, runtime, memory use and offline state"""

CHAT_HELP = """\
  /notes QUERY   force a notes lookup for this turn      /sources  show passages used last turn
  /reset         forget this conversation                 /exit     save transcript and quit"""


def wiki_topics(cfg: Config) -> str:
    try:
        from .notes import NoteStore
        by_folder: dict = {}
        for n in NoteStore(cfg).notes.values():
            by_folder.setdefault(n["folder"], []).append(n["title"])
        return "; ".join(f"{f}: {', '.join(sorted(ts))}" for f, ts in sorted(by_folder.items())) or "no notes yet"
    except (OSError, ValueError):
        return "unknown (run wiki ingest)"


def capabilities(cfg: Config) -> str:
    return (
        f"You run fully offline on this computer using the local model {cfg.model} through Ollama.\n"
        f"The user's wiki currently covers these notes (folder: titles): {wiki_topics(cfg)}\n"
        "What you can actually do in chat: brainstorm, draft, plan, rewrite and follow up using this "
        "conversation; look up the user's personal wiki when a request needs their notes, and cite them.\n"
        "What you cannot do: browse the internet, read files you were not given, remember past sessions, "
        "save anything the user tells you, run tools or change the wiki.\n"
        "Other commands the user can run in the terminal:\n" + COMMANDS
    )


# --- evidence formatting --------------------------------------------------------
def evidence_block(hits: list[Hit], max_chars: int) -> tuple[str, list[Hit]]:
    parts, used, total = [], [], 0
    for h in hits:
        text = h.passage.text
        if total + len(text) > max_chars and used:
            break
        text = text[: max_chars - total] if total + len(text) > max_chars else text
        n = len(used) + 1
        parts.append(f"[S{n}] {h.passage.location()}\n{text}")
        used.append(h)
        total += len(text)
    return "\n\n".join(parts), used


def check_citations(answer: str, n_sources: int) -> dict:
    cited = sorted({int(x) for x in re.findall(r"\[S(\d+)\]", answer)})
    insufficient = answer.strip().lower().startswith(INSUFFICIENT.lower())
    invalid = [c for c in cited if c < 1 or c > n_sources]
    status = "ok"
    if invalid:
        status = "invalid-citation"
    elif not cited and not insufficient:
        status = "uncited-answer"
    return {"cited": cited, "invalid": invalid, "insufficient": insufficient, "status": status}


def expand_citations(text: str, used: list[Hit]) -> str:
    """Replace [S1] with the passage location (used when storing chat history)."""
    def sub(m):
        i = int(m.group(1))
        return f"[{used[i - 1].passage.location()}]" if 1 <= i <= len(used) else m.group(0)
    return re.sub(r"\[S(\d+)\]", sub, text)


# --- ask ----------------------------------------------------------------------------
@dataclass
class AskResult:
    question: str
    hits: list[Hit]
    answer: str
    check: dict
    model_called: bool
    seconds: float = 0.0
    prompt_tokens: int = 0
    output_tokens: int = 0


def ask(cfg: Config, llm: OllamaClient, index: Index, question: str, on_token=None) -> AskResult:
    hits = [h for h in index.search(question, k=cfg.top_k, kinds=("raw",)) if h.score >= cfg.min_score]
    if not hits:
        answer = f"{INSUFFICIENT}: no passage in the wiki's sources matches this question."
        return AskResult(question, [], answer, check_citations(answer, 0), model_called=False)
    block, used = evidence_block(hits, cfg.evidence_chars)
    messages = [
        {"role": "system", "content": load_prompt(cfg, "wiki-instructions.md")},
        {"role": "user", "content": f"SOURCES:\n{block}\n\nQUESTION: {question}\n\n{ASK_REMINDER}"},
    ]
    r = llm.chat(messages, temperature=0.1, on_token=on_token)
    return AskResult(question, used, r.text, check_citations(r.text, len(used)), True,
                     r.seconds, r.prompt_tokens, r.output_tokens)


# --- chat ---------------------------------------------------------------------------
META = re.compile(r"\b(what can (you|we) do|what can you help|help me with\??$|who are you|what are you|"
                  r"your (commands|capabilities)|how do (i|you) use)\b|^(hi|hello|hey|thanks|thank you)\b", re.I)
FOLLOW_UP = re.compile(r"\b(make (it|that|this) |shorter|longer|rewrite|rephrase|simplify|expand (it|that|on that)|"
                       r"another version|as bullets?|bullet points|more (formal|casual|concise)|"
                       r"summari[sz]e (that|it|this)|translate (it|that)|try again|redo)\b", re.I)
DRAFT_REQUEST = re.compile(r"\b(draft|plan|write|outline|checklist|prepare|help me|ideas? for|tips)\b", re.I)
NOTE_INTENT = re.compile(r"\b(my notes|my wiki|my sources|according to|in my (notes|files|sources|wiki)|"
                         r"from my|based on my|what did i (write|note|say)|look (it )?up|check my|cite)\b", re.I)


@dataclass
class ChatTurn:
    user: str
    reply: str
    retrieved: bool
    reason: str
    hits: list[Hit] = field(default_factory=list)
    check: dict = field(default_factory=dict)
    seconds: float = 0.0


class ChatSession:
    def __init__(self, cfg: Config, llm: OllamaClient, index: Index | None):
        self.cfg, self.llm, self.index = cfg, llm, index
        self.history: list[dict] = []
        self.turns: list[ChatTurn] = []
        self.system = load_prompt(cfg, "persona.md") + "\n\n" + capabilities(cfg)

    def route(self, message: str) -> tuple[bool, str, list[Hit]]:
        """Decide whether this turn needs the notes. Returns (retrieve, reason, hits)."""
        if message.startswith("/notes "):
            query = message[len("/notes "):]
            return True, "forced by /notes", self._search(query)
        if META.search(message):
            return False, "question about the assistant itself", []
        if FOLLOW_UP.search(message) and self.history:
            return False, "follow-up on the conversation", []
        hits = self._search(message)
        if NOTE_INTENT.search(message):
            return bool(hits), "user asked about their notes" if hits else "asked about notes, none matched", hits
        strong = [h for h in hits if h.coverage >= self.cfg.chat_min_coverage and h.score >= self.cfg.min_score]
        if strong:
            return True, f"notes strongly match ({strong[0].coverage:.0%} of terms)", strong
        topical = [h for h in hits if len(h.matched) >= 2]
        if DRAFT_REQUEST.search(message) and topical:
            return True, f"drafting request on a topic in the notes ({', '.join(topical[0].matched)})", topical
        return False, "general conversation; no strong note match", []

    def _search(self, query: str) -> list[Hit]:
        if self.index is None:
            return []
        return [h for h in self.index.search(query, k=self.cfg.top_k, kinds=("raw",)) if h.score >= self.cfg.min_score]

    def send(self, message: str, on_token=None) -> ChatTurn:
        retrieve, reason, hits = self.route(message)
        text = message[len("/notes "):] if message.startswith("/notes ") else message
        used: list[Hit] = []
        content = text
        if retrieve:
            block, used = evidence_block(hits, self.cfg.evidence_chars)
            content = (f"{text}\n\n---\nNotes retrieved from the user's wiki for this turn. Cite any claim "
                       f"taken from them as [S1], [S2]... Do not claim anything about the user that is not "
                       f"in these notes or the conversation.\n{block}")
        elif NOTE_INTENT.search(message):
            content = (f"{text}\n\n---\n(The harness searched the wiki and found no matching notes. "
                       f"Say so rather than guessing.)")
        keep = self.cfg.chat_history_turns * 2
        messages = [{"role": "system", "content": self.system}] + self.history[-keep:] + \
                   [{"role": "user", "content": content}]
        r = self.llm.chat(messages, temperature=0.7, on_token=on_token)
        check = check_citations(r.text, len(used)) if used else {}
        self.history += [{"role": "user", "content": text},
                         {"role": "assistant", "content": expand_citations(r.text, used)}]
        turn = ChatTurn(message, r.text, bool(used), reason, used, check, r.seconds)
        self.turns.append(turn)
        return turn

    def reset(self) -> None:
        self.history.clear()
