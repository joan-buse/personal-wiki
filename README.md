# Personal Wiki CLI — local Gemma + RAG

> **Draft.** Every `TODO` must be replaced with real, measured results. Nothing here is a claimed result yet.

A command-line personal wiki that runs fully offline. Original notes in `vault/raw/` are turned into
linked Obsidian notes by a local Gemma model. You can then **chat** with a personal assistant,
**ask** grounded questions with citations, or **search** the original passages. The harness in
`wikicli/` is my own code. It uses only the Python standard library and calls Gemma through the
local Ollama HTTP API.

## Purpose and sources

A study wiki for my job search: my notes from the Haas career workshops (Aug 2025) on resumes, LinkedIn and networking, so I can look up advice quickly (e.g. how to structure a resume bullet or a cold outreach message) with citations back to the notes.

| Source (`vault/raw/`) | What it is | Permission to share |
|---|---|---|
| `Aug 2025 - Linkedin Training .pdf` (4 pp.) | My notes: LinkedIn exploration, positioning, job hunt, networking, research | TODO: confirm |
| `Aug 2025 - Resume & Networking Training.pdf` (6 pp.) | My notes: resume sessions 1–3, job-search process, networking | TODO: confirm |
| `Aug 2025 - Resume Only Workshop.pdf` (2 pp.) | My notes: resume workshops 1–2 (largely overlaps sessions 1–3 above) | TODO: confirm |

All three are Google Docs PDF exports with extractable text on every page (checked with `pdftotext`; no OCR needed, no warnings). Headings such as "Positioning" or "Session #2 - Resume Format" are detected so citations carry page and section.

Each original stays byte-for-byte unchanged. `data/source_catalog.json` maps each source ID
(e.g. `src-workshop`) to its file, its SHA-256 hash, and the wiki notes generated from it. Every wiki
note links back to its source(s) in its **Sources** section and in its `sources:` frontmatter.

## Setup and device

| | |
|---|---|
| Device | MacBook Air (M1, 2020), MacBookAir10,1 |
| OS | macOS 26.6.2 |
| CPU/GPU | Apple M1, 8 cores (4 performance + 4 efficiency), 8-core integrated GPU |
| Memory | 8 GB unified memory (shared by CPU and GPU; no dedicated VRAM) |
| Free disk | ~398 GB |
| Python | 3.13.15 (standard library only) |
| Runtime | Ollama TODO: version (`ollama --version`) |
| Model | `gemma4:e2b-it-qat` (Gemma 4 E2B instruction-tuned, 4-bit quantization-aware training; 4.3 GB download per Ollama library). TODO: digest from `ollama show` |

**Why this model.** The machine has 8 GB of unified memory, which the OS, Ollama, the context
window (KV cache) and other apps all share. Gemma E2B at 4-bit needs about 2.9 GB to load (official estimate for the weights). In Ollama the default `gemma4:e2b` tag is a 7.2 GB
download, too close to 8 GB, so I use the 4.3 GB QAT tag and a 4,096-token context. E4B (~4.5 GB) would be tight, and 26B A4B MoE (~14.4 GB) does not fit. MoE loads all
26B weights, even though only about 4B are active per token. TODO: confirm with the measurements below.

| Measurement | Value |
|---|---|
| Model memory reported by Ollama (`wiki status`) | TODO |
| Ollama process RSS during an answer | TODO |
| Ingest time (N sources) | TODO |
| Ask response time (T1–T4) | TODO |

### Install (once, while online)

```bash
brew install ollama        # or download the Ollama app
ollama serve               # leave running (the app does this automatically)
ollama pull gemma4:e2b-it-qat
./wiki status              # confirms the runtime and model are available locally
```

## Commands

```bash
./wiki --help
./wiki ingest                         # all of vault/raw/ (unchanged sources are skipped)
./wiki ingest vault/raw/notes.md --force
./wiki search "workshop location"     # original passages + paths; no model needed
./wiki ask "Where is the workshop?" --save
./wiki chat                           # /notes QUERY, /sources, /reset, /exit
./wiki test                           # 4 ask-mode tests -> evidence/ask/
./wiki modecheck                      # chat/search/ask boundary checks -> evidence/mode_checks/
python3 -m unittest discover tests    # harness tests with a fake model
```

## Architecture

- **Model:** local Gemma via Ollama. It only sees the messages the harness sends. It does not read files, remember sessions or run tools.
- **Retrieval tool** (`wikicli/retrieval.py`): BM25 keyword search over passages in `data/passages.jsonl`. It returns original text with its path, page, section and line range. `wiki search` shows this output directly.
- **RAG workflow** (`harness.ask`): retrieve the top passages (above a score floor), label them `[S1]…`, add the research rules from `prompts/wiki-instructions.md`, call Gemma, then check that every citation points to a passage that was actually retrieved.
- **Harness** (`wikicli/`): mode selection, instructions per mode, conversation context, the decision to retrieve, prompt assembly, model calls, citation checks, error messages and saved evidence.
- **CLI** (`wikicli/cli.py`): parses the command and dispatches it to the harness.

**One path traced, `./wiki ask "Where is the workshop?"`:**
1. `cli.main` parses `ask` and loads the config (`wiki.json`).
2. `cmd_ask` loads the passage index and checks that the model is downloaded.
3. `harness.ask` calls `Index.search`, which returns scored passages with their locations.
   - If none reach `min_score`, it returns "Insufficient evidence" without calling the model.
4. `evidence_block` labels the passages `[S1]…` and caps them at `evidence_chars`.
5. The messages are `[system: research rules, user: SOURCES + QUESTION]`, with no chat history. `OllamaClient.chat` sends them to `http://127.0.0.1:11434/api/chat` and streams the tokens back.
6. `check_citations` flags any invalid or missing citations. The CLI prints the answer, the source list and the timing, and `--save` writes an evidence card.

| Mode | Instructions | History | Retrieval | Temperature |
|---|---|---|---|---|
| chat | `prompts/persona.md` + a real capability list | last 6 exchanges | only when needed (see below) | 0.7 |
| ask | `prompts/wiki-instructions.md` | none | always | 0.1 |
| search | – | none | always; no model | – |

**When chat retrieves notes:**
- Never for questions about the assistant itself, or for follow-ups like "make that shorter".
- Always for `/notes QUERY`, or when the message refers to the user's notes.
- Also for drafting requests ("draft", "plan", "write", …) when a passage matches at least 2 topic words.
- Otherwise, only if a passage matches at least 50% of the query terms.

Chat history is stored with citations expanded to file locations. It is never used as evidence in ask mode.

## Design choices

- **Passages:** at most 150 words each. They break at Markdown headings and paragraphs, and PDFs split per page (`pdftotext`, which runs locally). Each passage keeps its path, page, section and line range. At most 6,000 characters of evidence go to Gemma per turn, within `num_ctx` 4096 (largest prompt ≈ 2.5k tokens).
- **Ingest:** each new or changed source (detected by SHA-256) goes to Gemma in pieces of at most 9,000 characters, along with `prompts/ingest-instructions.md` and the existing note titles. Gemma returns JSON topics. The harness then:
  - cleans each title into 2–6 words with no dates, hashes or punctuation;
  - files it in a topic folder (`Resume/`, `LinkedIn/`, `Networking/`, `Job Search/`);
  - keeps only `[[links]]` that point to notes that exist, each with a reason.
- **Merging and linking:** Gemma sees every existing note's title and summary, so it can reuse a title for the same subject. The harness also merges near-identical titles (≥2 shared content words covering ⅔ of the shorter title). A second, linking pass asks Gemma for up to 3 genuinely related notes per note, each with a reason (`prompts/link-instructions.md`). Links to notes that don't exist are dropped.
- **Model settings:** Gemma 4's thinking mode is off (`think: false`) for speed on the M1. The temperature is 0.1 for ask and linking, 0.2 for ingest and 0.7 for chat. Ask repeats the refusal rule right after the question (see Reflection).
- **No duplicates on re-ingest:** a note's filename is its title, and a source's old contributions are replaced rather than appended. Previous titles are passed back to Gemma so names stay stable. Machine IDs live only in frontmatter and in `data/`.
- **Manual corrections are safe:** a note edited by hand is never overwritten. The regenerated version goes to `data/pending/` for review.
- **Vault vs machine files:** only `vault/` is opened in Obsidian. Code, passages, the catalog, evidence and tests live outside it.

## Evidence

TODO: fill in after the offline run.
- Obsidian screenshots: an open note, `index.md`, and the graph view (filter `path:wiki/`, attachments off).
- The ingest log in `evidence/ingest/`, including a re-ingest that shows no duplicates.
- The four ask cards and a summary in `evidence/ask/`.
- The mode checks in `evidence/mode_checks/`.
- An offline recording or screenshots, with Wi-Fi off and the CLI restarted.

## Changes made after observed failures (online dry run, 2026-09-28)

Earlier results are kept in `evidence/first-ingest-backup/` and `evidence/online-dry-run/`.

| Observed | Cause | Change | Result |
|---|---|---|---|
| First ingest made near-duplicate notes ("Resume Bullet Components" / "Resume Bullet Structure Elements") | The Resume Only PDF repeats Sessions 1–3; Gemma saw only titles | Show summaries, merge similar titles, add linking pass | 17 notes, no duplicates; all 17 have related links (47 total) |
| Only 5/18 notes had related links | Links were proposed only while reading one source | Linking pass across all notes | see above |
| T4 answered "sources do not state…" with a bogus `[S1]` citation, not "Insufficient evidence:" | A 2B model drifts from system-prompt formatting | Repeat the refusal rule after the question | T4 returns `Insufficient evidence: …`, no citation; T2 still answered |
| Chat LinkedIn plan was generic and uncited | Drafting requests are mostly filler words, so term coverage < 50% | Drafting requests retrieve if the top passage matches ≥2 topic words | Plan built from notes with citations |
| Chat said "I have noted that" after a chat-only claim | Persona didn't state it cannot save | Persona: say it will keep it in mind for this conversation only | Fixed |
| T1's third sentence was a separate 22-word passage | Chunk boundary | Merge short tails into the previous passage from the same section | T1 passage complete |

## Reflection

TODO: one real failure or limitation you observed, its cause, and one concrete improvement.
