"""Harness tests with a fake model (no Ollama needed). Run: python3 -m unittest discover tests"""
from __future__ import annotations

import json
import shutil
import tempfile
import unittest
from pathlib import Path

from wikicli.config import PROJECT_ROOT, load_config
from wikicli.harness import ChatSession, ask, check_citations
from wikicli.ingest import ingest
from wikicli.llm import LLMResult
from wikicli.notes import clean_title
from wikicli.retrieval import Index

WORKSHOP = """# Workshop Plan

## Logistics
The workshop starts at 10 a.m. in Room 204. Bring a laptop with Python installed.

## Agenda
We will cover retrieval augmented generation and local models.
"""
GARDEN = """# Garden Notes

Tomatoes need six hours of sun. Water the basil every morning.
"""


class FakeLLM:
    model = "fake-gemma"

    def __init__(self):
        self.calls = []

    def check(self):
        pass

    def version(self):
        return "test"

    def memory_snapshot(self):
        return {}

    def chat(self, messages, temperature=0.2, json_mode=False, on_token=None):
        self.calls.append(messages)
        user = messages[-1]["content"]
        if json_mode:
            if "SOURCE TITLE: Workshop Plan" in user:
                topics = [
                    {"title": "workshop-plan: logistics 2026 notes for the class", "folder": "projects",
                     "summary": "The workshop starts at 10 a.m. in Room 204.",
                     "details": ["Starts 10 a.m.", "Room 204"],
                     "related": [{"title": "Retrieval Augmented Generation", "reason": "agenda topic"},
                                 {"title": "Nonexistent Note", "reason": "should be dropped"}]},
                    {"title": "Retrieval Augmented Generation", "folder": "Concepts",
                     "summary": "RAG is on the agenda.", "details": ["Covered at the workshop"],
                     "related": [{"title": "Workshop-Plan: Logistics 2026 Notes for", "reason": "where it is taught"}]},
                ]
            else:
                topics = [{"title": "Garden Care", "folder": "Resources", "summary": "Tomatoes need sun.",
                           "details": ["Six hours of sun"], "related": []}]
            text = json.dumps({"topics": topics})
        elif "SOURCES:" in user:
            text = "The workshop is in Room 204 [S1]."
        else:
            text = "Sure — here is a reply."
        if on_token:
            on_token(text)
        return LLMResult(text=text, seconds=0.01)


class HarnessTest(unittest.TestCase):
    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp())
        shutil.copytree(PROJECT_ROOT / "prompts", self.tmp / "prompts")
        (self.tmp / "vault" / "raw").mkdir(parents=True)
        (self.tmp / "vault" / "raw" / "workshop.md").write_text(WORKSHOP)
        (self.tmp / "vault" / "raw" / "garden.txt").write_text(GARDEN)
        self.cfg = load_config(str(self.tmp))
        self.llm = FakeLLM()
        self.log = []

    def tearDown(self):
        shutil.rmtree(self.tmp)

    def run_ingest(self, **kw):
        return ingest(self.cfg, self.llm, log=self.log.append, **kw)

    def wiki_files(self):
        return sorted(str(p.relative_to(self.cfg.wiki_dir)) for p in self.cfg.wiki_dir.rglob("*.md"))

    def test_ingest_readable_notes_links_and_index(self):
        self.run_ingest()
        files = self.wiki_files()
        self.assertIn("Concepts/Retrieval Augmented Generation.md", files)
        self.assertIn("Resources/Garden Care.md", files)
        self.assertTrue(any(f == "Projects/Workshop Plan Logistics Notes.md" for f in files), files)
        rag = (self.cfg.wiki_dir / "Concepts" / "Retrieval Augmented Generation.md").read_text()
        self.assertIn("# Retrieval Augmented Generation", rag)
        self.assertIn("[[raw/workshop.md|Workshop Plan]]", rag)
        self.assertIn("[[Workshop Plan Logistics Notes]]", rag)   # messy related title resolved
        ws = next(self.cfg.wiki_dir.glob("Projects/*.md")).read_text()
        self.assertNotIn("Nonexistent Note", ws)                            # broken links dropped
        index = self.cfg.index_md.read_text()
        self.assertIn("## Concepts", index)
        self.assertIn("[[raw/garden.txt|Garden Notes]]", index)
        catalog = json.loads(self.cfg.catalog_file.read_text())
        self.assertEqual(catalog["src-workshop"]["raw_path"], "vault/raw/workshop.md")
        self.assertEqual((self.cfg.raw_dir / "workshop.md").read_text(), WORKSHOP)  # original unchanged

    def test_reingest_skips_unchanged_and_never_duplicates(self):
        self.run_ingest()
        before = self.wiki_files()
        calls = len(self.llm.calls)
        self.run_ingest()
        self.assertEqual(len(self.llm.calls), calls)            # unchanged -> no model calls
        self.run_ingest(force=True)
        self.assertGreater(len(self.llm.calls), calls)
        self.assertEqual(self.wiki_files(), before)              # same notes, no duplicates

    def test_manual_edits_are_not_overwritten(self):
        self.run_ingest()
        note = self.cfg.wiki_dir / "Resources" / "Garden Care.md"
        note.write_text(note.read_text().replace("Tomatoes need sun.", "Tomatoes need 6h of sun (fixed)."))
        run = self.run_ingest(force=True)
        self.assertIn("(fixed)", note.read_text())
        self.assertIn("Garden Care", run["render"]["kept_manual_edits"])

    def test_search_needs_no_model(self):
        self.run_ingest()
        hits = Index.load(self.cfg.passages_file).search("workshop room", k=3)
        self.assertEqual(hits[0].passage.path, "vault/raw/workshop.md")
        self.assertEqual(hits[0].passage.section, "Logistics")
        self.assertIn("Room 204", hits[0].passage.text)

    def test_ask_cites_and_refuses_without_evidence(self):
        self.run_ingest()
        index = Index.load(self.cfg.passages_file)
        res = ask(self.cfg, self.llm, index, "Where is the workshop room?")
        self.assertTrue(res.model_called)
        self.assertEqual(res.check["status"], "ok")
        messages = self.llm.calls[-1]
        self.assertEqual([m["role"] for m in messages], ["system", "user"])   # no chat history in ask
        self.assertIn("Research rules", messages[0]["content"])
        n = len(self.llm.calls)
        res = ask(self.cfg, self.llm, index, "Who is catering the banquet?")
        self.assertFalse(res.model_called)
        self.assertEqual(len(self.llm.calls), n)
        self.assertTrue(res.check["insufficient"])

    def test_chat_routing(self):
        self.run_ingest()
        s = ChatSession(self.cfg, self.llm, Index.load(self.cfg.passages_file))
        self.assertFalse(s.send("what can you help me with?").retrieved)
        self.assertIn("Wren", self.llm.calls[-1][0]["content"])
        self.assertFalse(s.send("Draft a two-line plan for my week").retrieved)
        self.assertFalse(s.send("make that shorter").retrieved)
        self.assertEqual(len(self.llm.calls[-1]), 1 + 4 + 1)   # system + 2 prior exchanges + new message
        t = s.send("What do my notes say about the workshop room?")
        self.assertTrue(t.retrieved)
        self.assertIn("[S1]", self.llm.calls[-1][-1]["content"])

    def test_citation_check(self):
        self.assertEqual(check_citations("A [S1]. B [S4].", 2)["status"], "invalid-citation")
        self.assertEqual(check_citations("No cites here.", 2)["status"], "uncited-answer")
        self.assertTrue(check_citations("Insufficient evidence: nothing.", 0)["insufficient"])

    def test_similar_titles_merge_into_one_note(self):
        from wikicli.notes import NoteStore
        store = NoteStore(self.cfg)
        t = {"folder": "Resume", "summary": "s", "details": ["d"], "related": []}
        a = store.add_topic("src-a", {**t, "title": "Resume Bullet Components"}, "m")
        b = store.add_topic("src-b", {**t, "title": "Resume Bullet Structure Elements"}, "m")
        c = store.add_topic("src-b", {**t, "title": "Resume Word Repetition Strategy"}, "m")
        self.assertEqual(a, "Resume Bullet Components")
        self.assertEqual(b, "Resume Bullet Components")           # merged: same subject
        self.assertEqual(c, "Resume Word Repetition Strategy")    # distinct subject kept apart
        self.assertEqual(sorted(store.notes["resume bullet components"]["contributions"]), ["src-a", "src-b"])

    def test_clean_title(self):
        self.assertEqual(clean_title("developing a successful job search"), "Developing a Successful Job Search")
        self.assertEqual(clean_title("gpu parallel training"), "Gpu Parallel Training")
        self.assertEqual(clean_title("GPU Parallel Training"), "GPU Parallel Training")
        self.assertEqual(clean_title("notes: the [[plan]] for / week"), "Notes the Plan for Week")
        self.assertIsNone(clean_title("  ###  "))


if __name__ == "__main__":
    unittest.main()
