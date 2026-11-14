"""Tests for knowledge retrieval: reading files, chunking, indexing and search.

Most tests use a fake embedding model that counts words, so texts sharing words
come out similar. That's cruder than a real model, but it's predictable, and
it runs without the ai-pc.

Run from the local-ai folder:  python -m unittest discover tests
"""

import os
import re
import tempfile
import threading
import unittest
import zlib
from http.server import HTTPServer
from pathlib import Path

from app.conversation import Conversation
from app.ollama_client import OllamaClient, OllamaError
from retrieval.chunker import chunk_text
from retrieval.indexer import Indexer, KnowledgeSettings, location
from retrieval.readers import ReadError, read_document
from retrieval.retriever import Retriever, build_prompt
from retrieval.store import KnowledgeStore
from storage.database import ConversationStore
from tests.test_chat import FakeOllama

DIMENSIONS = 64


class WordCountEmbedder:
    """Stands in for OllamaClient.embed: one slot per word (by hash), counting uses."""

    def __init__(self):
        self.calls = []

    def embed(self, model, texts):
        self.calls.append((model, list(texts)))
        vectors = []
        for text in texts:
            text = re.sub(r"^search_(document|query): ", "", text)
            vector = [0.0] * DIMENSIONS
            for word in re.findall(r"[a-z]+", text.lower()):
                vector[zlib.crc32(word.encode()) % DIMENSIONS] += 1
            vectors.append(vector)
        return vectors

    def embedded_texts(self):
        return [t for _, texts in self.calls for t in texts]


def make_pdf(path: Path, pages: list[str]) -> None:
    """Write a minimal PDF with one line of text per page."""
    objects = ["<< /Type /Catalog /Pages 2 0 R >>", None,
               "<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>"]
    kids = []
    for text in pages:
        stream = f"BT /F1 12 Tf 72 720 Td ({text}) Tj ET"
        objects.append(f"<< /Length {len(stream)} >>\nstream\n{stream}\nendstream")
        objects.append(f"<< /Type /Page /Parent 2 0 R /MediaBox [0 0 612 792] "
                       f"/Resources << /Font << /F1 3 0 R >> >> /Contents {len(objects)} 0 R >>")
        kids.append(f"{len(objects)} 0 R")
    objects[1] = f"<< /Type /Pages /Kids [{' '.join(kids)}] /Count {len(kids)} >>"
    out = b"%PDF-1.4\n"
    offsets = []
    for number, body in enumerate(objects, start=1):
        offsets.append(len(out))
        out += f"{number} 0 obj\n{body}\nendobj\n".encode()
    xref = len(out)
    out += f"xref\n0 {len(objects) + 1}\n0000000000 65535 f \n".encode()
    out += "".join(f"{o:010d} 00000 n \n" for o in offsets).encode()
    out += f"trailer\n<< /Size {len(objects) + 1} /Root 1 0 R >>\nstartxref\n{xref}\n%%EOF\n".encode()
    path.write_bytes(out)


class ChunkerTests(unittest.TestCase):
    def test_short_text_is_one_chunk_with_line_numbers(self):
        [chunk] = chunk_text("Title\n\nFirst paragraph.\nStill first.\n", size=1000)
        self.assertEqual(chunk.text, "Title\n\nFirst paragraph.\nStill first.")
        self.assertEqual((chunk.start_line, chunk.end_line), (1, 4))

    def test_splits_on_paragraphs_and_stays_under_size(self):
        paragraphs = [f"Paragraph {i} " + "word " * 30 for i in range(10)]
        text = "\n\n".join(paragraphs)
        chunks = chunk_text(text, size=400, overlap=0)
        self.assertGreater(len(chunks), 1)
        for chunk in chunks:
            self.assertLessEqual(len(chunk.text), 400)
            # Every chunk starts at the beginning of a paragraph.
            self.assertTrue(chunk.text.startswith("Paragraph"))
        joined = "\n\n".join(c.text for c in chunks)
        self.assertEqual(joined, text.strip().replace(" \n", "\n").rstrip())

    def test_small_last_paragraph_overlaps_into_next_chunk(self):
        text = "A" * 300 + "\n\nshort bridge\n\n" + "B" * 300
        first, second = chunk_text(text, size=320, overlap=50)
        self.assertTrue(first.text.endswith("short bridge"))
        self.assertTrue(second.text.startswith("short bridge"))
        self.assertEqual(second.start_line, 3)

    def test_very_long_line_is_cut(self):
        chunks = chunk_text("x" * 2500, size=1000)
        self.assertEqual([len(c.text) for c in chunks], [1000, 1000, 500])
        self.assertTrue(all(c.start_line == 1 for c in chunks))

    def test_empty_text(self):
        self.assertEqual(chunk_text("\n\n  \n"), [])

    def test_location_labels(self):
        self.assertEqual(location(3, 9, "lines", None), "lines 3-9")
        self.assertEqual(location(4, 4, "lines", None), "line 4")
        self.assertEqual(location(2, 5, "paragraphs", None), "paragraphs 2-5")
        self.assertEqual(location(1, 3, "lines", 7), "page 7")


class ReaderTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.dir = Path(self.tmp.name)

    def tearDown(self):
        self.tmp.cleanup()

    def test_text_file(self):
        path = self.dir / "note.md"
        path.write_text("﻿# Heading\nBody", encoding="utf-8")
        [section] = read_document(path)
        self.assertEqual(section.text, "# Heading\nBody")

    def test_binary_file_is_refused(self):
        path = self.dir / "data.txt"
        path.write_bytes(b"abc\x00def")
        with self.assertRaises(ReadError):
            read_document(path)

    def test_pdf_pages(self):
        path = self.dir / "manual.pdf"
        make_pdf(path, ["Router setup instructions", "Firewall port rules"])
        sections = read_document(path)
        self.assertEqual([s.page for s in sections], [1, 2])
        self.assertIn("Firewall port rules", sections[1].text)

    def test_broken_pdf(self):
        path = self.dir / "broken.pdf"
        path.write_bytes(b"not really a pdf")
        with self.assertRaises(ReadError):
            read_document(path)

    def test_word_document_with_table(self):
        import docx
        document = docx.Document()
        document.add_paragraph("Backup plan")
        document.add_paragraph("Copy the database weekly.")
        table = document.add_table(rows=1, cols=2)
        table.rows[0].cells[0].text = "Drive"
        table.rows[0].cells[1].text = "D:"
        path = self.dir / "plan.docx"
        document.save(path)
        [section] = read_document(path)
        self.assertEqual(section.unit, "paragraphs")
        self.assertIn("Backup plan\nCopy the database weekly.", section.text)
        self.assertIn("Drive | D:", section.text)


class IndexerTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        root = Path(self.tmp.name)
        self.docs = root / "docs"
        self.docs.mkdir()
        (self.docs / "network.md").write_text(
            "The ai-pc listens on port 11434.\n\nThe firewall allows the primary pc only.\n")
        (self.docs / "garden.txt").write_text("Tomatoes need sun and water every day.\n")
        (self.docs / "photo.png").write_bytes(b"\x89PNG")  # not a supported type
        (self.docs / ".git").mkdir()
        (self.docs / ".git" / "secret.txt").write_text("hidden folders are skipped")
        self.settings = KnowledgeSettings(folders=[self.docs], database=root / "kb.db",
                                          document_prefix="search_document: ",
                                          query_prefix="search_query: ", min_score=0.1)
        self.store = KnowledgeStore(self.settings.database)
        self.embedder = WordCountEmbedder()

    def tearDown(self):
        self.store.close()
        self.tmp.cleanup()

    def index(self, **kwargs):
        return Indexer(self.embedder, self.store, self.settings).run(**kwargs)

    def test_indexes_supported_files_only(self):
        report = self.index()
        self.assertEqual(sorted(report.added), ["docs/garden.txt", "docs/network.md"])
        self.assertEqual(self.store.counts(), (2, 2))
        texts = self.embedder.embedded_texts()
        self.assertTrue(all(t.startswith("search_document: ") for t in texts))
        self.assertFalse(any("hidden" in t for t in texts))

    def test_search_finds_the_right_file(self):
        self.index()
        hits = Retriever(self.embedder, self.store, self.settings).search(
            "Which port does the ai-pc listen on?")
        self.assertEqual(hits[0].display, "docs/network.md")
        self.assertEqual(hits[0].location, "lines 1-3")
        self.assertEqual(self.embedder.calls[-1][1][0][:14], "search_query: ")

    def test_min_score_filters_weak_matches(self):
        self.index()
        self.settings.min_score = 0.99
        hits = Retriever(self.embedder, self.store, self.settings).search("tomatoes port")
        self.assertEqual(hits, [])

    def test_second_run_skips_unchanged_files(self):
        self.index()
        self.embedder.calls.clear()
        report = self.index()
        self.assertEqual(report.unchanged, 2)
        self.assertEqual(self.embedder.calls, [])

    def test_changed_and_deleted_files(self):
        self.index()
        path = self.docs / "garden.txt"
        path.write_text("Peppers like warm weather.\n")
        os.utime(path, (1, 1))  # a different modified time
        (self.docs / "network.md").unlink()
        report = self.index()
        self.assertEqual(report.updated, ["docs/garden.txt"])
        self.assertEqual(len(report.removed), 1)
        hits = self.store.search(self.embedder.embed("m", ["peppers weather"])[0])
        self.assertEqual([h.display for h in hits], ["docs/garden.txt"])

    def test_touched_but_unchanged_file_is_not_re_embedded(self):
        self.index()
        os.utime(self.docs / "garden.txt", (1, 1))
        self.embedder.calls.clear()
        report = self.index()
        self.assertEqual(report.unchanged, 2)
        self.assertEqual(self.embedder.calls, [])

    def test_missing_folder_keeps_its_entries(self):
        self.index()
        self.docs.rename(self.docs.parent / "elsewhere")  # like an unplugged drive
        report = self.index()
        self.assertEqual(report.missing_folders, [str(self.docs)])
        self.assertEqual(report.removed, [])
        self.assertEqual(self.store.counts(), (2, 2))

    def test_folder_removed_from_config_is_dropped(self):
        self.index()
        other = self.docs.parent / "other"
        other.mkdir()
        self.settings.folders = [other]
        report = self.index()
        self.assertEqual(len(report.removed), 2)
        self.assertEqual(self.store.counts(), (0, 0))

    def test_changing_embedding_model_rebuilds(self):
        self.index()
        self.settings.embedding_model = "other-model"
        self.embedder.calls.clear()
        report = self.index()
        self.assertEqual(len(report.added), 2)
        self.assertEqual(self.store.get_setting("embedding_model"), "other-model")

    def test_large_and_unreadable_files_are_reported(self):
        (self.docs / "big.txt").write_text("x" * 2048)
        self.settings.max_file_mb = 1 / 1024  # 1 KB
        (self.docs / "scan.pdf").write_bytes(b"not a pdf")
        report = self.index()
        skipped = dict(report.skipped)
        self.assertIn("docs/big.txt", skipped)
        self.assertIn("docs/scan.pdf", skipped)

    def test_pdf_chunks_point_to_pages(self):
        make_pdf(self.docs / "manual.pdf", ["Router setup", "Firewall port rules"])
        self.index()
        hits = self.store.search(self.embedder.embed("m", ["firewall rules"])[0], top_k=1)
        self.assertEqual(hits[0].reference, "docs/manual.pdf (page 2)")

    def test_settings_from_config(self):
        s = KnowledgeSettings.from_config({"knowledge": {"folders": ["knowledge"],
                                                         "extensions": ["md", ".PDF"]}})
        self.assertEqual(s.extensions, {".md", ".pdf"})
        self.assertEqual(s.query_prefix, "search_query: ")
        s = KnowledgeSettings.from_config({"knowledge": {"embedding_model": "all-minilm"}})
        self.assertEqual((s.document_prefix, s.query_prefix), ("", ""))
        s = KnowledgeSettings.from_config({"knowledge": {"embedding_model": "qwen3-embedding:0.6b"}})
        self.assertEqual(s.document_prefix, "")
        self.assertTrue(s.query_prefix.startswith("Instruct: "))


class ConversationWithKnowledgeTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.server = HTTPServer(("127.0.0.1", 0), FakeOllama)
        threading.Thread(target=cls.server.serve_forever, daemon=True).start()
        cls.client = OllamaClient(f"http://127.0.0.1:{cls.server.server_port}", timeout=5)

    @classmethod
    def tearDownClass(cls):
        cls.server.shutdown()

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        docs = Path(self.tmp.name) / "docs"
        docs.mkdir()
        (docs / "network.md").write_text("The ai-pc listens on port 11434.\n")
        self.settings = KnowledgeSettings(folders=[docs], database=Path(self.tmp.name) / "kb.db",
                                          min_score=0.1)
        self.knowledge = KnowledgeStore(self.settings.database)
        self.embedder = WordCountEmbedder()
        Indexer(self.embedder, self.knowledge, self.settings).run()
        self.retriever = Retriever(self.embedder, self.knowledge, self.settings)

    def tearDown(self):
        self.knowledge.close()
        self.tmp.cleanup()

    def test_client_embed_over_http(self):
        self.assertEqual(self.client.embed("nomic-embed-text", ["ab", "abcd"]),
                         [[2, 1.0, 0.0], [4, 1.0, 0.0]])
        self.assertEqual(self.client.embed("nomic-embed-text", []), [])
        with self.assertRaises(OllamaError) as ctx:
            self.client.embed("missing", ["hi"])
        self.assertIn("not found", str(ctx.exception))

    def test_sources_sent_once_and_question_kept_in_history(self):
        convo = Conversation(self.client, "qwen3:4b", retriever=self.retriever)
        convo.send("What port does the ai-pc use?")
        sent = FakeOllama.last_body["messages"][-1]["content"]
        self.assertIn("[1] docs/network.md (line 1)", sent)
        self.assertIn("Question: What port does the ai-pc use?", sent)
        self.assertEqual(convo.messages[0]["content"], "What port does the ai-pc use?")
        self.assertEqual([h.display for h in convo.last_sources], ["docs/network.md"])

        convo.send("thanks")
        first_question = FakeOllama.last_body["messages"][0]["content"]
        self.assertEqual(first_question, "What port does the ai-pc use?")

    def test_kb_off_sends_plain_question(self):
        convo = Conversation(self.client, "qwen3:4b", retriever=self.retriever)
        convo.use_knowledge = False
        convo.send("What port does the ai-pc use?")
        self.assertEqual(FakeOllama.last_body["messages"][-1]["content"],
                         "What port does the ai-pc use?")
        self.assertEqual(convo.last_sources, [])

    def test_failed_lookup_still_answers(self):
        self.settings.embedding_model = "missing"
        retriever = Retriever(self.client, self.knowledge, self.settings)
        convo = Conversation(self.client, "qwen3:4b", retriever=retriever)
        result = convo.send("hi")
        self.assertEqual(result.content, "Hello there")
        self.assertIn("not found", convo.knowledge_error)

    def test_sources_are_saved_with_the_reply(self):
        store = ConversationStore(":memory:")
        convo = Conversation(self.client, "qwen3:4b", store=store, retriever=self.retriever)
        convo.send("What port does the ai-pc use?")
        rows = store.db.execute("SELECT rank, file, location FROM message_sources").fetchall()
        self.assertEqual([tuple(r) for r in rows], [(1, "docs/network.md", "line 1")])
        store.close()

    def test_build_prompt_without_hits_is_just_the_question(self):
        self.assertEqual(build_prompt("hello", []), "hello")


if __name__ == "__main__":
    unittest.main()
