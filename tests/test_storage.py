"""Tests for saving conversations to SQLite. They use a temporary database
and the fake Ollama server from test_chat, so they run without the ai-pc.

Run from the local-ai folder:  python -m unittest discover tests
"""

import sqlite3
import tempfile
import threading
import unittest
from http.server import HTTPServer
from pathlib import Path
from unittest import mock

from app.conversation import Conversation
from app.main import database_path, open_store
from app.ollama_client import ChatResult, OllamaClient, OllamaError
from storage.database import ConversationStore, make_title
from tests.test_chat import FakeOllama


class StoreTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        # A path in a folder that doesn't exist yet, like data/databases/ on first run.
        self.path = Path(self.tmp.name) / "data" / "databases" / "test.db"
        self.store = ConversationStore(self.path)

    def tearDown(self):
        self.store.close()
        self.tmp.cleanup()

    def test_creates_folder_and_saves_a_turn(self):
        self.assertTrue(self.path.exists())
        cid = self.store.create_conversation("qwen3:4b", "Be brief.")
        result = ChatResult(content="Hello there", model="qwen3:4b", prompt_tokens=10,
                            completion_tokens=4, generation_seconds=2.0, total_seconds=3.0)
        self.store.add_turn(cid, "hi", result)

        saved = self.store.load(cid)
        self.assertEqual(saved.title, "hi")
        self.assertEqual(saved.system_prompt, "Be brief.")
        self.assertEqual(saved.messages, [{"role": "user", "content": "hi"},
                                          {"role": "assistant", "content": "Hello there"}])
        row = self.store.db.execute(
            "SELECT * FROM messages WHERE role = 'assistant'").fetchone()
        self.assertEqual(row["completion_tokens"], 4)
        self.assertAlmostEqual(row["tokens_per_second"], 2.0)

    def test_saved_data_survives_reopening(self):
        cid = self.store.create_conversation("qwen3:4b")
        self.store.add_turn(cid, "hi", ChatResult(content="yo", model="qwen3:4b"))
        self.store.close()
        self.store = ConversationStore(self.path)
        self.assertEqual(len(self.store.load(cid).messages), 2)

    def test_title_is_first_message_shortened(self):
        cid = self.store.create_conversation("qwen3:4b")
        self.store.add_turn(cid, "first\nmessage", ChatResult(content="a", model="m"))
        self.store.add_turn(cid, "second", ChatResult(content="b", model="m"))
        self.assertEqual(self.store.load(cid).title, "first message")
        long_title = make_title("word " * 50)
        self.assertLessEqual(len(long_title), 60)
        self.assertTrue(long_title.endswith("..."))

    def test_history_is_newest_first(self):
        first = self.store.create_conversation("qwen3:4b")
        second = self.store.create_conversation("qwen3:0.6b")
        self.store.add_turn(second, "two", ChatResult(content="b", model="qwen3:0.6b"))
        self.store.add_turn(first, "one", ChatResult(content="a", model="qwen3:4b"))
        ids = [c.id for c in self.store.list_conversations()]
        self.assertEqual(ids[0], first)
        self.assertEqual({c.id: c.message_count for c in self.store.list_conversations()},
                         {first: 2, second: 2})

    def test_load_unknown_id(self):
        self.assertIsNone(self.store.load(999))


class ConversationSavingTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.server = HTTPServer(("127.0.0.1", 0), FakeOllama)
        threading.Thread(target=cls.server.serve_forever, daemon=True).start()
        cls.client = OllamaClient(f"http://127.0.0.1:{cls.server.server_port}", timeout=5)

    @classmethod
    def tearDownClass(cls):
        cls.server.shutdown()

    def setUp(self):
        self.store = ConversationStore(":memory:")

    def tearDown(self):
        self.store.close()

    def test_replies_are_saved(self):
        convo = Conversation(self.client, "qwen3:4b", "Be brief.", store=self.store)
        convo.send("hi")
        convo.send("again")
        saved = self.store.load(convo.conversation_id)
        self.assertEqual([m["content"] for m in saved.messages],
                         ["hi", "Hello there", "again", "Hello there"])
        self.assertIsNone(convo.save_error)

    def test_failed_reply_saves_nothing(self):
        convo = Conversation(self.client, "missing", store=self.store)
        with self.assertRaises(OllamaError):
            convo.send("hi")
        self.assertIsNone(convo.conversation_id)
        self.assertEqual(self.store.list_conversations(), [])

    def test_thinking_is_not_saved(self):
        convo = Conversation(self.client, "qwen3:4b", think=True, store=self.store)
        convo.send("hi")
        contents = [r[0] for r in self.store.db.execute("SELECT content FROM messages")]
        self.assertNotIn("Let me think.", " ".join(contents))

    def test_reset_starts_a_new_saved_conversation(self):
        convo = Conversation(self.client, "qwen3:4b", store=self.store)
        convo.send("one")
        first = convo.conversation_id
        convo.reset()
        self.assertIsNone(convo.conversation_id)
        convo.send("two")
        self.assertNotEqual(convo.conversation_id, first)
        self.assertEqual(len(self.store.list_conversations()), 2)

    def test_load_and_continue(self):
        convo = Conversation(self.client, "qwen3:4b", "Old prompt.", store=self.store)
        convo.send("hi")
        cid = convo.conversation_id

        # A later session, with a different system prompt in the file.
        later = Conversation(self.client, "qwen3:4b", "New prompt.", store=self.store)
        self.assertTrue(later.load(cid))
        later.send("again")
        sent = FakeOllama.last_body["messages"]
        self.assertEqual([m["role"] for m in sent], ["system", "user", "assistant", "user"])
        self.assertEqual(sent[0]["content"], "Old prompt.")
        self.assertEqual(len(self.store.load(cid).messages), 4)
        self.assertFalse(later.load(999))

    def test_save_error_keeps_the_reply(self):
        convo = Conversation(self.client, "qwen3:4b", store=self.store)
        with mock.patch.object(self.store, "add_turn", side_effect=sqlite3.OperationalError("disk full")):
            result = convo.send("hi")
        self.assertEqual(result.content, "Hello there")
        self.assertIn("disk full", convo.save_error)
        self.assertEqual(len(convo.messages), 2)

    def test_without_a_store_nothing_is_saved(self):
        convo = Conversation(self.client, "qwen3:4b")
        convo.send("hi")
        self.assertIsNone(convo.conversation_id)
        self.assertFalse(convo.load(1))


class ConfigTests(unittest.TestCase):
    def test_relative_database_path_is_inside_project(self):
        path = database_path({"database": "data/databases/local_ai.db"})
        self.assertTrue(path.is_absolute())
        self.assertEqual(path.parts[-3:], ("data", "databases", "local_ai.db"))

    def test_saving_can_be_turned_off(self):
        self.assertIsNone(open_store({"save_conversations": False}, no_save=False))
        self.assertIsNone(open_store({}, no_save=True))


if __name__ == "__main__":
    unittest.main()
