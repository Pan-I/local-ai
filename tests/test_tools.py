"""Tests for the model's tools (agent/) and the tool loop in Conversation.

Run from the local-ai folder:  python -m unittest discover tests
"""

import json
import os
import tempfile
import threading
import unittest
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path

from agent.folders import AllowedFolders, ToolError
from agent.tools import Toolbox, ToolSettings, describe_call
from app.conversation import Conversation
from app.ollama_client import ChatResult, OllamaClient, OllamaError, ToolCall, parse_tool_call
from retrieval.store import Hit
from storage.database import ConversationStore


def call(name, **arguments):
    return ToolCall(name, arguments, {"function": {"name": name, "arguments": arguments}})


class ScriptedClient:
    """Stands in for OllamaClient: replies with the next scripted result."""

    def __init__(self, *replies, supports_tools=True):
        self.replies = list(replies)
        self.requests = []
        self.supports_tools = supports_tools

    def chat(self, model, messages, stream=True, on_token=None, think=None, tools=None):
        self.requests.append({"messages": [dict(m) for m in messages], "tools": tools})
        if tools and not self.supports_tools:
            raise OllamaError(f'Ollama returned HTTP 400: registry.ollama.ai/library/{model} '
                              'does not support tools')
        reply = self.replies.pop(0)
        if isinstance(reply, str):
            reply = ChatResult(content=reply, model=model, completion_tokens=5,
                               total_seconds=1.0)
        if on_token and reply.content:
            on_token(reply.content)
        return reply


def asks_for(*calls):
    return ChatResult(content="", model="qwen3:4b", tool_calls=list(calls),
                      completion_tokens=3, total_seconds=2.0)


class Workspace(unittest.TestCase):
    """A temporary app folder with a knowledge folder and a few files."""

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.base = Path(self.tmp.name).resolve()
        (self.base / "knowledge" / "notes").mkdir(parents=True)
        (self.base / "knowledge" / "todo.md").write_text("line 1\nline 2\nline 3\n", encoding="utf-8")
        (self.base / "knowledge" / "notes" / "a.txt").write_text("hello", encoding="utf-8")
        (self.base / "knowledge" / ".secret").write_text("hidden", encoding="utf-8")
        (self.base / "knowledge" / "data.db").write_bytes(b"SQLite")
        (self.base / "knowledge" / "image.txt").write_bytes(b"\x00\x01binary")
        (self.base / "outside.txt").write_text("not allowed", encoding="utf-8")
        self.toolbox = self.make_toolbox()

    def tearDown(self):
        self.tmp.cleanup()

    def make_toolbox(self, **settings):
        s = ToolSettings(folders=("knowledge",), blocked=("*.db",), **settings)
        box = Toolbox(s)
        box.folders = AllowedFolders([Path(f) for f in s.folders], s.blocked, base=self.base)
        return box


class AllowedFolderTests(Workspace):
    def test_inside_is_allowed(self):
        folders = self.toolbox.folders
        self.assertEqual(folders.check("knowledge/todo.md"), self.base / "knowledge" / "todo.md")
        self.assertEqual(folders.check(str(self.base / "knowledge")), self.base / "knowledge")

    def test_outside_and_tricks_are_refused(self):
        for path in ["outside.txt", "knowledge/../outside.txt", str(self.base / "outside.txt"),
                     "/etc/passwd", "C:/Windows/win.ini", "..", ""]:
            with self.subTest(path=path), self.assertRaises(ToolError):
                self.toolbox.folders.check(path)

    def test_hidden_and_blocked_are_refused(self):
        for path in ["knowledge/.secret", "knowledge/data.db"]:
            with self.subTest(path=path), self.assertRaises(ToolError) as ctx:
                self.toolbox.folders.check(path)
            self.assertIn("blocked", str(ctx.exception))

    def test_shortcut_pointing_outside_is_refused(self):
        link = self.base / "knowledge" / "escape"
        try:
            os.symlink(self.base, link, target_is_directory=True)
        except (OSError, NotImplementedError):
            self.skipTest("can't create symlinks here (Windows needs developer mode)")
        with self.assertRaises(ToolError):
            self.toolbox.folders.check("knowledge/escape/outside.txt")


class ToolTests(Workspace):
    def run_tool(self, name, **arguments):
        return self.toolbox.run(name, arguments)

    def test_definitions_describe_each_tool(self):
        names = [d["function"]["name"] for d in self.toolbox.definitions()]
        self.assertEqual(names, ["list_folder", "read_file", "search_knowledge"])
        read = self.toolbox.definitions()[1]["function"]
        self.assertEqual(read["parameters"]["required"], ["path"])

    def test_list_allowed_folders(self):
        outcome = self.run_tool("list_folder")
        self.assertTrue(outcome.ok)
        self.assertIn("knowledge/", outcome.text)

    def test_list_folder_hides_hidden_and_blocked(self):
        outcome = self.run_tool("list_folder", path="knowledge")
        self.assertTrue(outcome.ok, outcome.text)
        self.assertIn("notes/", outcome.text)
        self.assertIn("todo.md", outcome.text)
        self.assertNotIn(".secret", outcome.text)
        self.assertNotIn("data.db", outcome.text)
        self.assertIn("2 hidden or blocked", outcome.text)

    def test_list_folder_limit(self):
        self.toolbox = self.make_toolbox(max_list_entries=2)
        outcome = self.run_tool("list_folder", path="knowledge")
        self.assertIn("... and 1 more", outcome.text)

    def test_read_file(self):
        outcome = self.run_tool("read_file", path="knowledge/todo.md")
        self.assertTrue(outcome.ok, outcome.text)
        self.assertIn("lines 1-3 of 3", outcome.text)
        self.assertIn("line 2", outcome.text)
        self.assertNotIn("Cut off", outcome.text)

    def test_long_file_comes_in_pieces(self):
        self.toolbox = self.make_toolbox(max_read_chars=14)
        first = self.run_tool("read_file", path="knowledge/todo.md")
        self.assertIn("lines 1-2 of 3", first.text)
        self.assertIn("start_line=3", first.text)
        rest = self.run_tool("read_file", path="knowledge/todo.md", start_line=3)
        self.assertIn("lines 3-3 of 3", rest.text)
        self.assertIn("line 3", rest.text)

    def test_read_errors_go_back_to_the_model(self):
        for args, message in [({"path": "knowledge/missing.md"}, "doesn't exist"),
                              ({"path": "knowledge/notes"}, "is a folder"),
                              ({"path": "knowledge/image.txt"}, "binary"),
                              ({"path": "outside.txt"}, "outside the allowed folders"),
                              ({"path": "knowledge/todo.md", "start_line": 50}, "only 3 lines")]:
            with self.subTest(args=args):
                outcome = self.toolbox.run("read_file", args)
                self.assertFalse(outcome.ok)
                self.assertTrue(outcome.text.startswith("Error:"))
                self.assertIn(message, outcome.text)

    def test_arguments_are_checked(self):
        self.assertIn("missing required argument: path", self.run_tool("read_file").text)
        self.assertIn("must be a string", self.run_tool("read_file", path=5).text)
        self.assertIn("no tool called", self.run_tool("delete_everything").text)
        bad_json = self.toolbox.run("read_file", parse_tool_call(
            {"function": {"name": "read_file", "arguments": "{not json"}}).arguments)
        self.assertIn("valid JSON", bad_json.text)
        # "3" for 3 is accepted, and invented extra arguments are ignored.
        outcome = self.run_tool("read_file", path="knowledge/todo.md", start_line="3", mode="fast")
        self.assertTrue(outcome.ok, outcome.text)
        self.assertIn("lines 3-3", outcome.text)

    def test_search_knowledge(self):
        hit = Hit("knowledge/todo.md", "lines 1-3", "line 1\nline 2", 0.81)
        self.toolbox.search = lambda query: [hit] if query == "todo" else []
        outcome = self.run_tool("search_knowledge", query="todo")
        self.assertIn("[1] knowledge/todo.md (lines 1-3)  (match 0.81)", outcome.text)
        self.assertIn("No passages matched", self.run_tool("search_knowledge", query="x").text)
        self.toolbox.search = None
        self.assertIn("isn't available", self.run_tool("search_knowledge", query="x").text)

    def test_unexpected_failure_becomes_error_text(self):
        def broken(query):
            raise RuntimeError("index exploded")
        self.toolbox.search = broken
        outcome = self.run_tool("search_knowledge", query="x")
        self.assertFalse(outcome.ok)
        self.assertIn("index exploded", outcome.text)

    def test_describe_call(self):
        self.assertEqual(describe_call("read_file", {"path": "a.md"}), "read_file(path='a.md')")


class SettingsTests(unittest.TestCase):
    def test_from_config(self):
        s = ToolSettings.from_config({"tools": {"folders": ["docs"], "blocked": [],
                                                "max_read_chars": 100},
                                      "agents": {"max_iterations": 3}})
        self.assertEqual(s.folders, ("docs",))
        self.assertEqual(s.blocked, ())
        self.assertEqual(s.max_read_chars, 100)
        self.assertEqual(s.max_iterations, 3)
        self.assertEqual(ToolSettings.from_config({}).folders, ("knowledge", "."))


class ToolLoopTests(Workspace):
    def setUp(self):
        super().setUp()
        self.store = ConversationStore(":memory:")

    def tearDown(self):
        self.store.close()
        super().tearDown()

    def convo(self, client, **kwargs):
        return Conversation(client, "qwen3:4b", "Be brief.", store=self.store,
                            toolbox=self.toolbox, **kwargs)

    def test_runs_tool_then_answers(self):
        client = ScriptedClient(asks_for(call("read_file", path="knowledge/todo.md")),
                                "You have three lines.")
        convo = self.convo(client)
        seen = []
        result = convo.send("what's in my todo?", on_tool=lambda c: seen.append(c.name))
        self.assertEqual(result.content, "You have three lines.")
        self.assertEqual(seen, ["read_file"])
        # The second request carries the model's request and the tool's result.
        second = client.requests[1]["messages"]
        self.assertEqual([m["role"] for m in second], ["system", "user", "assistant", "tool"])
        self.assertEqual(second[2]["tool_calls"][0]["function"]["name"], "read_file")
        self.assertEqual(second[3]["tool_name"], "read_file")
        self.assertIn("line 2", second[3]["content"])
        # The history keeps only the question and answer.
        self.assertEqual([m["role"] for m in convo.messages], ["system", "user", "assistant"])
        # Stats add up across both model calls.
        self.assertEqual(result.completion_tokens, 8)
        self.assertEqual(result.total_seconds, 3.0)

    def test_tool_calls_are_logged_with_the_reply(self):
        client = ScriptedClient(asks_for(call("read_file", path="knowledge/todo.md"),
                                         call("read_file", path="outside.txt")),
                                "Done.")
        convo = self.convo(client)
        convo.send("read both")
        self.assertEqual([o.status for o in convo.last_tool_calls], ["ok", "error"])
        logged = list(reversed(self.store.recent_tool_calls()))
        self.assertEqual([(c.tool, c.status) for c in logged],
                         [("read_file", "ok"), ("read_file", "error")])
        self.assertEqual(logged[0].arguments, {"path": "knowledge/todo.md"})
        self.assertTrue(all(c.conversation_id == convo.conversation_id for c in logged))
        message_id = self.store.db.execute(
            "SELECT message_id FROM tool_calls WHERE id = ?", (logged[0].id,)).fetchone()[0]
        role = self.store.db.execute(
            "SELECT role FROM messages WHERE id = ?", (message_id,)).fetchone()[0]
        self.assertEqual(role, "assistant")

    def test_failed_chat_still_logs_tool_calls(self):
        client = ScriptedClient(asks_for(call("list_folder", path="knowledge")))
        convo = self.convo(client)
        with self.assertRaises(IndexError):  # the script runs out: a failed second call
            convo.send("hi")
        self.assertEqual(len(self.store.recent_tool_calls()), 1)
        self.assertIsNone(self.store.recent_tool_calls()[0].conversation_id)
        self.assertEqual(convo.messages, [{"role": "system", "content": "Be brief."}])

    def test_round_limit_makes_the_model_answer(self):
        loop = asks_for(call("list_folder", path="knowledge"))
        client = ScriptedClient(loop, loop, "Giving up and answering.")
        convo = self.convo(client, max_tool_rounds=2)
        result = convo.send("loop forever")
        self.assertEqual(result.content, "Giving up and answering.")
        self.assertEqual([r["tools"] is not None for r in client.requests], [True, True, False])

    def test_tools_off_sends_no_tools(self):
        client = ScriptedClient("Plain answer.")
        convo = self.convo(client)
        convo.use_tools = False
        convo.send("hi")
        self.assertIsNone(client.requests[0]["tools"])

    def test_model_without_tool_support_answers_anyway(self):
        client = ScriptedClient("Answer without tools.", supports_tools=False)
        convo = self.convo(client)
        result = convo.send("hi")
        self.assertEqual(result.content, "Answer without tools.")
        self.assertIn("does not support tools", convo.tools_unsupported_error)
        self.assertFalse(convo.use_tools)


class FakeToolOllama(BaseHTTPRequestHandler):
    last_body = None

    def log_message(self, *args):
        pass

    def do_POST(self):
        body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
        FakeToolOllama.last_body = body
        self.send_response(200)
        self.send_header("Content-Type", "application/x-ndjson")
        self.end_headers()
        chunks = [
            {"message": {"role": "assistant", "content": "Let me look."}, "done": False},
            {"message": {"role": "assistant", "content": "", "tool_calls": [
                {"function": {"name": "read_file", "arguments": {"path": "a.md"}}},
                {"function": {"name": "list_folder", "arguments": "{\"path\": \"knowledge\"}"}},
            ]}, "done": False},
            {"message": {"role": "assistant", "content": ""}, "done": True, "model": body["model"]},
        ]
        for chunk in chunks:
            self.wfile.write((json.dumps(chunk) + "\n").encode())


class ClientToolTests(unittest.TestCase):
    def test_tools_sent_and_tool_calls_parsed(self):
        server = HTTPServer(("127.0.0.1", 0), FakeToolOllama)
        threading.Thread(target=server.serve_forever, daemon=True).start()
        try:
            client = OllamaClient(f"http://127.0.0.1:{server.server_port}", timeout=5)
            tools = Toolbox(ToolSettings()).definitions()
            result = client.chat("qwen3:4b", [{"role": "user", "content": "hi"}], tools=tools)
        finally:
            server.shutdown()
            server.server_close()
        self.assertEqual(FakeToolOllama.last_body["tools"], tools)
        self.assertEqual(result.content, "Let me look.")
        self.assertEqual([(c.name, c.arguments) for c in result.tool_calls],
                         [("read_file", {"path": "a.md"}), ("list_folder", {"path": "knowledge"})])
        self.assertEqual(result.tool_calls[0].raw["function"]["name"], "read_file")


if __name__ == "__main__":
    unittest.main()
