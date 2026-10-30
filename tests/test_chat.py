"""Tests against a fake Ollama server, so they run without the ai-pc.

Run from the local-ai folder:  python -m unittest discover tests
"""

import json
import socket
import threading
import unittest
from http.server import BaseHTTPRequestHandler, HTTPServer

from app.config import load_config
from app.main import normalize_base_url
from app.conversation import Conversation
from app.ollama_client import OllamaClient, OllamaError


class FakeOllama(BaseHTTPRequestHandler):
    last_body = None

    def log_message(self, *args):
        pass

    def _json(self, code, obj):
        data = json.dumps(obj).encode()
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def do_GET(self):
        if self.path == "/api/version":
            self._json(200, {"version": "0.0-test"})
        elif self.path == "/api/tags":
            self._json(200, {"models": [{"name": "qwen3:4b"}, {"name": "qwen3:0.6b"}]})
        else:
            self._json(404, {"error": "not found"})

    def do_POST(self):
        body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
        FakeOllama.last_body = body
        if body["model"] == "missing":
            return self._json(404, {"error": "model 'missing' not found"})
        if body["model"] == "no-think" and "think" in body:
            return self._json(400, {"error": "\"no-think\" does not support thinking"})
        if body["model"] in ("cut-off", "stream-error", "bad-json"):
            self.send_response(200)
            self.send_header("Content-Type", "application/x-ndjson")
            self.end_headers()
            chunk = {"model": body["model"], "done": False,
                     "message": {"role": "assistant", "content": "Hel"}}
            self.wfile.write((json.dumps(chunk) + "\n").encode())
            if body["model"] == "stream-error":
                self.wfile.write((json.dumps({"error": "out of memory"}) + "\n").encode())
            elif body["model"] == "bad-json":
                self.wfile.write(b"{not json\n")
            # "cut-off" just stops without a done message.
            return
        done = {
            "model": body["model"], "done": True,
            "message": {"role": "assistant", "content": ""},
            "prompt_eval_count": 10, "prompt_eval_duration": 500_000_000,
            "eval_count": 4, "eval_duration": 2_000_000_000, "total_duration": 3_000_000_000,
            "load_duration": 250_000_000,
        }
        thinking = "Let me think." if body.get("think") else ""
        if not body["stream"]:
            done["message"]["content"] = "Hello there"
            done["message"]["thinking"] = thinking
            return self._json(200, done)
        self.send_response(200)
        self.send_header("Content-Type", "application/x-ndjson")
        self.end_headers()
        if thinking:
            chunk = {"model": body["model"], "done": False,
                     "message": {"role": "assistant", "content": "", "thinking": thinking}}
            self.wfile.write((json.dumps(chunk) + "\n").encode())
        for word in ["Hello", " there"]:
            chunk = {"model": body["model"], "done": False,
                     "message": {"role": "assistant", "content": word}}
            self.wfile.write((json.dumps(chunk) + "\n").encode())
        self.wfile.write((json.dumps(done) + "\n").encode())


class ChatTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.server = HTTPServer(("127.0.0.1", 0), FakeOllama)
        threading.Thread(target=cls.server.serve_forever, daemon=True).start()
        cls.client = OllamaClient(f"http://127.0.0.1:{cls.server.server_port}", timeout=5)

    @classmethod
    def tearDownClass(cls):
        cls.server.shutdown()

    def test_version_and_models(self):
        self.assertEqual(self.client.version(), "0.0-test")
        self.assertIn("qwen3:4b", self.client.list_models())

    def test_streaming_chat(self):
        tokens = []
        result = self.client.chat("qwen3:4b", [{"role": "user", "content": "hi"}],
                                  on_token=tokens.append)
        self.assertEqual(tokens, ["Hello", " there"])
        self.assertEqual(result.content, "Hello there")
        self.assertAlmostEqual(result.tokens_per_second, 2.0)
        self.assertAlmostEqual(result.prompt_tokens_per_second, 20.0)

    def test_non_streaming_chat(self):
        result = self.client.chat("qwen3:4b", [{"role": "user", "content": "hi"}], stream=False)
        self.assertEqual(result.content, "Hello there")

    def test_conversation_keeps_history(self):
        convo = Conversation(self.client, "qwen3:4b", system_prompt="Be brief.")
        convo.send("hi")
        convo.send("again")
        roles = [m["role"] for m in FakeOllama.last_body["messages"]]
        self.assertEqual(roles, ["system", "user", "assistant", "user"])
        self.assertEqual(len(convo.messages), 5)

    def test_error_does_not_leave_dangling_message(self):
        convo = Conversation(self.client, "missing")
        with self.assertRaises(OllamaError) as ctx:
            convo.send("hi")
        self.assertIn("not found", str(ctx.exception))
        self.assertEqual(convo.messages, [])

    def test_unreachable_host(self):
        client = OllamaClient("http://127.0.0.1:1", timeout=2)
        with self.assertRaises(OllamaError):
            client.version()

    def test_load_time_reported(self):
        result = self.client.chat("qwen3:4b", [{"role": "user", "content": "hi"}])
        self.assertAlmostEqual(result.load_seconds, 0.25)

    def test_think_setting_sent_and_thinking_kept_separate(self):
        tokens = []
        result = self.client.chat("qwen3:4b", [{"role": "user", "content": "hi"}],
                                  on_token=tokens.append, think=True)
        self.assertIs(FakeOllama.last_body["think"], True)
        self.assertEqual(result.thinking, "Let me think.")
        self.assertEqual("".join(tokens), "Hello there")
        self.assertEqual(result.content, "Hello there")

        self.client.chat("qwen3:4b", [{"role": "user", "content": "hi"}], think=False)
        self.assertIs(FakeOllama.last_body["think"], False)
        self.client.chat("qwen3:4b", [{"role": "user", "content": "hi"}])
        self.assertNotIn("think", FakeOllama.last_body)

    def test_think_off_works_with_models_without_thinking(self):
        result = self.client.chat("no-think", [{"role": "user", "content": "hi"}], think=False)
        self.assertEqual(result.content, "Hello there")
        with self.assertRaises(OllamaError):
            self.client.chat("no-think", [{"role": "user", "content": "hi"}], think=True)

    def test_conversation_does_not_resend_thinking(self):
        convo = Conversation(self.client, "qwen3:4b", think=True)
        convo.send("hi")
        convo.send("again")
        self.assertNotIn("thinking", FakeOllama.last_body["messages"][1])
        self.assertIs(FakeOllama.last_body["think"], True)

    def test_errors_while_reading_reply_become_ollama_errors(self):
        for model, message in [("cut-off", "ended before"),
                               ("stream-error", "out of memory"),
                               ("bad-json", "valid JSON")]:
            with self.subTest(model=model):
                convo = Conversation(self.client, model)
                with self.assertRaises(OllamaError) as ctx:
                    convo.send("hi")
                self.assertIn(message, str(ctx.exception))
                self.assertEqual(convo.messages, [])

    def test_read_timeout_becomes_ollama_error(self):
        # A server that accepts the request but never answers.
        listener = socket.socket()
        listener.bind(("127.0.0.1", 0))
        listener.listen(1)
        try:
            client = OllamaClient(f"http://127.0.0.1:{listener.getsockname()[1]}", timeout=0.5)
            with self.assertRaises(OllamaError) as ctx:
                client.chat("qwen3:4b", [{"role": "user", "content": "hi"}])
            self.assertIn("Timed out", str(ctx.exception))
        finally:
            listener.close()

    def test_normalize_base_url(self):
        self.assertEqual(normalize_base_url("192.168.1.86"), "http://192.168.1.86:11434")
        self.assertEqual(normalize_base_url("http://ai-pc:11434/"), "http://ai-pc:11434")


class ConfigTests(unittest.TestCase):
    def test_project_config_has_inference_settings(self):
        inference = load_config().get("inference", {})
        self.assertTrue(inference.get("base_url"))
        self.assertTrue(inference.get("default_model"))

if __name__ == "__main__":
    unittest.main()
