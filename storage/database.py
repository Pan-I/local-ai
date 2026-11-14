"""Saves conversations to a local SQLite database.

Uses Python's built-in sqlite3 module, so there is nothing extra to install.
Each conversation is a row in `conversations`, and each user message or model
reply is a row in `messages`. Each tool the model used is a row in `tool_calls`.
"""

import json
import sqlite3
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

from app.ollama_client import ChatResult

SCHEMA = """
CREATE TABLE IF NOT EXISTS conversations (
    id            INTEGER PRIMARY KEY AUTOINCREMENT,
    title         TEXT,
    model         TEXT NOT NULL,
    system_prompt TEXT NOT NULL DEFAULT '',
    created_at    TEXT NOT NULL,
    updated_at    TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS messages (
    id                INTEGER PRIMARY KEY AUTOINCREMENT,
    conversation_id   INTEGER NOT NULL REFERENCES conversations(id) ON DELETE CASCADE,
    role              TEXT NOT NULL CHECK (role IN ('user', 'assistant')),
    content           TEXT NOT NULL,
    model             TEXT,
    -- Stats for model replies; empty for user messages.
    prompt_tokens     INTEGER,
    completion_tokens INTEGER,
    tokens_per_second REAL,
    load_seconds      REAL,
    total_seconds     REAL,
    created_at        TEXT NOT NULL
);

CREATE INDEX IF NOT EXISTS messages_by_conversation ON messages (conversation_id, id);

-- The document passages a reply was given (see retrieval/), best match first.
CREATE TABLE IF NOT EXISTS message_sources (
    message_id INTEGER NOT NULL REFERENCES messages(id) ON DELETE CASCADE,
    rank       INTEGER NOT NULL,
    file       TEXT NOT NULL,
    location   TEXT NOT NULL,
    score      REAL NOT NULL,
    PRIMARY KEY (message_id, rank)
);

-- Every tool the model asked the app to run (see agent/), saved as it happens.
-- conversation_id and message_id are filled in when the reply is saved, and
-- stay empty if the chat failed or was interrupted before that.
CREATE TABLE IF NOT EXISTS tool_calls (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    conversation_id INTEGER REFERENCES conversations(id) ON DELETE SET NULL,
    message_id      INTEGER REFERENCES messages(id) ON DELETE SET NULL,
    model           TEXT NOT NULL,
    tool            TEXT NOT NULL,
    arguments       TEXT NOT NULL,   -- JSON, exactly as the model sent them
    status          TEXT NOT NULL,   -- ok or error
    result          TEXT NOT NULL,   -- what the model was sent (cut to 2,000 characters)
    result_chars    INTEGER NOT NULL,
    seconds         REAL NOT NULL,
    created_at      TEXT NOT NULL
);
"""

LOGGED_RESULT_CHARS = 2000

TITLE_LENGTH = 60


@dataclass
class ConversationSummary:
    id: int
    title: str
    model: str
    updated_at: str
    message_count: int


@dataclass
class SavedConversation:
    id: int
    title: str
    model: str
    system_prompt: str
    messages: list[dict]  # [{"role": ..., "content": ...}], oldest first


@dataclass
class ToolCallRecord:
    id: int
    conversation_id: int | None
    tool: str
    arguments: dict
    status: str
    result: str
    seconds: float
    created_at: str


def now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def make_title(text: str) -> str:
    title = " ".join(text.split())
    if len(title) > TITLE_LENGTH:
        title = title[: TITLE_LENGTH - 3].rstrip() + "..."
    return title


class ConversationStore:
    def __init__(self, path: str | Path):
        """Open (or create) the database. Pass ":memory:" for a throwaway one."""
        if str(path) != ":memory:":
            Path(path).parent.mkdir(parents=True, exist_ok=True)
        self.path = str(path)
        self.db = sqlite3.connect(self.path)
        self.db.row_factory = sqlite3.Row
        self.db.execute("PRAGMA foreign_keys = ON")
        self.db.executescript(SCHEMA)

    def close(self) -> None:
        self.db.close()

    def create_conversation(self, model: str, system_prompt: str = "") -> int:
        stamp = now()
        with self.db:
            cur = self.db.execute(
                "INSERT INTO conversations (model, system_prompt, created_at, updated_at)"
                " VALUES (?, ?, ?, ?)",
                (model, system_prompt, stamp, stamp),
            )
        return cur.lastrowid

    def add_turn(self, conversation_id: int, user_text: str, result: ChatResult,
                 sources=(), tool_call_ids=()) -> int:
        """Save a user message and the model's reply together (both or neither).

        sources are the retrieval Hits the reply was given, if any, and
        tool_call_ids the logged tool calls behind it. Returns the reply's id.
        """
        stamp = now()
        with self.db:  # one transaction
            self.db.execute(
                "INSERT INTO messages (conversation_id, role, content, created_at)"
                " VALUES (?, 'user', ?, ?)",
                (conversation_id, user_text, stamp),
            )
            cur = self.db.execute(
                "INSERT INTO messages (conversation_id, role, content, model, prompt_tokens,"
                " completion_tokens, tokens_per_second, load_seconds, total_seconds, created_at)"
                " VALUES (?, 'assistant', ?, ?, ?, ?, ?, ?, ?, ?)",
                (
                    conversation_id, result.content, result.model, result.prompt_tokens,
                    result.completion_tokens, result.tokens_per_second, result.load_seconds,
                    result.total_seconds, stamp,
                ),
            )
            self.db.executemany(
                "INSERT INTO message_sources (message_id, rank, file, location, score)"
                " VALUES (?, ?, ?, ?, ?)",
                [(cur.lastrowid, rank, h.display, h.location, h.score)
                 for rank, h in enumerate(sources, start=1)],
            )
            self.db.executemany(
                "UPDATE tool_calls SET conversation_id = ?, message_id = ? WHERE id = ?",
                [(conversation_id, cur.lastrowid, i) for i in tool_call_ids],
            )
            # The first message becomes the title.
            self.db.execute(
                "UPDATE conversations SET updated_at = ?, title = COALESCE(title, ?) WHERE id = ?",
                (stamp, make_title(user_text), conversation_id),
            )
        return cur.lastrowid

    def log_tool_call(self, model: str, tool: str, arguments: dict, status: str,
                      result: str, seconds: float) -> int:
        """Record one tool call right away, before the reply exists. Returns its id."""
        with self.db:
            cur = self.db.execute(
                "INSERT INTO tool_calls (model, tool, arguments, status, result, result_chars,"
                " seconds, created_at) VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
                (model, tool, json.dumps(arguments, ensure_ascii=False, default=str), status,
                 result[:LOGGED_RESULT_CHARS], len(result), seconds, now()),
            )
        return cur.lastrowid

    def recent_tool_calls(self, limit: int = 10) -> list[ToolCallRecord]:
        """Newest first."""
        rows = self.db.execute(
            "SELECT * FROM tool_calls ORDER BY id DESC LIMIT ?", (limit,)
        ).fetchall()
        return [
            ToolCallRecord(r["id"], r["conversation_id"], r["tool"], json.loads(r["arguments"]),
                           r["status"], r["result"], r["seconds"], r["created_at"])
            for r in rows
        ]

    def list_conversations(self, limit: int = 10) -> list[ConversationSummary]:
        """Most recently used first (message ids only go up, so the newest one wins)."""
        rows = self.db.execute(
            "SELECT c.id, c.title, c.model, c.updated_at, COUNT(m.id) AS message_count"
            " FROM conversations c LEFT JOIN messages m ON m.conversation_id = c.id"
            " GROUP BY c.id ORDER BY COALESCE(MAX(m.id), 0) DESC, c.id DESC LIMIT ?",
            (limit,),
        ).fetchall()
        return [
            ConversationSummary(r["id"], r["title"] or "(untitled)", r["model"],
                                r["updated_at"], r["message_count"])
            for r in rows
        ]

    def load(self, conversation_id: int) -> SavedConversation | None:
        row = self.db.execute(
            "SELECT * FROM conversations WHERE id = ?", (conversation_id,)
        ).fetchone()
        if row is None:
            return None
        messages = [
            {"role": m["role"], "content": m["content"]}
            for m in self.db.execute(
                "SELECT role, content FROM messages WHERE conversation_id = ? ORDER BY id",
                (conversation_id,),
            )
        ]
        return SavedConversation(row["id"], row["title"] or "(untitled)", row["model"],
                                 row["system_prompt"], messages)
