"""Keeps indexed documents, their chunks and the chunks' embeddings in SQLite.

An embedding is a list of numbers (768 for nomic-embed-text) that describes
what a piece of text is about. Texts with similar meaning get similar lists,
so finding passages that answer a question is a matter of finding the stored
lists closest to the question's list.

The lists are stored as raw bytes (4 bytes per number). Each one is scaled to
length 1 before saving, which makes "how similar" a plain dot product: the
cosine similarity, from about 0 (unrelated) to 1 (same meaning). Search checks
every chunk, which takes well under a second for a few thousand chunks and
needs no extra software. A dedicated vector database can replace this class
later if the collection grows much larger.
"""

import math
import sqlite3
from array import array
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

SCHEMA = """
CREATE TABLE IF NOT EXISTS files (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    path        TEXT NOT NULL UNIQUE,  -- full path on this computer
    display     TEXT NOT NULL,         -- shorter path shown in source lists
    folder      TEXT NOT NULL,         -- the approved folder it was found in
    size        INTEGER NOT NULL,
    mtime       REAL NOT NULL,
    sha256      TEXT NOT NULL,
    indexed_at  TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS chunks (
    id         INTEGER PRIMARY KEY AUTOINCREMENT,
    file_id    INTEGER NOT NULL REFERENCES files(id) ON DELETE CASCADE,
    position   INTEGER NOT NULL,       -- order within the file
    location   TEXT NOT NULL,          -- e.g. "lines 12-30" or "page 3"
    text       TEXT NOT NULL,
    embedding  BLOB NOT NULL
);

CREATE INDEX IF NOT EXISTS chunks_by_file ON chunks (file_id, position);

-- Which embedding model built the index. Vectors from different models
-- can't be compared, so changing the model means re-embedding everything.
CREATE TABLE IF NOT EXISTS settings (
    key   TEXT PRIMARY KEY,
    value TEXT NOT NULL
);
"""


@dataclass
class StoredFile:
    id: int
    path: str
    folder: str
    size: int
    mtime: float
    sha256: str


@dataclass
class NewChunk:
    location: str
    text: str
    embedding: list[float]


@dataclass
class Hit:
    """A chunk found by a search, with how similar it is to the question (0-1)."""
    display: str
    location: str
    text: str
    score: float

    @property
    def reference(self) -> str:
        return f"{self.display} ({self.location})"


def normalize(vector: list[float]) -> list[float]:
    length = math.sqrt(sum(x * x for x in vector))
    if length == 0:
        return list(vector)
    return [x / length for x in vector]


def to_bytes(vector: list[float]) -> bytes:
    return array("f", normalize(vector)).tobytes()


def from_bytes(blob: bytes) -> array:
    vector = array("f")
    vector.frombytes(blob)
    return vector


class KnowledgeStore:
    def __init__(self, path: str | Path):
        """Open (or create) the index. Pass ":memory:" for a throwaway one."""
        if str(path) != ":memory:":
            Path(path).parent.mkdir(parents=True, exist_ok=True)
        self.path = str(path)
        self.db = sqlite3.connect(self.path)
        self.db.row_factory = sqlite3.Row
        self.db.execute("PRAGMA foreign_keys = ON")
        self.db.executescript(SCHEMA)
        self._vectors = None  # search cache, rebuilt after any change

    def close(self) -> None:
        self.db.close()

    # Settings

    def get_setting(self, key: str) -> str | None:
        row = self.db.execute("SELECT value FROM settings WHERE key = ?", (key,)).fetchone()
        return row["value"] if row else None

    def set_setting(self, key: str, value: str) -> None:
        with self.db:
            self.db.execute(
                "INSERT INTO settings (key, value) VALUES (?, ?)"
                " ON CONFLICT(key) DO UPDATE SET value = excluded.value",
                (key, value),
            )

    # Files and chunks

    def files(self) -> dict[str, StoredFile]:
        rows = self.db.execute("SELECT id, path, folder, size, mtime, sha256 FROM files")
        return {r["path"]: StoredFile(*r) for r in rows}

    def touch_file(self, file_id: int, size: int, mtime: float) -> None:
        """Record a new modified time for a file whose content didn't change."""
        with self.db:
            self.db.execute("UPDATE files SET size = ?, mtime = ? WHERE id = ?",
                            (size, mtime, file_id))

    def replace_file(self, path: str, display: str, folder: str, size: int, mtime: float,
                     sha256: str, chunks: list[NewChunk]) -> None:
        """Save a file and its chunks, replacing any older version (all or nothing)."""
        stamp = datetime.now(timezone.utc).isoformat(timespec="seconds")
        with self.db:
            self.db.execute("DELETE FROM files WHERE path = ?", (path,))
            cur = self.db.execute(
                "INSERT INTO files (path, display, folder, size, mtime, sha256, indexed_at)"
                " VALUES (?, ?, ?, ?, ?, ?, ?)",
                (path, display, folder, size, mtime, sha256, stamp),
            )
            self.db.executemany(
                "INSERT INTO chunks (file_id, position, location, text, embedding)"
                " VALUES (?, ?, ?, ?, ?)",
                [(cur.lastrowid, i, c.location, c.text, to_bytes(c.embedding))
                 for i, c in enumerate(chunks)],
            )
        self._vectors = None

    def remove_file(self, path: str) -> None:
        with self.db:
            self.db.execute("DELETE FROM files WHERE path = ?", (path,))
        self._vectors = None

    def clear(self) -> None:
        with self.db:
            self.db.execute("DELETE FROM files")
        self._vectors = None

    def counts(self) -> tuple[int, int]:
        """(number of files, number of chunks)"""
        files = self.db.execute("SELECT COUNT(*) FROM files").fetchone()[0]
        chunks = self.db.execute("SELECT COUNT(*) FROM chunks").fetchone()[0]
        return files, chunks

    # Search

    def search(self, query: list[float], top_k: int = 4, min_score: float = 0.0) -> list[Hit]:
        """The top_k chunks most similar to the query vector, best first."""
        if self._vectors is None:
            self._vectors = [
                (r["id"], from_bytes(r["embedding"]))
                for r in self.db.execute("SELECT id, embedding FROM chunks")
            ]
        q = normalize(query)
        scored = []
        for chunk_id, vector in self._vectors:
            if len(vector) != len(q):
                continue  # from a different model; skipped until the next index
            score = sum(a * b for a, b in zip(q, vector))
            if score >= min_score:
                scored.append((score, chunk_id))
        scored.sort(reverse=True)
        hits = []
        for score, chunk_id in scored[:top_k]:
            row = self.db.execute(
                "SELECT f.display, c.location, c.text FROM chunks c"
                " JOIN files f ON f.id = c.file_id WHERE c.id = ?",
                (chunk_id,),
            ).fetchone()
            hits.append(Hit(row["display"], row["location"], row["text"], round(score, 3)))
        return hits
