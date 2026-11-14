"""Builds and updates the knowledge index from the approved folders.

Only folders listed under `knowledge.folders` in the config are read. For each
supported file the indexer reads the text, splits it into chunks, asks the
embedding model on the Ollama server for each chunk's embedding, and saves it
all with KnowledgeStore. Running it again only redoes files that changed, and
drops files that were deleted or whose folder was removed from the config.
"""

import hashlib
import re
import time
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Callable

from app.config import ROOT
from retrieval.chunker import chunk_text
from retrieval.readers import DOCUMENT_EXTENSIONS, TEXT_EXTENSIONS, ReadError, read_document
from retrieval.store import KnowledgeStore, NewChunk

# Folders that are never worth indexing, wherever they appear.
SKIP_DIRS = {"__pycache__", "node_modules", "venv", "env", "site-packages", "dist", "build"}

# nomic-embed-text was trained with these labels in front of the text and works
# best with them. Other models: set the prefixes in the config (often empty).
KNOWN_PREFIXES = {
    "nomic-embed-text": ("search_document: ", "search_query: "),
    # Qwen3-Embedding wants a one-line task description before questions only.
    "qwen3-embedding": ("", "Instruct: Given a question, retrieve passages from the "
                            "user's documents that answer it\nQuery: "),
}


@dataclass
class KnowledgeSettings:
    folders: list[Path]
    database: Path
    embedding_model: str = "nomic-embed-text"
    document_prefix: str = ""
    query_prefix: str = ""
    chunk_size: int = 1000
    chunk_overlap: int = 150
    batch_size: int = 16
    max_file_mb: float = 20
    extensions: set[str] = field(default_factory=lambda: TEXT_EXTENSIONS | DOCUMENT_EXTENSIONS)
    top_k: int = 4
    min_score: float = 0.5
    use_in_chat: bool = True
    # Prefixes set in the config apply to the configured model only.
    configured_prefixes: dict[str, tuple[str, str]] = field(default_factory=dict)

    @property
    def index_path(self) -> Path:
        """Each embedding model has its own index file next to `database`, e.g.
        knowledge-nomic-embed-text.db, so switching models doesn't erase the other's index."""
        slug = re.sub(r"[^A-Za-z0-9._-]+", "-", self.embedding_model.removesuffix(":latest"))
        return self.database.with_name(f"{self.database.stem}-{slug}{self.database.suffix}")

    def with_model(self, model: str) -> "KnowledgeSettings":
        """A copy of these settings for another embedding model, with its prefixes."""
        doc, query = self.configured_prefixes.get(model) or prefixes_for(model)
        return replace(self, embedding_model=model, document_prefix=doc, query_prefix=query)

    @classmethod
    def from_config(cls, config: dict) -> "KnowledgeSettings":
        k = config.get("knowledge") or {}
        model = k.get("embedding_model") or "nomic-embed-text"
        default_doc, default_query = prefixes_for(model)
        document_prefix = _or(k.get("document_prefix"), default_doc)
        query_prefix = _or(k.get("query_prefix"), default_query)
        extensions = k.get("extensions")
        return cls(
            folders=[resolve(f) for f in (k.get("folders") or ["knowledge"])],
            database=resolve(k.get("database") or "data/indexes/knowledge.db"),
            embedding_model=model,
            document_prefix=document_prefix,
            query_prefix=query_prefix,
            chunk_size=int(k.get("chunk_size", 1000)),
            chunk_overlap=int(k.get("chunk_overlap", 150)),
            batch_size=int(k.get("batch_size", 16)),
            max_file_mb=float(k.get("max_file_mb", 20)),
            extensions={e.lower() if e.startswith(".") else "." + e.lower() for e in extensions}
            if extensions else TEXT_EXTENSIONS | DOCUMENT_EXTENSIONS,
            top_k=int(k.get("top_k", 4)),
            min_score=float(k.get("min_score", 0.5)),
            use_in_chat=bool(k.get("use_in_chat", True)),
            configured_prefixes={model: (document_prefix, query_prefix)},
        )


def prefixes_for(model: str) -> tuple[str, str]:
    return KNOWN_PREFIXES.get(model.split(":")[0], ("", ""))


def _or(value, default):
    return default if value is None else value


def resolve(path: str | Path) -> Path:
    """A relative path in the config is inside the local-ai folder."""
    path = Path(path).expanduser()
    return (path if path.is_absolute() else ROOT / path).resolve()


def display_path(path: Path, folder: Path) -> str:
    """The folder's name plus the path inside it, e.g. knowledge/notes/setup.md"""
    return (Path(folder.name) / path.relative_to(folder)).as_posix()


def location(start: int, end: int, unit: str, page: int | None) -> str:
    if page is not None:
        return f"page {page}"
    label = unit if start != end else unit.rstrip("s")
    return f"{label} {start}-{end}" if start != end else f"{label} {start}"


@dataclass
class IndexReport:
    added: list[str] = field(default_factory=list)
    updated: list[str] = field(default_factory=list)
    unchanged: int = 0
    removed: list[str] = field(default_factory=list)
    skipped: list[tuple[str, str]] = field(default_factory=list)  # (file, reason)
    missing_folders: list[str] = field(default_factory=list)
    chunks: int = 0
    seconds: float = 0.0


class Indexer:
    def __init__(self, client, store: KnowledgeStore, settings: KnowledgeSettings,
                 progress: Callable[[str], None] | None = None):
        self.client = client  # anything with embed(model, texts), normally OllamaClient
        self.store = store
        self.settings = settings
        self.progress = progress or (lambda message: None)

    def run(self, rebuild: bool = False) -> IndexReport:
        report = IndexReport()
        started = time.monotonic()
        s = self.settings

        # Vectors from different models don't mix, so a new model means starting over.
        built_with = self.store.get_setting("embedding_model")
        if rebuild or (built_with and built_with != s.embedding_model):
            if built_with and built_with != s.embedding_model:
                self.progress(f"Embedding model changed from {built_with} to "
                              f"{s.embedding_model}, so everything is re-indexed.")
            self.store.clear()
        self.store.set_setting("embedding_model", s.embedding_model)

        stored = self.store.files()
        seen: set[str] = set()
        missing: list[Path] = []
        for folder in s.folders:
            if not folder.is_dir():
                missing.append(folder)
                report.missing_folders.append(str(folder))
                continue
            for path in self._files_in(folder):
                key = str(path)
                if key in seen:
                    continue  # the same file through two overlapping folders
                seen.add(key)
                self._index_file(path, folder, stored.get(key), report)

        # Forget files that are gone. A folder that's missing (say, an unplugged
        # drive) keeps its entries rather than being wiped.
        for key, old in stored.items():
            if key in seen or any(Path(key).is_relative_to(m) for m in missing):
                continue
            self.store.remove_file(key)
            report.removed.append(key)

        report.seconds = time.monotonic() - started
        return report

    def _files_in(self, folder: Path):
        for path in sorted(folder.rglob("*")):
            parts = path.relative_to(folder).parts
            # Skip hidden files and folders (.git, .venv, ...) and known clutter.
            if any(p.startswith(".") or p in SKIP_DIRS for p in parts):
                continue
            if path.is_symlink():
                # Don't follow links that lead outside the approved folder.
                if not path.resolve().is_relative_to(folder):
                    continue
            if path.is_file() and path.suffix.lower() in self.settings.extensions:
                yield path.resolve() if path.is_symlink() else path

    def _index_file(self, path: Path, folder: Path, old, report: IndexReport) -> None:
        s = self.settings
        shown = display_path(path, folder) if path.is_relative_to(folder) else str(path)
        try:
            stat = path.stat()
            if stat.st_size > s.max_file_mb * 1024 * 1024:
                raise ReadError(f"larger than {s.max_file_mb:g} MB (knowledge.max_file_mb)")
            # Same size and modified time: assume unchanged without reading it.
            if old and old.size == stat.st_size and old.mtime == stat.st_mtime:
                report.unchanged += 1
                return
            data = path.read_bytes()
            digest = hashlib.sha256(data).hexdigest()
            if old and old.sha256 == digest:
                self.store.touch_file(old.id, stat.st_size, stat.st_mtime)
                report.unchanged += 1
                return

            started = time.monotonic()
            pieces = []
            for section in read_document(path):
                for c in chunk_text(section.text, s.chunk_size, s.chunk_overlap):
                    pieces.append((location(c.start_line, c.end_line, section.unit, section.page),
                                   c.text))
            if not pieces:
                raise ReadError("no text found")
            vectors = []
            for i in range(0, len(pieces), s.batch_size):
                batch = [s.document_prefix + text for _, text in pieces[i:i + s.batch_size]]
                vectors.extend(self.client.embed(s.embedding_model, batch))
        except (ReadError, OSError) as e:
            report.skipped.append((shown, str(e)))
            self.progress(f"  skipped {shown}: {e}")
            return

        chunks = [NewChunk(loc, text, vec) for (loc, text), vec in zip(pieces, vectors)]
        self.store.replace_file(str(path), shown, str(folder), stat.st_size, stat.st_mtime,
                                digest, chunks)
        (report.updated if old else report.added).append(shown)
        report.chunks += len(chunks)
        self.progress(f"  {shown}: {len(chunks)} chunks in {time.monotonic() - started:.1f}s")
