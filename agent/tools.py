"""The tools the model can ask the app to run.

Each tool is a plain Python function plus a description of its inputs (a JSON
schema) that's sent to the model with every message. The model never runs
anything itself: it replies with a request such as
read_file(path="knowledge/notes.md"), and Toolbox.run() checks the request,
runs the function, and returns text to send back to the model.

These first tools only read. Errors (a missing file, a path outside the
allowed folders, a wrong argument) are returned as text starting with
"Error:", so the model can see what went wrong and try something else.
"""

import time
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

from agent.folders import AllowedFolders, ToolError
from retrieval.readers import ReadError, read_document

DEFAULT_MAX_READ_CHARS = 4000
DEFAULT_MAX_LIST_ENTRIES = 200
MAX_FILE_MB = 20


@dataclass
class ToolSettings:
    enabled: bool = True
    folders: tuple = ("knowledge", ".")
    blocked: tuple = ("config/local.yaml", "config/secrets.yaml", "*.db")
    max_read_chars: int = DEFAULT_MAX_READ_CHARS
    max_list_entries: int = DEFAULT_MAX_LIST_ENTRIES
    max_iterations: int = 10

    @classmethod
    def from_config(cls, config: dict) -> "ToolSettings":
        t = config.get("tools") or {}
        agents = config.get("agents") or {}
        defaults = cls()
        return cls(
            enabled=bool(t.get("enabled", True)),
            folders=tuple(t.get("folders") or defaults.folders),
            blocked=tuple(defaults.blocked if t.get("blocked") is None else t["blocked"]),
            max_read_chars=int(t.get("max_read_chars", DEFAULT_MAX_READ_CHARS)),
            max_list_entries=int(t.get("max_list_entries", DEFAULT_MAX_LIST_ENTRIES)),
            max_iterations=int(agents.get("max_iterations", 10)),
        )


@dataclass
class Tool:
    name: str
    description: str
    parameters: dict  # JSON schema: {"name": {"type": ..., "description": ...}}
    required: list[str]
    function: Callable[..., str]

    def definition(self) -> dict:
        """The description Ollama sends to the model."""
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": {
                    "type": "object",
                    "properties": self.parameters,
                    "required": self.required,
                },
            },
        }


@dataclass
class ToolOutcome:
    """What happened when a tool ran. `text` is what the model sees."""
    name: str
    arguments: dict
    ok: bool
    text: str
    seconds: float = 0.0

    @property
    def status(self) -> str:
        return "ok" if self.ok else "error"


JSON_TYPES = {"string": str, "integer": int}


class Toolbox:
    def __init__(self, settings: ToolSettings, search: Callable[[str], list] | None = None):
        """search(query) returns retrieval Hits, or is None if there's no index."""
        self.settings = settings
        self.folders = AllowedFolders([Path(f) for f in settings.folders], settings.blocked)
        self.search = search
        shown = self.folders.describe()
        self.tools = {t.name: t for t in [
            Tool(
                "list_folder",
                "List the files and subfolders in a folder on the user's computer. "
                f"Allowed folders: {shown} (paths are relative to the app folder). "
                "Leave path empty to list the allowed folders.",
                {"path": {"type": "string", "description": "Folder to list, e.g. knowledge"}},
                [],
                self.list_folder,
            ),
            Tool(
                "read_file",
                "Read a text, code, Markdown, PDF or Word file from the allowed folders. "
                "Long files are cut off; the reply says which line to start from to read more.",
                {
                    "path": {"type": "string", "description": "File to read, e.g. knowledge/notes.md"},
                    "start_line": {"type": "integer",
                                   "description": "First line to read, starting at 1 (optional)"},
                },
                ["path"],
                self.read_file,
            ),
            Tool(
                "search_knowledge",
                "Search the user's indexed documents for passages about a topic. "
                "Returns the best matching passages with their file and location.",
                {"query": {"type": "string", "description": "What to search for, in plain words"}},
                ["query"],
                self.search_knowledge,
            ),
        ]}

    def definitions(self) -> list[dict]:
        return [t.definition() for t in self.tools.values()]

    def run(self, name: str, arguments: dict) -> ToolOutcome:
        started = time.monotonic()
        try:
            tool = self.tools.get(name)
            if tool is None:
                raise ToolError(f"there's no tool called '{name}'. "
                                f"Available: {', '.join(self.tools)}")
            text = tool.function(**self._validate(tool, arguments))
            ok = True
        except ToolError as e:
            text, ok = f"Error: {e}", False
        except Exception as e:  # a bug or an odd file shouldn't end the chat
            text, ok = f"Error: {type(e).__name__}: {e}", False
        return ToolOutcome(name, arguments, ok, text, time.monotonic() - started)

    def _validate(self, tool: Tool, arguments: dict) -> dict:
        """Check the model's arguments against the tool's description. Extra
        arguments are dropped (small models sometimes invent some)."""
        if "_unparsed" in arguments:
            raise ToolError("the arguments weren't valid JSON")
        clean = {}
        for key, schema in tool.parameters.items():
            if key not in arguments or arguments[key] is None:
                continue
            value = arguments[key]
            expected = JSON_TYPES[schema["type"]]
            if expected is int and isinstance(value, str) and value.strip().isdigit():
                value = int(value)  # "12" for 12 is a common small-model slip
            if not isinstance(value, expected) or isinstance(value, bool):
                raise ToolError(f"'{key}' must be a {schema['type']}")
            clean[key] = value
        missing = [k for k in tool.required if k not in clean]
        if missing:
            raise ToolError(f"missing required argument: {', '.join(missing)}")
        return clean

    # The tools

    def list_folder(self, path: str = "") -> str:
        if not path.strip():
            return "Allowed folders:\n" + "\n".join(
                f"  {self.folders.display(f)}/" for f in self.folders.folders)
        folder = self.folders.check(path)
        if not folder.is_dir():
            raise ToolError(f"'{path}' is not a folder" if folder.exists()
                            else f"'{path}' doesn't exist")
        entries = sorted(folder.iterdir(), key=lambda p: (not p.is_dir(), p.name.lower()))
        lines, hidden = [], 0
        for entry in entries:
            if self.folders.is_blocked(entry.resolve()):
                hidden += 1
                continue
            if entry.is_dir():
                lines.append(f"  {entry.name}/")
            else:
                lines.append(f"  {entry.name}  ({format_size(entry.stat().st_size)})")
        limit = self.settings.max_list_entries
        header = f"{self.folders.display(folder)}/ has {len(lines)} items"
        if len(lines) > limit:
            lines = lines[:limit] + [f"  ... and {len(lines) - limit} more"]
        if not lines:
            lines = ["  (empty)"]
        note = f"\n({hidden} hidden or blocked items not shown)" if hidden else ""
        return header + ":\n" + "\n".join(lines) + note

    def read_file(self, path: str, start_line: int = 1) -> str:
        file = self.folders.check(path)
        if not file.is_file():
            raise ToolError(f"'{path}' is a folder; use list_folder" if file.is_dir()
                            else f"'{path}' doesn't exist")
        if file.stat().st_size > MAX_FILE_MB * 1024 * 1024:
            raise ToolError(f"'{path}' is larger than {MAX_FILE_MB} MB")
        try:
            sections = read_document(file)
        except ReadError as e:
            raise ToolError(f"can't read '{path}': {e}") from e
        text = "\n\n".join(
            (f"--- page {s.page} ---\n" if s.page else "") + s.text for s in sections)
        lines = text.splitlines()
        total = len(lines)
        start = max(1, start_line)
        if start > max(total, 1):
            raise ToolError(f"'{path}' has only {total} lines")
        # Take whole lines until the size limit, so a long file arrives in pieces.
        picked, size = [], 0
        for line in lines[start - 1:]:
            if picked and size + len(line) + 1 > self.settings.max_read_chars:
                break
            picked.append(line[: self.settings.max_read_chars])
            size += len(line) + 1
        end = start + len(picked) - 1
        shown = self.folders.display(file)
        header = f"{shown} (lines {start}-{end} of {total})"
        footer = (f"\n[Cut off here. Call read_file with start_line={end + 1} to read more.]"
                  if end < total else "")
        return header + ":\n" + "\n".join(picked) + footer

    def search_knowledge(self, query: str) -> str:
        if self.search is None:
            raise ToolError("the knowledge index isn't available")
        hits = self.search(query)
        if not hits:
            return f"No passages matched '{query}' closely enough."
        blocks = [f"[{i}] {h.reference}  (match {h.score:.2f})\n{h.text}"
                  for i, h in enumerate(hits, start=1)]
        return "\n\n".join(blocks)


def format_size(size: int) -> str:
    for unit in ("bytes", "KB", "MB"):
        if size < 1024 or unit == "MB":
            return f"{size} {unit}" if unit == "bytes" else f"{size:.1f} {unit}"
        size /= 1024
    return f"{size} bytes"


def describe_call(name: str, arguments: dict) -> str:
    """A short readable form like read_file(path='notes.md'), for the screen and log."""
    args = ", ".join(f"{k}={v!r}" for k, v in arguments.items())
    return f"{name}({args})"
