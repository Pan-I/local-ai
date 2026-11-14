"""Decides which files the tools may touch.

Only files inside the folders listed under `tools.folders` are allowed. A path
from the model is turned into a full path first (following any shortcuts or
"..") and then checked, so tricks like "knowledge/../../Windows" are refused.
Hidden files and folders (.git, .env, .venv, ...) and anything matching
`tools.blocked` are refused even inside an allowed folder.
"""

from fnmatch import fnmatch
from pathlib import Path

from app.config import ROOT

# Never useful to the model, wherever they are. Hidden names are blocked too.
SKIP_DIRS = {"__pycache__", "node_modules", "venv", "env", "site-packages"}


class ToolError(Exception):
    """A tool couldn't do what was asked. The message is sent to the model."""


class AllowedFolders:
    def __init__(self, folders: list[Path], blocked: list[str] = (), base: Path = ROOT):
        # Relative paths, in the config and from the model, start at `base`
        # (the local-ai folder).
        self.base = base.resolve()
        self.folders = [self._full(f) for f in folders]
        self.blocked = list(blocked)

    def _full(self, path: str | Path) -> Path:
        path = Path(path).expanduser()
        return (path if path.is_absolute() else self.base / path).resolve()

    def display(self, path: Path) -> str:
        """How a path is shown to the model and in the log: relative to the
        local-ai folder when it's inside it, otherwise the full path."""
        try:
            shown = path.relative_to(self.base).as_posix()
            return shown if shown != "." else "."
        except ValueError:
            return path.as_posix()

    def check(self, path: str) -> Path:
        """The full path for `path`, or ToolError if the tools may not use it."""
        if not isinstance(path, str) or not path.strip():
            raise ToolError("a path is required")
        full = self._full(path.strip())
        root = next((f for f in self.folders if full == f or full.is_relative_to(f)), None)
        if root is None:
            raise ToolError(f"'{path}' is outside the allowed folders ({self.describe()})")
        if self.is_blocked(full, root):
            raise ToolError(f"'{path}' is blocked by the app's rules")
        return full

    def is_blocked(self, full: Path, root: Path | None = None) -> bool:
        if root is None:
            root = next((f for f in self.folders if full == f or full.is_relative_to(f)), None)
            if root is None:
                return True
        inside = full.relative_to(root).parts
        if any(part.startswith(".") or part in SKIP_DIRS for part in inside):
            return True
        shown = self.display(full)
        return any(fnmatch(shown, pattern) or fnmatch(full.name, pattern)
                   for pattern in self.blocked)

    def describe(self) -> str:
        return ", ".join(self.display(f) for f in self.folders)
