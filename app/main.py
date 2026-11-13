"""Command-line chat with a model served by Ollama.

Run from the local-ai folder:
    python -m app.main
    python -m app.main --model qwen3:0.6b
    python -m app.main --check
    python -m app.main --no-save
"""

import argparse
import sqlite3
import sys
from datetime import datetime
from pathlib import Path

from app.config import ROOT, load_config
from app.conversation import Conversation
from app.ollama_client import OllamaClient, OllamaError
from storage.database import ConversationStore

HELP = """Commands:
  /help           show this help
  /models         list models installed on the Ollama server
  /model <name>   switch model (keeps the conversation)
  /think [on|off|default]  show or set whether thinking models reason first
  /history        list recent saved conversations
  /load <id>      continue a saved conversation
  /reset          start a new conversation (the old one stays saved)
  /exit           quit"""


SYSTEM_PROMPT_PATH = ROOT / "prompts" / "system" / "system.txt"

THINK_SETTINGS = {"on": True, "off": False, "default": None}

DEFAULT_DATABASE = "data/databases/local_ai.db"


def normalize_base_url(url: str) -> str:
    url = url.strip().rstrip("/")
    if not url.startswith(("http://", "https://")):
        url = "http://" + url
    # Allow an address without a port, e.g. "192.168.1.86".
    if url.count(":") == 1:
        url += ":11434"
    return url


def load_system_prompt() -> str:
    if SYSTEM_PROMPT_PATH.exists():
        return SYSTEM_PROMPT_PATH.read_text(encoding="utf-8").strip()
    return ""


def check(client: OllamaClient, model: str) -> int:
    try:
        print(f"Ollama {client.version()} at {client.base_url}")
        models = client.list_models()
    except OllamaError as e:
        print(f"Error: {e}")
        return 1
    print("Installed models: " + (", ".join(models) or "(none)"))
    if model not in models:
        print(f"Warning: configured model '{model}' is not installed.")
        return 1
    print(f"Configured model '{model}' is available.")
    return 0


def database_path(storage: dict) -> Path:
    """The database path from the config; a relative path is inside the local-ai folder."""
    path = Path(storage.get("database") or DEFAULT_DATABASE)
    return path if path.is_absolute() else ROOT / path


def open_store(storage: dict, no_save: bool) -> ConversationStore | None:
    if no_save or not storage.get("save_conversations", True):
        return None
    path = database_path(storage)
    try:
        return ConversationStore(path)
    except (sqlite3.Error, OSError) as e:
        print(f"Warning: can't open the database at {path} ({e}). This chat won't be saved.")
        return None


def print_history(store: ConversationStore | None) -> None:
    if store is None:
        print("Saving is off, so there's no history. Run without --no-save to use it.")
        return
    conversations = store.list_conversations()
    if not conversations:
        print("No saved conversations yet.")
        return
    for c in conversations:
        # Saved in UTC; shown in this computer's time zone.
        when = datetime.fromisoformat(c.updated_at).astimezone().strftime("%Y-%m-%d %H:%M")
        print(f"{c.id:>4}  {when}  {c.model:<16}  {c.message_count:>3} msgs  {c.title}")


def print_stats(result) -> None:
    print(
        f"[{result.model} | {result.completion_tokens} tokens, "
        f"{result.tokens_per_second:.1f} tok/s | prompt {result.prompt_tokens} tokens, "
        f"{result.prompt_tokens_per_second:.1f} tok/s | load {result.load_seconds:.1f}s | "
        f"total {result.total_seconds:.1f}s]"
    )


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="Chat with a local Ollama model.")
    parser.add_argument("--base-url", help="override inference.base_url from the config for this run")
    parser.add_argument("--model", help="model name, e.g. qwen3:4b")
    parser.add_argument("--check", action="store_true", help="test the connection and exit")
    parser.add_argument("--no-stream", action="store_true", help="wait for the full reply")
    parser.add_argument("--think", choices=["on", "off"], help="override inference.think for this run")
    parser.add_argument("--no-save", action="store_true", help="don't save this chat to the database")
    args = parser.parse_args(argv)

    config = load_config()
    inference = config.get("inference", {})
    base_url = args.base_url or inference.get("base_url")
    model = args.model or inference.get("default_model")
    if not base_url or not model:
        print("Config error: inference.base_url and inference.default_model must be set.")
        return 1
    base_url = normalize_base_url(base_url)
    timeout = float(inference.get("timeout", 300))
    stream = not args.no_stream
    # None leaves it to the model: qwen3 then reasons, and Ollama keeps the
    # reasoning out of the reply.
    think = {"on": True, "off": False}.get(args.think, inference.get("think"))

    client = OllamaClient(base_url, timeout=timeout)
    if args.check:
        return check(client, model)

    store = open_store(config.get("storage", {}), args.no_save)
    convo = Conversation(client, model, load_system_prompt(), think=think, store=store)
    print(f"Chatting with {model} at {base_url}. Type /help for commands.")
    if store:
        print(f"Saving conversations to {store.path}")
    else:
        print("This chat won't be saved.")

    while True:
        try:
            text = input("\nYou: ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            return 0
        if not text:
            continue

        if text.startswith("/"):
            cmd, _, arg = text.partition(" ")
            if cmd in ("/exit", "/quit"):
                return 0
            elif cmd == "/help":
                print(HELP)
            elif cmd == "/reset":
                convo.reset()
                print("Started a new conversation." + (" The old one is saved." if store else ""))
            elif cmd == "/history":
                print_history(store)
            elif cmd == "/load":
                if store is None:
                    print("Saving is off, so there's nothing to load.")
                elif not arg.strip().isdigit():
                    print("Use /load <id>. Type /history to see the ids.")
                elif convo.load(int(arg)):
                    turns = sum(1 for m in convo.messages if m["role"] == "user")
                    print(f"Loaded conversation {convo.conversation_id} ({turns} messages from you). "
                          f"Still using {convo.model}.")
                else:
                    print(f"No saved conversation with id {arg.strip()}.")
            elif cmd == "/models":
                try:
                    print("\n".join(client.list_models()) or "(no models installed)")
                except OllamaError as e:
                    print(f"Error: {e}")
            elif cmd == "/model":
                if arg.strip():
                    convo.model = arg.strip()
                    print(f"Now using {convo.model}.")
                else:
                    print(f"Current model: {convo.model}")
            elif cmd == "/think":
                setting = arg.strip().lower()
                if setting in THINK_SETTINGS:
                    convo.think = THINK_SETTINGS[setting]
                elif setting:
                    print("Use /think on, /think off, or /think default.")
                    continue
                current = {True: "on", False: "off", None: "the model's default"}[convo.think]
                print(f"Thinking is {current}.")
            else:
                print("Unknown command. Type /help.")
            continue

        print("\nAI: ", end="", flush=True)
        try:
            result = convo.send(
                text,
                stream=stream,
                on_token=lambda t: print(t, end="", flush=True),
            )
        except OllamaError as e:
            print(f"\nError: {e}")
            continue
        except KeyboardInterrupt:
            print("\n(interrupted)")
            continue
        if not stream:
            print(result.content, end="")
        print()
        print_stats(result)
        if convo.save_error:
            print(f"Warning: this reply wasn't saved ({convo.save_error}).")


if __name__ == "__main__":
    sys.exit(main())
