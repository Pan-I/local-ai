"""Command-line chat with a model served by Ollama.

Run from the local-ai folder:
    python -m app.main
    python -m app.main --model qwen3:0.6b
    python -m app.main --check
    python -m app.main --no-save
    python -m app.main --index       # read the approved folders into the knowledge index
    python -m app.main --no-tools    # chat without letting the model use tools
"""

import argparse
import sqlite3
import sys
from datetime import datetime
from pathlib import Path

from agent.folders import ToolError
from agent.tools import Toolbox, ToolOutcome, ToolSettings, describe_call
from app.config import ROOT, load_config
from app.conversation import Conversation
from app.ollama_client import OllamaClient, OllamaError, ToolCall
from retrieval.indexer import Indexer, IndexReport, KnowledgeSettings
from retrieval.retriever import Retriever
from retrieval.store import KnowledgeStore
from storage.database import ConversationStore

HELP = """Commands:
  /help           show this help
  /models         list models installed on the Ollama server
  /model <name>   switch model (keeps the conversation)
  /think [on|off|default]  show or set whether thinking models reason first
  /history        list recent saved conversations
  /load <id>      continue a saved conversation
  /reset          start a new conversation (the old one stays saved)
  /kb [on|off]    show or set whether answers use your documents
  /sources        show the passages the last answer was given
  /index          update the knowledge index from the approved folders
  /embed [name]   show or switch the embedding model (each has its own index)
  /tools [on|off] show or set whether the model can use tools (read files, search)
  /tools log      show the last 10 tool calls
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


def open_knowledge(settings: KnowledgeSettings) -> KnowledgeStore | None:
    adopt_old_index(settings)
    try:
        return KnowledgeStore(settings.index_path)
    except (sqlite3.Error, OSError) as e:
        print(f"Warning: can't open the knowledge index at {settings.index_path} ({e}).")
        return None


def run_index(client: OllamaClient, knowledge: KnowledgeStore, settings: KnowledgeSettings,
              rebuild: bool = False) -> int:
    print(f"Indexing with {settings.embedding_model} at {client.base_url}:")
    for folder in settings.folders:
        print(f"  {folder}")
    try:
        report = Indexer(client, knowledge, settings, progress=print).run(rebuild=rebuild)
    except OllamaError as e:
        print(f"Error: {e}")
        if "not found" in str(e):
            print(f"On the ai-pc, run:  ollama pull {settings.embedding_model}")
        print("Files finished before the error are saved; run it again to continue.")
        return 1
    print_report(report, knowledge)
    return 0


def adopt_old_index(settings: KnowledgeSettings) -> None:
    """Earlier versions kept one index at knowledge.database. If it was built with
    this model, rename it to this model's own file instead of re-embedding."""
    old, new = settings.database, settings.index_path
    if new.exists() or not old.is_file():
        return
    try:
        store = KnowledgeStore(old)
        built_with = store.get_setting("embedding_model")
        store.close()
        if built_with == settings.embedding_model:
            old.rename(new)
    except (sqlite3.Error, OSError):
        pass  # leave it; the model's index is simply built fresh


def is_installed(name: str, installed: list[str]) -> bool:
    return name in installed or f"{name}:latest" in installed


def looks_like_embedding_model(name: str) -> bool:
    return any(word in name.lower() for word in ("embed", "minilm", "bge", "e5", "gte"))


def print_report(report: IndexReport, knowledge: KnowledgeStore) -> None:
    for folder in report.missing_folders:
        print(f"  Folder not found, skipped: {folder}")
    for path in report.removed:
        print(f"  removed {path}")
    files, chunks = knowledge.counts()
    print(
        f"Done in {report.seconds:.1f}s: {len(report.added)} added, {len(report.updated)} updated, "
        f"{report.unchanged} unchanged, {len(report.removed)} removed, "
        f"{len(report.skipped)} skipped. The index has {chunks} chunks from {files} files."
    )


def print_sources(hits, verbose: bool = False) -> None:
    if not hits:
        print("No passages from your documents were used.")
        return
    print("Sources:")
    for i, hit in enumerate(hits, start=1):
        print(f"  [{i}] {hit.reference}  match {hit.score:.2f}")
        if verbose:
            preview = " ".join(hit.text.split())
            print(f"      {preview[:200]}{'...' if len(preview) > 200 else ''}")


class ReplyPrinter:
    """Prints a reply as it streams, with a line for each tool the model uses."""

    def __init__(self):
        self.after_tool = False

    def token(self, text: str) -> None:
        if self.after_tool:
            print("AI: ", end="")
            self.after_tool = False
        print(text, end="", flush=True)

    def tool(self, call: ToolCall) -> None:
        # Start a new line, unless the last tool's line already ended.
        start = "" if self.after_tool else "\n"
        print(f"{start}  [tool] {describe_call(call.name, call.arguments)}", end="", flush=True)

    def tool_done(self, outcome: ToolOutcome) -> None:
        if outcome.ok:
            print(f"  -> ok, {len(outcome.text):,} characters")
        else:
            print(f"  -> {outcome.text}")
        self.after_tool = True


def print_tool_log(store: ConversationStore | None) -> None:
    if store is None:
        print("Saving is off, so tool calls aren't logged. Run without --no-save to log them.")
        return
    calls = store.recent_tool_calls()
    if not calls:
        print("No tool calls logged yet.")
        return
    for c in reversed(calls):
        when = datetime.fromisoformat(c.created_at).astimezone().strftime("%Y-%m-%d %H:%M")
        convo = f"chat {c.conversation_id}" if c.conversation_id else "unsaved"
        print(f"{c.id:>4}  {when}  {convo:<9}  {c.status:<5}  {describe_call(c.tool, c.arguments)}")


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
    parser.add_argument("--index", action="store_true",
                        help="update the knowledge index from knowledge.folders and exit")
    parser.add_argument("--rebuild-index", action="store_true",
                        help="re-embed every document from scratch and exit")
    parser.add_argument("--embed-model", help="override knowledge.embedding_model for this run")
    parser.add_argument("--no-kb", action="store_true",
                        help="don't use your documents in this chat")
    parser.add_argument("--no-tools", action="store_true",
                        help="don't let the model use tools in this chat")
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

    settings = KnowledgeSettings.from_config(config)
    if args.embed_model:
        settings = settings.with_model(args.embed_model)
    knowledge = open_knowledge(settings)
    if args.index or args.rebuild_index:
        if knowledge is None:
            return 1
        try:
            return run_index(client, knowledge, settings, rebuild=args.rebuild_index)
        finally:
            knowledge.close()

    store = open_store(config.get("storage", {}), args.no_save)
    retriever = Retriever(client, knowledge, settings) if knowledge else None
    tool_settings = ToolSettings.from_config(config)
    toolbox = Toolbox(tool_settings)
    convo = Conversation(client, model, load_system_prompt(), think=think, store=store,
                         retriever=retriever, toolbox=toolbox,
                         max_tool_rounds=tool_settings.max_iterations)
    convo.use_tools = tool_settings.enabled and not args.no_tools

    def search_documents(query: str):
        # Uses whichever index /embed last switched to.
        if convo.retriever is None:
            raise ToolError("the knowledge index isn't available")
        return convo.retriever.search(query)

    toolbox.search = search_documents
    print(f"Chatting with {model} at {base_url}. Type /help for commands.")
    if store:
        print(f"Saving conversations to {store.path}")
    else:
        print("This chat won't be saved.")
    if knowledge:
        files, chunks = knowledge.counts()
        convo.use_knowledge = settings.use_in_chat and not args.no_kb and chunks > 0
        if chunks == 0:
            print("Your documents aren't indexed yet. Add files to an approved folder and "
                  "run /index (or python -m app.main --index).")
        else:
            state = "on" if convo.use_knowledge else "off (/kb on to use it)"
            print(f"Knowledge: {chunks} chunks from {files} files "
                  f"(embedded with {settings.embedding_model}), {state}.")
    if convo.use_tools:
        print(f"Tools: on. The model can read files in: {toolbox.folders.describe()}")
    else:
        print("Tools: off (/tools on to use them).")

    try:
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
                elif cmd == "/kb":
                    setting = arg.strip().lower()
                    if knowledge is None:
                        print("The knowledge index couldn't be opened, so this is unavailable.")
                        continue
                    if setting in ("on", "off"):
                        convo.use_knowledge = setting == "on"
                    elif setting:
                        print("Use /kb on or /kb off.")
                        continue
                    files, chunks = knowledge.counts()
                    state = "on" if convo.use_knowledge else "off"
                    print(f"Using your documents is {state}. The index has {chunks} chunks "
                          f"from {files} files, built with {settings.embedding_model}.")
                    if convo.use_knowledge and chunks == 0:
                        print("The index is empty. Run /index after adding files.")
                elif cmd == "/sources":
                    print_sources(convo.last_sources, verbose=True)
                elif cmd == "/index":
                    if knowledge is None:
                        print("The knowledge index couldn't be opened, so this is unavailable.")
                    else:
                        was_empty = knowledge.counts()[1] == 0
                        # Turn retrieval on if this filled an empty index.
                        if run_index(client, knowledge, settings) == 0 and was_empty \
                                and knowledge.counts()[1]:
                            convo.use_knowledge = settings.use_in_chat and not args.no_kb
                elif cmd == "/embed":
                    name = arg.strip()
                    if not name:
                        print(f"Embedding model: {settings.embedding_model}")
                        try:
                            others = [m for m in client.list_models() if looks_like_embedding_model(m)]
                            print("Installed embedding models: " + (", ".join(others) or "(none found)"))
                        except OllamaError as e:
                            print(f"Couldn't list the installed models ({e}).")
                        print("Switch with /embed <name>.")
                        continue
                    try:
                        installed = client.list_models()
                    except OllamaError as e:
                        print(f"Error: {e}")
                        continue
                    if not is_installed(name, installed):
                        print(f"'{name}' isn't installed on the Ollama server. "
                              f"On the ai-pc, run:  ollama pull {name}")
                        continue
                    new_settings = settings.with_model(name)
                    new_knowledge = open_knowledge(new_settings)
                    if new_knowledge is None:
                        continue
                    if knowledge:
                        knowledge.close()
                    settings, knowledge = new_settings, new_knowledge
                    convo.retriever = Retriever(client, knowledge, settings)
                    files, chunks = knowledge.counts()
                    print(f"Now using {name} for embeddings (this run only; set "
                          f"knowledge.embedding_model to keep it).")
                    if chunks:
                        print(f"Its index has {chunks} chunks from {files} files. "
                              f"Match scores differ between models, so check /sources.")
                        if not convo.use_knowledge:
                            print("Document lookup is off; /kb on turns it on.")
                    else:
                        convo.use_knowledge = False
                        print("It has no index yet, so document lookup is off. Run /index to "
                              "build it (every file gets embedded, so this can take a while).")
                elif cmd == "/tools":
                    setting = arg.strip().lower()
                    if setting == "log":
                        print_tool_log(store)
                        continue
                    if setting in ("on", "off"):
                        convo.use_tools = setting == "on"
                    elif setting:
                        print("Use /tools on, /tools off, or /tools log.")
                        continue
                    state = "on" if convo.use_tools else "off"
                    print(f"Tools are {state}. Allowed folders: {toolbox.folders.describe()}")
                    print("Tools: " + ", ".join(toolbox.tools))
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
            printer = ReplyPrinter()
            try:
                result = convo.send(
                    text,
                    stream=stream,
                    on_token=printer.token,
                    on_tool=printer.tool,
                    on_tool_done=printer.tool_done,
                )
            except OllamaError as e:
                print(f"\nError: {e}")
                continue
            except KeyboardInterrupt:
                print("\n(interrupted)")
                continue
            if not stream:
                printer.token(result.content)
            print()
            if convo.last_sources:
                print_sources(convo.last_sources)
            elif convo.use_knowledge and not convo.knowledge_error:
                print("(No closely matching passages in your documents.)")
            if convo.knowledge_error:
                print(f"Warning: couldn't search your documents ({convo.knowledge_error}).")
            if convo.tools_unsupported_error:
                print(f"Note: {convo.model} can't use tools, so it answered without them. "
                      "Tools are off until you turn them on with /tools on.")
            if convo.tool_log_error:
                print(f"Warning: a tool call wasn't logged ({convo.tool_log_error}).")
            print_stats(result)
            if convo.save_error:
                print(f"Warning: this reply wasn't saved ({convo.save_error}).")
    finally:
        # Close the database files so nothing is left locked (Windows cares).
        if knowledge:
            knowledge.close()
        if store:
            store.close()


if __name__ == "__main__":
    sys.exit(main())
