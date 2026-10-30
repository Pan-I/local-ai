"""Command-line chat with a model served by Ollama.

Run from the local-ai folder:
    python -m app.main
    python -m app.main --model qwen3:0.6b
    python -m app.main --check
"""

import argparse
import sys

from app.config import ROOT, load_config
from app.conversation import Conversation
from app.ollama_client import OllamaClient, OllamaError

HELP = """Commands:
  /help           show this help
  /models         list models installed on the Ollama server
  /model <name>   switch model (keeps the conversation)
  /think [on|off|default]  show or set whether thinking models reason first
  /reset          clear the conversation
  /exit           quit"""


SYSTEM_PROMPT_PATH = ROOT / "prompts" / "system" / "system.txt"

THINK_SETTINGS = {"on": True, "off": False, "default": None}


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
    args = parser.parse_args(argv)

    inference = load_config().get("inference", {})
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

    convo = Conversation(client, model, load_system_prompt(), think=think)
    print(f"Chatting with {model} at {base_url}. Type /help for commands.")

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
                print("Conversation cleared.")
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


if __name__ == "__main__":
    sys.exit(main())
