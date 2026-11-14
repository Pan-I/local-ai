"""Small client for the Ollama HTTP API.

The rest of the app talks to this module only, so the model runtime can be
swapped later without touching chat logic.
"""

import http.client
import json
import urllib.error
import urllib.request
from dataclasses import dataclass, field
from typing import Callable, Iterable


class OllamaError(Exception):
    pass


@dataclass
class ToolCall:
    """The model asking the app to run one tool, e.g. read_file(path="notes.md")."""
    name: str
    arguments: dict
    # The call exactly as Ollama sent it, to send back in the conversation.
    raw: dict = field(default_factory=dict)


@dataclass
class ChatResult:
    content: str
    model: str
    thinking: str = ""
    # Tools the model asked for instead of (or before) answering. Empty for a plain reply.
    tool_calls: list[ToolCall] = field(default_factory=list)
    prompt_tokens: int = 0
    completion_tokens: int = 0
    prompt_seconds: float = 0.0
    generation_seconds: float = 0.0
    load_seconds: float = 0.0
    total_seconds: float = 0.0

    @property
    def tokens_per_second(self) -> float:
        if self.generation_seconds <= 0:
            return 0.0
        return self.completion_tokens / self.generation_seconds

    @property
    def prompt_tokens_per_second(self) -> float:
        if self.prompt_seconds <= 0:
            return 0.0
        return self.prompt_tokens / self.prompt_seconds


class OllamaClient:
    def __init__(self, base_url: str, timeout: float = 300):
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout

    def _request(self, method: str, path: str, body: dict | None = None):
        data = json.dumps(body).encode("utf-8") if body is not None else None
        req = urllib.request.Request(
            self.base_url + path,
            data=data,
            method=method,
            headers={"Content-Type": "application/json"},
        )
        try:
            return urllib.request.urlopen(req, timeout=self.timeout)
        except urllib.error.HTTPError as e:
            detail = e.read().decode("utf-8", errors="replace")
            try:
                detail = json.loads(detail).get("error", detail)
            except (ValueError, AttributeError):
                pass
            raise OllamaError(f"Ollama returned HTTP {e.code}: {detail}") from e
        except urllib.error.URLError as e:
            raise OllamaError(f"Could not reach Ollama at {self.base_url}: {e.reason}") from e
        except TimeoutError as e:
            raise OllamaError(f"Timed out waiting for Ollama at {self.base_url}") from e

    def version(self) -> str:
        with self._request("GET", "/api/version") as resp:
            return json.load(resp).get("version", "unknown")

    def list_models(self) -> list[str]:
        with self._request("GET", "/api/tags") as resp:
            return [m["name"] for m in json.load(resp).get("models", [])]

    def embed(self, model: str, texts: list[str]) -> list[list[float]]:
        """Turn each text into an embedding (a list of numbers describing its meaning).

        Uses Ollama's /api/embed, which takes several texts in one request.
        Returns one vector per text, in the same order.
        """
        if not texts:
            return []
        body = {"model": model, "input": list(texts)}
        with self._request("POST", "/api/embed", body) as resp:
            try:
                data = json.load(resp)
            except TimeoutError as e:
                raise OllamaError(f"Timed out waiting for Ollama at {self.base_url}") from e
            except (OSError, http.client.HTTPException) as e:
                raise OllamaError(f"Lost connection to Ollama at {self.base_url}: {e}") from e
            except ValueError as e:
                raise OllamaError(f"Ollama sent a reply that isn't valid JSON: {e}") from e
        vectors = data.get("embeddings")
        if not isinstance(vectors, list) or len(vectors) != len(texts):
            raise OllamaError(f"Ollama returned {len(vectors or [])} embeddings for {len(texts)} texts")
        return vectors

    def chat(
        self,
        model: str,
        messages: Iterable[dict],
        stream: bool = True,
        on_token: Callable[[str], None] | None = None,
        think: bool | None = None,
        tools: list[dict] | None = None,
    ) -> ChatResult:
        """Send a chat request. If streaming, on_token is called with each text chunk.

        think=False asks a thinking model (like qwen3) to answer directly; None
        leaves it to the model's default. Thinking text is returned separately in
        ChatResult.thinking and never passed to on_token.

        tools describes the tools the model may ask for (see agent/tools.py). Any
        requests come back in ChatResult.tool_calls; running them is up to the caller.
        """
        body = {"model": model, "messages": list(messages), "stream": stream}
        if think is not None:
            body["think"] = think
        if tools:
            body["tools"] = tools
        parts: list[str] = []
        thinking: list[str] = []
        calls: list[dict] = []
        final: dict = {}
        try:
            resp = self._request("POST", "/api/chat", body)
        except OllamaError as e:
            # Models without thinking support may reject the setting; asking for
            # no thinking is then the same as not asking at all.
            if think is False and "does not support thinking" in str(e):
                del body["think"]
                resp = self._request("POST", "/api/chat", body)
            else:
                raise
        with resp:
            # Reading the reply can fail after the request succeeded (dropped
            # connection, timeout, garbled line), so report those as OllamaError too.
            try:
                if stream:
                    for line in resp:
                        line = line.strip()
                        if not line:
                            continue
                        chunk = json.loads(line)
                        if "error" in chunk:
                            raise OllamaError(chunk["error"])
                        message = chunk.get("message", {})
                        calls.extend(message.get("tool_calls") or [])
                        if message.get("thinking"):
                            thinking.append(message["thinking"])
                        text = message.get("content", "")
                        if text:
                            parts.append(text)
                            if on_token:
                                on_token(text)
                        if chunk.get("done"):
                            final = chunk
                else:
                    final = json.load(resp)
                    if "error" in final:
                        raise OllamaError(final["error"])
                    message = final.get("message", {})
                    calls.extend(message.get("tool_calls") or [])
                    thinking.append(message.get("thinking", ""))
                    parts.append(message.get("content", ""))
            except TimeoutError as e:
                raise OllamaError(f"Timed out waiting for Ollama at {self.base_url}") from e
            except (OSError, http.client.HTTPException) as e:
                raise OllamaError(f"Lost connection to Ollama at {self.base_url}: {e}") from e
            except ValueError as e:
                raise OllamaError(f"Ollama sent a reply that isn't valid JSON: {e}") from e
        if not final:
            raise OllamaError("Ollama's reply ended before it was complete")

        ns = 1e9
        return ChatResult(
            content="".join(parts),
            model=final.get("model", model),
            thinking="".join(thinking),
            tool_calls=[parse_tool_call(c) for c in calls],
            prompt_tokens=final.get("prompt_eval_count", 0),
            completion_tokens=final.get("eval_count", 0),
            prompt_seconds=final.get("prompt_eval_duration", 0) / ns,
            generation_seconds=final.get("eval_duration", 0) / ns,
            load_seconds=final.get("load_duration", 0) / ns,
            total_seconds=final.get("total_duration", 0) / ns,
        )


def parse_tool_call(raw: dict) -> ToolCall:
    """Ollama sends {"function": {"name": ..., "arguments": {...}}}. Some models
    send the arguments as a JSON string instead, so that's accepted too."""
    function = raw.get("function") or {}
    arguments = function.get("arguments") or {}
    if isinstance(arguments, str):
        try:
            arguments = json.loads(arguments)
        except ValueError:
            arguments = {"_unparsed": arguments}
    if not isinstance(arguments, dict):
        arguments = {"_unparsed": arguments}
    return ToolCall(name=str(function.get("name", "")), arguments=arguments, raw=raw)
