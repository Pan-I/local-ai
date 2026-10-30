"""Small client for the Ollama HTTP API.

The rest of the app talks to this module only, so the model runtime can be
swapped later without touching chat logic.
"""

import http.client
import json
import urllib.error
import urllib.request
from dataclasses import dataclass
from typing import Callable, Iterable


class OllamaError(Exception):
    pass


@dataclass
class ChatResult:
    content: str
    model: str
    thinking: str = ""
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

    def chat(
        self,
        model: str,
        messages: Iterable[dict],
        stream: bool = True,
        on_token: Callable[[str], None] | None = None,
        think: bool | None = None,
    ) -> ChatResult:
        """Send a chat request. If streaming, on_token is called with each text chunk.

        think=False asks a thinking model (like qwen3) to answer directly; None
        leaves it to the model's default. Thinking text is returned separately in
        ChatResult.thinking and never passed to on_token.
        """
        body = {"model": model, "messages": list(messages), "stream": stream}
        if think is not None:
            body["think"] = think
        parts: list[str] = []
        thinking: list[str] = []
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
            prompt_tokens=final.get("prompt_eval_count", 0),
            completion_tokens=final.get("eval_count", 0),
            prompt_seconds=final.get("prompt_eval_duration", 0) / ns,
            generation_seconds=final.get("eval_duration", 0) / ns,
            load_seconds=final.get("load_duration", 0) / ns,
            total_seconds=final.get("total_duration", 0) / ns,
        )
