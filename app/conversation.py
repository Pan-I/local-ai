"""Keeps the message history for one chat session (in memory for now)."""

from typing import Callable

from app.ollama_client import ChatResult, OllamaClient


class Conversation:
    def __init__(
        self,
        client: OllamaClient,
        model: str,
        system_prompt: str = "",
        think: bool | None = None,
    ):
        self.client = client
        self.model = model
        self.system_prompt = system_prompt
        self.think = think
        self.messages: list[dict] = []
        self.reset()

    def reset(self) -> None:
        self.messages = []
        if self.system_prompt:
            self.messages.append({"role": "system", "content": self.system_prompt})

    def send(
        self,
        text: str,
        stream: bool = True,
        on_token: Callable[[str], None] | None = None,
    ) -> ChatResult:
        self.messages.append({"role": "user", "content": text})
        try:
            result = self.client.chat(
                self.model, self.messages, stream=stream, on_token=on_token, think=self.think
            )
        except BaseException:
            # Drop the unanswered message so a retry doesn't send it twice.
            self.messages.pop()
            raise
        # Thinking text is kept on the result only; it isn't sent back in later turns.
        self.messages.append({"role": "assistant", "content": result.content})
        return result
