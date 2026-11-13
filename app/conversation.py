"""Keeps the message history for one chat session, and saves it if given a store."""

import sqlite3
from typing import Callable

from app.ollama_client import ChatResult, OllamaClient
from storage.database import ConversationStore


class Conversation:
    def __init__(
        self,
        client: OllamaClient,
        model: str,
        system_prompt: str = "",
        think: bool | None = None,
        store: ConversationStore | None = None,
    ):
        self.client = client
        self.model = model
        self.system_prompt = system_prompt
        self.think = think
        self.store = store
        # Set when the first reply is saved, so an empty chat isn't stored.
        self.conversation_id: int | None = None
        # The error from the last failed save, or None if it worked.
        self.save_error: str | None = None
        self.messages: list[dict] = []
        self.reset()

    def reset(self) -> None:
        """Start a new conversation. A saved one stays in the database."""
        self.conversation_id = None
        self.messages = []
        if self.system_prompt:
            self.messages.append({"role": "system", "content": self.system_prompt})

    def load(self, conversation_id: int) -> bool:
        """Continue a saved conversation. Returns False if there's no such id."""
        if self.store is None:
            return False
        saved = self.store.load(conversation_id)
        if saved is None:
            return False
        self.conversation_id = saved.id
        # Keep the system prompt the conversation started with.
        self.system_prompt = saved.system_prompt
        self.messages = []
        if saved.system_prompt:
            self.messages.append({"role": "system", "content": saved.system_prompt})
        self.messages.extend(saved.messages)
        return True

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
        self._save(text, result)
        return result

    def _save(self, text: str, result: ChatResult) -> None:
        # A failed save shouldn't lose the reply on screen, so it's reported
        # through save_error instead of raised.
        self.save_error = None
        if self.store is None:
            return
        try:
            if self.conversation_id is None:
                self.conversation_id = self.store.create_conversation(
                    self.model, self.system_prompt
                )
            self.store.add_turn(self.conversation_id, text, result)
        except sqlite3.Error as e:
            self.save_error = str(e)
