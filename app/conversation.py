"""Keeps the message history for one chat session, and saves it if given a store."""

import sqlite3
from typing import Callable

from app.ollama_client import ChatResult, OllamaClient, OllamaError
from retrieval.retriever import Retriever, build_prompt
from retrieval.store import Hit
from storage.database import ConversationStore


class Conversation:
    def __init__(
        self,
        client: OllamaClient,
        model: str,
        system_prompt: str = "",
        think: bool | None = None,
        store: ConversationStore | None = None,
        retriever: Retriever | None = None,
    ):
        self.client = client
        self.model = model
        self.system_prompt = system_prompt
        self.think = think
        self.store = store
        # Looks up passages from your documents for each message, when turned on.
        self.retriever = retriever
        self.use_knowledge = retriever is not None
        # The passages the last reply was given, and why the lookup failed if it did.
        self.last_sources: list[Hit] = []
        self.knowledge_error: str | None = None
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
        hits = self._retrieve(text)
        # The sources go into this request only. The history keeps just the
        # question, so earlier sources don't fill up the model's context later.
        outgoing = self.messages + [{"role": "user", "content": build_prompt(text, hits)}]
        self.messages.append({"role": "user", "content": text})
        try:
            result = self.client.chat(
                self.model, outgoing, stream=stream, on_token=on_token, think=self.think
            )
        except BaseException:
            # Drop the unanswered message so a retry doesn't send it twice.
            self.messages.pop()
            raise
        # Thinking text is kept on the result only; it isn't sent back in later turns.
        self.messages.append({"role": "assistant", "content": result.content})
        self.last_sources = hits
        self._save(text, result, hits)
        return result

    def _retrieve(self, text: str) -> list[Hit]:
        # If the lookup fails (say, the embedding model isn't installed), the
        # question is still answered, just without sources.
        self.knowledge_error = None
        if self.retriever is None or not self.use_knowledge:
            return []
        try:
            return self.retriever.search(text)
        except (OllamaError, sqlite3.Error) as e:
            self.knowledge_error = str(e)
            return []

    def _save(self, text: str, result: ChatResult, sources: list[Hit] = ()) -> None:
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
            self.store.add_turn(self.conversation_id, text, result, sources)
        except sqlite3.Error as e:
            self.save_error = str(e)
