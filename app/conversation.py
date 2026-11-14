"""Keeps the message history for one chat session, and saves it if given a store."""

import sqlite3
from typing import Callable

from agent.tools import Toolbox, ToolOutcome
from app.ollama_client import ChatResult, OllamaClient, OllamaError, ToolCall
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
        toolbox: Toolbox | None = None,
        max_tool_rounds: int = 10,
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
        # Tools the model may ask the app to run (agent/tools.py), when turned on.
        self.toolbox = toolbox
        self.use_tools = toolbox is not None
        # How many times one message may go back and forth with tool results.
        self.max_tool_rounds = max_tool_rounds
        # The tools the last reply used, and the log error if logging failed.
        self.last_tool_calls: list[ToolOutcome] = []
        self.tool_log_error: str | None = None
        # Set when the model turns out not to support tools.
        self.tools_unsupported_error: str | None = None
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
        on_tool: Callable[[ToolCall], None] | None = None,
        on_tool_done: Callable[[ToolOutcome], None] | None = None,
    ) -> ChatResult:
        """Send a message and return the model's answer.

        With tools on, the model may ask for tools instead of answering. Each
        request is run, its result is added to the messages, and the model is
        asked again, until it answers or max_tool_rounds is reached. on_tool is
        called before each tool runs and on_tool_done after.
        """
        hits = self._retrieve(text)
        # The sources go into this request only. The history keeps just the
        # question, so earlier sources don't fill up the model's context later.
        # Tool requests and results are kept out of the history the same way.
        outgoing = self.messages + [{"role": "user", "content": build_prompt(text, hits)}]
        self.messages.append({"role": "user", "content": text})
        self.last_tool_calls = []
        self.tool_log_error = None
        self.tools_unsupported_error = None
        log_ids: list[int] = []
        results: list[ChatResult] = []
        try:
            tools = self.toolbox.definitions() if self.toolbox and self.use_tools else None
            for round_number in range(self.max_tool_rounds + 1):
                # On the last round tools aren't offered, so the model has to answer.
                offer = tools if round_number < self.max_tool_rounds else None
                result = self._chat(outgoing, stream, on_token, offer)
                results.append(result)
                if not result.tool_calls or not offer:
                    break
                outgoing.append({"role": "assistant", "content": result.content,
                                 "tool_calls": [c.raw for c in result.tool_calls]})
                for call in result.tool_calls:
                    if on_tool:
                        on_tool(call)
                    outcome = self.toolbox.run(call.name, call.arguments)
                    self.last_tool_calls.append(outcome)
                    self._log_tool(outcome, log_ids)
                    if on_tool_done:
                        on_tool_done(outcome)
                    outgoing.append({"role": "tool", "tool_name": call.name,
                                     "content": outcome.text})
        except BaseException:
            # Drop the unanswered message so a retry doesn't send it twice.
            self.messages.pop()
            raise
        result = combine(results)
        # Thinking text is kept on the result only; it isn't sent back in later turns.
        self.messages.append({"role": "assistant", "content": result.content})
        self.last_sources = hits
        self._save(text, result, hits, log_ids)
        return result

    def _chat(self, messages, stream, on_token, tools) -> ChatResult:
        try:
            return self.client.chat(self.model, messages, stream=stream, on_token=on_token,
                                    think=self.think, tools=tools)
        except OllamaError as e:
            # Older or smaller models can't use tools. Answer without them.
            if tools and "does not support tools" in str(e):
                self.tools_unsupported_error = str(e)
                self.use_tools = False
                return self.client.chat(self.model, messages, stream=stream,
                                        on_token=on_token, think=self.think)
            raise

    def _log_tool(self, outcome: ToolOutcome, log_ids: list[int]) -> None:
        if self.store is None:
            return
        try:
            log_ids.append(self.store.log_tool_call(
                self.model, outcome.name, outcome.arguments, outcome.status,
                outcome.text, outcome.seconds))
        except sqlite3.Error as e:
            self.tool_log_error = str(e)

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

    def _save(self, text: str, result: ChatResult, sources: list[Hit] = (),
              tool_call_ids: list[int] = ()) -> None:
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
            self.store.add_turn(self.conversation_id, text, result, sources, tool_call_ids)
        except sqlite3.Error as e:
            self.save_error = str(e)


def combine(results: list[ChatResult]) -> ChatResult:
    """One result for a reply that took several model calls: the last call's
    text, with the token counts and times added up."""
    last = results[-1]
    if len(results) == 1:
        return last
    return ChatResult(
        content=last.content,
        model=last.model,
        thinking="".join(r.thinking for r in results),
        prompt_tokens=sum(r.prompt_tokens for r in results),
        completion_tokens=sum(r.completion_tokens for r in results),
        prompt_seconds=sum(r.prompt_seconds for r in results),
        generation_seconds=sum(r.generation_seconds for r in results),
        load_seconds=sum(r.load_seconds for r in results),
        total_seconds=sum(r.total_seconds for r in results),
    )
