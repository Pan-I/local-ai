"""Finds the passages that best match a question and formats them for the model.

The question is embedded with the same model as the documents, compared with
every stored chunk, and the closest ones (above knowledge.min_score) are pasted
into the prompt as numbered sources. The model is asked to cite them like [1],
but the app also prints the source list itself, so you can always see which
files informed an answer even if a small model forgets to cite.
"""

from retrieval.indexer import KnowledgeSettings
from retrieval.store import Hit, KnowledgeStore

INSTRUCTIONS = (
    "Below are numbered passages from the user's local documents, found by searching "
    "for their question. Use them if they help, and cite the ones you use like [1]. "
    "If they don't contain the answer, say so and answer from general knowledge, "
    "making clear which parts aren't from the documents."
)


class Retriever:
    def __init__(self, client, store: KnowledgeStore, settings: KnowledgeSettings):
        self.client = client
        self.store = store
        self.settings = settings

    def search(self, question: str) -> list[Hit]:
        s = self.settings
        [vector] = self.client.embed(s.embedding_model, [s.query_prefix + question])
        return self.store.search(vector, top_k=s.top_k, min_score=s.min_score)


def build_prompt(question: str, hits: list[Hit]) -> str:
    """The user message sent to the model: sources first, then the question."""
    if not hits:
        return question
    blocks = [f"[{i}] {hit.reference}\n{hit.text}" for i, hit in enumerate(hits, start=1)]
    return (
        INSTRUCTIONS
        + "\n\n<sources>\n" + "\n\n".join(blocks) + "\n</sources>\n\n"
        + "Question: " + question
    )
