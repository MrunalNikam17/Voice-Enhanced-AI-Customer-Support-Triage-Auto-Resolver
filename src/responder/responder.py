"""
src/responder/responder.py

The Responder Agent: takes a customer ticket's text + predicted category,
retrieves relevant knowledge base chunks (RAG), and asks an LLM to draft an
answer grounded ONLY in that retrieved context. If nothing relevant is
retrieved, it signals low confidence so the Escalation Agent can hand off
to a human instead of guessing.

Usage:
    from src.responder.responder import ResponderAgent
    agent = ResponderAgent()
    result = agent.respond("I was charged twice this month", category="PAYMENT")
"""

import os
import sys
import json

sys.path.append(os.getcwd())
from src.utils.config import config
from src.rag.build_kb import load_vector_store, retrieve


RESPONDER_SYSTEM_PROMPT = """You are a customer support response assistant. \
You must answer using ONLY the information provided in the retrieved context \
below. Do not use any outside knowledge or make up policy details.

If the retrieved context does NOT contain enough information to answer \
confidently, respond with exactly this JSON:
{{"answer": null, "confidence": 0.0, "reason": "insufficient context"}}

Otherwise respond with ONLY this JSON, no other text:
{{"answer": "<clear, polite, concise answer>", "confidence": <float 0-1>, \
"sources": ["<doc id>"]}}

Customer message: "{query}"
Category: {category}

Retrieved context:
{context}
"""


def format_context(chunks: list) -> str:
    if not chunks:
        return "(no relevant context retrieved)"
    parts = []
    for c in chunks:
        parts.append(f"[{c['id']}] {c['title']}: {c['content']} (relevance: {c['score']:.2f})")
    return "\n".join(parts)


class ResponderAgent:
    def __init__(self, llm_call_fn=None):
        """
        llm_call_fn: a function(system_prompt: str) -> str that calls your LLM
        of choice (local model, Anthropic API, OpenAI API, etc.) and returns
        the raw text response. Defaults to a simple local placeholder you
        should replace with your actual model call.
        """
        self.index, self.embedder, self.faq_entries = load_vector_store()
        self.llm_call_fn = llm_call_fn or self._default_llm_call

    def _default_llm_call(self, prompt: str) -> str:
        """
        Placeholder LLM call. Replace this with a real call to:
          - a local Llama/Mistral model (via transformers/ollama), or
          - the Anthropic/OpenAI API (LLM_PROVIDER in config.py)
        Must return the model's raw text output.
        """
        raise NotImplementedError(
            "Wire this up to your actual LLM (local model or API). "
            "See config.LLM_PROVIDER / config.LLM_MODEL_NAME."
        )

    def respond(self, query: str, category: str = None) -> dict:
        # Retrieve relevant KB chunks (optionally filter by category first)
        chunks = retrieve(query, self.index, self.embedder, self.faq_entries)

        if not chunks:
            return {
                "answer": None,
                "confidence": 0.0,
                "reason": "insufficient context",
                "sources": [],
                "retrieved_chunks": [],
            }

        prompt = RESPONDER_SYSTEM_PROMPT.format(
            query=query,
            category=category or "UNKNOWN",
            context=format_context(chunks),
        )

        raw_output = self.llm_call_fn(prompt)

        try:
            parsed = json.loads(raw_output)
        except (json.JSONDecodeError, TypeError):
            # LLM didn't return clean JSON — treat as low confidence, escalate
            parsed = {
                "answer": None,
                "confidence": 0.0,
                "reason": "unparseable LLM response",
            }

        parsed["retrieved_chunks"] = chunks
        return parsed


if __name__ == "__main__":
    agent = ResponderAgent()
    print("Retrieval test (no LLM call, just showing retrieved context):\n")
    chunks = retrieve(
        "I was charged twice this month, what should I do?",
        agent.index, agent.embedder, agent.faq_entries,
    )
    print(format_context(chunks))