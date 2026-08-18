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
from src.utils.llm import local_llm_call, extract_json


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
        llm_call_fn: a function(prompt: str) -> str that calls an LLM and
        returns raw text. Defaults to the local Mistral-7B wrapper in
        src/utils/llm.py. Pass a different function here if you swap models
        later (e.g. an API-based call) without changing this class.
        """
        self.index, self.embedder, self.faq_entries = load_vector_store()
        self.llm_call_fn = llm_call_fn or local_llm_call

    def respond(self, query: str, category: str = None) -> dict:
        # Retrieve relevant KB chunks
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
            parsed = json.loads(extract_json(raw_output))
        except (json.JSONDecodeError, TypeError):
            # LLM didn't return clean/parseable JSON — treat as low
            # confidence so the Escalation Agent hands this off to a human
            # instead of silently failing.
            parsed = {
                "answer": None,
                "confidence": 0.0,
                "reason": "unparseable LLM response",
                "sources": [],
            }

        parsed["retrieved_chunks"] = chunks
        return parsed


if __name__ == "__main__":
    agent = ResponderAgent()

    test_query = "I was charged twice this month, what should I do?"
    result = agent.respond(test_query, category="PAYMENT")

    print(f"Query: {test_query}\n")
    print(f"Answer: {result.get('answer')}")
    print(f"Confidence: {result.get('confidence')}")
    print(f"Sources: {result.get('sources')}")
    print(f"\nRetrieved chunks used:")
    for c in result.get("retrieved_chunks", []):
        print(f"  [{c['score']:.3f}] {c['id']} - {c['title']}")