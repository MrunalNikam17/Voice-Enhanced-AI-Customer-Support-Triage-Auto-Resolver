
"""
src/priority/priority.py

The Priority Agent: assigns an urgency level (low/medium/high) to a ticket
based on its text and predicted category, using rule-based keyword cues
combined with LLM reasoning. No emotion/audio model needed — urgency signals
come from the text itself (explicit urgency language, caps, category risk).

Usage:
    from src.priority.priority import PriorityAgent
    agent = PriorityAgent()
    result = agent.assess("URGENT!! unauthorized charge on my card", category="PAYMENT")
"""

import os
import sys
import json

sys.path.append(os.getcwd())
from src.utils.llm import local_llm_call, extract_json


PRIORITY_SYSTEM_PROMPT = """You are a support ticket prioritization assistant. \
Given a customer message and its category, assign an urgency level.

Rules:
- HIGH: mentions of fraud, unauthorized charges, data loss, service completely \
down, safety issues, explicit urgency language ("urgent", "immediately", \
"unacceptable"), excessive punctuation/caps suggesting distress
- MEDIUM: billing discrepancies, delayed delivery, account access issues
- LOW: general questions, feature requests, minor complaints

Respond ONLY with this JSON, no other text:
{{"urgency": "<low|medium|high>", "reason": "<one short sentence>"}}

Message: "{message}"
Category: {category}
"""

# Cheap, fast pre-check before calling the LLM — also acts as a safety net
# if the LLM output is unparseable, so urgency never silently defaults wrong.
HIGH_URGENCY_KEYWORDS = [
    "fraud", "unauthorized", "scam", "stolen", "hacked", "urgent",
    "immediately", "unacceptable", "lawsuit", "legal action", "asap",
]


def _keyword_signal(message: str) -> str:
    text = message.lower()
    has_high_keyword = any(kw in text for kw in HIGH_URGENCY_KEYWORDS)
    is_shouting = sum(1 for c in message if c.isupper()) > max(10, len(message) * 0.3)
    has_excessive_punct = "!!" in message or "???" in message

    if has_high_keyword or is_shouting or has_excessive_punct:
        return "high"
    return None  # no strong signal — defer to LLM judgment


class PriorityAgent:
    def __init__(self, llm_call_fn=None):
        self.llm_call_fn = llm_call_fn or local_llm_call

    def assess(self, message: str, category: str = None) -> dict:
        prompt = PRIORITY_SYSTEM_PROMPT.format(
            message=message, category=category or "UNKNOWN"
        )
        raw_output = self.llm_call_fn(prompt)

        try:
            parsed = json.loads(extract_json(raw_output))
            assert parsed.get("urgency") in ("low", "medium", "high")
        except (json.JSONDecodeError, TypeError, AssertionError):
            # Fallback: keyword signal, or default to medium (never silently
            # drop to "low" on a parse failure — safer to over-flag)
            keyword_urgency = _keyword_signal(message)
            parsed = {
                "urgency": keyword_urgency or "medium",
                "reason": "fallback: LLM response unparseable, used keyword/default rule",
            }
            return parsed

        # Keyword signal can only escalate urgency, never downgrade it
        keyword_urgency = _keyword_signal(message)
        if keyword_urgency == "high" and parsed["urgency"] != "high":
            parsed["urgency"] = "high"
            parsed["reason"] += " (escalated: high-urgency keyword/pattern detected)"

        return parsed


if __name__ == "__main__":
    agent = PriorityAgent()

    tests = [
        ("URGENT!! unauthorized charge on my card, please help immediately", "PAYMENT"),
        ("Hey just wondering when my order will arrive, no rush", "DELIVERY"),
        ("My subscription renewed but I wanted to downgrade first, can you fix this?", "SUBSCRIPTION"),
    ]

    for message, category in tests:
        result = agent.assess(message, category=category)
        print(f"Message: {message}")
        print(f"  -> Urgency: {result['urgency']}  |  Reason: {result['reason']}\n")