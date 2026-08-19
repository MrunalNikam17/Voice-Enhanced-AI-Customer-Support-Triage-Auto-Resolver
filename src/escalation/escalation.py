
"""
src/escalation/escalation.py

The Escalation Agent: the final gate. Decides whether a drafted response
from the Responder Agent is safe to auto-send to the customer, or must be
routed to a human agent instead.

Usage:
    from src.escalation.escalation import EscalationAgent
    agent = EscalationAgent()
    result = agent.decide(
        category="REFUND", urgency="high",
        draft_answer="...", confidence=0.55,
    )
"""

import os
import sys

sys.path.append(os.getcwd())
from src.utils.config import config

# Categories where an unclear/ambiguous amount should force human review,
# since these involve money leaving/entering the customer's account.
SENSITIVE_CATEGORIES = {"REFUND", "CANCEL", "PAYMENT"}


class EscalationAgent:
    def __init__(self, confidence_threshold: float = None):
        self.confidence_threshold = (
            confidence_threshold
            if confidence_threshold is not None
            else config.ESCALATE_CONFIDENCE_THRESHOLD
        )

    def decide(self, category: str, urgency: str, draft_answer, confidence: float,
               raw_message: str = "") -> dict:
        reasons = []

        if draft_answer is None:
            reasons.append("no grounded answer available (insufficient context)")

        if confidence is not None and confidence < self.confidence_threshold:
            reasons.append(f"answer confidence {confidence:.2f} below threshold {self.confidence_threshold}")

        if urgency == "high":
            reasons.append("urgency is high")

        if category in SENSITIVE_CATEGORIES and _amount_unclear(raw_message):
            reasons.append(f"sensitive category ({category}) with unclear amount mentioned")

        if reasons:
            return {
                "decision": "escalate",
                "escalation_summary": _build_summary(category, urgency, reasons, raw_message),
            }

        return {
            "decision": "auto_resolve",
            "escalation_summary": None,
        }


def _amount_unclear(raw_message: str) -> bool:
    """
    Very lightweight heuristic: if the message references money/charges but
    contains no digits, treat the amount as unclear/ambiguous.
    """
    text = raw_message.lower()
    money_words = ["charge", "charged", "refund", "amount", "paid", "bill"]
    mentions_money = any(w in text for w in money_words)
    has_digit = any(ch.isdigit() for ch in raw_message)
    return mentions_money and not has_digit


def _build_summary(category: str, urgency: str, reasons: list, raw_message: str) -> str:
    reason_text = "; ".join(reasons)
    return (
        f"[{category} | urgency: {urgency}] Escalated — {reason_text}. "
        f"Original message: \"{raw_message[:200]}\""
    )


if __name__ == "__main__":
    agent = EscalationAgent()

    print("Case 1: confident, grounded, low urgency -> should auto-resolve")
    r1 = agent.decide(
        category="DELIVERY", urgency="low",
        draft_answer="Your order will arrive in 4-6 business days.",
        confidence=0.85,
        raw_message="When will my order arrive?",
    )
    print(r1, "\n")

    print("Case 2: low confidence -> should escalate")
    r2 = agent.decide(
        category="REFUND", urgency="medium",
        draft_answer="Refunds take 5-7 days.",
        confidence=0.4,
        raw_message="I was charged twice, please help",
    )
    print(r2, "\n")

    print("Case 3: high urgency -> should escalate even if confident")
    r3 = agent.decide(
        category="PAYMENT", urgency="high",
        draft_answer="This looks like a duplicate charge, it will reverse automatically.",
        confidence=0.9,
        raw_message="URGENT unauthorized charge on my card!!",
    )
    print(r3, "\n")

    print("Case 4: no answer at all -> should escalate")
    r4 = agent.decide(
        category="ACCOUNT", urgency="medium",
        draft_answer=None,
        confidence=0.0,
        raw_message="Something weird happened to my account settings",
    )
    print(r4)