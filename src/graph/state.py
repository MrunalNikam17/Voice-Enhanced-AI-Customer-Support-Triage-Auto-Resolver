
"""
src/graph/state.py

The shared state object that flows through every node in the LangGraph
workflow. Each agent reads what it needs from this state and writes its
output back into it.
"""

from typing import TypedDict, Optional, List, Dict, Any


class TicketState(TypedDict, total=False):
    # --- Input ---
    raw_text: str                        # transcript or typed ticket text
    audio_path: Optional[str]            # set if input was voice

    # --- Node 1: Classifier Agent output ---
    category: Optional[str]
    category_confidence: Optional[float]

    # --- Node 2: Priority Agent output ---
    urgency: Optional[str]               # low / medium / high
    urgency_reason: Optional[str]

    # --- Node 3: Responder Agent output ---
    draft_answer: Optional[str]
    answer_confidence: Optional[float]
    sources: Optional[List[str]]
    retrieved_chunks: Optional[List[Dict[str, Any]]]

    # --- Node 4: Escalation Agent output ---
    decision: Optional[str]              # "auto_resolve" or "escalate"
    escalation_summary: Optional[str]