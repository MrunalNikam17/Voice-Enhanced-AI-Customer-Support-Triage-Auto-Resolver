
"""
src/graph/workflow.py

Wires the four agents (Classifier -> Priority -> Responder -> Escalation)
into a LangGraph StateGraph. This is the single entry point the dashboard
(dashboard/app.py) calls to process one ticket end-to-end.

Usage:
    from src.graph.workflow import run_ticket
    result = run_ticket("I was charged twice this month, please help immediately!")
"""

import os
import sys

sys.path.append(os.getcwd())

from langgraph.graph import StateGraph, END

from src.graph.state import TicketState
from src.classifier.classifier import ClassifierAgent
from src.priority.priority import PriorityAgent
from src.responder.responder import ResponderAgent
from src.escalation.escalation import EscalationAgent


# -----------------------------
# Lazy-loaded singleton agents
# (avoids reloading models on every ticket / every import)
# -----------------------------

_classifier = None
_priority = None
_responder = None
_escalation = None


def _get_agents():
    global _classifier, _priority, _responder, _escalation
    if _classifier is None:
        print("Loading agents (first call only)...")
        _classifier = ClassifierAgent()
        _priority = PriorityAgent()
        _responder = ResponderAgent()
        _escalation = EscalationAgent()
        print("All agents loaded.")
    return _classifier, _priority, _responder, _escalation


# -----------------------------
# Graph node functions
# -----------------------------

def classify_node(state: TicketState) -> TicketState:
    classifier, _, _, _ = _get_agents()
    result = classifier.predict(state["raw_text"])
    state["category"] = result["category"]
    state["category_confidence"] = result["confidence"]
    return state


def priority_node(state: TicketState) -> TicketState:
    _, priority, _, _ = _get_agents()
    result = priority.assess(state["raw_text"], category=state.get("category"))
    state["urgency"] = result["urgency"]
    state["urgency_reason"] = result["reason"]
    return state


def respond_node(state: TicketState) -> TicketState:
    _, _, responder, _ = _get_agents()
    result = responder.respond(state["raw_text"], category=state.get("category"))
    state["draft_answer"] = result.get("answer")
    state["answer_confidence"] = result.get("confidence")
    state["sources"] = result.get("sources", [])
    state["retrieved_chunks"] = result.get("retrieved_chunks", [])
    return state


def escalate_node(state: TicketState) -> TicketState:
    _, _, _, escalation = _get_agents()
    result = escalation.decide(
        category=state.get("category"),
        urgency=state.get("urgency"),
        draft_answer=state.get("draft_answer"),
        confidence=state.get("answer_confidence"),
        raw_message=state["raw_text"],
    )
    state["decision"] = result["decision"]
    state["escalation_summary"] = result["escalation_summary"]
    return state


# -----------------------------
# Build the graph
# -----------------------------

def build_graph():
    graph = StateGraph(TicketState)

    graph.add_node("classify", classify_node)
    graph.add_node("prioritize", priority_node)
    graph.add_node("respond", respond_node)
    graph.add_node("escalate", escalate_node)

    graph.set_entry_point("classify")
    graph.add_edge("classify", "prioritize")
    graph.add_edge("prioritize", "respond")
    graph.add_edge("respond", "escalate")
    graph.add_edge("escalate", END)

    return graph.compile()


_compiled_graph = None


def run_ticket(raw_text: str, audio_path: str = None) -> TicketState:
    """
    Main entry point. Runs one ticket through the full pipeline and returns
    the final state (category, urgency, draft answer, decision, etc).
    """
    global _compiled_graph
    if _compiled_graph is None:
        _compiled_graph = build_graph()

    initial_state: TicketState = {
        "raw_text": raw_text,
        "audio_path": audio_path,
    }

    final_state = _compiled_graph.invoke(initial_state)
    return final_state


if __name__ == "__main__":
    test_tickets = [
        "URGENT!! I was charged twice on my card this month, please fix immediately",
        "Hey, just wondering how long standard shipping usually takes",
        "My account got locked after too many login attempts, how do I fix it",
    ]

    for ticket in test_tickets:
        print(f"\n{'='*70}")
        print(f"Ticket: {ticket}")
        print('='*70)

        result = run_ticket(ticket)

        print(f"Category:   {result.get('category')} (conf: {result.get('category_confidence'):.2f})")
        print(f"Urgency:    {result.get('urgency')} — {result.get('urgency_reason')}")
        print(f"Answer:     {result.get('draft_answer')}")
        print(f"Confidence: {result.get('answer_confidence')}")
        print(f"Decision:   {result.get('decision')}")
        if result.get("escalation_summary"):
            print(f"Escalation: {result.get('escalation_summary')}")