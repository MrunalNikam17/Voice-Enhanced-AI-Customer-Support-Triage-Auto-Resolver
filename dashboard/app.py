
"""
dashboard/app.py

Streamlit dashboard for AutoTriage. Lets you submit a ticket (typed or
voice), watch it flow through Classifier -> Priority -> Responder ->
Escalation, and see a running analytics summary.

Run:
    streamlit run dashboard/app.py
"""

import os
import sys
import time

sys.path.append(os.getcwd())

import streamlit as st

from src.utils.config import config
from src.graph.workflow import run_ticket
from src.voice.whisper_service import transcribe

st.set_page_config(page_title=config.DASHBOARD_TITLE, layout="wide")

if "history" not in st.session_state:
    st.session_state.history = []

st.title(config.DASHBOARD_TITLE)
st.caption("Voice/text ticket triage: classify → prioritize → respond (RAG) → escalate")

tab_run, tab_analytics = st.tabs(["Run a ticket", "Analytics"])

# -----------------------------
# TAB 1: Run a ticket
# -----------------------------
with tab_run:
    input_mode = st.radio("Input type", ["Typed text", "Upload voice recording"], horizontal=True)

    raw_text = None

    if input_mode == "Typed text":
        raw_text = st.text_area(
            "Customer message",
            placeholder="e.g. I was charged twice this month, please help",
            height=100,
        )
    else:
        audio_file = st.file_uploader("Upload audio (mp3/wav)", type=["mp3", "wav", "m4a"])
        if audio_file is not None:
            tmp_path = os.path.join("data/raw/audio", audio_file.name)
            os.makedirs("data/raw/audio", exist_ok=True)
            with open(tmp_path, "wb") as f:
                f.write(audio_file.getbuffer())

            st.audio(tmp_path)

            with st.spinner("Transcribing with Whisper..."):
                transcript_result = transcribe(tmp_path)
            raw_text = transcript_result["text"]
            st.info(f"**Transcript:** {raw_text}")

    if st.button("Run through pipeline", type="primary", disabled=not raw_text):
        with st.spinner("Running agents (Classifier → Priority → Responder → Escalation)..."):
            start = time.time()
            result = run_ticket(raw_text)
            elapsed = time.time() - start

        st.session_state.history.append(result)

        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Category", result.get("category"))
        col2.metric("Category confidence", f"{result.get('category_confidence', 0):.2f}")
        col3.metric("Urgency", result.get("urgency"))
        col4.metric("Handling time", f"{elapsed:.1f}s")

        st.divider()

        decision = result.get("decision")
        if decision == "auto_resolve":
            st.success("✅ Auto-resolved")
            st.write(f"**Answer sent to customer:** {result.get('draft_answer')}")
            st.caption(f"Sources: {', '.join(result.get('sources', [])) or 'none'}")
        else:
            st.warning("🚩 Escalated to human agent")
            st.write(f"**Summary for human agent:** {result.get('escalation_summary')}")
            if result.get("draft_answer"):
                st.caption(f"(Draft answer for reference: {result.get('draft_answer')})")

        with st.expander("Retrieved knowledge base chunks"):
            for chunk in result.get("retrieved_chunks", []):
                st.write(f"**[{chunk['score']:.3f}]** {chunk['id']} — {chunk['title']}")
                st.caption(chunk["content"])

        with st.expander("Full raw state (debug)"):
            st.json(result)

# -----------------------------
# TAB 2: Analytics
# -----------------------------
with tab_analytics:
    history = st.session_state.history

    if not history:
        st.info("Run some tickets in the first tab to see analytics here.")
    else:
        total = len(history)
        auto_resolved = sum(1 for h in history if h.get("decision") == "auto_resolve")
        escalated = total - auto_resolved

        col1, col2, col3 = st.columns(3)
        col1.metric("Total tickets", total)
        col2.metric("Auto-resolved", f"{auto_resolved} ({auto_resolved/total*100:.0f}%)")
        col3.metric("Escalated", f"{escalated} ({escalated/total*100:.0f}%)")

        st.divider()
        st.subheader("Category breakdown")
        category_counts = {}
        for h in history:
            cat = h.get("category", "UNKNOWN")
            category_counts[cat] = category_counts.get(cat, 0) + 1
        st.bar_chart(category_counts)

        st.subheader("Urgency breakdown")
        urgency_counts = {}
        for h in history:
            urg = h.get("urgency", "unknown")
            urgency_counts[urg] = urgency_counts.get(urg, 0) + 1
        st.bar_chart(urgency_counts)

        st.subheader("Ticket log")
        for i, h in enumerate(reversed(history), 1):
            with st.expander(f"#{total - i + 1}: {h.get('raw_text', '')[:60]}..."):
                st.write(f"**Category:** {h.get('category')}  |  **Urgency:** {h.get('urgency')}  |  **Decision:** {h.get('decision')}")
                st.write(f"**Answer:** {h.get('draft_answer') or h.get('escalation_summary')}")