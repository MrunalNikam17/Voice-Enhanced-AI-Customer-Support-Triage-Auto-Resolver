# AutoTriage — Voice-Enabled AI Customer Support Triage & Auto-Resolver

An agentic system that classifies incoming customer support tickets (typed or
spoken), retrieves grounded answers from a knowledge base (RAG), decides
urgency, and either auto-resolves the ticket or escalates it to a human with
a clear summary.

## Pipeline

```
Voice/Text Input
   → [Whisper] Speech-to-Text (voice only)
   → [Classifier Agent] fine-tuned DistilBERT classifier → ticket category
   → [Priority Agent] rule + Mistral-7B reasoning → urgency (low/medium/high)
   → [Responder Agent] RAG over FAISS FAQ knowledge base → grounded draft answer
   → [Escalation Agent] decides: auto-resolve or hand off to human
   → [gTTS] Text-to-Speech voice reply (for auto-resolved tickets)
   → [Dashboard] live trace + analytics
```

Pipeline flow:
**Voice/Text → Whisper → Classifier → Priority → RAG+Mistral Responder → Escalation → gTTS voice reply**

Full architecture and agent prompts: see `docs/` (or the shared
`AutoTriage_Architecture_and_Prompts.md`).

> Note: Emotion/tone detection from voice was considered but intentionally
> left out of the core pipeline (high effort, unreliable accuracy for a
> student-project timeline). Voice input still works via Whisper
> transcription; urgency uses text-based cues instead. See the architecture
> doc's stretch-goal section if you want to add it later.

## How to run the dashboard

```bash
pip install -r requirements.txt
streamlit run dashboard/app.py
```

## Results

- **DistilBERT Classifier:** Achieves **~100% test accuracy** on the held-out test split.
  - *Note on Benchmark:* Bitext's templated phrasing makes this an upper bound.
  - *Data Augmentation Fix:* A duplicate-charge phrasing gap was found and fixed via data augmentation (20 additional examples added).
- **End-to-End Pipeline:** Fully integrated voice-in to voice-out triage and resolution pipeline.

## Project Structure

```
voice-customer-support-ai/
├── data/
│   ├── raw/
│   │   ├── audio/              # sample voice tickets
│   │   ├── bitext/             # Bitext customer support dataset (classifier training)
│   │   └── knowledge_base/     # curated FAQ source (knowledge_base_faq.json)
│   └── processed/              # cleaned/split datasets for training
├── models/                     # fine-tuned classifier + kb_store (FAISS index)
├── src/
│   ├── classifier/             # ticket category classifier (fine-tuning + inference)
│   ├── priority/                # urgency scoring agent
│   ├── responder/               # RAG-grounded answer generation
│   ├── escalation/              # auto-resolve vs escalate decision
│   ├── rag/                     # knowledge base builder + retrieval
│   ├── voice/                   # Whisper (STT) + TTS services
│   ├── graph/                   # LangGraph state + workflow wiring
│   └── utils/                   # config.py (central settings)
├── dashboard/                   # Streamlit demo UI
├── tests/                       # unit tests per component
├── notebook/                    # exploration / KB analysis notebooks
├── requirements.txt
└── .env                         # local config (not committed)
```

## Setup

```bash
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements.txt
cp .env.example .env             # then fill in values
```

## Build order (recommended)

1. **Knowledge base** — `python -m src.rag.build_kb`
   Builds the FAISS index from `data/raw/knowledge_base/knowledge_base_faq.json`.

2. **Classifier** — fine-tune on the Bitext dataset (`src/classifier/`).

3. **Responder Agent** — wires the classifier output + RAG retrieval into a
   grounded answer (`src/responder/responder.py`).

4. **Priority + Escalation Agents** — `src/priority/`, `src/escalation/`.

5. **Graph** — connect all agents into a LangGraph workflow
   (`src/graph/workflow.py`).

6. **Voice** — add Whisper transcription for voice-ticket input
   (`src/voice/whisper_service.py`) and gTTS voice replies (`src/voice/tts_service.py`).

7. **Dashboard** — `streamlit run dashboard/app.py`.

## Datasets

| Purpose | Source |
|---|---|
| Ticket classification | Bitext Customer Support Dataset (`data/raw/bitext/`) |
| Knowledge base | Curated FAQ (`data/raw/knowledge_base/knowledge_base_faq.json`) |

## Evaluation

- Classifier: accuracy / F1 on held-out Bitext split, compared against a
  prompted-LLM baseline.
- Responder: manual relevance scoring (1-5) on a held-out query set.
- End-to-end: auto-resolve rate and escalation precision on a mixed
  50-100 ticket sample.
