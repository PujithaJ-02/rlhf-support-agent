# RLHF-Aligned Customer Support Agent with RAG Pipeline

> An end-to-end AI system combining RAG, LLM-powered agents, and an RLHF loop
> built as a production-style AI engineering portfolio project.

---

## What This Project Does

A user asks a support question, the agent retrieves relevant documents,
the LLM generates 3 candidate answers, a reward model scores each answer,
the best answer is shown, and user feedback continuously retrains the reward model.

This is the same alignment loop used in production LLMs like ChatGPT and Claude.

---

## Tech Stack

| Layer | Technology |
|---|---|
| LLM | OpenAI GPT-3.5 / GPT-4o-mini |
| Embeddings | sentence-transformers (all-MiniLM-L6-v2) |
| Vector DB | ChromaDB |
| Agent | LangChain + ReAct |
| Reward Model | distilBERT (fine-tuned) |
| Training | PyTorch + HuggingFace Trainer |
| UI | Streamlit |
| Deployment | HuggingFace Spaces |

---

## Setup Instructions

1. Clone the repo and cd into it
2. Run: python3 -m venv venv
3. Run: source venv/bin/activate
4. Run: pip install -r requirements.txt
5. Copy .env.example to .env and add your API keys
6. Run: streamlit run app/streamlit_app.py

---

## Build Phases

| Phase | Description | Status |
|---|---|---|
| Phase 1 | Environment Setup | Complete |
| Phase 2 | RAG Pipeline | In Progress |
| Phase 3 | LLM Integration | Pending |
| Phase 4 | Agent + Tools | Pending |
| Phase 5 | Reward Model (RLHF) | Pending |
| Phase 6 | Human Feedback Loop | Pending |
| Phase 7 | Testing + Deployment | Pending |

---

## About

Built to demonstrate production-level AI engineering skills including
RAG pipelines, LLM agents, and RLHF — the alignment technique behind
modern LLMs like ChatGPT, Claude, and Gemini.