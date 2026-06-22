# RLHF-Aligned Customer Support Agent

A customer support chatbot that actually gets smarter the more people use it.

Most RAG chatbots just retrieve documents and generate answers. This one goes further — it generates multiple candidate answers, scores them using a reward model, picks the best one, and then learns from whether users found that answer helpful or not. That feedback loop is (a simplified version of) how ChatGPT and Claude are trained to be helpful.

Built this to understand RLHF from the inside, not just read papers about it.

---

## What it does

You ask a support question. The system:

1. Searches 26,872 real customer support conversations (Bitext dataset) stored in ChromaDB
2. Feeds the most relevant ones as context to Llama 3.2
3. Generates three candidate answers
4. Scores all three using a fine-tuned distilBERT reward model
5. Shows you the highest-scoring answer
6. Asks if it was helpful
7. Stores your feedback and uses it to retrain the reward model

The more people use it, the better the reward model gets at picking good answers. That is the flywheel.

---

## Why I built it this way

I wanted to understand why ChatGPT feels different from just calling GPT-4 directly. The answer is RLHF — the model is trained on human preference signals, not just next-token prediction. This project builds that mechanism from scratch at a small scale so I could actually see how it works.

The reward model starts trained on synthetic data (real answers vs degraded versions). As users rate answers, it gradually shifts toward what real humans actually find helpful. That gap between synthetic and real is the interesting part.

---

## Tech stack

| Component | Technology |
|---|---|
| LLM | Llama 3.2 via Ollama (runs locally, free) |
| Embeddings | nomic-embed-text via Ollama |
| Vector DB | ChromaDB |
| Agent | LangChain + LangGraph (ReAct) |
| Reward model | distilBERT fine-tuned on preference pairs |
| Training | PyTorch + HuggingFace Trainer |
| UI | Streamlit |

Everything runs locally. No OpenAI API key needed.

---

## Dataset

Bitext Customer Support Dataset from HuggingFace
(`bitext/Bitext-customer-support-llm-chatbot-training-dataset`)

26,872 real customer support conversations across 11 categories: orders, shipping, cancellations, refunds, payments, invoices, delivery, feedback, account management, subscriptions, and contact.

---

## Project structure
rlhf-support-agent/

├── data/

│   ├── raw/              # 11 category files from Bitext dataset

│   └── preferences/      # Human feedback pairs stored as JSON

├── src/

│   ├── ingestion/        # Document loading and Q&A chunking

│   ├── retrieval/        # ChromaDB vector store and search

│   ├── agent/            # LangChain ReAct agent with tools

│   ├── reward_model/     # distilBERT training, inference, dataset

│   └── pipeline/         # Connects retrieval to LLM generation

├── app/

│   └── streamlit_app.py  # Chat UI with feedback and analytics

├── tests/                # pytest unit tests

└── configs/

└── config.yaml       # All settings in one place
---

## Setup

You need Ollama installed first. Download it at ollama.com.

```bash
ollama pull llama3.2
ollama pull nomic-embed-text
```

Then clone and set up:

```bash
git clone https://github.com/PujithaJ-02/rlhf-support-agent.git
cd rlhf-support-agent

python3 -m venv venv
source venv/bin/activate

pip install -r requirements.txt
```

Download the dataset and build the vector store (one time, takes about 45 minutes on CPU):

```bash
python3 src/retrieval/vectorstore.py
```

Train the reward model (takes about 6 minutes on CPU):

```bash
python3 src/reward_model/dataset.py
python3 src/reward_model/train.py
```

Run the app:

```bash
streamlit run app/streamlit_app.py
```

---

## Running the tests

```bash
pytest tests/ -v
```

14 tests covering the retrieval pipeline and reward model. All passing.

---

## The RLHF loop
User asks a question

↓

ChromaDB finds relevant support conversations

↓

Llama 3.2 generates 3 candidate answers

↓

Reward model scores each one (0.0 to 1.0)

↓

Best answer shown to user

↓

User clicks thumbs up or thumbs down

↓

Feedback saved to data/preferences/feedback.json

↓

Reward model retrains on real human preferences

↓

Better answers next time
This is the same loop behind ChatGPT's alignment, minus the scale and the billion dollar GPU cluster.

---

## Known limitations

Response time is about 90 seconds per question because Llama 3.2 runs on CPU. On a GPU or using an API this drops to under 3 seconds. The architecture is the same either way — just swap the Ollama call in `src/pipeline/generator.py` for an OpenAI or Anthropic API call.

The reward model starts on synthetic data. It gets better as real feedback accumulates, but it needs a few hundred ratings before the improvement becomes noticeable.

---

## What I learned

Fine-tuning a reward model on synthetic preference pairs is a reasonable bootstrap strategy, but the interesting behavior only starts appearing once real human signals come in. The gap between "what we think users want" and "what users actually find helpful" is exactly what RLHF is designed to close — and you can see that gap clearly when you look at the feedback data accumulating in the JSON file.

Also: running a 2GB language model locally on a MacBook for free is genuinely impressive and I would recommend it to anyone who wants to understand what is actually happening inside these systems.