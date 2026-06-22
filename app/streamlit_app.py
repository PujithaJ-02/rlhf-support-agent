"""
app/streamlit_app.py
RLHF Support Agent — stable version with persistent question display
"""

import sys
import json
import time
import datetime
import warnings
import logging
import os
from pathlib import Path

warnings.filterwarnings("ignore")
logging.getLogger("transformers").setLevel(logging.ERROR)
os.environ["TOKENIZERS_PARALLELISM"] = "false"

sys.path.insert(0, ".")

import streamlit as st

st.set_page_config(
    page_title="ShopBot AI Support",
    page_icon="🛍️",
    layout="centered",
)

FEEDBACK_PATH = "data/preferences/feedback.json"

SHOPBOT_KEYWORDS = [
    "order", "cancel", "refund", "return", "shipping", "delivery",
    "payment", "account", "password", "invoice", "subscription",
    "track", "package", "charge", "billing", "address", "email",
    "login", "sign in", "reset", "purchase", "item", "product",
    "shopbot", "support", "help", "issue", "problem", "complaint",
    "website", "phone", "number", "contact", "receipt", "status",
]

SMALL_TALK = {
    "how are you":      "I'm doing great, thanks for asking! I'm here and ready to help you with any ShopBot support questions. What can I help you with today?",
    "how can you help": "I can help you with orders, shipping, returns, refunds, payments, subscriptions, and account management. What do you need help with?",
    "what can you do":  "I can help you with orders, shipping, returns, refunds, payments, subscriptions, and account management. Just ask!",
    "who are you":      "I'm ShopBot's AI support assistant, powered by a RAG pipeline and reward model. I can help with any ShopBot support questions.",
    "hello":            "Hello! Welcome to ShopBot support. How can I help you today?",
    "hi":               "Hi there! Welcome to ShopBot support. What can I help you with?",
    "hey":              "Hey! Welcome to ShopBot support. What can I help you with?",
    "good morning":     "Good morning! Hope you're having a great day. How can I help you with your ShopBot account today?",
    "good afternoon":   "Good afternoon! How can I help you with your ShopBot account today?",
    "good evening":     "Good evening! How can I help you with your ShopBot account today?",
    "bad morning":      "Oh no, sorry to hear your morning isn't going well! I hope I can make it a little better. How can I help you with your ShopBot account today?",
    "bad day":          "Sorry to hear that! I hope I can help make things a little easier. What ShopBot support question can I help you with?",
    "not working":      "I'm sorry you're having trouble! Could you describe the issue in more detail? Is it related to an order, payment, or your account?",
    "you are not working properly": "I apologize if I haven't been helpful! Could you tell me what you were trying to ask? I'll do my best to assist you.",
    "oh god":           "I sense some frustration! I'm here to help. What ShopBot support issue can I assist you with today?",
    "thank you":        "You're welcome! Is there anything else I can help you with?",
    "thanks":           "Happy to help! Is there anything else you need?",
    "bye":              "Goodbye! Feel free to come back if you have any ShopBot questions.",
    "goodbye":          "Goodbye! Have a great day!",
}

GIBBERISH_RESPONSE = (
    "I didn't quite understand that. Could you rephrase your question? "
    "I'm here to help with ShopBot support topics like orders, shipping, "
    "returns, refunds, payments, and account management."
)


def is_gibberish(text: str) -> bool:
    text = text.strip()
    if len(text) < 3:
        return True
    words = text.split()
    avg_len = sum(len(w) for w in words) / max(len(words), 1)
    if len(text) > 8 and len(words) == 1 and not text.isalpha():
        return True
    if avg_len > 12:
        return True
    return False


def get_small_talk_response(question: str) -> str | None:
    q = question.lower().strip()
    for phrase, response in SMALL_TALK.items():
        if phrase in q:
            return response
    return None


def is_shopbot_question(question: str) -> bool:
    q = question.lower()
    return any(kw in q for kw in SHOPBOT_KEYWORDS)


def load_feedback():
    if not Path(FEEDBACK_PATH).exists():
        return []
    with open(FEEDBACK_PATH, "r") as f:
        return json.load(f)


def save_feedback(question, answer, score, label):
    feedback = load_feedback()
    feedback.append({
        "question":  question,
        "answer":    answer,
        "score":     score,
        "label":     label,
        "timestamp": datetime.datetime.now().isoformat(),
    })
    Path("data/preferences").mkdir(parents=True, exist_ok=True)
    with open(FEEDBACK_PATH, "w") as f:
        json.dump(feedback, f, indent=2)


def init_session():
    defaults = {
        "messages":         [],
        "pending":          None,
        "page":             "chat",
        "processing":       False,
        "current_question": None,
    }
    for key, val in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = val


def get_pipeline_answer(question):
    from src.pipeline.generator import generate_candidates
    from src.reward_model.infer import score_candidates

    start   = time.time()
    result  = generate_candidates(question)
    scored  = score_candidates(question, result["candidates"])
    elapsed = round(time.time() - start, 1)
    best    = scored[0]
    return best["answer"], best["score"], elapsed


def render_sidebar():
    with st.sidebar:
        st.title("Navigation")
        if st.button("💬  Chat", use_container_width=True):
            st.session_state.page = "chat"
            st.rerun()
        if st.button("📊  Analytics", use_container_width=True):
            st.session_state.page = "analytics"
            st.rerun()
        st.divider()
        st.caption("Tech Stack")
        st.caption("• LLM: Llama 3.2 via Ollama")
        st.caption("• Embeddings: nomic-embed-text")
        st.caption("• Vector DB: ChromaDB (26,872 vectors)")
        st.caption("• Reward Model: distilBERT")
        st.caption("• Framework: LangChain + LangGraph")
        st.divider()
        feedback = load_feedback()
        st.metric("Total Feedback", len(feedback))
        if len(feedback) >= 10:
            st.caption(f"{len(feedback)} ratings collected.")
            st.caption("Run train.py to retrain.")


def render_chat_page():
    st.title("🛍️ ShopBot AI Support")
    st.caption(
        "Powered by RAG + Llama 3.2 + RLHF Reward Model · "
        "Running fully locally"
    )
    st.divider()

    # Render full chat history
    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])
            if msg.get("meta"):
                st.caption(msg["meta"])

    # Show current question AND a thinking indicator while processing
    # This keeps the question visible the entire time Llama is running
    if st.session_state.processing and st.session_state.current_question:
        with st.chat_message("user"):
            st.markdown(st.session_state.current_question)
        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                # Run the pipeline here while spinner is showing
                clean_q = st.session_state.current_question

                small_talk = get_small_talk_response(clean_q)
                if small_talk:
                    st.session_state.messages.append({
                        "role":    "user",
                        "content": clean_q,
                    })
                    st.session_state.messages.append({
                        "role":    "assistant",
                        "content": small_talk,
                        "meta":    "",
                    })
                    st.session_state.processing       = False
                    st.session_state.current_question = None
                    st.rerun()
                    return

                if not is_shopbot_question(clean_q):
                    st.session_state.messages.append({
                        "role":    "user",
                        "content": clean_q,
                    })
                    st.session_state.messages.append({
                        "role":    "assistant",
                        "content": (
                            "I can only help with ShopBot customer support questions "
                            "such as orders, shipping, returns, refunds, payments, "
                            "and account management. Could you ask something related to ShopBot?"
                        ),
                        "meta": "",
                    })
                    st.session_state.processing       = False
                    st.session_state.current_question = None
                    st.rerun()
                    return

                try:
                    answer, score, elapsed = get_pipeline_answer(clean_q)
                    st.session_state.messages.append({
                        "role":    "user",
                        "content": clean_q,
                    })
                    st.session_state.pending = {
                        "question": clean_q,
                        "answer":   answer,
                        "score":    score,
                        "time":     elapsed,
                    }
                except Exception:
                    st.session_state.messages.append({
                        "role":    "user",
                        "content": clean_q,
                    })
                    st.session_state.messages.append({
                        "role":    "assistant",
                        "content": (
                            "I had trouble generating an answer. "
                            "Please try again or rephrase your question."
                        ),
                        "meta": "",
                    })

                st.session_state.processing       = False
                st.session_state.current_question = None
                st.rerun()
        return

    # Show pending answer waiting for feedback
    if st.session_state.pending is not None:
        p = st.session_state.pending
        with st.chat_message("assistant"):
            st.markdown(p["answer"])
            if p.get("score") is not None:
                mins     = int(p["time"] // 60)
                secs     = int(p["time"] % 60)
                time_str = f"{mins}m {secs}s" if mins > 0 else f"{secs}s"
                st.caption(
                    f"Response time: {time_str}  "
                    f"·  Reward model confidence: {p['score']:.1%}"
                )
            col1, col2, col3 = st.columns([1, 1, 4])
            with col1:
                if st.button("👍 Helpful", key="up"):
                    if p.get("score") is not None:
                        save_feedback(
                            p["question"], p["answer"], p["score"], 1
                        )
                    st.session_state.messages.append({
                        "role":    "assistant",
                        "content": p["answer"],
                        "meta":    "Rated: Helpful",
                    })
                    st.session_state.pending = None
                    st.rerun()
            with col2:
                if st.button("👎 Not helpful", key="down"):
                    if p.get("score") is not None:
                        save_feedback(
                            p["question"], p["answer"], p["score"], 0
                        )
                    st.session_state.messages.append({
                        "role":    "assistant",
                        "content": p["answer"],
                        "meta":    "Rated: Not helpful",
                    })
                    st.session_state.pending = None
                    st.rerun()

    # Chat input -- only active when not processing
    if not st.session_state.processing:
        question = st.chat_input("Ask a ShopBot support question...")
    else:
        st.chat_input("Please wait...", disabled=True)
        question = None

    if question:
        clean_q = question.strip()

        if is_gibberish(clean_q):
            st.session_state.messages.append({
                "role":    "user",
                "content": clean_q,
            })
            st.session_state.messages.append({
                "role":    "assistant",
                "content": GIBBERISH_RESPONSE,
                "meta":    "",
            })
            st.rerun()
            return

        # Store question and trigger processing on next rerun
        st.session_state.current_question = clean_q
        st.session_state.processing       = True
        st.rerun()


def render_analytics_page():
    st.title("📊 Feedback Analytics")
    st.caption("Human preference data collected so far")
    st.divider()

    feedback = load_feedback()

    if not feedback:
        st.info("No feedback yet. Go to Chat and rate some answers.")
        return

    total    = len(feedback)
    positive = sum(1 for f in feedback if f["label"] == 1)
    negative = total - positive
    scored   = [f["score"] for f in feedback if f.get("score") is not None]
    avg      = sum(scored) / len(scored) if scored else 0

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Total Ratings",    total)
    c2.metric("Helpful",          positive)
    c3.metric("Not Helpful",      negative)
    c4.metric("Avg Reward Score", f"{avg:.1%}")

    st.divider()
    st.subheader("Recent Feedback")

    for item in reversed(feedback[-10:]):
        label    = "Helpful" if item["label"] == 1 else "Not helpful"
        score    = f"{item['score']:.1%}" if item.get("score") else "N/A"
        time_str = item["timestamp"][:19].replace("T", " ")
        with st.expander(f"{label}  ·  Score {score}  ·  {time_str}"):
            st.markdown(f"**Question:** {item['question']}")
            st.markdown(f"**Answer:** {item['answer'][:300]}...")
            st.markdown(f"**Reward score:** {score}")

    st.divider()
    st.subheader("Retrain the reward model")
    st.markdown(
        "Once you have enough feedback, run this to retrain on real preferences:"
    )
    st.code("python3 src/reward_model/train.py")


def main():
    init_session()
    render_sidebar()

    if st.session_state.page == "chat":
        render_chat_page()
    elif st.session_state.page == "analytics":
        render_analytics_page()


if __name__ == "__main__":
    main()