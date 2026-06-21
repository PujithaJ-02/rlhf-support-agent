"""
src/agent/agent.py

PURPOSE:
    A LangGraph ReAct agent that decides which tool to use
    for each user question.

    ReAct = Reasoning + Acting
    The agent reasons about what to do, acts by calling a tool,
    observes the result, then reasons again if needed.

TOOLS:
    1. support_retriever — searches our 26,872 Q&A chunks in ChromaDB
    2. calculator        — evaluates basic math expressions

USED BY:
    app/streamlit_app.py
"""

import sys
sys.path.insert(0, ".")

from langchain_ollama import OllamaLLM, ChatOllama
from langchain_core.tools import tool
from langchain.agents import create_react_agent
from src.retrieval.retriever import retrieve, format_context


LLM_MODEL = "llama3.2"


def get_llm() -> ChatOllama:
    """
    Loads Llama 3.2 for the agent via Ollama.
    LangGraph requires a chat model, not a plain LLM.
    Lower temperature = more consistent reasoning.
    """
    return ChatOllama(
        model=LLM_MODEL,
        temperature=0.3,
    )


@tool
def support_retriever(query: str) -> str:
    """
    Search the ShopBot knowledge base for customer support information.
    Use this for any question about orders, shipping, returns, refunds,
    payments, account management, delivery, subscriptions, cancellations,
    or any other ShopBot related topic.
    Input should be the customer question in plain English.
    """
    docs    = retrieve(query, k=4)
    context = format_context(docs)
    context = context.replace("{{", "[").replace("}}", "]")

    if not docs:
        return "No relevant information found in the knowledge base."

    return context


@tool
def calculator(expression: str) -> str:
    """
    Evaluate a basic math expression.
    Use this for any arithmetic calculation.
    Input should be a math expression like 45.99 + 8.00 or 100 * 0.15.
    Only use for math — not for support questions.
    """
    allowed = set("0123456789+-*/()., ")

    if not all(c in allowed for c in expression):
        return "Error: only basic math operations are allowed."

    try:
        result = eval(expression)
        return f"{expression} = {result}"
    except Exception as e:
        return f"Math error: {str(e)}"


def build_agent():
    """
    Builds and returns the LangGraph ReAct agent.

    LangGraph's create_react_agent handles the full
    reasoning loop automatically:
        1. Agent receives question
        2. Decides which tool to call
        3. Calls the tool
        4. Observes result
        5. Decides if more tool calls needed
        6. Returns final answer
    """
    llm   = get_llm()
    tools = [support_retriever, calculator]

    agent = create_react_agent(
        model=llm,
        tools=tools,
    )

    return agent


def ask_agent(question: str, agent) -> str:
    """
    Sends a question to the agent and returns the final answer.

    Args:
        question : user question in plain English
        agent    : the LangGraph agent from build_agent()

    Returns:
        the agent final answer as a string
    """
    try:
        messages = {"messages": [("human", question)]}
        result   = agent.invoke(messages)
        final    = result["messages"][-1].content
        return final
    except Exception as e:
        return f"Agent error: {str(e)}"


if __name__ == "__main__":
    print("Building agent...")
    agent = build_agent()
    print("Agent ready.\n")

    test_questions = [
        "How do I cancel my order?",
        "What is 49.99 plus 8.50 shipping?",
        "I forgot my account password, how do I reset it?",
    ]

    for question in test_questions:
        print("\n" + "=" * 60)
        print(f"QUESTION: {question}")
        print("=" * 60)
        answer = ask_agent(question, agent)
        print(f"\nFINAL ANSWER: {answer}")