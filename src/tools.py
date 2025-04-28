# --- Define RAG tools for HR and Strategy ---
import json
import os
from datetime import datetime, timedelta

from langchain.tools import DuckDuckGoSearchRun, Tool
from llama_index.core import SimpleDirectoryReader, VectorStoreIndex

from src.pain_point_detection import detect_pain_point
from src.rag import query_hr, query_strategy

WORKSHOP_MEMORY_LOGGER = "./workshop_memory.json"

def hr_rag_tool_func(query: str) -> str:
    return query_hr(query).response

def strategy_rag_tool_func(query: str) -> str:
    return query_strategy(query).response

hr_tool = Tool(
    name="HRPolicyRAG",
    func=hr_rag_tool_func,
    description="Use this to answer HR-related questions based on internal HR policies and workflows."
)

strategy_tool = Tool(
    name="StrategyPolicyRAG",
    func=strategy_rag_tool_func,
    description="Use this to answer strategy-related questions based on internal strategy policies and workflows."
)

# --- Web Search Tool ---

search = DuckDuckGoSearchRun()

web_search_tool = Tool(
        name="Search",
        func=search.run,
        description="Useful for searching information on the internet. Use this when you need to find current or factual information."
    )


# --- Question Generator Tool ---

DEFAULT_ROLES = ["facilitator", "hr", "strategy"]
from pathlib import Path


def load_workshop_log():
    try:
        # Try to read and parse the file
        text = Path(WORKSHOP_MEMORY_LOGGER).read_text()
        data = json.loads(text) if text.strip() else {}
    except FileNotFoundError:
        # File doesn’t exist yet → start fresh
        data = {}
    except json.JSONDecodeError:
        # File exists but is empty or invalid → also start fresh
        data = {}

    # Ensure each role key exists and is a list
    for role in DEFAULT_ROLES:
        data.setdefault(role, [])

    return data

# Usage
recent_conversation_log = load_workshop_log()


def question_generator(_: str) -> str:
    now = datetime.now()
    prompts = []

    for persona, logs in recent_conversation_log.items():
        if persona !='facilitator':
            if not logs:
                prompts.append(f"What are your current priorities, {persona}?")
                continue

            last_msg_time = datetime.strptime(logs[-1][0], "%Y-%m-%d %H:%M")
            if now - last_msg_time > timedelta(seconds=3):
                prompts.append(f"It's been a while since {persona} contributed. Ask them about current goals or blockers.")

    if not prompts:
        return "No question suggestions needed right now."
    return "\n".join(prompts)

question_tool = Tool(
    name="QuestionGenerator",
    func=question_generator,
    description="Use this to suggest facilitator questions when HR or Strategy have been quiet lately."
)

# ─── LangChain Tool Definition ─────────────────────────────────────────────────

def pain_point_tool_fn(text: str) -> str:
    result = detect_pain_point(text)
    return result or "No pain point detected."

PainPointDetector = Tool(
    name="PainPointDetector",
    func=pain_point_tool_fn,
    description=(
        "Analyzes an input utterance for potential pain points using sentiment analysis, "
        "emotion detection, intent classification, and optional NER. "
        "If a pain point is found, it logs it to pain_points.json and returns the reason."
    ),
)