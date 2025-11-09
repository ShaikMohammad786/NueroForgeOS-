# api/graph_core.py
from langgraph.graph import StateGraph, END
from typing import Dict, Any, TypedDict
import logging

from agents import code_writer, code_executor, code_fixer

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

# --- 1️⃣ Define State schema (LangGraph expects a type, not a function) ---
class NFState(TypedDict, total=False):
    task: str
    language: str
    code: str
    result: Dict[str, Any]
    error: str
    attempts: int


def initial_state(task: str) -> NFState:
    """Initialize the state dictionary."""
    return {
        "task": task,
        "language": None,
        "code": None,
        "result": None,
        "error": None,
        "attempts": 0,
    }


# --- 2️⃣ Writer Node ---
def node_writer(state: NFState) -> NFState:
    logger.info("🧠 Writing code...")
    code, language = code_writer.generate_code(state["task"], state.get("language"))
    state["code"] = code
    state["language"] = language
    state["attempts"] += 1
    return state


# --- 3️⃣ Executor Node ---
def node_executor(state: NFState) -> NFState:
    logger.info("⚙️ Executing code...")
    result = code_executor.execute(state["code"], language=state["language"])
    state["result"] = result
    if result["returncode"] == 0:
        logger.info("✅ Execution succeeded")
        state["error"] = None
    else:
        logger.info("❌ Execution failed")
        state["error"] = result["stderr"]
    return state


# --- 4️⃣ Fixer Node ---
def node_fixer(state: NFState) -> NFState:
    if not state.get("error"):
        return state
    logger.info("🔧 Fixing code...")
    fixed = code_fixer.fix_code(state["code"], state["error"], language=state["language"])
    state["code"] = fixed
    return state


# --- 5️⃣ Conditional routing ---
def decide_next(state: NFState):
    if state.get("error") and state["attempts"] < 3:
        logger.info("🔁 Retrying after fix (attempt %d)", state["attempts"])
        return "executor"
    logger.info("🏁 Ending flow")
    return END


# --- 6️⃣ Build Graph ---
def build_graph():
    graph = StateGraph(NFState)  # ✅ schema instead of function

    graph.add_node("writer", node_writer)
    graph.add_node("executor", node_executor)
    graph.add_node("fixer", node_fixer)

    graph.add_edge("writer", "executor")
    graph.add_edge("executor", "fixer")
    graph.add_conditional_edges("fixer", decide_next, {"executor": "executor", END: END})

    graph.set_entry_point("writer")

    # ❌ Remove checkpointer requirement for simplicity
    return graph.compile()


# --- 7️⃣ Runner ---
def run_task(task: str):
    flow = build_graph()
    state = flow.invoke(initial_state(task))
    result = state["result"]

    print("\n--- FINAL RESULT ---")
    if result["returncode"] == 0:
        print("✅ SUCCESS\n", result["stdout"])
    else:
        print("🚫 FAILED\n", result["stderr"])
