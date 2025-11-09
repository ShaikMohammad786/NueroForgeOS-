# api/graph_core.py
from langgraph.graph import StateGraph, END
from typing import Dict, Any, TypedDict
import logging
from memory import rag_manager
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
def node_writer(state):
    logger.info("🧠 Writing code...")
    # form a retrieval query from task + language hint
    query = state["task"]
    # fetch relevant tools/docs
    tools = rag_manager.retrieve_tools(query, top_k=5)
    docs = rag_manager.retrieve_docs(query, top_k=5)
    # build contextual prompt
    context_pieces = []
    for t in tools:
        context_pieces.append(f"Existing tool ({t['metadata'].get('language')}):\n{t['code']}")
    for d in docs:
        context_pieces.append(f"Doc: {d['title']}\n{d['content']}")
    context = "\n\n".join(context_pieces) if context_pieces else None

    code, language = code_writer.generate_code(state["task"], language=state.get("language"), context=context)
    state["code"] = code
    state["language"] = language
    state["attempts"] += 1
    return state



# --- 3️⃣ Executor Node ---
def node_executor(state):
    logger.info("⚙️ Executing code...")
    result = code_executor.execute(state["code"], language=state["language"])
    state["result"] = result
    if result["returncode"] == 0:
        logger.info("✅ Execution succeeded")
        state["error"] = None
        # persist the working tool into memory
        try:
            rag_manager.add_tool(name=None, language=state["language"], code=state["code"], metadata={"source":"auto_promote"})
        except Exception as e:
            logger.warning("Failed to persist tool: %s", e)
    else:
        logger.info("❌ Execution failed")
        state["error"] = result["stderr"] or result.get("stderr")
        # persist error for future retrieval
        try:
            rag_manager.add_error(error_text=state["error"], stderr=state["result"].get("stderr"), context=state["code"])
        except Exception as e:
            logger.warning("Failed to persist error: %s", e)
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
    print("=== NeuroForge LangGraph Orchestrator ===")
    flow = build_graph()
    state = flow.invoke(initial_state(task))

    # Debug print for clarity
    print("\n--- FINAL STATE ---")
    print(state)

    # Safely extract result
    result = {
        "language": state.get("language"),
        "attempts": state.get("attempts", 0),
        "stdout": state.get("result", {}).get("stdout", ""),
        "stderr": state.get("result", {}).get("stderr", ""),
        "returncode": state.get("result", {}).get("returncode", None),
    }

    return result

