from langgraph.graph import StateGraph, END
from typing import Dict, Any, TypedDict
import logging
from memory import rag_manager
from agents import code_writer, code_executor, code_fixer

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


# --- 1️⃣ Define State Schema ---
class NFState(TypedDict, total=False):
    task: str
    language: str
    code: str
    result: Dict[str, Any]
    error: str
    attempts: int


def initial_state(task: str) -> NFState:
    """Initialize the LangGraph state."""
    return {
        "task": task,
        "language": None,
        "code": None,
        "result": None,
        "error": None,
        "attempts": 0,
    }


# --- 2️⃣ Code Writer Node ---
def node_writer(state):
    logger.info("🧠 Writing code...")
    query = state["task"]

    # retrieve relevant tools and docs
    tools = rag_manager.retrieve_tools(query, top_k=5)
    docs = rag_manager.retrieve_docs(query, top_k=5)

    context_parts = []
    for t in tools:
        context_parts.append(f"Existing tool ({t['metadata'].get('language')}):\n{t.get('code', '')}")
    for d in docs:
        context_parts.append(f"Doc: {d.get('title')}\n{d.get('content')}")

    context = "\n\n".join(context_parts) if context_parts else None

    code, language = code_writer.generate_code(
        state["task"], language=state.get("language"), context=context
    )
    state["code"] = code
    state["language"] = language
    state["attempts"] += 1
    return state


# --- 3️⃣ Code Executor Node ---
def node_executor(state):
    logger.info("⚙️ Executing code...")
    result = code_executor.execute(state["code"], language=state["language"])
    state["result"] = result

    # extract actual return code from nested result
    returncode = result.get("result", {}).get("returncode", 1)

    if returncode == 0:
        logger.info("✅ Execution succeeded")
        state["error"] = None
        try:
            rid = rag_manager.add_tool(
                name=None,
                language=state["language"],
                code=state["code"],
                metadata={"source": "auto_promote"}
            )
            logger.info(f"🧩 Stored successful tool in Pinecone (id={rid})")
        except Exception as e:
            logger.warning(f"⚠️ Failed to persist tool: {e}")
    else:
        logger.info("❌ Execution failed")
        stderr = result.get("result", {}).get("stderr", "")
        state["error"] = stderr
        try:
            rag_manager.add_error(
                error_text=state["error"],
                stderr=stderr,
                context=state["code"]
            )
            logger.info("🧠 Logged error context for future retrieval")
        except Exception as e:
            logger.warning(f"⚠️ Failed to persist error: {e}")

    return state


# --- 4️⃣ Fixer Node ---
def node_fixer(state: NFState) -> NFState:
    if not state.get("error"):
        return state
    logger.info("🔧 Fixing code...")
    fixed = code_fixer.fix_code(state["code"], state["error"], language=state["language"])
    state["code"] = fixed
    return state


# --- 5️⃣ Conditional Routing ---
def decide_next(state: NFState):
    if state.get("error") and state["attempts"] < 3:
        logger.info(f"🔁 Retrying after fix (attempt {state['attempts']})")
        return "executor"
    logger.info("🏁 Ending flow")
    return END


# --- 6️⃣ Build LangGraph ---
def build_graph():
    graph = StateGraph(NFState)

    graph.add_node("writer", node_writer)
    graph.add_node("executor", node_executor)
    graph.add_node("fixer", node_fixer)

    graph.add_edge("writer", "executor")
    graph.add_edge("executor", "fixer")
    graph.add_conditional_edges("fixer", decide_next, {"executor": "executor", END: END})

    graph.set_entry_point("writer")
    return graph.compile()


# --- 7️⃣ Run a Full Task ---
def run_task(task: str):
    print("=== NeuroForge LangGraph Orchestrator ===")
    flow = build_graph()
    state = flow.invoke(initial_state(task))

    print("\n--- FINAL STATE ---")
    print(state)

    inner = state.get("result", {}).get("result", {})
    result = {
        "language": state.get("language"),
        "attempts": state.get("attempts", 0),
        "stdout": inner.get("stdout", ""),
        "stderr": inner.get("stderr", ""),
        "returncode": inner.get("returncode", None),
    }

    return result
