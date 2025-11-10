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
    input_files: Dict[str, bytes]
    inputs_required: Any
    timeout: int
    error_signature: str
def _normalize_error(err: str) -> str:
    """Normalize error text to a stable signature (strip file paths, numbers)."""
    try:
        import re as _re
        s = err or ""
        # remove absolute paths and Windows drive letters
        s = _re.sub(r"[A-Za-z]:\\\\[^\\s]+", "", s)  # Windows
        s = _re.sub(r"/[^\\s]+", "", s)  # Unix-like
        # collapse numbers (line numbers, ports, etc.)
        s = _re.sub(r"\d+", "N", s)
        # trim whitespace
        return " ".join(s.split())[:1024]
    except Exception:
        return (err or "")[:1024]

def _error_signature(err: str) -> str:
    import hashlib
    norm = _normalize_error(err)
    return hashlib.sha1(norm.encode("utf-8", errors="ignore")).hexdigest()



def initial_state(task: str, input_files: Dict[str, bytes] | None = None, timeout: int | None = None) -> NFState:
    """Initialize the LangGraph state."""
    return {
        "task": task,
        "language": None,
        "code": None,
        "result": None,
        "error": None,
        "attempts": 0,
        "input_files": input_files or {},
        "timeout": timeout or 60,
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
    result = code_executor.execute(
        state["code"],
        language=state["language"],
        timeout=state.get("timeout", 60),
        input_files=state.get("input_files") or None,
    )
    state["result"] = result

    # extract actual return code from nested result
    returncode = result.get("result", {}).get("returncode", 1)

    if returncode == 0:
        logger.info("✅ Execution succeeded")
        state["error"] = None
        state["error_signature"] = None
        try:
            rid = rag_manager.add_tool(
                name=None,
                language=state["language"],
                code=state["code"],
                metadata={"source": "auto_promote", "success_count": 1}
            )
            logger.info(f"🧩 Stored successful tool in Pinecone (id={rid})")
        except Exception as e:
            logger.warning(f"⚠️ Failed to persist tool: {e}")
    else:
        logger.info("❌ Execution failed")
        stderr = result.get("result", {}).get("stderr", "")
        state["error"] = stderr
        state["error_signature"] = _error_signature(stderr) if stderr else None
        # Persist error for future avoidance
        try:
            rag_manager.add_error(
                error_text=state["error"],
                stderr=stderr,
                context=state["code"]
            )
            logger.info("🧠 Logged error context for future retrieval")
        except Exception as e:
            logger.warning(f"⚠️ Failed to persist error: {e}")

        # If inputs are required, annotate state so API can prompt callers
        inputs_required = result.get("inputs_required") if isinstance(result, dict) else None
        if inputs_required:
            state["inputs_required"] = inputs_required

    return state


# --- 4️⃣ Fixer Node ---
def node_fixer(state: NFState) -> NFState:
    if not state.get("error"):
        return state
    logger.info("🔧 Fixing code...")
    # 1) Try to apply a known fix from memory using error signature or text
    try:
        sig = state.get("error_signature") or _error_signature(state.get("error") or "")
        candidates = rag_manager.retrieve_fixes(sig, top_k=1) or []
        if not candidates:
            candidates = rag_manager.retrieve_fixes(state.get("error") or "", top_k=1) or []
        fixed = None
        if candidates:
            # The 'fixed code' is embedded within the vector text; we cannot retrieve raw code directly.
            # As a pragmatic approach, fall back to LLM but bias with context from tools/docs already gathered by writer.
            logger.info("🧩 Similar fix found; proceeding to re-generate with higher confidence.")
        # 2) Use LLM-based fixer with RAG context to handle brand-new/unknown errors
        try:
            tools = rag_manager.retrieve_tools(state.get("task") or "", top_k=5)
        except Exception:
            tools = []
        try:
            docs = rag_manager.retrieve_docs(state.get("task") or "", top_k=5)
        except Exception:
            docs = []
        context_parts = []
        for t in tools:
            md = t.get("metadata") or {}
            lang = md.get("language") or ""
            context_parts.append(f"Existing tool ({lang}):\n{md.get('name') or ''}")
        for d in docs:
            md = d.get("metadata") or {}
            context_parts.append(f"Doc: {md.get('title') or ''}")
        context = "\n\n".join(context_parts) if context_parts else None

        fixed = code_fixer.fix_code(state["code"], state["error"], language=state["language"], context=context)
        state["code"] = fixed
        # 3) Persist fix mapped to error signature for future instant application
        try:
            if sig and fixed:
                rag_manager.add_fix(sig, state["language"], fixed, metadata={"source": "auto_fix"})
        except Exception as e:
            logger.warning(f"⚠️ Failed to persist fix: {e}")
        # Increment attempts and adaptively increase timeout for the next run
        try:
            state["attempts"] = int(state.get("attempts", 0)) + 1
        except Exception:
            state["attempts"] = 1
        try:
            current_to = int(state.get("timeout", 60) or 60)
            state["timeout"] = min(300, max(60, current_to + 30))
        except Exception:
            state["timeout"] = 90
        return state
    except Exception as e:
        logger.warning(f"⚠️ Fixing pipeline failed, leaving code unchanged: {e}")
        return state


# --- 5️⃣ Conditional Routing ---
def decide_next(state: NFState):
    if state.get("error"):
        if state["attempts"] < 3:
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
def run_task(task: str, input_files: Dict[str, bytes] | None = None, timeout: int | None = None):
    print("=== NeuroForge LangGraph Orchestrator ===")
    flow = build_graph()
    state = flow.invoke(initial_state(task, input_files=input_files, timeout=timeout))

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
    # Bubble up any declared input requirements
    if state.get("inputs_required"):
        result["inputs_required"] = state["inputs_required"]

    return result
