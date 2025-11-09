# api/main.py
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import os, logging
from contextlib import asynccontextmanager
from graph_core import run_task
from memory.db_init import init_chroma_client, init_embedding_model

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("NeuroForgeKernel")

# --- TEMP FIX for huggingface_hub ImportError ---
import huggingface_hub
if not hasattr(huggingface_hub, "cached_download"):
    try:
        from huggingface_hub import hf_hub_download
        huggingface_hub.cached_download = hf_hub_download
        print("✅ patched huggingface_hub.cached_download (startup fix)")
    except Exception as e:
        print("⚠️ Patch failed:", e)
# ------------------------------------------------


@asynccontextmanager
async def lifespan(app: FastAPI):
    print("🚀 Starting NeuroForge Memory subsystem...")
    init_embedding_model()
    client = init_chroma_client()
    for col in ["tools", "errors", "docs", "patterns"]:
        if col not in [c.name for c in client.list_collections()]:
            client.create_collection(name=col)
    yield
    print("🧹 Shutting down NeuroForge (cleanup if needed)...")

app = FastAPI(
    title="NeuroForge Kernel",
    description="Self-Improving Runtime for AI Agents",
    version="1.0.0",
    lifespan=lifespan
)

class TaskRequest(BaseModel):
    task: str

@app.get("/")
def root():
    return {"message": "🧠 NeuroForge Kernel is alive"}


@app.post("/run_task")
def run_task_api(req: TaskRequest):
    try:
        logger.info("Received new task: %s", req.task)
        result = run_task(req.task)
        return {"status": "success", "result": result}
    except Exception as e:
        logger.exception("Task failed: %s", e)
        raise HTTPException(status_code=500, detail=str(e))
