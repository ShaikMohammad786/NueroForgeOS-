# api/main.py
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import logging
from contextlib import asynccontextmanager
from graph_core import run_task
from memory.db_init import init_pinecone_client, init_embedding_model

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("NeuroForgeKernel")

@asynccontextmanager
async def lifespan(app: FastAPI):
    print("🚀 Starting NeuroForge Memory subsystem...")
    # Initialize embeddings and Pinecone index
    init_embedding_model()
    init_pinecone_client()
    print("✅ Pinecone client and embedding model initialized.")
    yield
    print("🧹 Shutting down NeuroForge (cleanup if needed)...")

app = FastAPI(
    title="NeuroForge Kernel",
    description="Self-Improving Runtime for AI Agents (Pinecone version)",
    version="1.0.0",
    lifespan=lifespan
)

class TaskRequest(BaseModel):
    task: str

@app.get("/")
def root():
    return {"message": "🧠 NeuroForge Kernel (Pinecone) is alive"}

@app.post("/run_task")
def run_task_api(req: TaskRequest):
    try:
        logger.info("Received new task: %s", req.task)
        result = run_task(req.task)
        return {"status": "success", "result": result}
    except Exception as e:
        logger.exception("Task failed: %s", e)
        raise HTTPException(status_code=500, detail=str(e))
