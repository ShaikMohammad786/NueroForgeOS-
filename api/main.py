# api/main.py
import base64
import importlib
import logging
import subprocess
import sys
from contextlib import asynccontextmanager
from typing import Optional, Dict

from fastapi import FastAPI, HTTPException
from fastapi import UploadFile, File, Form
from fastapi.responses import FileResponse
from starlette.background import BackgroundTask
import tempfile
import os
from pydantic import BaseModel

from memory.db_init import init_pinecone_client, init_embedding_model

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("NeuroForgeKernel")


def ensure_dependency(package: str, import_name: Optional[str] = None) -> None:
    """
    Lazily install a dependency inside the container when it's first needed.
    Avoids having to bake every optional package into the base image.
    """
    module_name = import_name or package
    try:
        importlib.import_module(module_name)
        return
    except ModuleNotFoundError:
        logger.info("📦 Installing missing dependency '%s' on-demand...", package)
        try:
            subprocess.check_call(
                [sys.executable, "-m", "pip", "install", package],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )
        except subprocess.CalledProcessError as exc:
            logger.error("Failed to install %s: %s", package, exc)
            raise
        importlib.import_module(module_name)
        logger.info("✅ Installed '%s'", package)


# Ensure core multipart support
try:
    ensure_dependency("python-multipart", import_name="multipart")
except Exception as exc:
    logger.warning("⚠️ Failed to ensure python-multipart: %s", exc)

# Defer importing graph_core until after auto-install bootstrap is configured
from graph_core import run_task

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
    files_b64: Optional[Dict[str, str]] = None  # filename -> base64-encoded content
    timeout: Optional[int] = None

@app.get("/")
def root():
    return {"message": "🧠 NeuroForge Kernel (Pinecone) is alive"}

@app.post("/run_task")
def run_task_api(req: TaskRequest):
    try:
        logger.info("Received new task: %s", req.task)
        # Decode optional files
        input_files: Optional[Dict[str, bytes]] = None
        if req.files_b64:
            input_files = {}
            for name, b64 in req.files_b64.items():
                try:
                    input_files[name] = base64.b64decode(b64)
                except Exception as e:
                    raise HTTPException(status_code=400, detail=f"Invalid base64 for file {name}: {e}")

        result = run_task(req.task, input_files=input_files, timeout=req.timeout)
        return {"status": "success", "result": result}
    except Exception as e:
        logger.exception("Task failed: %s", e)
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/run_task_multipart")
async def run_task_multipart(
    task: str = Form(...),
    timeout: Optional[int] = Form(default=None),
    files: Optional[list[UploadFile]] = File(default=None),
):
    """
    Accept multipart/form-data with one or more files and a task.
    The uploaded files are provided to the runner as input workspace files.
    """
    try:
        logger.info("Received new multipart task: %s", task)
        input_files: Optional[Dict[str, bytes]] = None
        if files:
            input_files = {}
            for f in files:
                try:
                    content = await f.read()
                    input_files[f.filename] = content
                except Exception as fe:
                    raise HTTPException(status_code=400, detail=f"Failed to read file {getattr(f,'filename','(unknown)')}: {fe}")

        result = run_task(task, input_files=input_files, timeout=timeout)
        return {"status": "success", "result": result}
    except HTTPException:
        raise
    except Exception as e:
        logger.exception("Multipart task failed: %s", e)
        raise HTTPException(status_code=500, detail=str(e))


# --- Dedicated, production-grade PDF -> DOCX converter (no LLM involved) ---
@app.post("/convert/pdf-to-docx")
async def convert_pdf_to_docx(file: UploadFile = File(...)) -> FileResponse:
    """
    Convert an uploaded PDF to DOCX and stream the result back.
    This bypasses the generic LLM code runner for reliability.
    """
    # Ensure runtime dependencies exist inside the container
    try:
        ensure_dependency("pdf2docx")
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Failed to install pdf2docx: {exc}")

    # Validate content-type and filename
    filename = file.filename or "input.pdf"
    if not filename.lower().endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Uploaded file must be a .pdf")

    # Materialize the upload to a temp file
    try:
        tmp_pdf = tempfile.NamedTemporaryFile(delete=False, suffix=".pdf")
        pdf_path = tmp_pdf.name
        content = await file.read()
        tmp_pdf.write(content)
        tmp_pdf.close()
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Failed to persist uploaded PDF: {exc}")

    # Prepare output DOCX temp path
    base_name = os.path.splitext(os.path.basename(filename))[0]
    try:
        tmp_docx = tempfile.NamedTemporaryFile(delete=False, suffix=".docx")
        docx_path = tmp_docx.name
        tmp_docx.close()
    except Exception as exc:
        # Cleanup pdf if docx temp creation fails
        try:
            os.unlink(pdf_path)
        except Exception:
            pass
        raise HTTPException(status_code=500, detail=f"Failed to allocate output DOCX: {exc}")

    # Perform conversion
    try:
        from pdf2docx import Converter
        conv = Converter(pdf_path)
        conv.convert(docx_path)
        conv.close()
    except Exception as exc:
        # Cleanup on failure
        try:
            os.unlink(pdf_path)
        except Exception:
            pass
        try:
            os.unlink(docx_path)
        except Exception:
            pass
        raise HTTPException(status_code=500, detail=f"Conversion failed: {exc}")

    # Schedule temp cleanup after response is sent
    def _cleanup_paths() -> None:
        for p in (pdf_path, docx_path):
            try:
                if os.path.exists(p):
                    os.unlink(p)
            except Exception:
                pass

    output_name = f"{base_name}.docx"
    return FileResponse(
        path=docx_path,
        filename=output_name,
        media_type="application/vnd.openxmlformats-officedocument.wordprocessingml.document",
        background=BackgroundTask(_cleanup_paths),
    )
