# runner/app.py
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import subprocess
import tempfile
import os
import shutil
import uuid

app = FastAPI(title="NeuroForge Sandbox Runner")

# Supported languages and commands
RUN_COMMANDS = {
    "python": ["python3", "main.py"],
    "javascript": ["node", "main.js"],
    "c": ["bash", "-c", "gcc main.c -o main && ./main"],
    "cpp": ["bash", "-c", "g++ main.cpp -o main && ./main"],
    "java": ["bash", "-c", "javac Main.java && java Main"],
}

class RunRequest(BaseModel):
    language: str
    code: str
    timeout: int = 8


@app.post("/run")
async def run_code(req: RunRequest):
    if req.language not in RUN_COMMANDS:
        raise HTTPException(400, f"Unsupported language: {req.language}")

    temp_dir = tempfile.mkdtemp(prefix="nf_")
    try:
        filename = {
            "python": "main.py",
            "javascript": "main.js",
            "c": "main.c",
            "cpp": "main.cpp",
            "java": "Main.java",
        }[req.language]

        file_path = os.path.join(temp_dir, filename)
        with open(file_path, "w", encoding="utf-8") as f:
            f.write(req.code)

        result = subprocess.run(
            RUN_COMMANDS[req.language],
            capture_output=True,
            text=True,
            timeout=req.timeout,
            cwd=temp_dir,
        )

        return {
            "returncode": result.returncode,
            "stdout": result.stdout,
            "stderr": result.stderr,
        }

    except subprocess.TimeoutExpired:
        return {
            "returncode": 124,
            "stdout": "",
            "stderr": "Execution timed out.",
        }
    except Exception as e:
        return {
            "returncode": 1,
            "stdout": "",
            "stderr": f"Runner error: {e}",
        }
    finally:
        shutil.rmtree(temp_dir, ignore_errors=True)
