# api/agents/code_executor.py
import os
import re
import shutil
import tempfile
import subprocess
import logging
from typing import Dict, Any, Optional
import requests
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

LOCAL_TIMEOUT = int(os.getenv("LOCAL_RUN_TIMEOUT", "8"))

# Basic banned patterns that could cause network/OS usage; extend as needed.
BANNED_PATTERNS = [
    r"\bexec\b", r"\beval\b", r"system\(", r"fork\(", r"socket\.", r"subprocess\.", r"popen\(",
    r"#include\s*<sys/", r"#include\s*<netinet", r"import\s+socket", r"Runtime\.getRuntime",
]

LANG_EXT = {
    "python": ".py",
    "javascript": ".js",
    "c": ".c",
    "cpp": ".cpp",
    "java": ".java",
}

def _contains_banned(code: str) -> Optional[str]:
    for pat in BANNED_PATTERNS:
        if re.search(pat, code):
            return pat
    return None

def _write_temp_file(contents: str, suffix: str) -> str:
    tmpdir = tempfile.mkdtemp(prefix="nf_run_")
    path = os.path.join(tmpdir, "Main" + suffix)
    with open(path, "w", encoding="utf-8") as f:
        f.write(contents)
    return tmpdir, path

def _run_subprocess(cmd, cwd, timeout) -> Dict[str, Any]:
    try:
        proc = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout, cwd=cwd)
        return {"returncode": proc.returncode, "stdout": proc.stdout or "", "stderr": proc.stderr or ""}
    except subprocess.TimeoutExpired as e:
        return {"returncode": 124, "stdout": "", "stderr": f"TimeoutExpired: {e}"}
    except Exception as e:
        return {"returncode": 1, "stdout": "", "stderr": f"Execution exception: {e}"}

# api/agents/code_executor.py (only modify execute())

# logger = logging.getLogger(__name__)

RUNNER_URL = os.getenv("RUNNER_URL", "http://localhost:8001/run")

def execute(code: str, language: str = "python", timeout: int = 8) -> dict:
    """
    Send code to the sandboxed Runner microservice.
    """
    payload = {"language": language, "code": code, "timeout": timeout}
    try:
        resp = requests.post(RUNNER_URL, json=payload, timeout=timeout + 5)
        resp.raise_for_status()
        return resp.json()
    except Exception as e:
        logger.error("Runner request failed: %s", e)
        return {"returncode": 1, "stdout": "", "stderr": f"Runner error: {e}"}
