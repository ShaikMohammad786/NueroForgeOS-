import os
import re
import tempfile
import subprocess
import logging
from typing import Dict, Any, Optional
import requests

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

LOCAL_TIMEOUT = int(os.getenv("LOCAL_RUN_TIMEOUT", "8"))
RUNNER_URL = os.getenv("RUNNER_URL", "http://localhost:8001/run")

# Security guardrails
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
    """Return the first banned pattern match, or None."""
    for pat in BANNED_PATTERNS:
        if re.search(pat, code):
            return pat
    return None


def _write_temp_file(contents: str, suffix: str) -> str:
    """Write a temporary code file to disk for local sandbox use."""
    tmpdir = tempfile.mkdtemp(prefix="nf_run_")
    path = os.path.join(tmpdir, "Main" + suffix)
    with open(path, "w", encoding="utf-8") as f:
        f.write(contents)
    return tmpdir, path


def _run_subprocess(cmd, cwd, timeout) -> Dict[str, Any]:
    """Execute a command locally and capture output."""
    try:
        proc = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout, cwd=cwd)
        return {"returncode": proc.returncode, "stdout": proc.stdout or "", "stderr": proc.stderr or ""}
    except subprocess.TimeoutExpired as e:
        return {"returncode": 124, "stdout": "", "stderr": f"TimeoutExpired: {e}"}
    except Exception as e:
        return {"returncode": 1, "stdout": "", "stderr": f"Execution exception: {e}"}


# -------------------------------
# 🌐 Runner-based Remote Execution
# -------------------------------

def execute(
    code: str,
    language: str = "python",
    timeout: int = 8,
    requirements: Optional[list[str]] = None,
) -> dict:
    """
    Send code to the isolated Runner microservice.
    Always returns:
        {
            "result": {
                "returncode": int,
                "stdout": str,
                "stderr": str
            }
        }
    """
    payload = {"language": language, "code": code, "timeout": timeout}
    if requirements:
        payload["requirements"] = requirements

    try:
        resp = requests.post(RUNNER_URL, json=payload, timeout=timeout + 5)
        resp.raise_for_status()
        raw = resp.json()
        logger.info(f"🧠 Runner response: {raw}")

        # ✅ Normalize response
        if all(k in raw for k in ("returncode", "stdout", "stderr")):
            result = raw
        elif isinstance(raw, dict) and "result" in raw and isinstance(raw["result"], dict):
            result = raw["result"]
        else:
            # Unknown response shape → fallback
            result = {"returncode": 1, "stdout": "", "stderr": str(raw)}

        return {"result": result}

    except Exception as e:
        logger.error(f"Runner request failed: {e}")
        return {
            "result": {
                "returncode": 1,
                "stdout": "",
                "stderr": f"Runner error: {e}",
            }
        }
