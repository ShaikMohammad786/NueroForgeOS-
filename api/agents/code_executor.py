import os
import re
import tempfile
import subprocess
import logging
from typing import Dict, Any, Optional, Set
import requests
import ast
from memory import rag_manager
import base64

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

LOCAL_TIMEOUT = int(os.getenv("LOCAL_RUN_TIMEOUT", "120"))
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
    timeout: int = 60,
    requirements: Optional[list[str]] = None,
    allow_network: bool = True,
    auto_requirements: bool = True,
    input_files: Optional[Dict[str, bytes]] = None,
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
    # Pre-sanitize trivially broken model output (e.g., stray 'python' line or markdown fences)
    if isinstance(code, str):
        stripped = code.lstrip("\ufeff").strip()
        lines = stripped.splitlines()
        # Remove leading language token lines and markdown fences
        while lines and (lines[0].strip().lower() in {"python", "c", "cpp", "c++", "javascript", "java"} or lines[0].strip().startswith("```")):
            lines = lines[1:]
        code = "\n".join(lines).strip()

    # Infer requirements early to decide dynamic timeout
    inferred: Set[str] = set()
    if language == "python" and auto_requirements:
        inferred = _infer_python_requirements_from_code(code)

    # Heuristic timeout: more time if we need to install packages or heavy libs are present
    heavy_libs = {"pandas", "numpy", "torch", "opencv-python", "pdfplumber", "tabula-py", "openpyxl"}
    base_timeout = timeout or 60
    install_penalty = 20 if inferred else 0
    heavy_bonus = 20 if (inferred & heavy_libs) else 0
    timeout_final = max(base_timeout, 30 + install_penalty + heavy_bonus)

    payload = {"language": language, "code": code, "timeout": timeout_final}
    if requirements:
        payload["requirements"] = requirements
    if allow_network:
        payload["network"] = os.getenv("SANDBOX_DEFAULT_NETWORK", "bridge")
    else:
        payload["network"] = "none"

    # Attach input files, if any (base64)
    if input_files:
        files_b64 = {}
        for name, data in input_files.items():
            try:
                files_b64[name] = base64.b64encode(data).decode("ascii")
            except Exception:
                continue
        if files_b64:
            payload["files_b64"] = files_b64

    # Optionally pre-resolve python imports to requirements
    if language == "python" and auto_requirements and inferred:
        existing = set(payload.get("requirements", []))
        merged = list(existing.union(inferred))
        if merged:
            payload["requirements"] = merged

    try:
        resp = requests.post(RUNNER_URL, json=payload, timeout=timeout_final + 60)
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

        # If python code failed due to missing input file, surface the required filenames
        inputs_required = _extract_missing_filenames(result.get("stderr") or "") if isinstance(result.get("stderr"), str) else []
        if inputs_required:
            return {"result": result, "inputs_required": inputs_required}

        # If python code failed due to missing module, retry once with auto-install
        if (
            language == "python"
            and result.get("returncode", 0) != 0
            and isinstance(result.get("stderr"), str)
            and ("ModuleNotFoundError: No module named" in result["stderr"] or "No module named" in result["stderr"])
        ):
            # Short-circuit if we've already seen a very similar error to avoid loops
            try:
                similar = rag_manager.retrieve_similar_errors(result["stderr"], top_k=1)
                if similar:
                    logger.info("⚠️ Similar error found in memory; skipping auto-install retry to avoid repetition.")
                    return {"result": result}
            except Exception:
                pass

            # Extract all missing modules mentioned and retry once installing all
            missing_pkgs = set()
            for match in re.findall(r"No module named ['\"]([^'\"]+)['\"]", result["stderr"]):
                missing_pkgs.add(_map_import_to_pypi(match))
            single = _extract_missing_module(result["stderr"])
            if single:
                missing_pkgs.add(_map_import_to_pypi(single))

            if missing_pkgs:
                logger.info(f"📦 Auto-install retry for missing modules: {sorted(missing_pkgs)}")
                retry_payload = dict(payload)
                retry_payload["requirements"] = list(set(retry_payload.get("requirements", []) | missing_pkgs))
                # Give extra time for installs
                retry_timeout = max(timeout_final, 60) + 60
                retry_payload["timeout"] = retry_timeout
                try:
                    retry_resp = requests.post(RUNNER_URL, json=retry_payload, timeout=retry_timeout + 60)
                    retry_resp.raise_for_status()
                    retry_raw = retry_resp.json()
                    logger.info(f"🧠 Runner response (retry): {retry_raw}")
                    if all(k in retry_raw for k in ("returncode", "stdout", "stderr")):
                        retry_result = retry_raw
                    elif isinstance(retry_raw, dict) and "result" in retry_raw and isinstance(retry_raw["result"], dict):
                        retry_result = retry_raw["result"]
                    else:
                        retry_result = {"returncode": 1, "stdout": "", "stderr": str(retry_raw)}
                    return {"result": retry_result}
                except Exception as e2:
                    logger.error(f"Retry after installing {sorted(missing_pkgs)} failed: {e2}")
                    # fallthrough to original result

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


_STDLIB_LIKE: Set[str] = {
    # common stdlib modules to ignore
    "sys","os","json","re","math","itertools","functools","collections","subprocess","pathlib",
    "typing","dataclasses","datetime","time","random","logging","argparse","shutil","tempfile",
    "uuid","hashlib","base64","gzip","bz2","lzma","csv","configparser","enum","statistics",
}


def _infer_python_requirements_from_code(code: str) -> Set[str]:
    try:
        tree = ast.parse(code)
    except Exception:
        return set()
    imports: Set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for n in node.names:
                top = (n.name or "").split(".")[0]
                if top:
                    imports.add(top)
        elif isinstance(node, ast.ImportFrom):
            if node.module:
                top = node.module.split(".")[0]
                imports.add(top)
    # Map to PyPI names
    pkgs = set()
    for mod in imports:
        if mod in _STDLIB_LIKE:
            continue
        pkgs.add(_map_import_to_pypi(mod))
    return pkgs


def _extract_missing_module(stderr: str) -> Optional[str]:
    m = re.search(r"ModuleNotFoundError: No module named ['\"]([^'\"]+)['\"]", stderr)
    if not m:
        return None
    return m.group(1)


def _extract_missing_filenames(stderr: str) -> list[str]:
    """
    Heuristically extract filenames referenced in common missing-file errors.
    """
    names: Set[str] = set()
    # quoted filenames
    for m in re.findall(r"['\"]([^'\"]+\.(?:pdf|csv|xlsx?|txt|json|xml|jpg|png))['\"]", stderr, flags=re.IGNORECASE):
        names.add(m)
    # common phrases
    patterns = [
        r"file\s+not\s+found:\s+([^\s]+)",
        r"no such file or directory:\s+['\"]?([^\s'\"\\]+)",
        r"Input .* file ['\"]([^'\"]+)['\"] not found",
    ]
    for pat in patterns:
        for m in re.findall(pat, stderr, flags=re.IGNORECASE):
            if any(ext in m.lower() for ext in (".pdf", ".csv", ".xls", ".xlsx", ".txt", ".json", ".xml", ".jpg", ".png")):
                names.add(m)
    return sorted(names)


_PY_IMPORT_TO_PYPI = {
    "cv2": "opencv-python",
    "PIL": "Pillow",
    "sklearn": "scikit-learn",
    "bs4": "beautifulsoup4",
    "yaml": "PyYAML",
    "Crypto": "pycryptodome",
    "dateutil": "python-dateutil",
    "pdf2image": "pdf2image",
    "pdfplumber": "pdfplumber",
    "PyPDF2": "PyPDF2",
    "openpyxl": "openpyxl",
    "reportlab": "reportlab",
    "tabula": "tabula-py",
    "pandas": "pandas",
    "numpy": "numpy",
}


def _map_import_to_pypi(module_name: str) -> str:
    # Top-level package name (e.g., x.y -> x)
    top = module_name.split(".")[0]
    return _PY_IMPORT_TO_PYPI.get(top, top)
