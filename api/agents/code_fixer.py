# api/agents/code_fixer.py
import os
import logging
from typing import Optional
import google.generativeai as genai

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

API_MODEL = os.getenv("GEMINI_MODEL", "gemini-2.5-pro")
FIX_RETRIES = int(os.getenv("FIX_RETRIES", "2"))
def _configure_gemini():
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        raise RuntimeError("❌ Missing GEMINI_API_KEY in .env")
    genai.configure(api_key=api_key)
    
def _strip_code_fences(text: Optional[str]) -> str:
    if not text:
        return ""
    s = text.strip()
    if s.startswith("python\n"):
        s = s[len("python\n"):].lstrip()
    if s.startswith("```"):
        parts = s.split("```")
        candidate = max((p for p in parts if p.strip()), key=len, default=s)
        s = candidate.strip()
    return s

def fix_code(code: str, error: str, language: str = "python", context: Optional[str] = None, max_tokens: int = 1024) -> str:
    """
    Ask Gemini to fix code for given language and runtime error.
    Returns the fixed code string.
    """
    if not code or not error:
        raise ValueError("code and error required")

    prompt_lines = [
        f"You are an assistant that fixes {language} programs.",
        "The user will provide the original script and the runtime error. Provide only corrected, runnable code with minimal changes.",
        "Constraints:",
        "- Do not add network or filesystem calls unless necessary.",
        "- Avoid use of dangerous system calls.",
        "",
        "Original code:",
        code,
        "",
        "Runtime error / traceback:",
        error
    ]
    if language.lower() == "java":
        prompt_lines.append("Ensure the public class is named Main (public class Main { ... }).")
    if context:
        prompt_lines.append("\nContext:\n" + context)

    prompt = "\n".join(prompt_lines)
    last_exc = None
    for attempt in range(1, FIX_RETRIES + 1):
        try:
            model = genai.GenerativeModel(API_MODEL)
            resp = model.generate_content(prompt)
            raw = getattr(resp, "text", "")
            fixed = _strip_code_fences(raw)
            if not fixed.strip():
                raise RuntimeError("Empty fix returned")
            logger.info("Fix attempt %d successful", attempt)
            return fixed
        except Exception as e:
            logger.warning("Fix attempt %d failed: %s", attempt, e)
            last_exc = e

    raise RuntimeError("Code fixing failed") from last_exc
