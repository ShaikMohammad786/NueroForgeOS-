# api/agents/code_writer.py
import os
import logging
from typing import Optional, Tuple
import google.generativeai as genai
from dotenv import load_dotenv
from pathlib import Path

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

API_MODEL = os.getenv("GEMINI_MODEL", "gemini-2.5-flash")
GEN_CALL_RETRIES = int(os.getenv("GEN_RETRIES", "2"))

LANG_HINTS = {
    "python": "Python 3.10+ script (run with `python file.py`)",
    "javascript": "JavaScript for Node.js (use console.log)",
    "c": "C program (compile with gcc, standard C99)",
    "cpp": "C++ program (compile with g++, standard C++17)",
    "java": "Java program (public class Main, compile with javac Main.java)",
}


def _configure_gemini():
    # Look upward from agents/ to project root
    root_env = Path(__file__).resolve().parents[2] / ".env"
    load_dotenv(root_env)

    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        raise RuntimeError(f"❌ Missing GEMINI_API_KEY (checked {root_env})")
    import google.generativeai as genai
    genai.configure(api_key=api_key)


def _strip_code_fences(text: Optional[str]) -> str:
    """Remove markdown fences and language labels like ```python ...```, and stray leading language tokens."""
    if not text:
        return ""
    raw = text.strip()
    # Fast path: if a fenced block exists anywhere, extract its inner content
    if "```" in raw:
        parts = raw.split("```")
        # Take the first non-empty inner block after a fence
        for i in range(1, len(parts), 2):
            block = parts[i]
            # Drop an initial language label on the same line, e.g., "python\n"
            block_lines = block.splitlines()
            if block_lines:
                first = block_lines[0].strip().lower()
                if first in {"python", "c", "cpp", "c++", "javascript", "java"}:
                    block_lines = block_lines[1:]
            code = "\n".join(block_lines).strip()
            if code:
                return code
        # If nothing extracted, fall through to non-fence cleanup
    # Non-fenced content: drop a stray first-line language token
    lines = raw.splitlines()
    while lines and lines[0].strip().lower() in {"python", "c", "cpp", "c++", "javascript", "java"}:
        lines = lines[1:]
    # Also drop any residual lone backtick fence lines that might have slipped through
    lines = [ln for ln in lines if not ln.strip().startswith("```")]
    return "\n".join(lines).strip()


def _detect_language_with_gemini(task: str) -> str:
    """
    Ask Gemini which programming language is implied by the task.
    Returns one of: python, javascript, c, cpp, java
    Defaults to python if uncertain.
    """
    _configure_gemini()
    model = genai.GenerativeModel(API_MODEL)

    prompt = f"""
You are a language detection assistant.

The user will describe a coding task. 
Your job is to determine the programming language they are referring to.

Supported options: Python, JavaScript, C, C++, Java.

Respond with only the language name in lowercase (e.g., "python", "c", "cpp", "java", "javascript").

User task:
{task}
"""
    try:
        resp = model.generate_content(prompt)
        text = (resp.text or "").strip().lower()
        for lang in LANG_HINTS.keys():
            if lang in text:
                return lang
        if "c++" in text:
            return "cpp"
        return "python"
    except Exception as e:
        logger.warning("Language detection failed, defaulting to python: %s", e)
        return "python"


def generate_code(task: str, language: Optional[str] = None, context: Optional[str] = None) -> Tuple[str, str]:
    """
    Generate executable code for the given task using Gemini.
    Step 1: detect language (if not provided)
    Step 2: generate code in that language
    """
    _configure_gemini()

    if not task:
        raise ValueError("Task cannot be empty")

    # Step 1: Ask Gemini what language the user wants
    language = language or _detect_language_with_gemini(task)
    if language not in LANG_HINTS:
        language = "python"  # fallback

    # Step 2: Ask Gemini to generate code in that language
    prompt = (
        f"Write a {language} program to {task}.\n"
        f"Rules:\n"
        f"- Return only executable {language} code (no explanations).\n"
        f"- Must print or output results to STDOUT.\n"
        f"- {LANG_HINTS[language]}"
    )
    if context:
        prompt += f"\nContext:\n{context}"

    logger.info("Generating %s code for task: %s", language, task)
    last_exc = None

    for attempt in range(1, GEN_CALL_RETRIES + 1):
        try:
            model = genai.GenerativeModel(API_MODEL)
            resp = model.generate_content(prompt)
            raw = getattr(resp, "text", "")
            code = _strip_code_fences(raw)
            if not code.strip():
                raise RuntimeError("Empty code returned by model")
            logger.info("✅ Code generation successful for %s", language)
            return code, language
        except Exception as e:
            logger.warning("⚠️ Generation attempt %d failed: %s", attempt, e)
            last_exc = e

    raise RuntimeError("Code generation failed") from last_exc
