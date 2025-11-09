# main.py
import logging
from dotenv import load_dotenv
from agents import code_writer, code_executor, code_fixer

load_dotenv()
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def self_correcting_flow(task: str, attempts: int = 3):
    context = None
    code = None
    language = None

    for attempt in range(1, attempts + 1):
        print(f"\n🧠 Attempt {attempt}: Generating code...")
        try:
            code, language = code_writer.generate_code(task, language=language, context=context)
        except Exception as e:
            print("Generation failed:", e)
            return

        print(f"-> Detected language: {language}")
        print("-> Executing...")

        res = code_executor.execute(code, language=language)
        if res["returncode"] == 0:
            print("\n✅ Success! Output:\n")
            print(res["stdout"])
            return
        else:
            print("\n❌ Failed. Stderr:\n", res["stderr"])
            try:
                fixed = code_fixer.fix_code(code, res["stderr"], language=language, context=context)
            except Exception as e:
                print("Auto-fix failed:", e)
                return
            context = f"Previous attempt failed with error:\n{res['stderr']}\nPlease fix and retry."

    print("\n🚫 All attempts exhausted. Try refining your prompt.")

def main():
    print("=== NeuroForge Mini: Multi-language Self-Correcting Runtime ===")
    task = input("🧩 Describe your task (e.g., 'in c print hello world'): ").strip()
    if not task:
        print("No task provided.")
        return
    self_correcting_flow(task)

if __name__ == "__main__":
    main()
