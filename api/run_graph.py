from graph_core import run_task

if __name__ == "__main__":
    print("=== NeuroForge LangGraph Orchestrator ===")
    task = input("Describe your coding task (multi-language supported): ")
    run_task(task)
