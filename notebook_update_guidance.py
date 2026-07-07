"""Notebook guidance for 105.ipynb updates.

This file contains the exact notebook text and code adjustments needed to
keep the notebook aligned with the current repo layout and root-based runtime
behavior.
"""

# Replace any hardcoded repository-folder references with generic root-based
# execution guidance. For example, replace:
#
#   cd heavy-machinery-predictive-maintenance-pipeline
#
# with:
#
#   python train_and_save.py
#   python -m uvicorn src.api.server:app --reload --port 8000
#
# The codebase now resolves `src` from the repository root and does not depend
# on the folder name.

NOTEBOOK_UPDATE_GUIDANCE = {
    "folder_independence": (
        "The project code is folder-name independent. The only requirement is "
        "that the repository root is on Python's import path when running "
        "commands like `python train_and_save.py` or `python -m uvicorn src.api.server:app --reload --port 8000`."
    ),
    "run_commands": (
        "Run from the repository root, for example:\n"
        "```bash\n"
        "python train_and_save.py\n"
        "python -m uvicorn src.api.server:app --reload --port 8000\n"
        "```"
    ),
    "path_setup": (
        "from pathlib import Path\n"
        "import sys\n\n"
        "REPO_ROOT = Path('.').resolve()\n"
        "sys.path.insert(0, str(REPO_ROOT))\n\n"
        "from src.model_registry import ModelRegistry\n"
    ),
}

if __name__ == "__main__":
    for key, value in NOTEBOOK_UPDATE_GUIDANCE.items():
        print(f"--- {key} ---")
        print(value)
        print()
