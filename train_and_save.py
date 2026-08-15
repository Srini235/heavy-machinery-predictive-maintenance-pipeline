"""Train the hydraulic predictive-maintenance models and save them for the API.

Thin CLI wrapper — the implementation lives in `src/ml/train.py` (the
production package), keeping a single source of truth for training (DRY).
This entry point exists because Docker, DVC (`dvc.yaml`) and CI all invoke
`python train_and_save.py` from the repository root.

Run:
    python train_and_save.py             # idempotent: skips if artifacts current
    python train_and_save.py --force     # retrain regardless
    python train_and_save.py --profile   # also write cProfile hotspots report
"""

from src.ml.train import main

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Train and save hydraulic predictive-maintenance models"
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Force retraining even when artifacts are up-to-date in the registry",
    )
    parser.add_argument(
        "--profile",
        action="store_true",
        help="Run cProfile during training and write model_registry/training_profile.txt",
    )
    args = parser.parse_args()

    main(force=args.force, profile=args.profile)
