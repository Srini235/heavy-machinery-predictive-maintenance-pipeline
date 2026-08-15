"""Canonical training pipeline for the hydraulic predictive-maintenance models.

Author: Group 105

This module is the single source of truth for training (the repo-root
`train_and_save.py` is a thin CLI wrapper around it — DRY).

Design notes:

1. Idempotent training — before retraining, artifact checksums are verified
   against `model_registry/index.json`; up-to-date artifacts skip training
   unless `force=True`.
2. Pipe-and-Filter — a `ColumnTransformer` + `Pipeline` separates
   preprocessing from model logic so the exact transform is reused at
   inference time.
3. Model Registry — artifacts are registered with SHA-256 checksums enabling
   runtime integrity checks by the API.
4. Profiling & observability — every stage (load, impute, fit, evaluate,
   save) is timed with `perf_counter`, peak memory is tracked with
   `tracemalloc`, and the profile is logged and persisted into
   `schema.json["training_profile"]`. `profile=True` additionally runs
   cProfile and writes the top hotspots to `model_registry/training_profile.txt`.
   Optional MLflow logging activates when `MLFLOW_TRACKING_URI` is set.
"""

from __future__ import annotations

import json
import logging
import os
import tracemalloc
from contextlib import contextmanager
from pathlib import Path
from time import perf_counter

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import train_test_split
from sklearn.multioutput import MultiOutputClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from src.model_registry import ModelArtifact, ModelRegistry
from src.security_layer import compute_file_sha256

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [TRAIN] %(levelname)s %(message)s",
)
logger = logging.getLogger(__name__)

_here = Path(__file__).resolve().parent
# Find a reasonable repository root that contains the `data` directory.
ROOT = _here
for _ in range(4):
    if (ROOT / "data" / "hydraulic_fleet_telemetry.csv").exists():
        break
    ROOT = ROOT.parent
DATA = ROOT / "data" / "hydraulic_fleet_telemetry.csv"
REGISTRY = ROOT / "model_registry"

NUMERIC_FEATURES = [
    "operating_hours",
    "pressure_mean_bar",
    "pressure_std_bar",
    "flow_mean_lpm",
    "oil_temp_mean_c",
    "vibration_rms_mms",
    "motor_power_kw",
    "pump_speed_mean_rpm",
    "cooling_efficiency_pct",
]
CATEGORICAL_FEATURES = ["machine_type"]
TARGETS = ["cooler_condition", "valve_condition", "pump_leakage", "accumulator_pressure"]
RT_TARGET = "stability_flag"
NOISY_COLS = ["pressure_mean_bar", "flow_mean_lpm", "oil_temp_mean_c", "vibration_rms_mms"]

ARTIFACT_NAMES = ["condition_model.joblib", "stability_model.joblib", "schema.json"]


class StageTimer:
    """Collects wall-clock timings per pipeline stage (profiling requirement).

    Usage:
        timer = StageTimer()
        with timer.stage("load_data"):
            df = pd.read_csv(...)
        timer.timings  ->  {"load_data": 0.41, ...}
    """

    def __init__(self) -> None:
        self.timings: dict[str, float] = {}

    @contextmanager
    def stage(self, name: str):
        start = perf_counter()
        try:
            yield
        finally:
            elapsed = perf_counter() - start
            self.timings[name] = round(elapsed, 4)
            logger.info("Stage %-18s finished in %.3fs", name, elapsed)


def build_preprocessor() -> ColumnTransformer:
    """The shared preprocessing filter (Pipe-and-Filter pattern)."""
    return ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), NUMERIC_FEATURES),
            ("cat", OneHotEncoder(handle_unknown="ignore"), CATEGORICAL_FEATURES),
        ]
    )


def build_condition_model() -> Pipeline:
    return Pipeline(
        [
            ("prep", build_preprocessor()),
            (
                "clf",
                MultiOutputClassifier(
                    RandomForestClassifier(
                        n_estimators=200, max_depth=12, random_state=42, n_jobs=-1
                    )
                ),
            ),
        ]
    )


def build_stability_model() -> Pipeline:
    return Pipeline(
        [
            ("prep", build_preprocessor()),
            (
                "clf",
                RandomForestClassifier(n_estimators=60, max_depth=6, random_state=42, n_jobs=-1),
            ),
        ]
    )


def artifacts_up_to_date(registry: ModelRegistry) -> bool:
    """Idempotency guard: True when all artifacts exist and checksums match."""
    for name in ARTIFACT_NAMES:
        path = REGISTRY / name
        if not path.exists():
            return False
        entry = registry.get(name)
        if entry is None or entry.sha256 != compute_file_sha256(path):
            return False
    return True


def _log_mlflow_run(
    condition_path: Path, stability_path: Path, schema_path: Path, cond_metrics: dict, rt_acc: float
) -> None:
    mlflow_uri = os.getenv("MLFLOW_TRACKING_URI")
    if not mlflow_uri:
        logger.info("MLflow not configured; skipping MLflow logging")
        return

    try:
        import mlflow
        import mlflow.sklearn
    except ImportError:
        logger.warning("Mlflow is not installed; skipping MLflow logging")
        return

    experiment_name = os.getenv("MLFLOW_EXPERIMENT_NAME", "hydraulics-predictive-maintenance")
    mlflow.set_tracking_uri(mlflow_uri)
    mlflow.set_experiment(experiment_name)
    with mlflow.start_run(run_name=os.getenv("MLFLOW_RUN_NAME", "train_and_save")):
        mlflow.log_param("num_numeric_features", len(NUMERIC_FEATURES))
        mlflow.log_param("num_categorical_features", len(CATEGORICAL_FEATURES))
        mlflow.log_param("num_targets", len(TARGETS))
        mlflow.log_metric("stability_accuracy", float(rt_acc))
        for target, metrics in cond_metrics.items():
            mlflow.log_metric(f"accuracy_{target}", float(metrics["accuracy"]))
            mlflow.log_metric(f"macro_f1_{target}", float(metrics["macro_f1"]))
        for path in (condition_path, stability_path, schema_path):
            mlflow.log_artifact(str(path), artifact_path="model_registry")


def _train(timer: StageTimer) -> None:
    """The actual training run: load -> clean -> fit -> evaluate -> persist."""
    with timer.stage("load_data"):
        if not DATA.exists():
            raise FileNotFoundError(
                f"{DATA} missing — run merge_notebook.py to export the dataset."
            )
        df = pd.read_csv(DATA)

    with timer.stage("impute"):
        imputer = SimpleImputer(strategy="median")
        df[NOISY_COLS] = imputer.fit_transform(df[NOISY_COLS])

    X = df[NUMERIC_FEATURES + CATEGORICAL_FEATURES]
    X_tr, X_te, ym_tr, ym_te, yr_tr, yr_te = train_test_split(
        X, df[TARGETS], df[RT_TARGET], test_size=0.2, random_state=42, stratify=df[RT_TARGET]
    )

    condition_model = build_condition_model()
    with timer.stage("fit_condition"):
        condition_model.fit(X_tr, ym_tr)

    with timer.stage("eval_condition"):
        ym_pred_arr = np.asarray(condition_model.predict(X_te))
        cond_metrics = {
            t: {
                "accuracy": round(float(accuracy_score(ym_te[t].to_numpy(), ym_pred_arr[:, i])), 4),
                "macro_f1": round(
                    float(f1_score(ym_te[t].to_numpy(), ym_pred_arr[:, i], average="macro")), 4
                ),
            }
            for i, t in enumerate(TARGETS)
        }

    stability_model = build_stability_model()
    with timer.stage("fit_stability"):
        stability_model.fit(X_tr, yr_tr)

    with timer.stage("eval_stability"):
        rt_acc = accuracy_score(yr_te, stability_model.predict(X_te))

    with timer.stage("save_artifacts"):
        condition_path = REGISTRY / "condition_model.joblib"
        stability_path = REGISTRY / "stability_model.joblib"
        schema_path = REGISTRY / "schema.json"

        joblib.dump(condition_model, condition_path)
        joblib.dump(stability_model, stability_path)

        schema = {
            "numeric_features": NUMERIC_FEATURES,
            "categorical_features": CATEGORICAL_FEATURES,
            "machine_types": sorted(df["machine_type"].unique().tolist()),
            "targets": TARGETS,
            "rt_target": RT_TARGET,
            "healthy_class": {
                "cooler_condition": 100,
                "valve_condition": 100,
                "pump_leakage": 0,
                "accumulator_pressure": 130,
            },
            "condition_metrics": cond_metrics,
            "stability_accuracy": round(float(rt_acc), 4),
            # timings so far; save_artifacts itself finishes after this write
            "training_profile": dict(timer.timings),
        }
        schema_path.write_text(json.dumps(schema, indent=2), encoding="utf-8")

        registry = ModelRegistry(REGISTRY)
        artifact_types = {
            "condition_model.joblib": "condition_model",
            "stability_model.joblib": "stability_model",
            "schema.json": "schema",
        }
        for name, kind in artifact_types.items():
            registry.register(
                ModelArtifact(
                    path=name,
                    sha256=compute_file_sha256(REGISTRY / name),
                    metadata={"type": kind},
                )
            )

    _log_mlflow_run(condition_path, stability_path, schema_path, cond_metrics, float(rt_acc))

    logger.info("Saved %s", ", ".join(ARTIFACT_NAMES))
    for t, m in cond_metrics.items():
        logger.info("%s acc=%.3f macroF1=%.3f", t, m["accuracy"], m["macro_f1"])
    logger.info("stability_flag acc=%.3f", rt_acc)


def main(force: bool = False, profile: bool = False) -> None:
    """Train and persist both models.

    Args:
        force:   retrain even when registry checksums say artifacts are current.
        profile: additionally run cProfile and write the top-25 cumulative-time
                 hotspots to model_registry/training_profile.txt.
    """
    logger.info("Train run started; force=%s profile=%s", force, profile)
    REGISTRY.mkdir(parents=True, exist_ok=True)

    if artifacts_up_to_date(ModelRegistry(REGISTRY)) and not force:
        logger.info(
            "Artifacts are up-to-date according to model_registry/index.json — skipping training."
        )
        return

    timer = StageTimer()
    tracemalloc.start()
    total_start = perf_counter()

    if profile:
        import cProfile
        import io
        import pstats

        profiler = cProfile.Profile()
        profiler.enable()
        _train(timer)
        profiler.disable()
        buffer = io.StringIO()
        pstats.Stats(profiler, stream=buffer).sort_stats("cumulative").print_stats(25)
        profile_path = REGISTRY / "training_profile.txt"
        profile_path.write_text(buffer.getvalue(), encoding="utf-8")
        logger.info("cProfile hotspots written to %s", profile_path)
    else:
        _train(timer)

    total = perf_counter() - total_start
    _, peak_bytes = tracemalloc.get_traced_memory()
    tracemalloc.stop()

    logger.info(
        "Training complete in %.2fs (peak traced memory %.1f MB); stage timings: %s",
        total,
        peak_bytes / 1e6,
        timer.timings,
    )
