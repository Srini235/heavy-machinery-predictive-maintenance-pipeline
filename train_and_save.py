"""Train the hydraulic predictive-maintenance models and save them for the API.

Stepwise architecture / design notes (implemented here):

1. Repository layout & separation of concerns
    - `data/` contains input CSVs; `model_registry/` holds trained artifacts and an
      `index.json` artifact registry. This module implements the training stage only.

2. Idempotent training pattern
    - Before loading large datasets or re-training, the script checks the registry
      (`model_registry/index.json`) and artifact checksums. If artifacts exist and
      checksums match, training is skipped unless `--force` is passed.

3. Pipeline / preprocessor (Pipe-and-Filter)
    - A `ColumnTransformer` + `Pipeline` separates preprocessing from model logic,
      enabling consistent transform reuse at inference time.

4. Model registry pattern
    - Artifacts are saved under `model_registry/` and registered with a sha256
      checksum via `ModelRegistry`. This enables runtime integrity checks by the API.

5. Logging & observability
    - Optional MLflow logging is available if `MLFLOW_TRACKING_URI` is configured.

6. Single responsibility and testability
    - This module exposes `main(force: bool=False)` for programmatic invocation and
      a CLI wrapper for convenience. Each step is isolated to make unit-testing
      (dataset missing, idempotency, registration) straightforward.
"""

from __future__ import annotations

import json
from pathlib import Path

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

_here = Path(__file__).resolve().parent
# Find a reasonable repository root that contains the `data` directory.
ROOT = _here
for _ in range(4):
    if (ROOT / "data" / "hydraulic_fleet_telemetry.csv").exists():
        break
    ROOT = ROOT.parent
DATA = ROOT / "data" / "hydraulic_fleet_telemetry.csv"
REGISTRY = ROOT / "model_registry"

NUMERIC_FEATURES = ["operating_hours", "pressure_mean_bar", "pressure_std_bar", "flow_mean_lpm",
                    "oil_temp_mean_c", "vibration_rms_mms", "motor_power_kw",
                    "pump_speed_mean_rpm", "cooling_efficiency_pct"]
CATEGORICAL_FEATURES = ["machine_type"]
TARGETS = ["cooler_condition", "valve_condition", "pump_leakage", "accumulator_pressure"]
RT_TARGET = "stability_flag"
NOISY_COLS = ["pressure_mean_bar", "flow_mean_lpm", "oil_temp_mean_c", "vibration_rms_mms"]


def main(force: bool = False):
    REGISTRY.mkdir(parents=True, exist_ok=True)
    # Do not require the dataset unless we actually need to retrain; check later.

    # Idempotency guard: if artifacts + registry entries already exist and checksums
    # match, skip expensive retraining and notify the user.
    condition_path = REGISTRY / "condition_model.joblib"
    stability_path = REGISTRY / "stability_model.joblib"
    schema_path = REGISTRY / "schema.json"

    registry = ModelRegistry(REGISTRY)
    up_to_date = True
    for name, path in [("condition_model.joblib", condition_path),
                       ("stability_model.joblib", stability_path),
                       ("schema.json", schema_path)]:
        if not path.exists():
            up_to_date = False
            break
        entry = registry.get(name)
        if entry is None:
            up_to_date = False
            break
        if entry.sha256 != compute_file_sha256(path):
            up_to_date = False
            break

    if up_to_date and not force:
        print("Artifacts are up-to-date according to model_registry/index.json — skipping training.")
        return

    # Load dataset only if we need to train
    if not DATA.exists():
        raise FileNotFoundError(f"{DATA} missing — run merge_notebook.py to export the dataset.")
    df = pd.read_csv(DATA)

    imputer = SimpleImputer(strategy="median")
    df[NOISY_COLS] = imputer.fit_transform(df[NOISY_COLS])

    preprocessor = ColumnTransformer(transformers=[
        ("num", StandardScaler(), NUMERIC_FEATURES),
        ("cat", OneHotEncoder(handle_unknown="ignore"), CATEGORICAL_FEATURES),
    ])

    X = df[NUMERIC_FEATURES + CATEGORICAL_FEATURES]
    y_multi = df[TARGETS]
    y_rt = df[RT_TARGET]
    X_tr, X_te, ym_tr, ym_te, yr_tr, yr_te = train_test_split(
        X, y_multi, y_rt, test_size=0.2, random_state=42, stratify=df[RT_TARGET])

    condition_model = Pipeline([
        ("prep", preprocessor),
        ("clf", MultiOutputClassifier(RandomForestClassifier(
            n_estimators=200, max_depth=12, random_state=42, n_jobs=-1)))]
    )
    condition_model.fit(X_tr, ym_tr)
    ym_pred = condition_model.predict(X_te)
    ym_pred_arr = np.asarray(ym_pred)
    cond_metrics = {t: {"accuracy": round(float(accuracy_score(ym_te[t].to_numpy(), ym_pred_arr[:, i])), 4),
                        "macro_f1": round(float(f1_score(ym_te[t].to_numpy(), ym_pred_arr[:, i], average="macro")), 4)}
                    for i, t in enumerate(TARGETS)}

    stability_model = Pipeline([
        ("prep", preprocessor),
        ("clf", RandomForestClassifier(n_estimators=60, max_depth=6, random_state=42, n_jobs=-1))])
    stability_model.fit(X_tr, yr_tr)
    rt_acc = accuracy_score(yr_te, stability_model.predict(X_te))

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
        "healthy_class": {"cooler_condition": 100, "valve_condition": 100,
                          "pump_leakage": 0, "accumulator_pressure": 130},
        "condition_metrics": cond_metrics,
        "stability_accuracy": round(float(rt_acc), 4),
    }
    schema_path.write_text(json.dumps(schema, indent=2), encoding="utf-8")

    registry = ModelRegistry(REGISTRY)
    registry.register(ModelArtifact(path="condition_model.joblib",
                                    sha256=compute_file_sha256(condition_path),
                                    metadata={"type": "condition_model"}))
    registry.register(ModelArtifact(path="stability_model.joblib",
                                    sha256=compute_file_sha256(stability_path),
                                    metadata={"type": "stability_model"}))
    registry.register(ModelArtifact(path="schema.json",
                                    sha256=compute_file_sha256(schema_path),
                                    metadata={"type": "schema"}))

    _log_mlflow_run(condition_path, stability_path, schema_path, cond_metrics, float(rt_acc))

    print("Saved condition_model.joblib, stability_model.joblib, schema.json")
    for t, m in cond_metrics.items():
        print(f"  {t:22s} acc={m['accuracy']:.3f}  macroF1={m['macro_f1']:.3f}")
    print(f"  stability_flag         acc={rt_acc:.3f}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Train and save hydraulic predictive-maintenance models")
    parser.add_argument("--force", action="store_true",
                        help="Force retraining even when artifacts are up-to-date in the registry")
    args = parser.parse_args()

    main(force=args.force)


def _log_mlflow_run(condition_path: Path, stability_path: Path,
                    schema_path: Path, cond_metrics: dict, rt_acc: float) -> None:
    mlflow_uri = os.getenv("MLFLOW_TRACKING_URI")
    if not mlflow_uri:
        return

    try:
        import mlflow
        import mlflow.sklearn
    except ImportError:
        print("MLflow is not installed; skipping MLflow logging")
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
        mlflow.log_artifact(str(condition_path), artifact_path="model_registry")
        mlflow.log_artifact(str(stability_path), artifact_path="model_registry")
        mlflow.log_artifact(str(schema_path), artifact_path="model_registry")
