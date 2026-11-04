"""Train the locked RiskForge model and record the run in MLflow.

This is the tracking entry point. It does not define a second training pipeline:
the modeling frame, the chronological split, the XGBoost configuration, the
scoring and the evaluation all come from the modules that already own them, and
:mod:`tracking.run` only adds a record of what happened.

It also does not produce anything the API serves. The Phase 6 serving artifact is
built by ``scripts/build_serving_artifact.py`` and is untouched here, so running
this script cannot change a prediction.

Usage::

    PYTHONPATH=. python scripts/train_tracked_model.py

Configuration comes from the environment (see ``tracking/config.py``):
``MLFLOW_TRACKING_URI``, ``MLFLOW_EXPERIMENT_NAME`` and
``MLFLOW_REGISTERED_MODEL_NAME``. The default is a local ``file:./mlruns`` store,
which is generated state and is gitignored.
"""

from __future__ import annotations

import time
from pathlib import Path

from tracking.config import TrackingConfig
from tracking.run import (
    DEFAULT_BEHAVIORAL_PATH,
    DEFAULT_TRANSACTION_PATH,
    run_tracked_training,
)


REPO_ROOT = Path(__file__).resolve().parents[1]

# The locked raw metrics, restated here only as guards, in the same idiom as
# scripts/build_serving_artifact.py: MLflow may record a run, but it may not
# record a run for a model that is no longer the locked one. These are raw
# XGBoost scores; the API serves isotonic-calibrated probabilities, which are
# fitted downstream and score slightly lower by design.
LOCKED_VALIDATION_PR_AUC = 0.474352
LOCKED_VALIDATION_ROC_AUC = 0.884513
LOCKED_TEST_PR_AUC = 0.449254
LOCKED_TEST_ROC_AUC = 0.865318
LOCKED_METRIC_TOLERANCE = 1e-6


def print_section(title: str) -> None:
    print("\n" + "=" * 80)
    print(title)
    print("=" * 80)


def print_header(step: str, title: str) -> None:
    print(f"\n[{step}] {title}...")


def main() -> None:
    started = time.perf_counter()

    print_section("RISKFORGE — PHASE 8 MLFLOW TRACKED TRAINING")
    print(
        "Trains the locked pipeline and records parameters, metrics, the feature "
        "contract, dataset identity and the fitted model in MLflow. It writes no "
        "serving artifact and changes no prediction."
    )

    config = TrackingConfig.from_environment()

    print_header("1/4", "Resolving the tracking configuration")
    print(f"experiment       : {config.experiment_name}")
    print(f"store location   : {config.resolved_against(REPO_ROOT).store_location}")
    print(f"registered model : {config.registered_model_name or '(registry disabled)'}")
    print(f"behavioral input : {DEFAULT_BEHAVIORAL_PATH}")
    print(f"transaction input: {DEFAULT_TRANSACTION_PATH}")

    print_header("2/4", "Training the locked model and opening the MLflow run")
    print(
        "Hashing the real input files takes a few seconds; the transaction CSV "
        "is ~683 MB and its digest is what makes the run's dataset identity real."
    )

    tracked = run_tracked_training(
        config,
        repo_root=REPO_ROOT,
    )

    print(f"run id           : {tracked.run_id}")
    print(f"run name         : {tracked.run_name}")
    print(f"tracking uri     : {tracked.tracking_uri}")
    print(f"model uri        : {tracked.model_uri}")
    print(f"file-store opt-out applied: {tracked.file_store_opt_out_applied}")

    print_header("3/4", "Checking the recorded run against the locked values")

    checks = (
        ("VALIDATION PR-AUC", tracked.validation_pr_auc, LOCKED_VALIDATION_PR_AUC),
        ("VALIDATION ROC-AUC", tracked.validation_roc_auc, LOCKED_VALIDATION_ROC_AUC),
        ("TEST PR-AUC", tracked.test_pr_auc, LOCKED_TEST_PR_AUC),
        ("TEST ROC-AUC", tracked.test_roc_auc, LOCKED_TEST_ROC_AUC),
    )

    for label, recorded, locked in checks:
        print(f"{label:18s}: recorded={recorded:.6f} locked={locked:.6f}")

        if abs(recorded - locked) > LOCKED_METRIC_TOLERANCE:
            raise AssertionError(
                f"The tracked run does not reproduce the locked {label} "
                f"({recorded:.6f} != {locked:.6f}). Refusing to record a model "
                "that is not the locked one."
            )

    print(f"commit_sha       : {tracked.commit_sha}")
    print(f"dataset identity : {tracked.dataset_fingerprint}")

    if tracked.registered_model_version is not None:
        print(
            f"registry version : {tracked.registered_model_name} "
            f"v{tracked.registered_model_version} (tagged, not promoted)"
        )

    print_header("4/4", "Done")
    print(f"model_version    : {tracked.model_version}")
    print(f"feature_version  : {tracked.feature_version}")
    print(f"policy_version   : {tracked.policy_version}")
    print(f"elapsed          : {time.perf_counter() - started:.1f}s")
    print_section("MLFLOW TRACKED TRAINING COMPLETE")
    print(
        "The run is recorded under the store location printed above; read its "
        "artifacts there. `mlflow ui` is not available because mlflow-skinny does "
        "not ship the tracking server. Nothing was promoted and the serving "
        "artifact was not modified."
    )


if __name__ == "__main__":
    main()
