"""Build the frozen serving artifact the Phase 6 API loads.

This is the only place the serving bundle is produced, and it reproduces the
locked pipeline exactly rather than defining a second one: the same 28-feature
contract, the same chronological split, the same XGBoost configuration fitted on
TRAIN only, and the same isotonic calibration fitted on VALIDATION.

The locked TEST partition is never scored, calibrated or read here.
"""

from __future__ import annotations

import time
from pathlib import Path

import pandas as pd

from explain.feature_attribution import build_feature_attribution_map
from ml.data.modeling import (
    CATEGORICAL_FEATURES,
    CD_FEATURES,
    MODEL_FEATURES,
    TARGET_COLUMN,
    build_modeling_frame,
)
from ml.models.artifact import (
    ARTIFACT_FILENAME,
    FEATURE_VERSION,
    MODEL_VERSION,
    RiskForgeArtifact,
    current_library_versions,
    save_artifact,
    utc_now_iso,
)
from ml.models.calibration import apply_calibrator, fit_calibrator
from ml.models.xgboost_model import (
    CLASSIFIER_STEP_NAME,
    PREPROCESSOR_STEP_NAME,
    fit_xgboost_model,
    predict_fraud_probability,
)
from ml.split.temporal import temporal_split
from policy.locked import build_locked_policy


BEHAVIORAL_PATH = Path("data/processed/train_behavioral_v1.parquet")
TRANSACTION_PATH = Path("data/raw/ieee_cis/train_transaction.csv")
ARTIFACT_PATH = Path("artifacts/serving") / ARTIFACT_FILENAME

TRANSACTION_COLUMNS = [
    "TransactionID",
    TARGET_COLUMN,
    *CATEGORICAL_FEATURES,
    *CD_FEATURES,
]

# The Phase 3 locked configuration, restated here only as a guard: if a future
# edit to the model changed the fitted imbalance ratio, this script must refuse
# to publish an artifact that is no longer the locked one.
LOCKED_CALIBRATION_METHOD = "isotonic"
LOCKED_SCALE_POS_WEIGHT = 27.410063
SCALE_POS_WEIGHT_TOLERANCE = 1e-6


def print_section(title: str) -> None:
    print("\n" + "=" * 80)
    print(title)
    print("=" * 80)


def print_header(step: str, title: str) -> None:
    print(f"\n[{step}] {title}...")


def main() -> None:
    started = time.perf_counter()

    print_section("RISKFORGE — PHASE 6 SERVING ARTIFACT BUILD")
    print(
        "Reproduces the locked pipeline and writes it to disk. Nothing here "
        "tunes, re-selects or re-derives a Phase 3/4 result; the guards below "
        "refuse to publish an artifact that is not the locked one."
    )

    print_header("1/6", "Loading the locked data artifacts")

    behavioral = pd.read_parquet(BEHAVIORAL_PATH)
    transactions = pd.read_csv(TRANSACTION_PATH, usecols=TRANSACTION_COLUMNS)

    modeling = build_modeling_frame(
        behavioral=behavioral,
        transaction_features=transactions,
        labels=transactions,
    )
    split = temporal_split(modeling.data)

    print(f"Modeling rows   : {len(modeling.data):,}")
    print(f"Model features  : {len(modeling.feature_columns)} (locked contract)")
    print(f"TRAIN rows      : {len(split.train):,}")
    print(f"VALIDATION rows : {len(split.validation):,}")
    print("TEST partition  : never scored, calibrated or read by this script.")

    print_header("2/6", "Fitting the locked XGBoost model on TRAIN only")

    model = fit_xgboost_model(split.train)

    print(f"scale_pos_weight: {model.scale_pos_weight:.6f}")
    print(f"locked value    : {LOCKED_SCALE_POS_WEIGHT:.6f}")

    if abs(model.scale_pos_weight - LOCKED_SCALE_POS_WEIGHT) > SCALE_POS_WEIGHT_TOLERANCE:
        raise AssertionError(
            "The fitted imbalance ratio is not the locked one "
            f"({model.scale_pos_weight:.6f} != {LOCKED_SCALE_POS_WEIGHT:.6f}). "
            "Refusing to publish an artifact for a changed model."
        )

    print_header("3/6", "Fitting the locked isotonic calibration on VALIDATION")

    raw_validation = predict_fraud_probability(model, split.validation)
    calibrator = fit_calibrator(
        LOCKED_CALIBRATION_METHOD,
        raw_validation,
        split.validation[TARGET_COLUMN],
    )
    calibrated = apply_calibrator(calibrator, raw_validation)

    print(f"Method          : {LOCKED_CALIBRATION_METHOD}")
    print(f"Raw mean        : {raw_validation.mean():.6f}")
    print(f"Calibrated mean : {calibrated.mean():.6f}")

    print_header("4/6", "Cross-checking the preprocessing width")

    pipeline = model.model
    classifier = pipeline.named_steps[CLASSIFIER_STEP_NAME]
    attribution = build_feature_attribution_map(
        pipeline.named_steps[PREPROCESSOR_STEP_NAME]
    )

    print(f"Transformed columns : {attribution.n_transformed_columns}")
    print(f"Classifier width    : {classifier.n_features_in_}")
    print(f"Contract features   : {len(attribution.contract_features)}")

    if attribution.n_transformed_columns != classifier.n_features_in_:
        raise AssertionError(
            "The attribution map does not span the model's actual input width."
        )

    if list(attribution.contract_features) != list(MODEL_FEATURES):
        raise AssertionError("The attribution map does not cover the locked contract.")

    print_header("5/6", "Assembling the artifact")

    policy = build_locked_policy()

    artifact = RiskForgeArtifact(
        pipeline=pipeline,
        calibrator=calibrator,
        model_version=MODEL_VERSION,
        feature_version=FEATURE_VERSION,
        policy_version=policy.config.policy_version,
        calibration_method=LOCKED_CALIBRATION_METHOD,
        review_threshold=float(policy.config.review_threshold),
        block_threshold=float(policy.config.block_threshold),
        scale_pos_weight=float(model.scale_pos_weight),
        n_transformed_columns=int(attribution.n_transformed_columns),
        feature_columns=tuple(MODEL_FEATURES),
        n_train_rows=int(len(split.train)),
        n_calibration_rows=int(len(split.validation)),
        library_versions=current_library_versions(),
        built_at_utc=utc_now_iso(),
    )

    print(f"model_version   : {artifact.model_version}")
    print(f"feature_version : {artifact.feature_version}")
    print(f"policy_version  : {artifact.policy_version}")
    print(f"T_REVIEW        : {artifact.review_threshold}")
    print(f"T_BLOCK         : {artifact.block_threshold}")

    print_header("6/6", "Writing the artifact")

    destination = save_artifact(artifact, ARTIFACT_PATH)

    print(f"Written to      : {destination}")
    print(f"Size            : {destination.stat().st_size / 1_048_576:.2f} MiB")

    print_section("DISCIPLINE")
    print("1. The TEST partition was never scored, calibrated or read.")
    print("2. Preprocessing and XGBoost were fitted on TRAIN only.")
    print("3. Calibration is isotonic fitted on VALIDATION, as locked in Phase 3.")
    print("4. Policy thresholds come from policy.locked, the single frozen definition.")
    print("5. The XGBoost configuration was not modified; the scale_pos_weight")
    print("   guard above proves the fitted model is still the locked one.")
    print("6. artifacts/ is gitignored, so no model binary enters version control.")

    print(f"\nTotal seconds: {time.perf_counter() - started:.2f}")

    print_section("PHASE 6 SERVING ARTIFACT BUILD COMPLETE")


if __name__ == "__main__":
    main()
