from __future__ import annotations

import time
from pathlib import Path

import pandas as pd

from explain.explanation import explain_transaction
from explain.model_explanation import (
    ATTRIBUTION_SPACE_NOTE,
    DEFAULT_TOP_K,
    FLOAT32_ADDITIVITY_TOLERANCE,
    build_model_explainer,
)
from ml.data.modeling import (
    CATEGORICAL_FEATURES,
    CD_FEATURES,
    TARGET_COLUMN,
    build_modeling_frame,
)
from ml.models.calibration import apply_calibrator, fit_calibrator
from ml.models.xgboost_model import fit_xgboost_model, predict_fraud_probability
from ml.split.temporal import temporal_split
from policy.decision import Decision

from policy.locked import (
    LOCKED_BLOCK_THRESHOLD,
    LOCKED_REVIEW_THRESHOLD,
    build_locked_policy,
)


BEHAVIORAL_PATH = Path("data/processed/train_behavioral_v1.parquet")
TRANSACTION_PATH = Path("data/raw/ieee_cis/train_transaction.csv")

TRANSACTION_COLUMNS = [
    "TransactionID",
    TARGET_COLUMN,
    *CATEGORICAL_FEATURES,
    *CD_FEATURES,
]

# The frozen Phase 3 deployment mapping and the frozen Phase 4 policy. Neither
# is re-derived here; both are read from the locked artifacts.
LOCKED_CALIBRATION_METHOD = "isotonic"

SELECTION_NOTE = (
    "The three transactions below are chosen from VALIDATION using only their "
    "calibrated risk probabilities and the frozen policy thresholds. No label "
    "is read to pick, rank or tune them. The TEST partition is never scored or "
    "explained; only its row count is reported, to make the omission explicit."
)


def print_section(title: str) -> None:
    print("\n" + "=" * 80)
    print(title)
    print("=" * 80)


def print_header(step: str, title: str) -> None:
    print(f"\n[{step}] {title}...")




def select_examples(
    calibrated: pd.Series,
    decisions: pd.Series,
) -> list[tuple[str, int]]:
    """Pick one ALLOW, one REVIEW and one BLOCK row from probability alone.

    Deterministic by construction: the lowest ALLOW, the REVIEW closest to the
    centre of its band, and the highest BLOCK. Labels play no part. The
    returned integers are index labels, not positions, because ``calibrated``
    keeps the VALIDATION frame's own non-contiguous index.
    """
    allowed = calibrated[decisions == Decision.ALLOW]
    reviewed = calibrated[decisions == Decision.REVIEW]
    blocked = calibrated[decisions == Decision.BLOCK]

    if allowed.empty or reviewed.empty or blocked.empty:
        raise ValueError(
            "VALIDATION does not contain at least one row in each decision "
            "band, so the smoke test cannot demonstrate all three."
        )

    band_centre = (LOCKED_REVIEW_THRESHOLD + LOCKED_BLOCK_THRESHOLD) / 2.0
    central = (reviewed - band_centre).abs()

    return [
        ("LOWEST-RISK ALLOW", int(allowed.idxmin())),
        ("MID-BAND REVIEW", int(central.idxmin())),
        ("HIGHEST-RISK BLOCK", int(blocked.idxmax())),
    ]


def print_explanation(label: str, explained, position: int) -> None:
    model = explained.model_explanation
    decision = explained.decision_explanation

    residual = abs(
        model.base_value
        + sum(entry.contribution for entry in model.contributions)
        - model.raw_margin
    )

    if residual > FLOAT32_ADDITIVITY_TOLERANCE:
        raise AssertionError(
            f"{label}: the TreeSHAP decomposition does not close against the "
            f"model's own raw margin (residual {residual:.3e})."
        )

    print("\n" + "-" * 80)
    print(f"{position}. {label}")
    print("-" * 80)
    print(f"transaction_id             : {explained.transaction_id}")
    print(f"risk_probability           : "
          f"{explained.calibrated_risk_probability:.6f}  (calibrated)")
    print(f"raw_model_probability      : {model.raw_model_probability:.6f}  "
          f"(XGBoost, pre-calibration)")
    print(f"raw_margin                 : {model.raw_margin:+.6f}  (log-odds)")
    print(f"base_value                 : {model.base_value:+.6f}")
    print(f"additivity_residual        : {residual:.3e}  "
          f"(tolerance {FLOAT32_ADDITIVITY_TOLERANCE:.0e})")
    print(f"decision                   : {explained.decision}")
    print(f"decision_explanation       : {decision.explanation}")
    print(f"decision_reason_code       : {decision.reason_code}")
    print(f"top reason codes (k={model.top_k})    : "
          f"{', '.join(model.top_reason_codes) or '(none)'}")

    print("\n  risk-increasing model signals (contributions in log-odds):")

    if not model.risk_increasing:
        print("    (no feature pushed the model score upward)")

    for entry in model.risk_increasing:
        print(
            f"    {entry.reason_code:38s} {entry.contribution:+.6f}  "
            f"{entry.feature:22s} cols={entry.n_transformed_columns}"
        )
        print(f"      {entry.reason_title}")

    print("\n  risk-decreasing model signals (reported, never as fraud evidence):")

    if not model.risk_decreasing:
        print("    (no feature pushed the model score downward)")

    for entry in model.risk_decreasing:
        print(
            f"    {entry.reason_code:38s} {entry.contribution:+.6f}  "
            f"{entry.feature:22s} cols={entry.n_transformed_columns}"
        )

    print(f"\n  total risk-increasing     : {model.total_risk_increasing:+.6f}")
    print(f"  total risk-decreasing     : {model.total_risk_decreasing:+.6f}")


def main() -> None:
    started = time.perf_counter()

    print_section("RISKFORGE — PHASE 5 EXPLAINABILITY SMOKE TEST")
    print(
        "This script demonstrates the explanation layer end to end on real "
        "IEEE-CIS rows. It does not train, tune, calibrate or threshold "
        "anything: it reuses the locked pipeline and the frozen policy-001."
    )
    print(f"\n{SELECTION_NOTE}")

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
    print(f"TEST rows       : {len(split.test):,}  (never scored by this script)")

    print_header("2/6", "Fitting the locked XGBoost model (unmodified)")

    model = fit_xgboost_model(split.train)

    print(f"scale_pos_weight: {model.scale_pos_weight:.6f} (locked: 27.410063)")

    print_header("3/6", "Applying the locked isotonic calibration to VALIDATION")

    raw_validation = predict_fraud_probability(model, split.validation)
    calibrator = fit_calibrator(
        LOCKED_CALIBRATION_METHOD,
        raw_validation,
        split.validation[TARGET_COLUMN],
    )
    calibrated = apply_calibrator(calibrator, raw_validation)

    print(f"Method          : {LOCKED_CALIBRATION_METHOD} (refit on all VALIDATION)")
    print(f"Raw mean        : {raw_validation.mean():.6f}")
    print(f"Calibrated mean : {calibrated.mean():.6f}")

    print_header("4/6", "Building the explainer for the fitted model")

    explainer = build_model_explainer(model.model)
    attribution = explainer.attribution

    print(f"Transformed columns : {attribution.n_transformed_columns}")
    print(f"Contract features   : {len(attribution.contract_features)}")
    print(f"Classifier width    : {explainer.classifier.n_features_in_}")

    if attribution.n_transformed_columns != explainer.classifier.n_features_in_:
        raise AssertionError(
            "The attribution map does not span the model's actual input width."
        )

    print(f"\n{ATTRIBUTION_SPACE_NOTE}")

    print_header("5/6", "Selecting one ALLOW, one REVIEW and one BLOCK row")

    policy = build_locked_policy()
    decisions = policy.decide_many(calibrated)

    for decision in (Decision.ALLOW, Decision.REVIEW, Decision.BLOCK):
        count = int((decisions == decision).sum())
        print(f"{decision.value:8s}: {count:7,}  ({count / len(decisions):.4%})")

    print(f"\nPolicy {policy.config.policy_version}: "
          f"T_REVIEW={policy.config.review_threshold}, "
          f"T_BLOCK={policy.config.block_threshold}")
    print("\nSelection inputs: calibrated risk probability only.")

    examples = select_examples(calibrated, decisions)

    print_header("6/6", f"Explaining each transaction (top_k={DEFAULT_TOP_K})")

    validation = split.validation

    for position, (label, row_label) in enumerate(examples, start=1):
        transaction = validation.loc[[row_label]]

        explained = explain_transaction(
            explainer,
            transaction,
            calibrated_risk_probability=float(calibrated.loc[row_label]),
            policy=policy,
            top_k=DEFAULT_TOP_K,
        )

        if explained.transaction_id != transaction["TransactionID"].iloc[0]:
            raise AssertionError("The explanation reported the wrong transaction.")

        print_explanation(label, explained, position)

    if (
        policy.config.review_threshold != LOCKED_REVIEW_THRESHOLD
        or policy.config.block_threshold != LOCKED_BLOCK_THRESHOLD
    ):
        raise AssertionError("The frozen policy thresholds were modified.")

    print_section("DISCIPLINE")
    print("1. The TEST partition was loaded for its row count only and was "
          "never scored, explained or labelled by this script.")
    print("2. The three examples were selected from calibrated risk "
          "probabilities and the frozen thresholds alone; no label was read.")
    print("3. Contributions come from TreeSHAP on the same fitted booster that "
          "produces the production score — no surrogate, no second model.")
    print("4. Reason codes are deterministic strings from a static registry. No "
          "LLM, no template generation, no external call.")
    print("5. Model signals and the decision rule are reported separately; the "
          "strongest model signal does not by itself explain the band.")
    print("6. The XGBoost configuration, preprocessing, calibration and policy "
          "thresholds were not modified by this phase.")

    print(f"\nTotal seconds: {time.perf_counter() - started:.2f}")

    print_section("PHASE 5 EXPLAINABILITY SMOKE TEST COMPLETE")


if __name__ == "__main__":
    main()
