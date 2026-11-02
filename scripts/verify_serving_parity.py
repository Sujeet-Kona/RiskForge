"""Prove the API and the offline pipeline are the same computation.

For a real VALIDATION transaction this script scores it twice:

* **offline** — the behavioral features already materialised in the locked
  artifact by replaying all 590,540 transactions in chronological order, then
  the frozen pipeline, calibration, policy and Phase 5 explanation;
* **API** — the same transaction posted to ``/v1/risk/score`` with only the
  prior 24 hours of its own two entities as historical context, which makes the
  service regenerate those features through
  :func:`ml.features.behavioral.build_behavioral_features`.

If online feature generation had drifted from the offline contract, the two
results would diverge. They are compared feature by feature and then end to end.

The locked TEST partition is never scored, and no label is read to choose, rank
or tune anything.
"""

from __future__ import annotations

import math
import time
from pathlib import Path

import pandas as pd
from fastapi.testclient import TestClient

from app.config import DEFAULT_ARTIFACT_PATH, MAX_HISTORY_TRANSACTIONS
from app.features import build_serving_features
from app.main import create_app
from app.schemas import HistoryTransaction, RiskScoreRequest, TransactionPayload
from explain.explanation import explain_transaction
from explain.model_explanation import DEFAULT_TOP_K, build_model_explainer
from ml.data.modeling import (
    BEHAVIORAL_FEATURES,
    CATEGORICAL_FEATURES,
    CD_FEATURES,
    MODEL_FEATURES,
    TARGET_COLUMN,
    build_modeling_frame,
)
from ml.features.behavioral import ONE_DAY
from ml.models.artifact import load_artifact
from ml.models.calibration import apply_calibrator
from ml.models.xgboost_model import predict_fraud_probability
from ml.split.temporal import temporal_split
from policy.decision import Decision
from policy.locked import build_locked_policy


BEHAVIORAL_PATH = Path("data/processed/train_behavioral_v1.parquet")
TRANSACTION_PATH = Path("data/raw/ieee_cis/train_transaction.csv")

TRANSACTION_COLUMNS = [
    "TransactionID",
    TARGET_COLUMN,
    *CATEGORICAL_FEATURES,
    *CD_FEATURES,
]

HISTORY_ENTITY_COLUMNS = [
    "TransactionID",
    "TransactionDT",
    "TransactionAmt",
    "card1",
    "addr1",
    "addr2",
]

# Counts and flags are integers and booleans and must match exactly. The two
# amount sums are floating-point accumulations: offline they are built by
# replaying 590,540 rows with running additions and subtractions, online by
# summing only the trailing window. Addition is not associative, so the last
# bits may legitimately differ. The tolerance is far below any split threshold
# the model could sit on.
EXACT_MATCH_FEATURES = [
    feature
    for feature in MODEL_FEATURES
    if feature not in ("card_amount_sum_24h", "address_amount_sum_24h")
]
TOLERANT_MATCH_FEATURES = ["card_amount_sum_24h", "address_amount_sum_24h"]
AMOUNT_SUM_ABSOLUTE_TOLERANCE = 1e-9
AMOUNT_SUM_RELATIVE_TOLERANCE = 1e-12
# Populated per comparison so the run reports the delta it actually measured
# rather than only asserting it stayed inside the bound.
AMOUNT_SUM_DELTAS: list[tuple[str, float]] = []

# The locked Phase 3 calibrated VALIDATION statistics, reproduced by the
# published artifact. If these move, the artifact is not the locked model.
LOCKED_CALIBRATED_VALIDATION_MEAN = 0.034199
LOCKED_VALIDATION_BAND_COUNTS = {
    Decision.ALLOW: 81_584,
    Decision.REVIEW: 2_235,
    Decision.BLOCK: 4_194,
}
LOCKED_SCALE_POS_WEIGHT = 27.410063


def print_section(title: str) -> None:
    print("\n" + "=" * 80)
    print(title)
    print("=" * 80)


def print_header(step: str, title: str) -> None:
    print(f"\n[{step}] {title}...")


def _json_value(value):
    """A pandas/numpy scalar as a JSON-native value, with gaps as None."""
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return None

    if pd.isna(value):
        return None

    if isinstance(value, (bool,)):
        return value

    if hasattr(value, "item"):
        return value.item()

    return value


def build_transaction_payload(row: pd.Series, entities: pd.Series) -> TransactionPayload:
    """The wire payload for one real transaction, gaps preserved as None."""
    values = {
        "TransactionID": str(int(row["TransactionID"])),
        "TransactionDT": int(entities["TransactionDT"]),
        "TransactionAmt": float(entities["TransactionAmt"]),
        "card1": int(entities["card1"]),
        "addr1": _json_value(entities["addr1"]),
        "addr2": _json_value(entities["addr2"]),
    }

    for feature in (*CATEGORICAL_FEATURES, *CD_FEATURES):
        values[feature] = _json_value(row[feature])

    return TransactionPayload.model_validate(values)


def build_history_payload(history: pd.DataFrame) -> list[HistoryTransaction]:
    """The prior-24h rows of this transaction's own card and address entities."""
    return [
        HistoryTransaction.model_validate(
            {
                "TransactionDT": int(entry["TransactionDT"]),
                "TransactionAmt": float(entry["TransactionAmt"]),
                "card1": _json_value(entry["card1"]),
                "addr1": _json_value(entry["addr1"]),
                "addr2": _json_value(entry["addr2"]),
            }
        )
        for _, entry in history.iterrows()
    ]


def select_prior_history(entities_by_id: pd.DataFrame, current: pd.Series) -> pd.DataFrame:
    """Every transaction that may legitimately influence this one.

    The window is open at both ends: ``TransactionDT`` strictly greater than
    ``current - 24h`` and strictly less than ``current``. That is exactly the
    range the offline rolling computation retains, so this is the same set of
    rows rather than an approximation of it.
    """
    timestamp = int(current["TransactionDT"])
    card = current["card1"]
    addr1 = current["addr1"]
    addr2 = current["addr2"]

    window = entities_by_id[
        (entities_by_id["TransactionDT"] > timestamp - ONE_DAY)
        & (entities_by_id["TransactionDT"] < timestamp)
    ]

    matches_card = window["card1"] == card

    if pd.isna(addr1) or pd.isna(addr2):
        matches_address = pd.Series(False, index=window.index)
    else:
        matches_address = (window["addr1"] == addr1) & (window["addr2"] == addr2)

    return window[matches_card | matches_address]


def compare_features(offline: pd.Series, online: pd.DataFrame) -> list[str]:
    """Report every feature where the two paths disagree."""
    disagreements: list[str] = []

    row = online.iloc[0]

    for feature in EXACT_MATCH_FEATURES:
        expected = offline[feature]
        actual = row[feature]

        expected_missing = bool(pd.isna(expected))
        actual_missing = bool(pd.isna(actual))

        if expected_missing or actual_missing:
            # A gap is a value in this contract: the preprocessor imputes it and
            # raises a missing indicator. Both paths have to agree that it is
            # missing, which ``==`` alone cannot express because NaN != NaN.
            agreed = expected_missing == actual_missing
        elif isinstance(expected, (bool,)) or isinstance(actual, (bool,)):
            agreed = bool(expected) == bool(actual)
        else:
            agreed = expected == actual

        if not agreed:
            disagreements.append(
                f"{feature}: offline={expected!r} online={actual!r}"
            )

    for feature in TOLERANT_MATCH_FEATURES:
        expected = float(offline[feature])
        actual = float(row[feature])
        delta = abs(actual - expected)

        AMOUNT_SUM_DELTAS.append((feature, delta))

        if not math.isclose(
            actual,
            expected,
            rel_tol=AMOUNT_SUM_RELATIVE_TOLERANCE,
            abs_tol=AMOUNT_SUM_ABSOLUTE_TOLERANCE,
        ):
            disagreements.append(
                f"{feature}: offline={expected!r} online={actual!r} "
                f"(delta={actual - expected:.3e})"
            )

    return disagreements


def select_examples(
    calibrated: pd.Series,
    decisions: pd.Series,
) -> list[tuple[str, int]]:
    """One ALLOW, one REVIEW and one BLOCK row, chosen from probability alone.

    The lowest ALLOW, the REVIEW closest to the centre of its band and the
    highest BLOCK. No label is read. Returned integers are index labels, not
    positions, because VALIDATION keeps a non-contiguous index.
    """
    allowed = calibrated[decisions == Decision.ALLOW]
    reviewed = calibrated[decisions == Decision.REVIEW]
    blocked = calibrated[decisions == Decision.BLOCK]

    if allowed.empty or reviewed.empty or blocked.empty:
        raise ValueError(
            "VALIDATION does not contain at least one row in each decision band."
        )

    policy = build_locked_policy()
    centre = (
        policy.config.review_threshold + policy.config.block_threshold
    ) / 2.0

    return [
        ("LOWEST-RISK ALLOW", int(allowed.idxmin())),
        ("MID-BAND REVIEW", int((reviewed - centre).abs().idxmin())),
        ("HIGHEST-RISK BLOCK", int(blocked.idxmax())),
    ]


def main() -> None:
    started = time.perf_counter()

    print_section("RISKFORGE — PHASE 6 OFFLINE vs API PARITY")
    print(
        "Scores real VALIDATION transactions twice — once through the offline "
        "pipeline's materialised features, once through the HTTP API with "
        "caller-supplied prior history — and requires the two to agree."
    )

    print_header("1/6", "Loading the published serving artifact")

    artifact = load_artifact(DEFAULT_ARTIFACT_PATH)
    policy = build_locked_policy()
    explainer = build_model_explainer(artifact.pipeline)

    print(f"model_version   : {artifact.model_version}")
    print(f"feature_version : {artifact.feature_version}")
    print(f"policy_version  : {artifact.policy_version}")
    print(f"built_at_utc    : {artifact.built_at_utc}")

    if abs(artifact.scale_pos_weight - LOCKED_SCALE_POS_WEIGHT) > 1e-6:
        raise AssertionError("The artifact is not the locked XGBoost model.")

    print_header("2/6", "Loading the real data and rebuilding the locked split")

    behavioral = pd.read_parquet(BEHAVIORAL_PATH)
    transactions = pd.read_csv(TRANSACTION_PATH, usecols=TRANSACTION_COLUMNS)

    modeling = build_modeling_frame(
        behavioral=behavioral,
        transaction_features=transactions,
        labels=transactions,
    )
    split = temporal_split(modeling.data)
    validation = split.validation

    entities_by_id = behavioral.set_index("TransactionID")[
        HISTORY_ENTITY_COLUMNS[1:]
    ]

    print(f"Modeling rows   : {len(modeling.data):,}")
    print(f"VALIDATION rows : {len(validation):,}")
    print(f"TEST rows       : {len(split.test):,}  (never scored by this script)")

    print_header("3/6", "Checking the artifact reproduces the locked statistics")

    raw = predict_fraud_probability(artifact_pipeline_model(artifact), validation)
    calibrated = apply_calibrator(artifact.calibrator, raw)
    decisions = policy.decide_many(calibrated)

    print(f"scale_pos_weight: {artifact.scale_pos_weight:.6f}")
    print(f"Calibrated mean : {calibrated.mean():.6f} "
          f"(locked: {LOCKED_CALIBRATED_VALIDATION_MEAN:.6f})")

    if abs(calibrated.mean() - LOCKED_CALIBRATED_VALIDATION_MEAN) > 5e-7:
        raise AssertionError(
            "The artifact does not reproduce the locked calibrated VALIDATION mean."
        )

    for decision, expected in LOCKED_VALIDATION_BAND_COUNTS.items():
        actual = int((decisions == decision).sum())
        print(f"{decision.value:8s}: {actual:7,}  (locked: {expected:,})")

        if actual != expected:
            raise AssertionError(
                f"The artifact does not reproduce the locked {decision.value} count."
            )

    print_header("4/6", "Selecting one ALLOW, one REVIEW and one BLOCK row")

    examples = select_examples(calibrated, decisions)

    print("Selection inputs: calibrated risk probability only. No label was read.")
    for label, row_label in examples:
        print(f"  {label:20s} index_label={row_label} "
              f"p={calibrated.loc[row_label]:.6f} "
              f"decision={decisions.loc[row_label].value}")

    print_header("5/6", "Comparing the offline path with the API path")

    application = create_app()
    failures: list[str] = []

    with TestClient(application) as client:
        health = client.get("/health")

        if health.status_code != 200 or health.json()["status"] != "ok":
            raise AssertionError(f"/health did not report ok: {health.status_code}")

        for position, (label, row_label) in enumerate(examples, start=1):
            row = validation.loc[row_label]
            transaction_id = int(row["TransactionID"])
            entities = entities_by_id.loc[transaction_id]

            history = select_prior_history(entities_by_id, entities)

            payload = build_transaction_payload(row, entities)
            history_payload = build_history_payload(history)

            request = RiskScoreRequest(
                transaction=payload,
                history=history_payload,
                top_k=DEFAULT_TOP_K,
            )

            print("\n" + "-" * 80)
            print(f"{position}. {label}  (TransactionID {transaction_id})")
            print("-" * 80)
            print(f"history rows supplied : {len(history):,} "
                  f"(bound {MAX_HISTORY_TRANSACTIONS:,})")

            if len(history) > MAX_HISTORY_TRANSACTIONS:
                failures.append(
                    f"{label}: real history of {len(history)} rows exceeds the "
                    f"API bound of {MAX_HISTORY_TRANSACTIONS}"
                )

            offline_features = row[MODEL_FEATURES]
            online_features = build_serving_features(payload, history_payload)

            disagreements = compare_features(offline_features, online_features)

            print(f"features compared     : {len(MODEL_FEATURES)}")
            print(f"feature mismatches    : {len(disagreements)}")

            for disagreement in disagreements:
                print(f"    MISMATCH {disagreement}")

            failures.extend(f"{label}: {item}" for item in disagreements)

            offline_probability = float(calibrated.loc[row_label])
            offline_explained = explain_transaction(
                explainer,
                validation.loc[[row_label]],
                calibrated_risk_probability=offline_probability,
                policy=policy,
                top_k=DEFAULT_TOP_K,
            )

            response = client.post("/v1/risk/score", json=request.model_dump())

            if response.status_code != 200:
                failures.append(f"{label}: API returned {response.status_code}")
                print(f"API status            : {response.status_code}")
                print(response.text[:400])
                continue

            body = response.json()
            api_probability = body["risk_probability"]

            print(f"offline probability   : {offline_probability!r}")
            print(f"api probability       : {api_probability!r}")
            print(f"probability identical : {offline_probability == api_probability}")

            if offline_probability != api_probability:
                failures.append(
                    f"{label}: calibrated probability differs "
                    f"({offline_probability!r} vs {api_probability!r})"
                )

            offline_decision = offline_explained.decision
            print(f"offline decision      : {offline_decision}")
            print(f"api decision          : {body['decision']}")

            if body["decision"] != offline_decision:
                failures.append(
                    f"{label}: decision differs ({offline_decision} vs "
                    f"{body['decision']})"
                )

            offline_codes = list(offline_explained.top_reason_codes)
            api_codes = [entry["code"] for entry in body["reason_codes"]]
            api_codes.append(body["decision_reason_code"]["code"])

            print(f"offline reason codes  : {offline_codes}")
            print(f"api reason codes      : {api_codes}")

            if offline_codes != api_codes:
                failures.append(f"{label}: reason codes differ")

            offline_sentence = offline_explained.decision_explanation.explanation
            print(f"offline explanation   : {offline_sentence}")
            print(f"api explanation       : {body['decision_explanation']}")

            if body["decision_explanation"] != offline_sentence:
                failures.append(f"{label}: decision explanation differs")

            for field, expected in (
                ("model_version", artifact.model_version),
                ("feature_version", artifact.feature_version),
                ("policy_version", policy.config.policy_version),
            ):
                if body[field] != expected:
                    failures.append(
                        f"{label}: {field} is {body[field]!r}, expected {expected!r}"
                    )

            print(f"versions              : {body['model_version']} / "
                  f"{body['feature_version']} / {body['policy_version']}")
            print(f"history used          : {body['history_transactions_used']}")

    print_header("6/6", "Result")

    worst_feature, worst_delta = max(AMOUNT_SUM_DELTAS, key=lambda item: item[1])
    print(f"amount-sum comparisons : {len(AMOUNT_SUM_DELTAS)}")
    print(f"worst amount-sum delta : {worst_delta:.3e} ({worst_feature})")
    print(f"tolerance              : abs {AMOUNT_SUM_ABSOLUTE_TOLERANCE:.0e} / "
          f"rel {AMOUNT_SUM_RELATIVE_TOLERANCE:.0e}")

    if failures:
        print(f"PARITY FAILURES: {len(failures)}")

        for failure in failures:
            print(f"  - {failure}")

        raise SystemExit(1)

    print("PARITY CONFIRMED: the API and the offline pipeline agree on every")
    print("compared value for all three transactions.")

    print_section("DISCIPLINE")
    print("1. The TEST partition was loaded for its row count only; it was never")
    print("   scored, calibrated or explained by this script.")
    print("2. The three examples were selected from calibrated risk probabilities")
    print("   and the frozen thresholds alone. No label was read.")
    print("3. Nothing was trained, calibrated or thresholded here; the artifact")
    print("   was read from disk exactly as the API reads it.")
    print("4. Historical context was restricted to the trailing 24 hours of the")
    print("   transaction's own card1 and (addr1, addr2) entities — the same rows")
    print("   the offline rolling computation retains. No synthetic history.")
    print("5. Counts and flags were required to match exactly; only the two")
    print("   floating-point amount sums were compared with a tolerance, because")
    print("   their summation order differs between a full replay and a window.")

    print(f"\nTotal seconds: {time.perf_counter() - started:.2f}")

    print_section("PHASE 6 OFFLINE vs API PARITY COMPLETE")


def artifact_pipeline_model(artifact):
    """The artifact's pipeline in the type ``predict_fraud_probability`` takes."""
    from app.service import _model_view

    return _model_view(artifact)


if __name__ == "__main__":
    main()
