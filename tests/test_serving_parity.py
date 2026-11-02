"""The regression the brief calls very important: offline versus API.

The same transaction, the same historical context, the same artifact, the same
calibration and the same policy are pushed through two independent routes — the
offline feature and scoring functions called directly, and the HTTP API — and
the two must agree. Nothing is retrained here; both routes read the one bundle.

``scripts/verify_serving_parity.py`` performs the same comparison against real
IEEE-CIS VALIDATION rows. This test is the synthetic counterpart that runs in
the ordinary suite, so the guarantee is checked even where the dataset is absent.
"""

from __future__ import annotations

import math

import pandas as pd
import pytest
from fastapi.testclient import TestClient

from app.features import BEHAVIORAL_INPUT_COLUMNS, SERVING_DTYPES
from app.main import create_app
from explain.explanation import explain_transaction
from ml.data.modeling import CATEGORICAL_FEATURES, CD_FEATURES, MODEL_FEATURES
from ml.features.behavioral import build_behavioral_features
from ml.models.calibration import apply_calibrator
from ml.models.xgboost_model import predict_fraud_probability


# The two amount sums are floating-point accumulations whose summation order
# differs between a full replay and a window, so they are compared with a
# tolerance. Every other feature must match exactly.
AMOUNT_SUMS = ("card_amount_sum_24h", "address_amount_sum_24h")
AMOUNT_SUM_ABSOLUTE_TOLERANCE = 1e-9
AMOUNT_SUM_RELATIVE_TOLERANCE = 1e-12

CURRENT = {
    "TransactionID": "4000123",
    "TransactionDT": 500_000,
    "TransactionAmt": 210.75,
    "card1": 9500,
    "addr1": 299.0,
    "addr2": 87.0,
    "ProductCD": "W",
    "card6": "debit",
    "P_emaildomain": "gmail.com",
    "R_emaildomain": None,
    "C1": 3.0,
    "C2": 5.0,
    "C11": 2.0,
    "C13": 7.0,
    "D2": None,
    "D3": None,
    "D4": 1.0,
    "D8": None,
    "D10": 12.0,
    "D11": 3.0,
    "D15": 0.0,
}

def _prior(
    transaction_dt: int,
    amount: float,
    *,
    card1: int | None = 9500,
    addr1: float | None = 299.0,
    addr2: float | None = 87.0,
) -> dict:
    return {
        "TransactionDT": transaction_dt,
        "TransactionAmt": amount,
        "card1": card1,
        "addr1": addr1,
        "addr2": addr2,
    }


HISTORY_CASES = {
    "no_history": [],
    "one_prior_second": [_prior(499_999, 30.0)],
    "both_windows": [
        _prior(499_000, 30.0),
        _prior(497_000, 12.5),
        _prior(420_000, 88.0),
    ],
    "on_the_window_edges": [
        _prior(500_000 - 3_600, 5.0),
        _prior(500_000 - 86_400, 5.0),
    ],
    "other_entities_only": [
        _prior(499_500, 70.0, card1=1111, addr1=100.0, addr2=10.0),
        _prior(498_000, 70.0, card1=None, addr1=None, addr2=None),
    ],
    "unsorted_input": [_prior(420_000, 88.0), _prior(499_000, 30.0)],
}


def _offline_features(history: list[dict]) -> pd.DataFrame:
    """Build the 28 features the way the offline pipeline builds them."""
    rows = [
        {column: entry.get(column) for column in BEHAVIORAL_INPUT_COLUMNS}
        for entry in history
    ]
    rows.append({column: CURRENT.get(column) for column in BEHAVIORAL_INPUT_COLUMNS})

    frame = pd.DataFrame(rows, columns=list(BEHAVIORAL_INPUT_COLUMNS))
    frame = frame.astype(
        {column: SERVING_DTYPES[column] for column in BEHAVIORAL_INPUT_COLUMNS}
    )
    frame = frame.sort_values("TransactionDT", kind="stable").reset_index(drop=True)

    behavioral = build_behavioral_features(frame)

    current = behavioral.iloc[[-1]].reset_index(drop=True).assign(
        **{
            feature: CURRENT.get(feature)
            for feature in (*CATEGORICAL_FEATURES, *CD_FEATURES)
        }
    )
    current = current.astype(
        {
            feature: SERVING_DTYPES[feature]
            for feature in (*CATEGORICAL_FEATURES, *CD_FEATURES)
        }
    )

    return current[MODEL_FEATURES]


def _offline_score(loaded_artifact, serving_service, features: pd.DataFrame) -> dict:
    raw = predict_fraud_probability(serving_service.model, features)
    probability = float(apply_calibrator(loaded_artifact.calibrator, raw).iloc[0])

    explanation = explain_transaction(
        serving_service.explainer,
        features,
        calibrated_risk_probability=probability,
        policy=serving_service.policy,
        top_k=3,
    )

    return {
        "probability": probability,
        "decision": explanation.decision_explanation.decision.value,
        "reason_codes": [
            entry.reason_code for entry in explanation.model_explanation.risk_increasing
        ],
        "decreasing": [
            entry.reason_code for entry in explanation.model_explanation.risk_decreasing
        ],
        "explanation": explanation.decision_explanation.explanation,
    }


@pytest.mark.parametrize("case", sorted(HISTORY_CASES))
def test_the_offline_and_api_routes_agree(
    client, loaded_artifact, serving_service, case
):
    history = HISTORY_CASES[case]

    offline_features = _offline_features(history)
    expected = _offline_score(loaded_artifact, serving_service, offline_features)

    response = client.post(
        "/v1/risk/score",
        json={"transaction": CURRENT, "history": history, "top_k": 3},
    )

    assert response.status_code == 200, response.text

    body = response.json()

    assert body["risk_probability"] == expected["probability"]
    assert body["decision"] == expected["decision"]
    assert [entry["code"] for entry in body["reason_codes"]] == expected["reason_codes"]
    assert [
        entry["code"] for entry in body["risk_decreasing_signals"]
    ] == expected["decreasing"]
    assert body["decision_explanation"] == expected["explanation"]
    assert body["model_version"] == loaded_artifact.model_version
    assert body["feature_version"] == loaded_artifact.feature_version
    assert body["policy_version"] == loaded_artifact.policy_version
    assert body["history_transactions_used"] == len(history)


@pytest.mark.parametrize("case", sorted(HISTORY_CASES))
def test_every_feature_matches_between_the_two_routes(
    client, loaded_artifact, serving_service, case, monkeypatch
):
    """The score agreeing is not enough; the 28 inputs must agree too."""
    history = HISTORY_CASES[case]
    captured: dict[str, pd.DataFrame] = {}

    import app.service as service_module

    original = service_module.build_serving_features

    def capturing(transaction, entries):
        features = original(transaction, entries)
        captured["features"] = features

        return features

    monkeypatch.setattr(service_module, "build_serving_features", capturing)

    assert client.post(
        "/v1/risk/score",
        json={"transaction": CURRENT, "history": history},
    ).status_code == 200

    offline = _offline_features(history).iloc[0]
    online = captured["features"].iloc[0]

    for feature in MODEL_FEATURES:
        expected = offline[feature]
        actual = online[feature]

        if pd.isna(expected) or pd.isna(actual):
            assert bool(pd.isna(expected)) == bool(pd.isna(actual)), feature
        elif feature in AMOUNT_SUMS:
            assert math.isclose(
                float(actual),
                float(expected),
                rel_tol=AMOUNT_SUM_RELATIVE_TOLERANCE,
                abs_tol=AMOUNT_SUM_ABSOLUTE_TOLERANCE,
            ), feature
        elif isinstance(expected, (bool,)) or isinstance(actual, (bool,)):
            assert bool(expected) == bool(actual), feature
        else:
            assert expected == actual, feature


def test_a_second_boot_of_the_same_artifact_gives_the_same_score(
    tiny_artifact_path, loaded_artifact, serving_service
):
    """Parity is a property of the bundle, not of one process."""
    expected = _offline_score(
        loaded_artifact, serving_service, _offline_features(HISTORY_CASES["both_windows"])
    )

    with TestClient(create_app(artifact_path=tiny_artifact_path)) as other:
        body = other.post(
            "/v1/risk/score",
            json={
                "transaction": CURRENT,
                "history": HISTORY_CASES["both_windows"],
                "top_k": 3,
            },
        ).json()

    assert body["risk_probability"] == expected["probability"]
    assert body["decision"] == expected["decision"]
