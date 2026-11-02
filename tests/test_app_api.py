"""The API over HTTP, which is how every requirement in the brief is observed.

Schema-level rules are covered in ``test_app_schemas.py``; what is asserted here
is the wire behaviour — status codes, response shape, the artifact lifecycle, and
that a failure is reported as a failure rather than as a score.
"""

from __future__ import annotations

import copy

import pytest
from fastapi.testclient import TestClient

from app.dependencies import ARTIFACT_INVALID, ARTIFACT_NOT_FOUND
from app.main import (
    HEALTH_ROUTE,
    INFERENCE_FAILED,
    SCORE_ROUTE,
    create_app,
)
from app.schemas import RiskScoreResponse
from app.service import RiskScoringService
from policy.decision import Decision


RESPONSE_FIELDS = set(RiskScoreResponse.model_fields)


def _post(client: TestClient, payload: dict):
    return client.post(SCORE_ROUTE, json=payload)


# --- requirements 1, 7, 8, 9: health and identity ----------------------------


def test_health_returns_200(client):
    response = client.get(HEALTH_ROUTE)

    assert response.status_code == 200
    assert response.json()["status"] == "ok"


def test_health_reports_the_artifact_identity(client):
    body = client.get(HEALTH_ROUTE).json()

    assert body["model_version"] == "xgb-001"
    assert body["feature_version"] == "behavioral-v1"
    assert body["policy_version"] == "policy-001"


def test_health_performs_no_inference(client, monkeypatch):
    def refuse(*args, **kwargs):
        raise AssertionError("health must not score anything")

    monkeypatch.setattr(RiskScoringService, "score", refuse)

    assert client.get(HEALTH_ROUTE).status_code == 200


def test_health_is_unavailable_when_the_artifact_could_not_be_loaded(tmp_path):
    with TestClient(create_app(artifact_path=tmp_path / "absent.joblib")) as client:
        response = client.get(HEALTH_ROUTE)

    assert response.status_code == 503

    body = response.json()

    assert body["status"] == "unavailable"
    assert body["model_version"] is None
    assert body["feature_version"] is None
    assert body["policy_version"] is None
    assert body["artifact"] is None
    assert str(tmp_path) not in response.text


# --- requirements 2, 3: a valid transaction scores, and the shape is fixed ---


def test_a_valid_transaction_returns_200(client, score_payload):
    assert _post(client, score_payload).status_code == 200


def test_the_response_carries_exactly_the_declared_fields(client, score_payload):
    assert set(_post(client, score_payload).json()) == RESPONSE_FIELDS


@pytest.mark.parametrize(
    "field",
    [
        "risk_probability",
        "decision",
        "reason_codes",
        "decision_reason_code",
        "decision_explanation",
        "model_version",
        "feature_version",
        "policy_version",
    ],
)
def test_every_field_the_brief_requires_is_present(client, score_payload, field):
    assert field in _post(client, score_payload).json()


def test_a_transaction_with_no_history_at_all_is_scored(client, score_payload):
    payload = copy.deepcopy(score_payload)
    payload["history"] = []
    payload["transaction"].update(
        {"R_emaildomain": None, "D2": None, "D3": None, "D8": None}
    )

    assert _post(client, payload).status_code == 200


# --- requirements 4 and 5 ----------------------------------------------------


def test_the_probability_is_inside_the_unit_interval(client, score_payload):
    probability = _post(client, score_payload).json()["risk_probability"]

    assert 0.0 <= probability <= 1.0


def test_the_decision_is_one_of_the_three_locked_bands(client, score_payload):
    assert _post(client, score_payload).json()["decision"] in {
        decision.value for decision in Decision
    }


# --- requirements 7, 8, 9: versions on the scoring response ------------------


def test_the_scoring_response_names_the_model_version(client, score_payload):
    assert _post(client, score_payload).json()["model_version"] == "xgb-001"


def test_the_scoring_response_names_the_feature_version(client, score_payload):
    assert _post(client, score_payload).json()["feature_version"] == "behavioral-v1"


def test_the_scoring_response_names_the_policy_version(client, score_payload):
    assert _post(client, score_payload).json()["policy_version"] == "policy-001"


def test_the_response_states_the_size_of_the_feature_contract(client, score_payload):
    assert _post(client, score_payload).json()["feature_count"] == 28


# --- requirements 10 and 11 --------------------------------------------------


def test_reason_codes_are_returned_with_their_model_attribution(client, score_payload):
    body = _post(client, score_payload).json()

    assert isinstance(body["reason_codes"], list)

    for entry in body["reason_codes"]:
        assert entry["code"]
        assert entry["title"]
        assert entry["feature"]
        assert isinstance(entry["contribution"], float)
        assert entry["direction"] == "risk_increasing"
        assert entry["n_transformed_columns"] >= 1


def test_the_decision_explanation_is_a_sentence_about_the_band(client, score_payload):
    body = _post(client, score_payload).json()

    assert body["decision_explanation"].endswith(".")
    assert body["decision_reason_code"]["code"].startswith("POLICY_")
    assert body["decision_reason_code"]["description"]


def test_risk_decreasing_signals_are_returned_separately(client, score_payload):
    body = _post(client, score_payload).json()

    for entry in body["risk_decreasing_signals"]:
        assert entry["direction"] == "risk_decreasing"
        assert entry["contribution"] <= 0.0


# --- requirements 12, 13, 14: invalid requests ------------------------------


def test_a_malformed_body_is_rejected(client):
    response = client.post(
        SCORE_ROUTE,
        content="{not json",
        headers={"Content-Type": "application/json"},
    )

    assert response.status_code == 422


def test_an_empty_body_is_rejected(client):
    assert client.post(SCORE_ROUTE, json={}).status_code == 422


@pytest.mark.parametrize(
    "field", ["TransactionID", "TransactionDT", "TransactionAmt", "card1"]
)
def test_a_missing_required_field_is_rejected(client, score_payload, field):
    payload = copy.deepcopy(score_payload)
    del payload["transaction"][field]

    response = _post(client, payload)

    assert response.status_code == 422
    assert field in response.text


@pytest.mark.parametrize("amount", [-1.0, -0.01])
def test_a_negative_amount_is_rejected(client, score_payload, amount):
    payload = copy.deepcopy(score_payload)
    payload["transaction"]["TransactionAmt"] = amount

    assert _post(client, payload).status_code == 422


def test_an_unknown_request_field_is_rejected(client, score_payload):
    payload = copy.deepcopy(score_payload)
    payload["transaction"]["product_cd"] = "W"

    assert _post(client, payload).status_code == 422


def test_a_non_numeric_c_d_value_is_rejected(client, score_payload):
    payload = copy.deepcopy(score_payload)
    payload["transaction"]["C1"] = "high"

    assert _post(client, payload).status_code == 422


# --- requirements 15, 16, 17: the point-in-time rule over HTTP --------------


def test_history_at_the_current_timestamp_is_rejected(client, score_payload):
    payload = copy.deepcopy(score_payload)
    payload["history"] = [
        {
            "TransactionDT": payload["transaction"]["TransactionDT"],
            "TransactionAmt": 10.0,
            "card1": 13926,
        }
    ]

    response = _post(client, payload)

    assert response.status_code == 422
    assert "strictly less than" in response.text


def test_future_history_is_rejected(client, score_payload):
    payload = copy.deepcopy(score_payload)
    payload["history"] = [
        {
            "TransactionDT": payload["transaction"]["TransactionDT"] + 3_600,
            "TransactionAmt": 10.0,
            "card1": 13926,
        }
    ]

    response = _post(client, payload)

    assert response.status_code == 422
    assert "strictly less than" in response.text


def test_valid_prior_history_is_accepted(client, score_payload):
    current = score_payload["transaction"]["TransactionDT"]
    payload = copy.deepcopy(score_payload)
    payload["history"] = [
        {"TransactionDT": current - 1, "TransactionAmt": 10.0, "card1": 13926},
        {"TransactionDT": current - 3_601, "TransactionAmt": 20.0, "card1": 13926},
        {"TransactionDT": current - 86_401, "TransactionAmt": 30.0, "card1": 13926},
    ]

    response = _post(client, payload)

    assert response.status_code == 200
    assert response.json()["history_transactions_used"] == 3


def test_history_changes_the_score_because_it_changes_the_features(client, score_payload):
    """The caller-supplied context is genuinely consumed, not merely accepted."""
    without = copy.deepcopy(score_payload)
    without["history"] = []

    with_history = copy.deepcopy(score_payload)
    current = with_history["transaction"]["TransactionDT"]
    with_history["history"] = [
        {
            "TransactionDT": current - offset,
            "TransactionAmt": 900.0,
            "card1": with_history["transaction"]["card1"],
            "addr1": with_history["transaction"]["addr1"],
            "addr2": with_history["transaction"]["addr2"],
        }
        for offset in (5, 60, 600, 3_000, 20_000)
    ]

    assert _post(client, without).json()["model_detail"]["raw_model_probability"] != (
        _post(client, with_history).json()["model_detail"]["raw_model_probability"]
    )


def test_history_beyond_the_declared_bound_is_rejected(client, score_payload):
    from app.config import MAX_HISTORY_TRANSACTIONS

    current = score_payload["transaction"]["TransactionDT"]
    payload = copy.deepcopy(score_payload)
    payload["history"] = [
        {"TransactionDT": current - 1, "TransactionAmt": 1.0, "card1": 13926}
    ] * (MAX_HISTORY_TRANSACTIONS + 1)

    assert _post(client, payload).status_code == 422


# --- requirement 23: unknown categorical levels ------------------------------


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("ProductCD", "ZZZ"),
        ("card6", "never_seen_in_training"),
        ("P_emaildomain", "unseen.example"),
        ("R_emaildomain", "unseen.example"),
    ],
)
def test_an_unseen_categorical_level_is_scored_not_rejected(client, score_payload, field, value):
    """``handle_unknown="ignore"`` makes this a valid input, so it must be a 200."""
    payload = copy.deepcopy(score_payload)
    payload["transaction"][field] = value

    response = _post(client, payload)

    assert response.status_code == 200
    assert 0.0 <= response.json()["risk_probability"] <= 1.0


def test_an_unseen_categorical_level_scores_like_a_missing_one(client, score_payload):
    """An ignored level contributes no one-hot column, exactly as a gap does."""
    unseen = copy.deepcopy(score_payload)
    unseen["transaction"]["ProductCD"] = "ZZZ"

    missing = copy.deepcopy(score_payload)
    missing["transaction"]["ProductCD"] = None

    assert _post(client, unseen).json()["risk_probability"] == _post(
        client, missing
    ).json()["risk_probability"]


# --- requirement 24: determinism --------------------------------------------


def test_the_same_request_twice_gives_a_byte_identical_response(client, score_payload):
    first = _post(client, score_payload).json()
    second = _post(client, score_payload).json()

    assert first == second


def test_two_clients_on_the_same_artifact_agree(
    tiny_artifact_path, client, score_payload
):
    """A second boot of the same bundle is not a second model."""
    expected = _post(client, score_payload).json()

    with TestClient(create_app(artifact_path=tiny_artifact_path)) as other:
        assert other.post(SCORE_ROUTE, json=score_payload).json() == expected


# --- requirements 19, 20: nothing is fitted at request time -----------------


def test_scoring_never_trains_a_model(client, score_payload, monkeypatch):
    import ml.models.xgboost_model as xgboost_model

    def refuse(*args, **kwargs):
        raise AssertionError("the API must not train a model at request time")

    monkeypatch.setattr(xgboost_model, "build_xgboost_model", refuse)
    monkeypatch.setattr(xgboost_model, "fit_xgboost_model", refuse)

    assert _post(client, score_payload).status_code == 200


def test_scoring_never_fits_a_calibrator(client, score_payload, monkeypatch):
    import ml.models.calibration as calibration

    def refuse(*args, **kwargs):
        raise AssertionError("the API must not fit calibration at request time")

    monkeypatch.setattr(calibration, "fit_calibrator", refuse)

    assert _post(client, score_payload).status_code == 200


def test_scoring_never_fits_preprocessing(client, score_payload, monkeypatch):
    from sklearn.pipeline import Pipeline

    def refuse(self, *args, **kwargs):
        raise AssertionError("the API must not fit anything at request time")

    monkeypatch.setattr(Pipeline, "fit", refuse)

    assert _post(client, score_payload).status_code == 200


# --- requirement 21: loaded once --------------------------------------------


def test_the_artifact_is_loaded_once_and_not_per_request(tiny_artifact_path, monkeypatch):
    import app.dependencies as dependencies

    calls: list = []
    original = dependencies.load_artifact

    def counting(path):
        calls.append(path)

        return original(path)

    monkeypatch.setattr(dependencies, "load_artifact", counting)

    with TestClient(create_app(artifact_path=tiny_artifact_path)) as test_client:
        for _ in range(3):
            test_client.post(SCORE_ROUTE, json={"transaction": {
                "TransactionID": "1",
                "TransactionDT": 1_000,
                "TransactionAmt": 10.0,
                "card1": 1,
            }})

    assert len(calls) == 1


def test_the_service_object_is_reused_across_requests(client, score_payload):
    seen = {id(client.app.state.startup.service)}

    for _ in range(3):
        _post(client, score_payload)
        seen.add(id(client.app.state.startup.service))

    assert len(seen) == 1


# --- requirement 22: a missing artifact fails clearly -----------------------


def test_scoring_returns_503_when_the_artifact_is_absent(tmp_path, score_payload):
    with TestClient(create_app(artifact_path=tmp_path / "absent.joblib")) as client:
        response = client.post(SCORE_ROUTE, json=score_payload)

    assert response.status_code == 503
    assert response.json()["error"]["code"] == ARTIFACT_NOT_FOUND


def test_scoring_returns_503_when_the_artifact_is_not_a_bundle(tmp_path, score_payload):
    import joblib

    path = tmp_path / "broken.joblib"
    joblib.dump({"not": "an artifact"}, path)

    with TestClient(create_app(artifact_path=path)) as client:
        response = client.post(SCORE_ROUTE, json=score_payload)

    assert response.status_code == 503
    assert response.json()["error"]["code"] == ARTIFACT_INVALID


def test_the_503_body_says_the_service_never_trains(tmp_path):
    payload = {
        "transaction": {
            "TransactionID": "1",
            "TransactionDT": 1,
            "TransactionAmt": 1.0,
            "card1": 1,
        }
    }

    with TestClient(create_app(artifact_path=tmp_path / "absent.joblib")) as client:
        health = client.get(HEALTH_ROUTE)
        scored = client.post(SCORE_ROUTE, json=payload)

    assert health.status_code == 503
    assert scored.status_code == 503
    assert "never trains" in scored.json()["error"]["message"]


# --- requirements 26, 27: no database, no network ---------------------------


def test_scoring_needs_no_database_or_network_stack(client, score_payload, monkeypatch):
    """Blocking the sockets proves inference is a local computation."""
    import socket

    def refuse(*args, **kwargs):
        raise AssertionError("inference must not open a connection")

    monkeypatch.setattr(socket.socket, "connect", refuse)
    monkeypatch.setattr(socket, "create_connection", refuse)

    assert _post(client, score_payload).status_code == 200


# --- requirement 28: errors stay structured ---------------------------------


def test_an_inference_failure_returns_500_with_a_structured_body(
    tiny_artifact_path, score_payload, monkeypatch
):
    def explode(self, request):
        raise RuntimeError("/Users/somebody/secret/path/weights.bin: boom")

    monkeypatch.setattr(RiskScoringService, "score", explode)

    with TestClient(
        create_app(artifact_path=tiny_artifact_path), raise_server_exceptions=False
    ) as client:
        response = client.post(SCORE_ROUTE, json=score_payload)

    assert response.status_code == 500

    body = response.json()

    assert set(body) == {"error"}
    assert set(body["error"]) == {"code", "message"}
    assert body["error"]["code"] == INFERENCE_FAILED
    assert body["error"]["message"]


def test_an_inference_failure_leaks_no_traceback_path_or_exception_text(
    tiny_artifact_path, score_payload, monkeypatch
):
    def explode(self, request):
        raise RuntimeError("/Users/somebody/secret/path/weights.bin: boom")

    monkeypatch.setattr(RiskScoringService, "score", explode)

    with TestClient(
        create_app(artifact_path=tiny_artifact_path), raise_server_exceptions=False
    ) as client:
        body = client.post(SCORE_ROUTE, json=score_payload).text

    assert "Traceback" not in body
    assert "/Users/" not in body
    assert "weights.bin" not in body
    assert "RuntimeError" not in body


def test_an_inference_failure_is_never_reported_as_a_score(
    tiny_artifact_path, score_payload, monkeypatch
):
    def explode(self, request):
        raise ValueError("cannot score")

    monkeypatch.setattr(RiskScoringService, "score", explode)

    with TestClient(
        create_app(artifact_path=tiny_artifact_path), raise_server_exceptions=False
    ) as client:
        body = client.post(SCORE_ROUTE, json=score_payload).json()

    assert "risk_probability" not in body
    assert "decision" not in body


def test_a_validation_error_body_is_structured_and_names_the_field(client, score_payload):
    payload = copy.deepcopy(score_payload)
    del payload["transaction"]["TransactionAmt"]

    body = _post(client, payload).json()

    assert "detail" in body
    assert body["detail"][0]["loc"][-1] == "TransactionAmt"
    assert "Traceback" not in str(body)
