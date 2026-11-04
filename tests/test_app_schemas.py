"""Request validation, checked at the schema layer where the rules live.

The same rules are exercised over HTTP in ``test_app_api.py``; here they are
asserted directly so a rejection is attributable to the field that caused it
rather than to whatever FastAPI decided to report.
"""

from __future__ import annotations

import copy

import pytest
from pydantic import ValidationError

from app.config import MAX_HISTORY_TRANSACTIONS
from app.schemas import HistoryTransaction, RiskScoreRequest, TransactionPayload


def _history(transaction_dt: int) -> dict:
    return {
        "TransactionDT": transaction_dt,
        "TransactionAmt": 10.0,
        "card1": 13926,
        "addr1": 299.0,
        "addr2": 87.0,
    }


def test_a_well_formed_request_is_accepted(score_payload):
    request = RiskScoreRequest(**score_payload)

    assert request.transaction.TransactionID == "3000001"
    assert len(request.history) == 2
    assert request.top_k == 3


def test_history_defaults_to_empty_rather_than_being_required(score_payload):
    payload = copy.deepcopy(score_payload)
    del payload["history"]

    assert RiskScoreRequest(**payload).history == []


def test_a_transaction_without_history_is_a_complete_request():
    request = RiskScoreRequest(
        transaction=TransactionPayload(
            TransactionID="1", TransactionDT=0, TransactionAmt=0.0, card1=1
        )
    )

    assert request.history == []


@pytest.mark.parametrize(
    "field",
    ["TransactionID", "TransactionDT", "TransactionAmt", "card1"],
)
def test_every_required_transaction_field_is_required(score_payload, field):
    payload = copy.deepcopy(score_payload)
    del payload["transaction"][field]

    with pytest.raises(ValidationError) as error:
        RiskScoreRequest(**payload)

    assert field in str(error.value)


def test_the_transaction_block_itself_is_required():
    with pytest.raises(ValidationError):
        RiskScoreRequest(history=[])


@pytest.mark.parametrize(
    "field",
    [
        "ProductCD",
        "card6",
        "P_emaildomain",
        "R_emaildomain",
        "C1",
        "C2",
        "C11",
        "C13",
        "D2",
        "D3",
        "D4",
        "D8",
        "D10",
        "D11",
        "D15",
        "addr1",
        "addr2",
    ],
)
def test_every_optional_field_may_be_omitted_entirely(field):
    values = {
        "TransactionID": "1",
        "TransactionDT": 100,
        "TransactionAmt": 5.0,
        "card1": 7,
    }
    values.pop(field, None)

    assert RiskScoreRequest(transaction=values).transaction is not None


def test_an_unknown_field_is_rejected_rather_than_silently_ignored(score_payload):
    """A caller who sends ``product_cd`` gets told, not a silently imputed gap."""
    payload = copy.deepcopy(score_payload)
    payload["transaction"]["product_cd"] = "W"

    with pytest.raises(ValidationError, match="extra"):
        RiskScoreRequest(**payload)


def test_an_unknown_top_level_field_is_rejected(score_payload):
    payload = copy.deepcopy(score_payload)
    payload["customer_id"] = "abc"

    with pytest.raises(ValidationError, match="extra"):
        RiskScoreRequest(**payload)


def test_an_unknown_history_field_is_rejected(score_payload):
    payload = copy.deepcopy(score_payload)
    payload["history"][0]["ProductCD"] = "W"

    with pytest.raises(ValidationError, match="extra"):
        RiskScoreRequest(**payload)


@pytest.mark.parametrize("amount", [-0.01, -1.0, -1000.0])
def test_a_negative_amount_is_rejected(score_payload, amount):
    payload = copy.deepcopy(score_payload)
    payload["transaction"]["TransactionAmt"] = amount

    with pytest.raises(ValidationError):
        RiskScoreRequest(**payload)


def test_a_zero_amount_is_accepted(score_payload):
    payload = copy.deepcopy(score_payload)
    payload["transaction"]["TransactionAmt"] = 0.0

    assert RiskScoreRequest(**payload).transaction.TransactionAmt == 0.0


@pytest.mark.parametrize("amount", [float("nan"), float("inf"), float("-inf")])
def test_a_non_finite_amount_is_rejected(score_payload, amount):
    payload = copy.deepcopy(score_payload)
    payload["transaction"]["TransactionAmt"] = amount

    with pytest.raises(ValidationError):
        RiskScoreRequest(**payload)


@pytest.mark.parametrize("value", [float("nan"), float("inf")])
def test_a_non_finite_c_d_value_is_rejected(score_payload, value):
    payload = copy.deepcopy(score_payload)
    payload["transaction"]["C1"] = value

    with pytest.raises(ValidationError):
        RiskScoreRequest(**payload)


def test_a_negative_transaction_dt_is_rejected(score_payload):
    payload = copy.deepcopy(score_payload)
    payload["transaction"]["TransactionDT"] = -1

    with pytest.raises(ValidationError):
        RiskScoreRequest(**payload)


def test_a_negative_history_dt_is_rejected(score_payload):
    payload = copy.deepcopy(score_payload)
    payload["history"][0]["TransactionDT"] = -5

    with pytest.raises(ValidationError):
        RiskScoreRequest(**payload)


@pytest.mark.parametrize("transaction_id", ["", None])
def test_a_blank_transaction_id_is_rejected(score_payload, transaction_id):
    payload = copy.deepcopy(score_payload)
    payload["transaction"]["TransactionID"] = transaction_id

    with pytest.raises(ValidationError):
        RiskScoreRequest(**payload)


def test_a_transaction_id_at_the_database_limit_is_accepted(score_payload):
    payload = copy.deepcopy(score_payload)
    payload["transaction"]["TransactionID"] = "x" * 64

    request = RiskScoreRequest(**payload)

    assert len(request.transaction.TransactionID) == 64


def test_a_transaction_id_longer_than_the_database_limit_is_rejected(score_payload):
    payload = copy.deepcopy(score_payload)
    payload["transaction"]["TransactionID"] = "x" * 65

    with pytest.raises(ValidationError, match="at most 64 characters"):
        RiskScoreRequest(**payload)


def test_a_non_integer_card1_is_rejected(score_payload):
    payload = copy.deepcopy(score_payload)
    payload["transaction"]["card1"] = "thirteen"

    with pytest.raises(ValidationError):
        RiskScoreRequest(**payload)


# --- requirement 15 and 16: the point-in-time rule is enforced, not filtered ---


def test_history_at_the_current_timestamp_is_rejected(score_payload):
    """Same-timestamp rows would let two transactions see each other."""
    current = score_payload["transaction"]["TransactionDT"]
    payload = copy.deepcopy(score_payload)
    payload["history"] = [_history(current)]

    with pytest.raises(ValidationError, match="strictly less than"):
        RiskScoreRequest(**payload)


def test_history_after_the_current_timestamp_is_rejected(score_payload):
    current = score_payload["transaction"]["TransactionDT"]
    payload = copy.deepcopy(score_payload)
    payload["history"] = [_history(current + 1)]

    with pytest.raises(ValidationError, match="strictly less than"):
        RiskScoreRequest(**payload)


def test_one_invalid_history_row_fails_the_whole_request(score_payload):
    """Invalid history is reported, never silently dropped."""
    current = score_payload["transaction"]["TransactionDT"]
    payload = copy.deepcopy(score_payload)
    payload["history"] = [_history(current - 10), _history(current), _history(current - 20)]

    with pytest.raises(ValidationError, match=r"history\[1\]"):
        RiskScoreRequest(**payload)


def test_the_rejection_names_the_offending_position_and_both_timestamps(score_payload):
    current = score_payload["transaction"]["TransactionDT"]
    payload = copy.deepcopy(score_payload)
    payload["history"] = [_history(current + 3600)]

    with pytest.raises(ValidationError) as error:
        RiskScoreRequest(**payload)

    message = str(error.value)

    assert "history[0].TransactionDT=" in message
    assert str(current + 3600) in message
    assert str(current) in message


def test_history_one_second_before_the_current_timestamp_is_accepted(score_payload):
    current = score_payload["transaction"]["TransactionDT"]
    payload = copy.deepcopy(score_payload)
    payload["history"] = [_history(current - 1)]

    assert len(RiskScoreRequest(**payload).history) == 1


def test_history_spanning_both_windows_is_accepted(score_payload):
    """Rows older than 24h stay valid; the feature layer is what bounds them."""
    current = score_payload["transaction"]["TransactionDT"]
    payload = copy.deepcopy(score_payload)
    payload["history"] = [
        _history(current - 1),
        _history(current - 3_600),
        _history(current - 86_400),
        _history(current - 100_000),
    ]

    assert len(RiskScoreRequest(**payload).history) == 4


def test_unsorted_history_is_accepted_because_the_features_sort_it(score_payload):
    current = score_payload["transaction"]["TransactionDT"]
    payload = copy.deepcopy(score_payload)
    payload["history"] = [_history(current - 10), _history(current - 500)]

    assert [entry.TransactionDT for entry in RiskScoreRequest(**payload).history] == [
        current - 10,
        current - 500,
    ]


def test_history_at_the_declared_bound_is_accepted(score_payload):
    current = score_payload["transaction"]["TransactionDT"]
    payload = copy.deepcopy(score_payload)
    payload["history"] = [_history(current - 1)] * MAX_HISTORY_TRANSACTIONS

    assert len(RiskScoreRequest(**payload).history) == MAX_HISTORY_TRANSACTIONS


def test_history_above_the_declared_bound_is_rejected_not_truncated(score_payload):
    """Trimming would change the behavioral features and break offline parity."""
    current = score_payload["transaction"]["TransactionDT"]
    payload = copy.deepcopy(score_payload)
    payload["history"] = [_history(current - 1)] * (MAX_HISTORY_TRANSACTIONS + 1)

    with pytest.raises(ValidationError):
        RiskScoreRequest(**payload)


# --- requirement 23: unknown categorical levels stay valid ---------------------


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("ProductCD", "ZZZ"),
        ("card6", "a_card_type_never_seen_in_training"),
        ("P_emaildomain", "brand-new-mail.example"),
        ("R_emaildomain", "also-unseen.example"),
    ],
)
def test_an_unseen_categorical_level_is_a_valid_request(score_payload, field, value):
    """The encoder uses ``handle_unknown="ignore"``, so this must not be a 422."""
    payload = copy.deepcopy(score_payload)
    payload["transaction"][field] = value

    assert getattr(RiskScoreRequest(**payload).transaction, field) == value


def test_an_empty_string_categorical_is_accepted_as_a_level(score_payload):
    payload = copy.deepcopy(score_payload)
    payload["transaction"]["ProductCD"] = ""

    assert RiskScoreRequest(**payload).transaction.ProductCD == ""


# --- top_k -------------------------------------------------------------------


@pytest.mark.parametrize("top_k", [0, -1])
def test_top_k_must_be_at_least_one(score_payload, top_k):
    payload = copy.deepcopy(score_payload)
    payload["top_k"] = top_k

    with pytest.raises(ValidationError):
        RiskScoreRequest(**payload)


def test_top_k_may_ask_for_every_feature(score_payload):
    from ml.data.modeling import MODEL_FEATURES

    payload = copy.deepcopy(score_payload)
    payload["top_k"] = len(MODEL_FEATURES)

    assert RiskScoreRequest(**payload).top_k == len(MODEL_FEATURES)


def test_top_k_above_the_feature_count_is_rejected(score_payload):
    from ml.data.modeling import MODEL_FEATURES

    payload = copy.deepcopy(score_payload)
    payload["top_k"] = len(MODEL_FEATURES) + 1

    with pytest.raises(ValidationError):
        RiskScoreRequest(**payload)


# --- the history model on its own --------------------------------------------


def test_history_needs_only_the_five_columns_the_features_actually_read():
    entry = HistoryTransaction(TransactionDT=10, TransactionAmt=1.0)

    assert entry.card1 is None
    assert entry.addr1 is None
    assert entry.addr2 is None


def test_a_history_row_without_an_amount_is_rejected():
    with pytest.raises(ValidationError):
        HistoryTransaction(TransactionDT=10)


def test_a_history_row_without_a_timestamp_is_rejected():
    with pytest.raises(ValidationError):
        HistoryTransaction(TransactionAmt=1.0)
