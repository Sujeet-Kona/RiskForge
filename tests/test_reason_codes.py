from __future__ import annotations

import re

import pytest

from explain.reason_codes import (
    DECISION_REASON_CODES,
    DECISION_RULE,
    MODEL_SIGNAL,
    MODEL_SIGNAL_REASON_CODES,
    UNDOCUMENTED_SEMANTICS_NOTE,
    all_reason_codes,
    decision_reason_code,
    reason_code_for_feature,
)
from ml.data.modeling import CD_FEATURES, MODEL_FEATURES


CODE_PATTERN = re.compile(r"^[A-Z][A-Z0-9_]*$")

# Causal / certainty language the brief forbids. Explanations may describe what
# the model did; they may not describe what the world did.
FORBIDDEN_CAUSAL_LANGUAGE = (
    "caus",
    "because",
    "prove",
    "proves",
    "guarantee",
    "is fraudulent",
    "confirms fraud",
    "definitely",
    "certain",
)


def test_every_model_signal_code_covers_only_contract_features():
    covered = {
        feature
        for reason in MODEL_SIGNAL_REASON_CODES
        for feature in reason.features
    }

    assert covered == set(MODEL_FEATURES)


def test_no_feature_is_claimed_by_two_different_codes():
    seen: dict[str, str] = {}

    for reason in MODEL_SIGNAL_REASON_CODES:
        for feature in reason.features:
            assert feature not in seen, f"{feature} claimed by two codes"
            seen[feature] = reason.code


def test_every_contract_feature_resolves_to_a_code():
    for feature in MODEL_FEATURES:
        assert reason_code_for_feature(feature).code


def test_codes_are_stable_upper_snake_strings():
    for reason in all_reason_codes():
        assert isinstance(reason.code, str)
        assert CODE_PATTERN.match(reason.code), reason.code


def test_codes_are_unique():
    codes = [reason.code for reason in all_reason_codes()]

    assert len(codes) == len(set(codes))


def test_every_reason_has_all_required_fields():
    for reason in all_reason_codes():
        assert reason.code
        assert reason.title
        assert reason.type in {MODEL_SIGNAL, DECISION_RULE}
        assert reason.description

        if reason.type == MODEL_SIGNAL:
            assert reason.features, f"{reason.code} names no feature"
        else:
            # A decision rule is about thresholds, not features.
            assert reason.features == ()


def test_model_signal_descriptions_are_non_empty_prose():
    for reason in MODEL_SIGNAL_REASON_CODES:
        assert len(reason.description.split()) >= 6
        assert "model" in reason.description.lower()


def test_no_reason_code_uses_causal_language():
    for reason in all_reason_codes():
        text = f"{reason.title} {reason.description}".lower()

        for forbidden in FORBIDDEN_CAUSAL_LANGUAGE:
            assert forbidden not in text, f"{reason.code} says {forbidden!r}"


# --- requirement 9: undocumented C/D features stay neutral -----------------


@pytest.mark.parametrize("feature", CD_FEATURES)
def test_cd_features_get_neutral_model_signal_codes(feature):
    reason = reason_code_for_feature(feature)

    assert reason.code == f"MODEL_SIGNAL_{feature}"
    assert reason.features == (feature,)


@pytest.mark.parametrize("feature", CD_FEATURES)
def test_cd_features_claim_no_business_semantics(feature):
    reason = reason_code_for_feature(feature)
    text = f"{reason.title} {reason.description}"

    assert UNDOCUMENTED_SEMANTICS_NOTE in reason.description
    assert "no business meaning is claimed" in reason.description.lower()

    for invented in ("device", "browser", "country", "velocity", "address"):
        assert invented not in text.lower(), f"{reason.code} invents {invented!r}"


def test_cd_codes_never_outnumber_the_cd_contract():
    cd_codes = {
        reason.code for feature in CD_FEATURES
        for reason in [reason_code_for_feature(feature)]
    }

    assert len(cd_codes) == len(CD_FEATURES)


# --- requirement 8: no fabricated semantics for unknown features -----------


@pytest.mark.parametrize(
    "feature",
    ["card4", "id_01", "V1", "D5", "device_type", "", "transaction_day_index"],
)
def test_an_unsupported_feature_is_rejected_rather_than_invented(feature):
    with pytest.raises(ValueError, match="No reason code is defined"):
        reason_code_for_feature(feature)


def test_a_non_string_feature_is_rejected():
    with pytest.raises(TypeError, match="must be a string"):
        reason_code_for_feature(17)


def test_the_registry_has_no_fallback_entry():
    assert "*" not in reason_code_for_feature.__globals__["REASON_CODE_BY_FEATURE"]
    assert "unknown" not in {
        key.lower() for key in reason_code_for_feature.__globals__["REASON_CODE_BY_FEATURE"]
    }


# --- decision-rule codes ---------------------------------------------------


@pytest.mark.parametrize("decision_value", ["ALLOW", "REVIEW", "BLOCK"])
def test_each_decision_has_its_own_rule_code(decision_value):
    reason = decision_reason_code(decision_value)

    assert reason.type == DECISION_RULE
    assert CODE_PATTERN.match(reason.code)


def test_decision_codes_mention_thresholds_not_features():
    for name in ("ALLOW", "REVIEW", "BLOCK"):
        assert "threshold" in DECISION_REASON_CODES[name].description.lower()


def test_an_unknown_decision_is_rejected():
    with pytest.raises(ValueError, match="No decision reason code"):
        decision_reason_code("APPROVE")


def test_all_reason_codes_lists_model_signals_then_decision_rules():
    codes = [reason.code for reason in all_reason_codes()]

    assert len(codes) == len(MODEL_SIGNAL_REASON_CODES) + 3
    assert codes[-3:] == [
        DECISION_REASON_CODES[name].code for name in ("ALLOW", "REVIEW", "BLOCK")
    ]


def test_reason_code_to_dict_is_serializable():
    payload = reason_code_for_feature("card_txn_count_1h").to_dict()

    assert payload == {
        "code": "CARD_TRANSACTION_VELOCITY_SIGNAL",
        "title": "Recent card transaction activity",
        "type": MODEL_SIGNAL,
        "features": ["card_txn_count_1h", "card_txn_count_24h"],
        "description": reason_code_for_feature("card_txn_count_1h").description,
    }


def test_reason_codes_are_frozen():
    reason = reason_code_for_feature("C1")

    with pytest.raises(Exception):
        reason.code = "SOMETHING_ELSE"


def test_elapsed_time_reason_codes_do_not_claim_calendar_semantics():
    hour = reason_code_for_feature("transaction_hour_bucket")
    weekday = reason_code_for_feature("transaction_day_mod7")

    assert "elapsed" in hour.title.lower()
    assert "wall-clock" in hour.description.lower()
    assert "time-of-day" not in hour.title.lower()

    assert "elapsed" in weekday.title.lower()
    assert "calendar weekday" in weekday.description.lower()
    assert "day-of-week" not in weekday.title.lower()
