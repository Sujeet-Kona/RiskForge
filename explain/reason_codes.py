from __future__ import annotations

from dataclasses import dataclass

from ml.data.modeling import CD_FEATURES


MODEL_SIGNAL = "model_signal"
DECISION_RULE = "decision_rule"

UNDOCUMENTED_SEMANTICS_NOTE = (
    "a numeric benchmark feature from the IEEE-CIS dataset with undocumented "
    "upstream semantics"
)


@dataclass(frozen=True)
class ReasonCode:
    """One stable, human-readable explanation entry.

    ``type`` separates the two things a reader must not confuse: a
    ``model_signal`` says which contract feature moved the model's fraud-risk
    score, while a ``decision_rule`` says which frozen policy threshold turned
    that risk into an action.

    Every description is deliberately non-causal. It reports what the fitted
    model did with a feature, never what the feature means in the world and
    never whether the transaction is fraudulent.
    """

    code: str
    title: str
    type: str
    features: tuple[str, ...]
    description: str

    def to_dict(self) -> dict[str, object]:
        return {
            "code": self.code,
            "title": self.title,
            "type": self.type,
            "features": list(self.features),
            "description": self.description,
        }


def _undocumented(feature: str) -> ReasonCode:
    """Neutral code for a C/D feature whose meaning the dataset does not define."""
    return ReasonCode(
        code=f"MODEL_SIGNAL_{feature}",
        title=f"Model signal from {feature}",
        type=MODEL_SIGNAL,
        features=(feature,),
        description=(
            f"{feature} is {UNDOCUMENTED_SEMANTICS_NOTE}. Its imputed value and "
            "missing indicator contributed to the model's fraud-risk score. No "
            "business meaning is claimed for it."
        ),
    )


# Grouped by contract feature rather than one code per feature. Correlated
# members of a family — TransactionAmt and amount_log1p, or the 1h and 24h
# velocity counters — split SHAP credit between them, so a family-level code is
# a more stable description of the same underlying model signal than either
# column reported alone.
MODEL_SIGNAL_REASON_CODES: tuple[ReasonCode, ...] = (
    ReasonCode(
        code="CARD_TRANSACTION_VELOCITY_SIGNAL",
        title="Recent card transaction activity",
        type=MODEL_SIGNAL,
        features=("card_txn_count_1h", "card_txn_count_24h"),
        description=(
            "The count of prior transactions observed for this card in the "
            "point-in-time 1h and 24h windows contributed to the model's "
            "fraud-risk score."
        ),
    ),
    ReasonCode(
        code="CARD_AMOUNT_VOLUME_SIGNAL",
        title="Prior card spend volume",
        type=MODEL_SIGNAL,
        features=("card_amount_sum_24h",),
        description=(
            "The summed transaction amount observed for this card in the "
            "preceding 24h window contributed to the model's fraud-risk score."
        ),
    ),
    ReasonCode(
        code="CARD_HISTORY_SIGNAL",
        title="Card activity-history indicator",
        type=MODEL_SIGNAL,
        features=("card_has_history_1h", "card_has_history_24h"),
        description=(
            "The presence-or-absence flags for prior card activity in the 1h "
            "and 24h windows contributed to the model's fraud-risk score."
        ),
    ),
    ReasonCode(
        code="ADDRESS_TRANSACTION_VELOCITY_SIGNAL",
        title="Recent address transaction activity",
        type=MODEL_SIGNAL,
        features=("address_txn_count_1h", "address_txn_count_24h"),
        description=(
            "The count of prior transactions observed for this address in the "
            "point-in-time 1h and 24h windows contributed to the model's "
            "fraud-risk score."
        ),
    ),
    ReasonCode(
        code="ADDRESS_AMOUNT_VOLUME_SIGNAL",
        title="Prior address spend volume",
        type=MODEL_SIGNAL,
        features=("address_amount_sum_24h",),
        description=(
            "The summed transaction amount observed for this address in the "
            "preceding 24h window contributed to the model's fraud-risk score."
        ),
    ),
    ReasonCode(
        code="ADDRESS_HISTORY_SIGNAL",
        title="Address activity-history indicator",
        type=MODEL_SIGNAL,
        features=("address_has_history_1h",),
        description=(
            "The presence-or-absence flag for prior address activity in the 1h "
            "window contributed to the model's fraud-risk score."
        ),
    ),
    ReasonCode(
        code="TRANSACTION_AMOUNT_SIGNAL",
        title="Transaction amount signal",
        type=MODEL_SIGNAL,
        features=("TransactionAmt", "amount_log1p"),
        description=(
            "The transaction amount, in both raw and log1p form, contributed to "
            "the model's fraud-risk score."
        ),
    ),
    ReasonCode(
        code="TRANSACTION_HOUR_SIGNAL",
        title="Elapsed-time hour-bucket signal",
        type=MODEL_SIGNAL,
        features=("transaction_hour_bucket",),
        description=(
            "The 24-hour bucket derived from elapsed TransactionDT seconds "
            "contributed to the model's fraud-risk score; it is not a "
            "wall-clock time-of-day measurement."
        ),
    ),
    ReasonCode(
        code="TRANSACTION_WEEKDAY_SIGNAL",
        title="Elapsed-day modulo-seven signal",
        type=MODEL_SIGNAL,
        features=("transaction_day_mod7",),
        description=(
            "The elapsed-day index modulo 7 derived from TransactionDT "
            "contributed to the model's fraud-risk score; it is not a "
            "calendar weekday."
        ),
    ),
    ReasonCode(
        code="PRODUCT_CD_SIGNAL",
        title="Product code signal",
        type=MODEL_SIGNAL,
        features=("ProductCD",),
        description=(
            "The one-hot encoded ProductCD level present on this transaction "
            "contributed to the model's fraud-risk score."
        ),
    ),
    ReasonCode(
        code="CARD6_SIGNAL",
        title="Card product-type signal",
        type=MODEL_SIGNAL,
        features=("card6",),
        description=(
            "The one-hot encoded card6 level present on this transaction "
            "contributed to the model's fraud-risk score."
        ),
    ),
    ReasonCode(
        code="P_EMAIL_DOMAIN_SIGNAL",
        title="Purchaser email domain signal",
        type=MODEL_SIGNAL,
        features=("P_emaildomain",),
        description=(
            "The one-hot encoded P_emaildomain level present on this "
            "transaction, including the reserved __MISSING__ level, contributed "
            "to the model's fraud-risk score."
        ),
    ),
    ReasonCode(
        code="R_EMAIL_DOMAIN_SIGNAL",
        title="Recipient email domain signal",
        type=MODEL_SIGNAL,
        features=("R_emaildomain",),
        description=(
            "The one-hot encoded R_emaildomain level present on this "
            "transaction, including the reserved __MISSING__ level, contributed "
            "to the model's fraud-risk score."
        ),
    ),
    *(_undocumented(feature) for feature in CD_FEATURES),
)

DECISION_REASON_CODES: dict[str, ReasonCode] = {
    "ALLOW": ReasonCode(
        code="POLICY_BELOW_REVIEW_THRESHOLD",
        title="Risk below the review threshold",
        type=DECISION_RULE,
        features=(),
        description=(
            "The calibrated risk probability fell below the frozen policy's "
            "REVIEW threshold, so the policy returned ALLOW."
        ),
    ),
    "REVIEW": ReasonCode(
        code="POLICY_WITHIN_REVIEW_BAND",
        title="Risk inside the review band",
        type=DECISION_RULE,
        features=(),
        description=(
            "The calibrated risk probability fell inside the frozen policy's "
            "REVIEW band, at or above the REVIEW threshold and below the BLOCK "
            "threshold, so the policy returned REVIEW."
        ),
    ),
    "BLOCK": ReasonCode(
        code="POLICY_AT_OR_ABOVE_BLOCK_THRESHOLD",
        title="Risk at or above the block threshold",
        type=DECISION_RULE,
        features=(),
        description=(
            "The calibrated risk probability met or exceeded the frozen "
            "policy's BLOCK threshold, so the policy returned BLOCK."
        ),
    ),
}


REASON_CODE_BY_FEATURE: dict[str, ReasonCode] = {
    feature: reason_code
    for reason_code in MODEL_SIGNAL_REASON_CODES
    for feature in reason_code.features
}


def reason_code_for_feature(feature: str) -> ReasonCode:
    """Look up the model-signal code for a contract feature.

    A feature outside the taxonomy raises rather than falling back to a
    generated name, so an unsupported feature can never be given invented
    semantics.
    """
    if not isinstance(feature, str):
        raise TypeError("feature must be a string.")

    try:
        return REASON_CODE_BY_FEATURE[feature]
    except KeyError:
        raise ValueError(
            f"No reason code is defined for feature {feature!r}. Refusing to "
            "invent semantics for an unsupported feature."
        ) from None


def decision_reason_code(decision_value: str) -> ReasonCode:
    """Look up the decision-rule code for an ALLOW / REVIEW / BLOCK outcome."""
    try:
        return DECISION_REASON_CODES[decision_value]
    except KeyError:
        raise ValueError(
            f"No decision reason code is defined for {decision_value!r}."
        ) from None


def all_reason_codes() -> tuple[ReasonCode, ...]:
    """Every code in the taxonomy, model signals followed by decision rules."""
    return (
        *MODEL_SIGNAL_REASON_CODES,
        *(DECISION_REASON_CODES[name] for name in ("ALLOW", "REVIEW", "BLOCK")),
    )
