"""Request and response contracts for the inference API.

Request field names are the locked contract names verbatim — ``TransactionDT``,
``ProductCD``, ``C13`` — rather than API-styled renamings. That is deliberate:
a translation table between the wire format and :data:`MODEL_FEATURES` would be
a second place for the feature contract to be expressed, and therefore a second
place for it to drift. Response field names use the conventional snake_case the
brief specifies, because the response is a report about the scoring rather than
an input to it.

Unknown fields are rejected everywhere. A caller who sends ``product_cd``
instead of ``ProductCD`` must get a 422, not a silently missing feature scored
as an imputed gap.
"""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field, model_validator

from app.config import DEFAULT_TOP_K, MAX_HISTORY_TRANSACTIONS
from ml.data.modeling import MODEL_FEATURES


STRICT = ConfigDict(extra="forbid")


class TransactionPayload(BaseModel):
    """The transaction being scored.

    Required fields are the behavioral entity keys plus the amount and elapsed
    time the feature pipeline consumes. ``addr1``/``addr2`` and the C/D family
    are optional because the locked dataset genuinely has gaps in them
    (65,706 real rows carry no address), and the fitted preprocessing encodes
    those gaps as signal rather than rejecting them.

    Categorical levels are deliberately not restricted to a fixed vocabulary:
    the locked encoder uses ``handle_unknown="ignore"``, so an unseen level is a
    valid input that simply lights no one-hot column.
    """

    model_config = STRICT

    TransactionID: str = Field(min_length=1)
    TransactionDT: int = Field(
        ge=0,
        description="Elapsed seconds, matching the offline TransactionDT scale.",
    )
    TransactionAmt: float = Field(ge=0.0, allow_inf_nan=False)

    card1: int = Field(description="Card entity key for behavioral history.")
    addr1: float | None = Field(default=None, allow_inf_nan=False)
    addr2: float | None = Field(default=None, allow_inf_nan=False)

    ProductCD: str | None = None
    card6: str | None = None
    P_emaildomain: str | None = None
    R_emaildomain: str | None = None

    C1: float | None = Field(default=None, allow_inf_nan=False)
    C2: float | None = Field(default=None, allow_inf_nan=False)
    C11: float | None = Field(default=None, allow_inf_nan=False)
    C13: float | None = Field(default=None, allow_inf_nan=False)
    D2: float | None = Field(default=None, allow_inf_nan=False)
    D3: float | None = Field(default=None, allow_inf_nan=False)
    D4: float | None = Field(default=None, allow_inf_nan=False)
    D8: float | None = Field(default=None, allow_inf_nan=False)
    D10: float | None = Field(default=None, allow_inf_nan=False)
    D11: float | None = Field(default=None, allow_inf_nan=False)
    D15: float | None = Field(default=None, allow_inf_nan=False)


class HistoryTransaction(BaseModel):
    """One prior transaction supplied as behavioral context.

    These five fields are the complete input to
    :func:`ml.features.behavioral.build_behavioral_features`; no categorical or
    C/D value of a *historical* transaction contributes to any feature of the
    current one, so none is requested. Asking for them would imply they matter.
    """

    model_config = STRICT

    TransactionDT: int = Field(ge=0)
    TransactionAmt: float = Field(ge=0.0, allow_inf_nan=False)

    card1: int | None = None
    addr1: float | None = Field(default=None, allow_inf_nan=False)
    addr2: float | None = Field(default=None, allow_inf_nan=False)


class RiskScoreRequest(BaseModel):
    """A transaction plus the strictly-prior context it is scored against."""

    model_config = STRICT

    transaction: TransactionPayload
    history: list[HistoryTransaction] = Field(
        default_factory=list,
        max_length=MAX_HISTORY_TRANSACTIONS,
        description=(
            "Prior transactions for the same card1 / (addr1, addr2) entities. "
            "Only rows inside the trailing 24 hours can affect any feature."
        ),
    )
    top_k: int = Field(default=DEFAULT_TOP_K, ge=1, le=len(MODEL_FEATURES))

    @model_validator(mode="after")
    def _reject_history_that_is_not_strictly_prior(self) -> RiskScoreRequest:
        """Enforce the locked point-in-time rule at the API boundary.

        The offline feature pipeline only ever reads rows with
        ``TransactionDT < current TransactionDT``, and same-timestamp rows cannot
        see each other. A history row that violates this is a malformed request,
        not something to filter out quietly: silently dropping it would hide a
        caller bug and still return a confident-looking score.
        """
        current = self.transaction.TransactionDT

        for position, entry in enumerate(self.history):
            if entry.TransactionDT >= current:
                raise ValueError(
                    f"history[{position}].TransactionDT={entry.TransactionDT} "
                    f"must be strictly less than transaction.TransactionDT="
                    f"{current}; historical context may not include the current "
                    "timestamp or anything after it."
                )

        return self


class ReasonCodeOut(BaseModel):
    """One model-signal reason code, reusing the Phase 5 structures verbatim."""

    model_config = STRICT

    code: str
    title: str
    type: str
    description: str
    feature: str
    contribution: float
    direction: str
    n_transformed_columns: int


class DecisionReasonCodeOut(BaseModel):
    """The single decision-rule code, reported separately from model signals."""

    model_config = STRICT

    code: str
    title: str
    type: str
    description: str


class ModelDetailOut(BaseModel):
    model_config = STRICT

    base_value: float
    raw_margin: float
    raw_model_probability: float
    attribution_space: str
    attribution_space_note: str
    total_risk_increasing: float
    total_risk_decreasing: float


class PolicyDetailOut(BaseModel):
    model_config = STRICT

    review_threshold: float
    block_threshold: float


class RiskScoreResponse(BaseModel):
    """The scoring result.

    ``reason_codes`` holds model signals only and ``decision_reason_code`` holds
    the decision rule. They are never merged into one ranked list: the strongest
    model signal on a transaction says nothing on its own about which band the
    frozen policy put it in. ``risk_decreasing_signals`` is reported so features
    that pushed risk *down* are visible without being mistaken for fraud
    evidence.

    ``risk_probability`` is the full-precision calibrated value that
    ``DecisionPolicy.decide`` actually consumed. It is not rounded; the rounded
    rendering exists only inside ``decision_explanation``.
    """

    model_config = STRICT

    transaction_id: str
    risk_probability: float = Field(ge=0.0, le=1.0)
    decision: str
    reason_codes: list[ReasonCodeOut]
    risk_decreasing_signals: list[ReasonCodeOut]
    decision_reason_code: DecisionReasonCodeOut
    decision_explanation: str
    model_detail: ModelDetailOut
    policy_detail: PolicyDetailOut
    model_version: str
    feature_version: str
    policy_version: str
    feature_count: int
    history_transactions_used: int


class HealthResponse(BaseModel):
    """Liveness, artifact identity, and audit-store readiness.

    ``status`` describes the model artifact, exactly as it did before the audit
    store existed: ``ok`` or ``unavailable``. ``database`` is reported separately
    as ``ready`` or ``unavailable``, because the two can disagree — a service
    whose artifact is fine but whose audit store is down cannot return a decision,
    and a caller needs to be able to tell those two failures apart.
    """

    model_config = STRICT

    status: str
    model_version: str | None = None
    feature_version: str | None = None
    policy_version: str | None = None
    artifact: dict | None = None
    database: str | None = None


class ErrorOut(BaseModel):
    """A structured error. Never carries a stack trace or a filesystem path."""

    model_config = STRICT

    code: str
    message: str


class ErrorResponse(BaseModel):
    model_config = STRICT

    error: ErrorOut
