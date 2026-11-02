# RiskForge explainability (Phase 5)

Deterministic, per-transaction explanations for the locked XGBoost fraud model
and the frozen `policy-001` decision rule.

Run the real-data demonstration with:

```bash
PYTHONPATH=. python scripts/explain_transactions.py
```

---

## 1. What the explanation layer means

It answers two questions about one transaction:

- **Model explanation** — which of the 28 contract features moved the model's
  fraud-risk score, in which direction, and by how much.
- **Decision explanation** — why the resulting calibrated risk probability
  landed in the ALLOW, REVIEW or BLOCK band under `policy-001`.

Both answers are pure functions of the frozen artifacts (the fitted booster,
the fitted preprocessor, the `PolicyConfig`) and the input row. Same model +
same row → same ranked reason codes, every time. There is no randomness, no
sampling, no background dataset, no LLM, no generated prose and no network
call. The wording comes from static templates and a static reason-code
registry.

## 2. What it does NOT mean

The layer describes **model behaviour, not the world**. It makes no causal
claim.

Say: *"risk driver"*, *"model signal"*, *"contributing feature"*.
Do not say: *"this feature caused fraud"*, *"this transaction is fraudulent
because…"*, *"this feature proves fraud"*.

Concretely:

- A positive contribution means the model's score was higher with this feature
  taking the value it took. It does not mean the feature produced fraud.
- Correlation with the training labels is what the trees encode; nothing here
  distinguishes a causal driver from a proxy.
- Reason codes are not evidence, and they are not a verdict. The verdict is the
  decision band, which is a threshold comparison and nothing more.
- The layer never reads a label. It cannot tell you whether a transaction
  actually was fraudulent.

## 3. Raw XGBoost score vs calibrated probability

These are two different numbers and the explanation keeps them apart.

| Quantity | Produced by | Space | Used for |
| --- | --- | --- | --- |
| raw margin | XGBoost booster | log-odds | TreeSHAP attribution |
| raw model probability | `sigmoid(raw margin)` | [0, 1] | reported alongside, not acted on |
| calibrated risk probability | locked isotonic calibrator (fitted on all of VALIDATION) | [0, 1] | the decision policy |

Contributions are computed in the **raw margin (log-odds) space**, because that
is the quantity the booster actually decomposes:

```
base_value + sum(all 28 feature contributions) == raw_margin
sigmoid(raw_margin)                            == raw_model_probability
```

The smoke test asserts this closure on every explained row
(`FLOAT32_ADDITIVITY_TOLERANCE = 1e-4`; the measured residual on real IEEE-CIS
rows was exactly `0.0`).

Isotonic calibration is **monotonic**, so the *sign* of a contribution still
tells you whether that feature pushed the deployed risk probability up or down.
The *magnitude* does not carry over — calibration rescales the margin
non-linearly. The MID-BAND REVIEW row printed in §11 is a clear case: a raw
model probability of `0.656952` becomes a calibrated risk probability of
`0.090909`, roughly a 7× rescale, and the row lands in REVIEW rather than
BLOCK.

**Reason codes therefore describe model-level risk signals. They do not explain
the calibration mapping, and the ranking of reason codes must not be read as a
ranking of calibrated-probability movement.**

## 4. How model contributions are calculated

**Method: TreeSHAP with path-dependent feature perturbation, evaluated on the
already-fitted booster.**

```python
shap.TreeExplainer(
    classifier,
    feature_perturbation="tree_path_dependent",
    model_output="raw",
)
```

Why this and not an approximation:

- Path-dependent TreeSHAP needs **no background dataset**, so there is no
  sampling choice that could make the output non-deterministic or
  data-dependent.
- It was verified to be **bit-identical** to XGBoost's own native
  `booster.predict(DMatrix(Xt), pred_contribs=True)`, including the bias column
  (`expected_value == np.float32(0.034338623)` on the locked model). This is
  asserted in `tests/test_model_explanation.py`, so a future SHAP or XGBoost
  version that diverges fails the suite rather than silently changing
  explanations.
- No surrogate model, no second model, no retraining. The explainer is
  read-only with respect to the locked artifact — asserted by comparing the
  booster's serialized form before and after an explanation.

### The 162 → 28 attribution map

The booster never sees the 28 contract features. The fitted §14
`ColumnTransformer` expands them into **162 columns**:

```
 24  numeric values
  7  numeric missing indicators (D2, D3, D4, D8, D10, D11, D15)
131  one-hot levels (ProductCD 5, card6 5, P_emaildomain 60, R_emaildomain 61)
```

`explain/feature_attribution.py` maps every one of those 162 columns back to
the contract feature it came from, and a feature's contribution is the **sum of
its columns**. Because SHAP values are exactly additive over the columns the
model consumes, this sum is an exact attribution, not an approximation — a
feature's imputed value, its missing indicator and all of its one-hot levels
are one signal to the reader.

The mapping is derived **structurally from the fitted estimators**, not by
parsing feature-name strings:

- numeric block: `list(columns)` followed by `imputer.indicator_.features_`;
- categorical block: contiguous slices of `len(encoder.categories_[i])` per
  input feature.

That slicing is only valid because the locked encoder uses `drop=None`,
`min_frequency=None` and `max_categories=None`. All three are asserted; any of
them being set makes the map **raise** rather than guess. The finished column
order is also cross-checked against `preprocessor.get_feature_names_out()`, and
the map is rejected if any column traces to a feature outside the contract or
if any contract feature has no column. A future preprocessing change therefore
fails loudly instead of quietly producing a wrong explanation.

## 5. How top-N ranking works

Default `top_k = 3`, configurable per call. `top_k` must be a positive `int`;
`0`, negatives, floats, strings, `None` and `bool` are all rejected.

The ranking is **sign-aware, not magnitude-aware**:

1. Keep only contributions `> 0`. Sort by `(-contribution, feature)`. Take the
   first `top_k` → `risk_increasing`.
2. Keep only contributions `< 0`. Sort by `(contribution, feature)`. Take the
   first `top_k` → `risk_decreasing`.
3. `top_reason_codes` is the `risk_increasing` list only.

The feature name is the secondary sort key, so ties break deterministically.

**Absolute-value ranking is deliberately not used.** Returning the three
largest `|contribution|` values would present a strongly risk-*reducing*
feature as fraud evidence. The real ALLOW row in the smoke test shows why: its
largest magnitude was `CARD_TRANSACTION_VELOCITY_SIGNAL` at `-0.797339`, while
its strongest *upward* driver was `P_EMAIL_DOMAIN_SIGNAL` at only `+0.447131`.
Magnitude ranking would have led with a feature that was pushing the score
down.

No magnitude floor is applied either — that would be an invented threshold
manufacturing an explanation. Materiality is exposed instead as
`total_risk_increasing` and `total_risk_decreasing`, so a reader can see that a
`+0.447131` top signal on a row whose total downward push is `-5.188420` is not
a strong risk story — which is consistent with the policy allowing it.

`risk_decreasing` is reported, clearly labelled, and never merged into the
reason-code list.

## 6. How undocumented C/D features are handled

The project audit established that local dataset documentation does not
reliably define the semantics of the IEEE-CIS `C` and `D` columns. The
explanation layer therefore **invents nothing**.

Each of the 11 C/D contract features gets a neutral code — `MODEL_SIGNAL_C1`,
`MODEL_SIGNAL_C2`, `MODEL_SIGNAL_C11`, `MODEL_SIGNAL_C13`, `MODEL_SIGNAL_D2`,
`MODEL_SIGNAL_D3`, `MODEL_SIGNAL_D4`, `MODEL_SIGNAL_D8`, `MODEL_SIGNAL_D10`,
`MODEL_SIGNAL_D11`, `MODEL_SIGNAL_D15` — with the title `Model signal from <F>`
and a description that states the column is *"a numeric benchmark feature from
the IEEE-CIS dataset with undocumented upstream semantics"* and that *"no
business meaning is claimed for it."*

Codes such as `UNUSUAL_PAYMENT_DEVICE` are never produced from a C/D feature.
Tests assert that no C/D description contains `device`, `browser`, `country`,
`velocity` or `address`.

Features outside the contract (`card4`, `id_01`, `V1`, `D5`, …) have **no**
code. `reason_code_for_feature` raises rather than falling back to a generic
`UNKNOWN` entry — there is no wildcard key in the registry, and this is
asserted.

## 7. Reason-code taxonomy

Every code carries: a stable UPPER_SNAKE `code`, a human-readable `title`, the
`features` involved, a `type` of `model_signal` or `decision_rule`, and a
short `description`. No causal language appears anywhere in the registry
(asserted against a forbidden-substring list).

**24 model-signal codes covering all 28 contract features exactly**, with no
feature claimed by two codes.

Correlated members of a family are grouped into one code, because they split
SHAP credit between them and a family-level code is the more stable description
of the same underlying signal:

| Code | Features |
| --- | --- |
| `CARD_TRANSACTION_VELOCITY_SIGNAL` | `card_txn_count_1h`, `card_txn_count_24h` |
| `CARD_AMOUNT_VOLUME_SIGNAL` | `card_amount_sum_24h` |
| `CARD_HISTORY_SIGNAL` | `card_has_history_1h`, `card_has_history_24h` |
| `ADDRESS_TRANSACTION_VELOCITY_SIGNAL` | `address_txn_count_1h`, `address_txn_count_24h` |
| `ADDRESS_AMOUNT_VOLUME_SIGNAL` | `address_amount_sum_24h` |
| `ADDRESS_HISTORY_SIGNAL` | `address_has_history_1h` |
| `TRANSACTION_AMOUNT_SIGNAL` | `TransactionAmt`, `amount_log1p` |
| `TRANSACTION_HOUR_SIGNAL` | `transaction_hour_bucket` |
| `TRANSACTION_WEEKDAY_SIGNAL` | `transaction_day_mod7` |
| `PRODUCT_CD_SIGNAL` | `ProductCD` |
| `CARD6_SIGNAL` | `card6` |
| `P_EMAIL_DOMAIN_SIGNAL` | `P_emaildomain` |
| `R_EMAIL_DOMAIN_SIGNAL` | `R_emaildomain` |
| `MODEL_SIGNAL_C1` … `MODEL_SIGNAL_D15` | one C/D feature each (11 codes) |

**3 decision-rule codes**, keyed by band, with an empty `features` tuple —
a decision rule is about thresholds, not features:

| Decision | Code |
| --- | --- |
| ALLOW | `POLICY_BELOW_REVIEW_THRESHOLD` |
| REVIEW | `POLICY_WITHIN_REVIEW_BAND` |
| BLOCK | `POLICY_AT_OR_ABOVE_BLOCK_THRESHOLD` |

## 8. How decision explanations differ from model explanations

They are separate objects, computed by separate modules, and never merged into
one ranked list.

`explain/decision_explanation.py` calls `DecisionPolicy.decide(...)` for the
band and reads `PolicyConfig` for the thresholds, so the wording **cannot drift
from the logic that actually ran**. Nothing re-implements the comparison and no
threshold value is hardcoded in the explanation layer.

Templates (filled with the live frozen values `T_REVIEW = 0.075`,
`T_BLOCK = 0.110`):

```
ALLOW : Risk probability {p} is below the REVIEW threshold {T_REVIEW}.
REVIEW: Risk probability {p} falls between the REVIEW threshold {T_REVIEW} and BLOCK threshold {T_BLOCK}.
BLOCK : Risk probability {p} meets or exceeds the BLOCK threshold {T_BLOCK}.
```

Probabilities render at six decimals, not three: with `T_REVIEW = 0.075` a
three-decimal rendering of `0.0751` would print as `0.075` and appear to
contradict the band the policy chose. Digits are added beyond six only when the
rounded text would land on a threshold the probability is not actually sitting
on, so a value like `0.0749999` prints in full rather than as a misleading
`0.075000`.

The model explanation says *what moved the score*. The decision explanation
says *why that score became this action*. The strongest model signal on a
transaction says nothing on its own about which band the policy put it in.

## 9. Missing values

Nothing is re-implemented. The row is passed through the **fitted** §14
preprocessor before attribution, so it is handled exactly as it was at scoring
time:

- **Numeric gaps** (the C/D family): TRAIN median fill plus the explicit
  missing indicator. Both columns belong to the same contract feature, so the
  feature's contribution includes the signal from its own absence.
- **Categorical gaps**: mapped to the reserved `__MISSING__` level and one-hot
  encoded alongside the observed levels. Absence is signal in this dataset.
- **A level the encoder never saw**: `handle_unknown="ignore"` lights no
  one-hot column, so that feature contributes zero. Inherited from the locked
  preprocessing, not chosen here.
- **Behavioral features** are computed for every row, so a NaN there is a
  pipeline failure and is rejected upstream by `build_modeling_frame`.

## 10. Architecture

```
ml/        model, preprocessing, calibration
explain/   feature_attribution, reason_codes, model_explanation,
           decision_explanation, explanation
policy/    costs, decision, evaluation
scripts/   orchestration
```

Dependency direction is `explain → ml` (contract + preprocessing metadata) and
`explain → policy` (the frozen decision rule). `policy/` imports **nothing**
from `explain/`, `ml/`, `xgboost`, `sklearn`, `scipy` or `shap`. These
boundaries are enforced by AST-based tests, not by convention: substring scans
were tried first and tripped on the modules' own prose describing the absence
of exactly what was being tested for.

| File | Purpose |
| --- | --- |
| `explain/feature_attribution.py` | the structural 162 → 28 column map and exact contribution grouping |
| `explain/reason_codes.py` | the static taxonomy; pure data, no model or dataset dependency |
| `explain/model_explanation.py` | TreeSHAP attribution and sign-aware top-k ranking |
| `explain/decision_explanation.py` | threshold wording driven by the live `PolicyConfig` |
| `explain/explanation.py` | composes both halves for one transaction, keeping them separate |
| `scripts/explain_transactions.py` | real-data demonstration on VALIDATION |

## 11. Example output

From `scripts/explain_transactions.py` on real IEEE-CIS VALIDATION rows, with
the locked model and frozen `policy-001`. Selection used calibrated
probabilities only; no label was read.

```
--------------------------------------------------------------------------------
2. MID-BAND REVIEW
--------------------------------------------------------------------------------
transaction_id             : 3401505
risk_probability           : 0.090909  (calibrated)
raw_model_probability      : 0.656952  (XGBoost, pre-calibration)
raw_margin                 : +0.649742  (log-odds)
base_value                 : +0.034339
additivity_residual        : 0.000e+00  (tolerance 1e-04)
decision                   : REVIEW
decision_explanation       : Risk probability 0.090909 falls between the REVIEW threshold 0.075000 and BLOCK threshold 0.110000.
decision_reason_code       : POLICY_WITHIN_REVIEW_BAND
top reason codes (k=3)    : MODEL_SIGNAL_D3, MODEL_SIGNAL_C11, MODEL_SIGNAL_D15

  risk-increasing model signals (contributions in log-odds):
    MODEL_SIGNAL_D3                        +0.680639  D3                     cols=2
      Model signal from D3
    MODEL_SIGNAL_C11                       +0.263049  C11                    cols=1
      Model signal from C11
    MODEL_SIGNAL_D15                       +0.216755  D15                    cols=2
      Model signal from D15

  risk-decreasing model signals (reported, never as fraud evidence):
    CARD6_SIGNAL                           -0.235375  card6                  cols=5
    MODEL_SIGNAL_C1                        -0.188475  C1                     cols=1
    MODEL_SIGNAL_C13                       -0.130370  C13                    cols=1

  total risk-increasing     : +1.490573
  total risk-decreasing     : -0.875170
```

Note what this row demonstrates: `D3` is the leading reason code, and the
documentation for `D3` claims no business meaning for it. The calibrated
probability (`0.090909`) and the raw model probability (`0.656952`) differ by
an order of magnitude, and the decision sentence quotes the calibrated one —
the number the policy actually compared against its thresholds.

## 12. Test-partition discipline

- The TEST partition is **never** used for reason-code design, reason-code
  thresholds, feature selection, policy selection or explanation tuning.
- `scripts/explain_transactions.py` selects its three examples from
  **VALIDATION** rows, using calibrated risk probabilities and the frozen
  thresholds only. It prints the TEST row count to make the omission explicit
  and never scores or explains a TEST row.
- No explanation module references a label. `tests/test_explainability_discipline.py`
  AST-scans `explain/*.py` for label identifiers (`isFraud`, `TARGET_COLUMN`,
  `y_test`, `y_true`, …) and for imports of the split, evaluation and
  calibration layers, and scans the explainability tests for any access to the
  locked TEST partition.
- The locked TEST evaluation from Phases 3 and 4 was not modified.
- Unit tests run on a small synthetic frame built through the *real*
  `build_modeling_frame` and *real* `build_feature_preprocessor`, so the suite
  needs no dataset. That fixture uses deliberately different XGBoost
  hyperparameters — it exists to give the tests a real fitted booster to
  decompose, and is **not** a surrogate for the production model. Requirement
  19 (the explanation corresponds to the actual fitted model) is tested
  separately, against the native booster's own `pred_contribs`.

## 13. Known limitations

1. **Contribution magnitudes live in log-odds, not probability.** They cannot
   be added to, or compared against, the calibrated risk probability. Only
   their sign survives calibration.
2. **Grouping hides within-family detail.** `CARD_TRANSACTION_VELOCITY_SIGNAL`
   merges `card_txn_count_1h` and `card_txn_count_24h`. The per-column values
   are not exposed, because correlated members split SHAP credit and a
   family-level code is the more stable description.
3. **C/D codes carry no business meaning by design.** A reader who needs
   semantics for `MODEL_SIGNAL_D3` will not get them from this layer, and
   should not.
4. **Attribution is local, not global.** This explains one transaction. It says
   nothing about which features matter to the model overall.
5. **The 162 → 28 map is coupled to the locked preprocessing.** If
   `OneHotEncoder` gains `drop`, `min_frequency` or `max_categories`, or the
   `ColumnTransformer` gains a third block, the map raises instead of guessing.
   That is intentional, but it means a preprocessing change requires a
   matching change here.
