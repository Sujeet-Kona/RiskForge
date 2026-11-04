# Model evaluation and benchmark results

## Scope

RiskForge is evaluated as an offline fraud-risk research project using the
IEEE-CIS Fraud Detection benchmark. The dataset is not committed to the
repository. Reported metrics describe this dataset and split only; they do not
establish expected performance on live payment traffic.

The modeling frame contains 590,540 transactions and 28 model features:

| Partition | Rows | Fraud cases |
| --- | ---: | ---: |
| TRAIN | 414,446 | — |
| VALIDATION | 88,013 | 3,010 |
| TEST | 88,081 | 3,065 |

The split is chronological, and equal transaction timestamps are kept together.
Preprocessing and XGBoost are fitted on TRAIN. Calibration method selection and
policy search use VALIDATION. TEST is reserved for final reporting.

## Model implementations

The repository contains:

- Logistic Regression baseline: `ml/models/logistic_baseline.py`,
  executed by `scripts/train_logistic_baseline.py`.
- XGBoost model: `ml/models/xgboost_model.py`, executed for diagnostic
  training by `scripts/train_xgboost.py` and rebuilt for serving by
  `scripts/build_serving_artifact.py`.

The Logistic Regression baseline uses the same 28-feature contract and
chronological split, with median imputation, numeric standardization,
categorical preprocessing and class-balanced fitting.

A numeric Logistic Regression-versus-XGBoost comparison is intentionally not
published here because a verified baseline run output has not been committed.
Do not claim that XGBoost outperformed the baseline without recording the
validation metrics from both runs under the same data and evaluation protocol.
Use VALIDATION for model selection; do not use TEST results to choose a model
or tune hyperparameters.

## Calibration method selection

Calibration candidates were compared using expanding-window forward-chaining
out-of-fold predictions on VALIDATION. The first chronological fold is a
warm-up period and is excluded from scored metrics.

- VALIDATION rows: 88,013
- OOF-scored rows: 70,334
- Warm-up rows excluded: 17,679

| Method | Brier | Log loss | ECE | PR-AUC | ROC-AUC |
| --- | ---: | ---: | ---: | ---: | ---: |
| Sigmoid | 0.025368 | 0.104336 | 0.010320 | 0.426233 | 0.876543 |
| Isotonic | 0.024703 | 0.108550 | 0.004678 | 0.422378 | 0.874798 |

Brier score was the predeclared selection criterion, so isotonic was selected.
The trade-off matters: sigmoid had better OOF log loss and ranking metrics,
while isotonic had better OOF Brier score and ECE. The result should not be
summarized as isotonic being universally superior.

## Held-out TEST metrics

The selected isotonic calibrator was refitted on all VALIDATION rows and then
applied once to TEST.

| Metric | Raw XGBoost | Isotonic calibrated |
| --- | ---: | ---: |
| PR-AUC | 0.449254 | 0.438919 |
| ROC-AUC | 0.865318 | 0.864850 |
| Brier | Not reported | 0.024204 |
| Log loss | Not reported | 0.102851 |
| ECE | Not reported | 0.003553 |

Calibration changes probability quality, not just ranking. The calibrated
model's slightly lower ranking metrics are not hidden; calibration was selected
on its stated probability-scoring criterion.

## Decision policy: candidate is not the serving policy

The active API policy remains:

- Version: `policy-001`
- Review threshold: `0.075`
- Block threshold: `0.110`

The corrected forward-chaining policy search produces a separate experimental
candidate. Under the project's synthetic cost assumptions
(`C_FN=100`, `C_FP=10`, `C_REVIEW=5`, assumed review fraud capture rate
`e=0.70`), the selected candidate was `T_REVIEW=0.050`,
`T_BLOCK=0.110`. Its forward-OOF validation cost was 142,250 units; the
runner-up cost was 142,285, a margin of only 35 units over 70,334 scored rows.

These values are not measured payment-network economics. The small difference
between the top candidates, sensitivity to review-effectiveness assumptions,
and the difference between forward-fold calibrators and the final all-
VALIDATION refit make automatic promotion unjustified. The candidate is not
the active API policy and must not be described as deployed or production-
validated.

## Reproduction

With the IEEE-CIS source files in `data/raw/ieee_cis/` and the behavioral
artifact generated locally:

```bash
PYTHONPATH=. python scripts/evaluate_calibration.py
PYTHONPATH=. python scripts/select_decision_policy.py
PYTHONPATH=. python scripts/build_serving_artifact.py
```

The Logistic Regression diagnostic is:

```bash
PYTHONPATH=. python scripts/train_logistic_baseline.py
```

The scripts require the local dataset and are not part of ordinary CI. Generated
datasets and model artifacts are gitignored and should not be committed.

## Interpretation limits

- IEEE-CIS is a public benchmark, not a live payment network.
- Synthetic policy costs are assumptions, not business measurements.
- Caller-supplied history is not independently authenticated or guaranteed
  complete by the API.
- Audit rows do not retain all prior-history records needed to replay a score.
- The reported TEST metrics are one chronological holdout, not confidence
  intervals or evidence of stable performance across future time periods.
- No production deployment, online feature store, drift monitoring or human
  review operation is claimed.
