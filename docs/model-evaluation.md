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

## Three-model benchmark and robustness results

The final comparison is implemented in `scripts/run_model_benchmark.py`.
It was executed locally against the IEEE-CIS dataset using the same
chronological split and 28-feature contract for all three models.

Reproduce it with:

```bash
PYTHONPATH=. python scripts/run_model_benchmark.py
```

The default bootstrap uses 500 replicates; increase this with
`--bootstrap-replicates` for a longer run.

### Held-out model comparison

Average Precision (AP) is the primary ranking metric because fraud is rare.
ROC-AUC is complementary. Brier score and log loss are descriptive diagnostics
because class weighting can distort raw probability calibration.

| Model | TEST AP | TEST ROC-AUC |
| --- | ---: | ---: |
| Logistic Regression | 0.145789 | 0.780255 |
| Random Forest | 0.426033 | 0.861220 |
| XGBoost | 0.449254 | 0.865318 |

XGBoost achieved the highest overall TEST AP and ROC-AUC on this fixed
chronological holdout. This is evidence for this dataset and split, not proof
that it will generalize best to future payment traffic.

### Paired bootstrap uncertainty

The script uses 500 paired, class-stratified percentile-bootstrap replicates
with seed `20261104`. The same sampled TEST rows are used for each model.

| Model | TEST AP 95% CI | TEST ROC-AUC 95% CI |
| --- | --- | --- |
| Logistic Regression | [0.136244, 0.155149] | [0.770834, 0.789119] |
| Random Forest | [0.408384, 0.443758] | [0.854032, 0.868409] |
| XGBoost | [0.430757, 0.467033] | [0.858336, 0.872366] |

| Paired difference | AP difference 95% CI | ROC-AUC difference 95% CI |
| --- | --- | --- |
| XGBoost minus Logistic Regression | [+0.287405, +0.318510] | [+0.078614, +0.091628] |
| XGBoost minus Random Forest | [+0.015090, +0.030551] | [+0.001328, +0.007241] |

These intervals quantify sampling uncertainty under this empirical holdout.
They do not account for temporal dependence, dataset shift, model retraining
variability, or threshold-selection uncertainty.

### Segment robustness findings

XGBoost had the highest AP in all four chronological TEST periods:

| Period | XGBoost AP | Random Forest AP |
| --- | ---: | ---: |
| 1 | 0.462382 | 0.437336 |
| 2 | 0.398156 | 0.397353 |
| 3 | 0.460684 | 0.441258 |
| 4 | 0.471802 | 0.434679 |

The advantage in period 2 was very small. XGBoost also had higher AP in all
four TRAIN-defined amount bands and all three prior-24-hour card-history
groups.

ProductCD results were not uniformly in its favor: Random Forest narrowly
led AP for category H (0.377120 versus XGBoost 0.376596). XGBoost led AP for
category S (0.473999 versus Random Forest 0.443186), while Random Forest led
ROC-AUC for S (0.787611 versus XGBoost 0.762257). H and S have small fraud
counts, so these differences should not be overinterpreted.

ProductCD W is difficult for both models: XGBoost AP was 0.085566 and Random
Forest AP was 0.073854. These are descriptive slices of one TEST window,
not independent future deployments. Segment-level confidence intervals are
not reported.

### Generated outputs

The benchmark writes files under `reports/model_benchmark/`:

- `model_metrics.csv`: VALIDATION and TEST metrics for all three models.
- `test_segment_metrics.csv`: TEST results by chronological period,
  TRAIN-defined amount band, ProductCD and prior 24-hour card activity.
- `test_bootstrap_intervals.csv`: paired bootstrap intervals and model
  differences for TEST AP and ROC-AUC.
- `run_metadata.json`: split, feature contract, model settings, timings,
  bootstrap seed and execution metadata.

Temporal groups preserve equal timestamps. Amount cut points are calculated
from TRAIN, not TEST. Segment metrics include sample and fraud counts;
metrics requiring both classes are unavailable when a segment lacks one.

### Model implementations

- Logistic Regression: `ml/models/logistic_baseline.py`, executed by
  `scripts/train_logistic_baseline.py`.
- Random Forest: fixed baseline configuration in
  `scripts/run_model_benchmark.py`; it is not a tuned optimum.
- XGBoost: `ml/models/xgboost_model.py`, used by diagnostic training and
  serving-artifact workflows.

All three benchmark pipelines use the same 28-feature contract and
chronological partitions. Preprocessing is fitted on TRAIN only. TEST is
reserved for final reporting and must not be used to select models or tune
parameters.

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
