# Model evaluation

## Scope and split

RiskForge was evaluated on the IEEE-CIS Fraud Detection dataset (590,540 transactions) using 28 model features. The chronological split keeps equal `TransactionDT` values together.

| Partition | Rows | Fraud cases |
| --- | ---: | ---: |
| TRAIN | 414,446 | — |
| VALIDATION | 88,013 | 3,010 |
| TEST | 88,081 | 3,065 |

Preprocessing and estimators are fitted on TRAIN. Model comparison, calibration selection, and policy research use VALIDATION. TEST is reserved for final reporting.

## Model comparison

All three estimators use the same split and feature contract. The reported TEST metrics are from the final benchmark.

| Model | Average Precision | ROC-AUC | Brier score | Log loss |
| --- | ---: | ---: | ---: | ---: |
| Logistic Regression | 0.145789 | 0.780255 | 0.178173 | 0.538330 |
| Random Forest | 0.426033 | 0.861220 | 0.077925 | 0.296063 |
| XGBoost | **0.449254** | **0.865318** | 0.101228 | 0.344462 |

Average Precision is emphasized because fraud is the minority class. ROC-AUC is included as a complementary ranking metric. Brier score and log loss measure probability quality and should not be interpreted as ranking metrics.

A paired, class-stratified bootstrap with 500 replicates estimated the XGBoost-minus-Random-Forest difference as:

| Metric difference | 95% interval |
| --- | ---: |
| Average Precision | +0.015090 to +0.030551 |
| ROC-AUC | +0.001328 to +0.007241 |

XGBoost led Average Precision in all four time slices, four TRAIN-defined amount bands, and three card-history groups. It did not lead every ProductCD slice or every metric. These subgroup results are descriptive, not independent validation datasets.

The bootstrap resamples transactions within the observed TEST distribution. It does not account for temporal dependence, dataset shift, retraining variability, or threshold-selection uncertainty.

## Calibration

Calibration methods were compared with expanding-window, forward-chaining out-of-fold predictions on VALIDATION. The first 17,679 rows were used as warm-up and excluded from scored OOF metrics; 70,334 rows were scored.

| Method | OOF Brier score | OOF log loss |
| --- | ---: | ---: |
| Sigmoid | 0.025368 | Better than isotonic |
| Isotonic | **0.024703** | Worse than sigmoid |

Isotonic was selected because Brier score was the stated selection criterion. It did not win every metric.

On TEST, raw versus calibrated XGBoost results were:

| Metric | Raw | Isotonic |
| --- | ---: | ---: |
| Average Precision | 0.449254 | 0.438919 |
| ROC-AUC | 0.865318 | 0.864850 |
| Brier score | — | 0.024204 |
| Log loss | — | 0.102851 |
| ECE | — | 0.003553 |

Calibration targets probability reliability, not ranking. Isotonic's stepwise mapping can create tied probabilities and slightly change ranking metrics.

## Decision policy

The serving API uses the frozen `policy-001` thresholds:

- `ALLOW`: probability below 0.075
- `REVIEW`: probability from 0.075 up to (but not including) 0.110
- `BLOCK`: probability at or above 0.110

The policy research uses synthetic false-positive, false-negative, and review assumptions. These are not measured payment-network costs.

A forward-chaining validation search produced an experimental candidate with a 0.050 review threshold and 0.110 block threshold. It has not replaced `policy-001`; the active serving configuration remains unchanged.

## Reproduction

With the IEEE-CIS files under `data/raw/ieee_cis/`, run:

```bash
PYTHONPATH=. python scripts/run_model_benchmark.py
PYTHONPATH=. python scripts/evaluate_calibration.py
PYTHONPATH=. python scripts/select_decision_policy.py
```

The benchmark writes:

- `reports/model_benchmark/model_metrics.csv`
- `reports/model_benchmark/test_segment_metrics.csv`
- `reports/model_benchmark/test_bootstrap_intervals.csv`
- `reports/model_benchmark/run_metadata.json`

## Interpretation limits

These measurements describe one chronological split of IEEE-CIS. They do not establish performance on current or future payment traffic. The bootstrap intervals quantify resampling uncertainty within this holdout, not production uncertainty. The project does not implement live drift monitoring or continuous retraining.
