# Model evaluation

## Scope and split

RiskForge was evaluated on the IEEE-CIS Fraud Detection dataset (590,540 transactions) using 28 model features. The chronological split keeps equal `TransactionDT` values together.

| Partition | Rows | Fraud cases |
| --- | ---: | ---: |
| TRAIN | 414,446 | — |
| VALIDATION | 88,013 | 3,010 |
| TEST | 88,081 | 3,065 |

Preprocessing and estimators are fitted on TRAIN. Model comparison, calibration selection, and policy research use VALIDATION. TEST is reserved for final reporting. VALIDATION is therefore a **selection** partition, not an independent estimate — see [Reuse of the VALIDATION partition](#reuse-of-the-validation-partition).

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
| Brier score | 0.101228 | 0.024204 |
| Log loss | 0.344462 | 0.102851 |
| ECE (10 bins) | 0.222200 | 0.003553 |

Calibration targets probability reliability, not ranking. Isotonic's stepwise mapping can create tied probabilities and slightly change ranking metrics.

## Decision policy

The serving API uses the frozen `policy-001` thresholds:

- `ALLOW`: probability below 0.075
- `REVIEW`: probability from 0.075 up to (but not including) 0.110
- `BLOCK`: probability at or above 0.110

The policy research uses synthetic false-positive, false-negative, and review assumptions. These are not measured payment-network costs.

A forward-chaining validation search produced an experimental candidate with a 0.050 review threshold and 0.110 block threshold. It has not replaced `policy-001`; the active serving configuration remains unchanged.

### Provenance of the frozen thresholds

`policy-001` is immutable. This section records what the repository can and
cannot establish about where its two thresholds came from, so that the gap is
explicit rather than papered over.

**Known and verifiable.**

- The frozen values are 0.075 and 0.110, written in exactly one place:
  `policy/locked.py:22-23`.
- `policy/locked.py` was introduced already-locked in commit `34aceca`
  ("Add FastAPI inference service"), the same commit that added the serving
  layer, the artifact loader, and the parity script. There is no earlier commit
  that derives, searches, or motivates 0.075.
- The values are enforced at runtime: `app/service.py:196-235` refuses to start
  against an artifact whose thresholds differ from `policy.locked`, and
  `tests/test_serving_discipline.py:476-481` forbids the literals `0.075` and
  `0.110` from appearing anywhere in `app/` or `db/`.
- Boundary behaviour is exact and tested: ALLOW for `p < 0.075`, REVIEW for
  `0.075 <= p < 0.110`, BLOCK for `p >= 0.110`.

**Not established by this repository.**

- The original rationale for choosing 0.075 as the REVIEW threshold is **not
  recorded anywhere** — not in a commit message, a report, a script output, or
  a document. It cannot be recovered from the code, and none is claimed here.
- The 0.075 value is **not** the output of the cost-sensitive search described
  below. That search selects 0.050.
- No artifact links `scripts/select_decision_policy.py` to `policy/locked.py`.
  The script does not import it, does not assert that its winner matches the
  deployed thresholds, and **writes nothing to disk** — it is print-only, so
  the search leaves no committed evidence.

**The separate cost-search experiment.**

`scripts/select_decision_policy.py` performs an exhaustive (not sampled) search
over every feasible `(T_REVIEW, T_BLOCK)` pair with `T_REVIEW < T_BLOCK`, drawn
from the empirical percentiles of forward-chaining out-of-fold VALIDATION
scores plus a fixed interpretable grid. It uses the synthetic cost model in
`policy/costs.py:88-93` — `C_FN=100`, `C_FP=10`, `C_REVIEW=5`, review fraud
capture `e=0.70` — which that module states explicitly are project assumptions
and not industry facts. TEST labels are read only after the search completes,
and the script asserts the thresholds did not change (`:481-483`).

Its selected candidate is `0.050 / 0.110`. The repository records only that
this candidate "has not replaced `policy-001`". **No reason for that decision
is recorded, and none is inferred here.**

Adopting the search winner would change deployed behaviour, so it is treated as
a separate future experiment requiring its own evidence, not as a correction to
make silently.

## Reproduction

With the IEEE-CIS files under `data/raw/ieee_cis/`, run:

```bash
PYTHONPATH=. python scripts/run_model_benchmark.py
PYTHONPATH=. python scripts/evaluate_calibration.py
PYTHONPATH=. python scripts/select_decision_policy.py
```

The calibration evaluation also writes
`reports/calibration/evaluation_metrics.json`. It records the raw and
calibrated TEST Brier score, ECE, ranking metrics, the forward-chaining OOF
metrics, and the exact relative Brier/ECE reductions computed from the same
run. Use this generated artifact to verify percentage-improvement claims; do
not infer the raw ECE baseline from the calibrated-only summary table.

**This file is generated on demand and is not committed.** A clean clone does
not contain `reports/calibration/`; regenerating it requires the gitignored
IEEE-CIS inputs and refits the model. The raw-versus-calibrated TEST table in
[Calibration](#calibration) above is therefore the committed historical record
of those five numbers, and it is what the serving artifact was built against.

The benchmark writes:

- `reports/model_benchmark/model_metrics.csv`
- `reports/model_benchmark/test_segment_metrics.csv`
- `reports/model_benchmark/test_bootstrap_intervals.csv`
- `reports/model_benchmark/run_metadata.json`

New benchmark runs now write source-input fingerprints (relative path, byte
size, and SHA-256), Python/platform details, key numerical-library versions,
and the Git commit (when run inside a Git checkout) into
`run_metadata.json`. The committed metadata predates this addition and will
gain those fields when the benchmark is next rerun with the local IEEE-CIS
inputs. This does not package or redistribute the dataset.

## Reuse of the VALIDATION partition

VALIDATION was consulted repeatedly during development, and each consultation
made a decision that changed what was subsequently measured on it:

| Decision taken on VALIDATION | Where | Effect on VALIDATION scores |
| --- | --- | --- |
| Drop `D5` from the 28-feature contract | `ml/data/modeling.py:35-37` | feature set chosen to raise VALIDATION PR-AUC |
| Choose isotonic over sigmoid calibration | `scripts/evaluate_calibration.py`, forward-chaining OOF | calibrator chosen to lower OOF Brier score |
| Search the review/block threshold grid | `scripts/select_decision_policy.py` | thresholds chosen to minimize VALIDATION expected cost |
| Rank XGBoost against Random Forest and Logistic Regression | `scripts/run_model_benchmark.py` | estimator chosen on the same partition |

The consequence is that **VALIDATION PR-AUC is not an independent estimate of
generalization.** It is a score on the data that selected the features, the
calibrator, the estimator and the thresholds. Repeated selection on one finite
partition biases that score upward, and nothing in this repository quantifies
the bias: there is no nested cross-validation and no separate selection
partition, so the optimistic component cannot be separated from the real one.

The clearest instance is the `D5` removal, which was accepted because VALIDATION
PR-AUC moved from 0.471864 to 0.474352 (+0.002488). Some of that gain is real
and some is selection noise on the same 88,013 rows. The repository records the
decision and its measured effect; it does not claim to know the split between
the two.

TEST is the only partition that no selection decision consulted. Its labels are
withheld until the final reporting step (`scripts/select_decision_policy.py:163`
skips TEST when labels are summarized, and the first TEST label read is
`test_labels = split.test[TARGET_COLUMN]` at line 419 — after the winning
thresholds have already been printed at lines 411-412). Lines 481-483 then abort
if the thresholds differ from those pre-TEST values. **The TEST figures are the
ones to quote.** The VALIDATION figure belongs in the record of how the model was
built, not in a claim about how well it performs.

The walk-forward experiment in [temporal evaluation](temporal_evaluation.md) is
not a remedy for this. It retrains over expanding windows that overlap both
VALIDATION and TEST, so it reuses the same rows a second time rather than
providing a fresh partition. Its non-comparability with the locked benchmark is
documented in `README.md`.

Correcting this properly would mean a fourth partition or nested
cross-validation for selection. Neither is implemented, and neither can be
retrofitted without re-running feature and threshold selection — which would
change the frozen model and invalidate every locked number above. It is recorded
here as a known limitation of the evidence, not as a pending fix.

## Interpretation limits

These measurements describe one chronological split of IEEE-CIS. They do not establish performance on current or future payment traffic. The bootstrap intervals quantify resampling uncertainty within this holdout, not production uncertainty. The project does not implement live drift monitoring or continuous retraining.
