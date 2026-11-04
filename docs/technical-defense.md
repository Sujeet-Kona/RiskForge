# RiskForge technical defense

This guide is a preparation aid, not a script to memorize. You should be able to derive the reasoning, point to the implementation, and state the limitations without reading this document.

## One-minute project explanation

RiskForge is an offline fraud-risk decisioning project built on the IEEE-CIS Fraud Detection benchmark. It assembles a 28-feature modeling contract, derives point-in-time behavioral features, uses a chronological train/validation/test split, trains a fraud classifier, evaluates probability calibration, explores a cost-sensitive ALLOW/REVIEW/BLOCK policy, and serves scores through FastAPI with PostgreSQL decision auditing.

It demonstrates an applied ML lifecycle and a serving boundary. It is not a live payment-network system, and its synthetic policy economics are not business measurements.

## Data and leakage

### Why split chronologically instead of randomly?

Fraud patterns, customers, devices and attacker behavior evolve over time. A random split can place transactions from similar time periods on both sides and make evaluation look easier than predicting later transactions. RiskForge sorts by `TransactionDT` and partitions distinct timestamps chronologically (70% TRAIN, 15% VALIDATION, 15% TEST). Equal timestamps are kept together.

**You should be able to explain:** chronological splitting better approximates deployment on future transactions, but one holdout still does not prove stability under future concept drift.

### What is point-in-time feature engineering?

A feature for transaction (t) must be computed only from information available before (t). For example, a 24-hour card transaction count must exclude the current transaction and all future transactions. A correct temporal window is not enough if joins, sorting, or entity keys accidentally include future information.

The serving API accepts prior history from its caller. It cannot independently prove that this history is complete, fresh, authentic, or associated with the correct entity. That is an explicit serving limitation.

### Why are entity identifiers excluded?

Raw identifiers such as card and address keys are used to group prior activity when building behavioral aggregates, but are not directly included as model predictors. This reduces direct memorization of high-cardinality identities. The derived counts and amount aggregates represent behavior rather than the raw identity itself.

### Why 28 features?

The contract contains 13 behavioral/transaction-derived numeric features, 4 categorical transaction fields, and 11 selected C/D numeric fields. The feature contract is explicit and shared across model implementations. `TransactionDT` is retained for splitting and temporal analysis, not used as a model feature. The fraud label is never a feature.

Be ready to explain why each feature family exists and how missing values are interpreted. Do not claim that every feature is causally predictive.

## Split, preprocessing and model comparison

### Why fit preprocessing on TRAIN only?

Imputation statistics, scaling parameters and one-hot vocabularies are learned from TRAIN. Fitting these on validation or test would leak information about future distributions into training. The benchmark script fits each complete pipeline on TRAIN only.

### Why compare Logistic Regression, Random Forest and XGBoost?

They provide different capacity/inductive-bias baselines:

- Logistic Regression is a linear baseline. Numeric scaling helps optimization; class weighting addresses imbalance.
- Random Forest captures nonlinear interactions through bagged decision trees and is less sensitive to numeric scale.
- XGBoost builds boosted trees sequentially to reduce residual errors and can model nonlinear feature interactions.

All three use the same rows, chronological split and 28 input features. Their preprocessing is fit only on TRAIN. The Random Forest configuration is a fixed baseline, not a hyperparameter search. On the recorded TEST holdout, XGBoost achieved AP 0.449254 and ROC-AUC 0.865318, Random Forest achieved AP 0.426033 and ROC-AUC 0.861220, and Logistic Regression achieved AP 0.145789 and ROC-AUC 0.780255. XGBoost led overall on this holdout; do not generalize that result to future payment traffic.

### Why class weighting?

Fraud is rare. Without compensation, a learner can reduce overall error by favoring the majority non-fraud class. Logistic Regression uses `class_weight="balanced"`; Random Forest uses `balanced_subsample`; XGBoost computes `scale_pos_weight = negatives / positives` from TRAIN.

Class weighting changes the effective training objective. The resulting scores should not automatically be interpreted as calibrated real-world fraud probabilities. This is why ranking metrics and probability-quality metrics must be discussed separately.

### Why not accuracy?

With a low fraud prevalence, predicting every transaction as non-fraud can achieve high accuracy while detecting no fraud. Accuracy hides the minority-class failure. Precision-recall performance, recall, false-positive burden and the operational cost of actions are more informative.

### PR-AUC versus ROC-AUC

Average precision (the implementation's PR-AUC-style ranking summary) emphasizes precision/recall trade-offs for the positive class and is sensitive to the fraud prevalence. ROC-AUC measures ranking across true-positive and false-positive rates and can look strong even when precision is poor under severe imbalance. Report both, but lead with average precision for this task.

The exact metric implementation matters: RiskForge uses scikit-learn's `average_precision_score`, not trapezoidal integration of a plotted precision-recall curve. Call it Average Precision (AP) when precision is required.

### Why Brier score and log loss?

They assess probability quality, not only ranking. Brier score is mean squared probability error. Log loss penalizes confident wrong predictions more heavily. Because the models use class weighting, their raw scores may be miscalibrated; report these values as diagnostics and do not use them alone to declare a winner.

## Calibration and decision policy

### What is calibration?

A model is calibrated when, among predictions assigned probability (p), approximately fraction (p) are positive over the relevant population. Ranking and calibration are different: a monotonic calibration map can improve probability reliability while slightly changing ranking metrics.

### Why compare sigmoid and isotonic calibration with forward-chaining OOF predictions?

A calibrator must be evaluated on predictions from examples that were not used to fit the corresponding base model. Expanding-window forward chaining preserves temporal order. The first fold is warm-up and is excluded from scored OOF metrics.

The recorded validation comparison used 70,334 OOF-scored rows and excluded 17,679 warm-up rows. Isotonic had lower Brier score (0.024703 versus 0.025368), the predeclared selection criterion. Sigmoid had better log loss and ranking metrics. Therefore, isotonic was selected for the stated Brier objective—not because it was better on every metric.

### Why can calibration lower PR-AUC or ROC-AUC?

Calibration targets probability reliability, not ranking. Isotonic is a non-decreasing stepwise mapping and can create ties. Ties can change ranking metrics. The recorded TEST PR-AUC decreased from 0.449254 raw to 0.438919 calibrated; ROC-AUC changed from 0.865318 to 0.864850. These trade-offs must be disclosed.

### Why is the selected policy not simply threshold 0.5?

The threshold depends on the relative costs of false negatives, false positives and manual review, plus review effectiveness and operational capacity. A default 0.5 threshold is not inherently meaningful for an imbalanced fraud workflow. RiskForge explores a three-action policy, but its cost assumptions are synthetic.

The active serving policy is still `policy-001` (review 0.075, block 0.110). The forward-OOF candidate (review 0.050, block 0.110) is not promoted. The candidate's cost advantage over the runner-up was only 35 synthetic cost units, so it is not a robust business conclusion.

## Robustness and uncertainty

### What did the final benchmark show?

On the single chronological TEST holdout, average precision / ROC-AUC were:

- Logistic Regression: 0.145789 / 0.780255.
- Random Forest: 0.426033 / 0.861220.
- XGBoost: 0.449254 / 0.865318.

The paired class-stratified bootstrap 95% interval for XGBoost minus Random
Forest was [+0.015090, +0.030551] for AP and [+0.001328, +0.007241] for
ROC-AUC. The bootstrap treats transactions as independent sampling units; it
does not account for temporal dependence, dataset shift or model retraining
variability.

XGBoost led AP in all four chronological periods, four amount bands and three
card-history groups, but it did not win every ProductCD slice or every metric.
Random Forest narrowly led AP in ProductCD H; it led ROC-AUC in ProductCD S.
ProductCD H and S have small fraud counts. ProductCD W was weak for both
models. These are descriptive slices of one holdout, not independent future
deployments.

### What does the final benchmark report?

Run `scripts/run_model_benchmark.py`. It writes:
- `model_metrics.csv`: validation and test metrics for all three estimators.
- `test_segment_metrics.csv`: test results by chronological period, transaction amount band, ProductCD and prior 24-hour card activity.
- `test_bootstrap_intervals.csv`: 95% paired, class-stratified percentile bootstrap intervals for TEST AP and ROC-AUC, plus paired XGBoost-minus-baseline differences.
- `run_metadata.json`: split, feature contract, fixed model settings, fit timings and bootstrap seed.

Temporal groups are formed using distinct timestamp ranks so equal timestamps stay together. Amount bands use TRAIN quantiles, not TEST labels. Segment metrics are descriptive; small groups or groups containing one class cannot support AP/ROC-AUC and are marked accordingly.

### What does a confidence interval mean here?

The bootstrap resamples positive and negative TEST examples separately and uses the same sampled indices for each model. This preserves class counts and makes model-difference intervals paired. The interval describes sampling uncertainty under the empirical TEST distribution; it does not account for dataset shift, model retraining variability, threshold-selection uncertainty, or dependence between related transactions. It is not a guarantee about future production performance.

### What if two confidence intervals overlap?

Overlapping marginal confidence intervals are not a formal test of the difference. The paired bootstrap difference interval is more directly relevant. If the difference interval includes zero, the available TEST sample does not establish a clear ranking advantage. Do not overstate small metric differences.

### What is concept drift?

The relationship between inputs and fraud outcomes can change over time. A temporal performance decline may reflect changes in fraud prevalence, customer behavior, merchant mix, attacker tactics, or feature availability. This project reports within-holdout temporal slices; it does not implement live drift detection or monitoring.

## Serving, explanations and audit

### What happens at inference?

The FastAPI service validates the request, constructs the feature vector from the transaction and caller-supplied prior history, applies the locked serving artifact, assigns ALLOW/REVIEW/BLOCK using the active policy thresholds, produces explanation/reason-code output, and persists the decision audit record before returning success.

### SHAP versus reason codes

SHAP explains model contribution to a score. The threshold policy maps a score to an action. A reason code is a user-facing, controlled explanation taxonomy. These are related but not interchangeable: a feature with a large SHAP contribution is not itself the rule that selected BLOCK.

### Can an audit record replay the exact decision?

Not fully. Audit rows preserve the scored transaction inputs and decision details, but not every prior-history row used to derive behavioral features. Therefore, the database record alone cannot reconstruct the exact original score. The API's append-only behavior is an application convention, not a database-enforced immutability guarantee. Idempotency and duplicate suppression are not implemented.

## Limitations you must state without being prompted

- IEEE-CIS is an offline public benchmark, not live payment-network data.
- The results describe one chronological holdout and its subgroups, not future deployment.
- Policy costs and manual-review effectiveness are synthetic assumptions.
- Caller-supplied history is not independently authenticated or guaranteed complete.
- Audit persistence is not a full replayable event-sourcing system.
- No production deployment, authentication, TLS, rate limiting, drift monitoring or human-review operation is implemented.

## Oral defense checklist

Answer each aloud in 60–90 seconds, then verify the relevant implementation:

1. Walk through one transaction from raw data to model-ready row.
2. Explain how you prevent future transactions from entering a behavioral feature.
3. Defend the 70/15/15 chronological split and its limitations.
4. Name all 28 features by family and explain the excluded fields.
5. Explain exactly which preprocessing objects are fit on TRAIN.
6. Compare the assumptions and strengths of Logistic Regression, Random Forest and XGBoost.
7. Explain why class weighting is used and how it affects probability interpretation.
8. Explain why AP is the primary ranking metric in this imbalanced problem.
9. Explain why ROC-AUC can be misleading when fraud prevalence is low.
10. Explain the difference between discrimination and calibration.
11. Explain forward-chaining OOF calibration and the warm-up period.
12. Defend isotonic selection while acknowledging where sigmoid performed better.
13. Explain why calibration can reduce ranking metrics.
14. Derive the ALLOW/REVIEW/BLOCK policy and identify every synthetic assumption.
15. Explain the active policy versus the experimental candidate.
16. Explain how the final benchmark avoids test leakage.
17. Explain what the paired stratified bootstrap estimates—and what it does not.
18. Interpret one temporal slice and one transaction segment without overclaiming.
19. Explain why SHAP, reason codes and action thresholds are different.
20. State the exact serving-history and audit-replay limitations.

You are not ready to defend the project if you can only recite these answers. Be able to trace each claim to code, reproduce the benchmark, and explain what evidence would change your conclusion.
