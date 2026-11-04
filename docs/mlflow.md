# RiskForge model lifecycle tracking

> MLflow is used for experiment/model lifecycle tracking. It is not the runtime
> model provider for the FastAPI inference path in this phase.

A training run is recorded in MLflow: its parameters, its metrics, the exact
feature contract that produced it, the identity of the data it was fitted on,
the environment that fitted it, the git commit it was fitted from, and the
fitted pipeline itself as a loadable artifact.

Nothing in `app/`, `db/`, `ml/`, `policy/` or `explain/` imports MLflow. The
only module in the repository that does is `tracking/run.py`, and
`tests/test_serving_discipline.py` enforces that mechanically rather than by
convention.

---

## 1. Why MLflow is used

The model was chosen by experiment: 162 candidate features were attributed,
ablated and reduced to 28; calibration was selected on out-of-fold VALIDATION
Brier score; `policy-001` was frozen against a cost function. Those conclusions
live in the repository as locked constants, but the record of the run that
produced them did not. Before this phase there was no answer to "which commit,
which data and which library versions fitted the model that is being served"
other than reading the code and hoping nothing had drifted.

MLflow supplies that record. It is not used for:

- **model serving.** The API loads
  `artifacts/serving/riskforge_serving_artifact.joblib` from disk, as section 9
  explains.
- **feature storage.** Features are computed by `ml/features/behavioral.py` at
  request time from the caller-supplied history.
- **orchestration.** There is no scheduler, no pipeline engine and no workflow.
  One script runs one training and records it.

`mlflow-skinny==3.17.2` is the pinned dependency. The full `mlflow` package was
rejected on the resolver's own output: `pip install --dry-run mlflow==3.17.2`
reported it would install `pyarrow-25.0.1`, downgrading the pinned
`pyarrow==26.0.0` that the parquet behavioral pipeline reads with, and would
also pull in Flask, gunicorn, graphene, matplotlib, aiohttp and `docker`, which
is a model-serving stack this phase does not add. `mlflow-skinny` reported no
change to any pinned package. A consequence is that `mlflow ui` and
`mlflow server` are not available; skinny does not ship the tracking server.
Runs are inspected by reading the store directory or through the `MlflowClient`
API.

---

## 2. Local tracking setup

Configuration is read from the environment by `tracking/config.py`, never
hard-coded. `.env.example` documents all of it:

| Variable | Default | Meaning |
| --- | --- | --- |
| `MLFLOW_TRACKING_URI` | `file:./mlruns` | Where runs are stored |
| `MLFLOW_EXPERIMENT_NAME` | `riskforge-xgboost-lifecycle` | Experiment the runs belong to |
| `MLFLOW_REGISTERED_MODEL_NAME` | `riskforge-fraud-model` | Registry name; blank disables the registry |

Two backends are accepted, both local: a `file:` directory and an absolute
`sqlite:////path/to/mlflow.db`. The filesystem store is the default because it
keeps every piece of generated state (run metadata, model artifacts and the
local registry) inside one directory that `.gitignore` already covers.

Everything else is refused with a `TrackingConfigurationError`:

- a non-local scheme (`http`, `https`, `s3`, `dbfs`, `postgresql`), because this
  phase adds no cloud registry and no remote tracking server;
- any URI that names a host, or that carries a query, a fragment or credentials;
- a relative `sqlite:` URI, whose location would then depend on the working
  directory the script happened to be launched from;
- a `file:` URI that does not name a directory.

The error message deliberately does not echo the URI it rejected, because a
tracking URI can contain credentials. `tests/test_tracking_config.py` asserts
that a rejected URI with an embedded secret does not appear in the message.

### The file-store opt-out

MLflow 3.x places its filesystem backend in maintenance mode and raises an
`MlflowException` naming that maintenance mode unless `MLFLOW_ALLOW_FILE_STORE=true`
is set. `apply_file_store_opt_out` sets it for a `file:` store, and
`TrackedRun.file_store_opt_out_applied` reports whether it was needed. This is a
deviation from the plain `MLFLOW_TRACKING_URI=file:./mlruns` that MLflow's own
documentation shows, and it is not optional: without it the default
configuration cannot start a run at all. The variable is also in `.env.example`
so that a tool other than the training script can reach the same store.

### Generated state is not committed

`.gitignore` covers `mlruns/` and `mlflow.db`. The repository carries the
configuration, the tracking package and the script; it never carries an
experiment database or a logged model artifact.

---

## 3. Running the tracked training

```bash
PYTHONPATH=. python scripts/train_tracked_model.py
```

It needs the real inputs, which are gitignored:
`data/processed/train_behavioral_v1.parquet` and
`data/raw/ieee_cis/train_transaction.csv`. If either is missing the script
raises `TrackingError` and records nothing; it does not fall back to a
substitute dataset.

The script prints four steps and then stops: resolve the configuration, train
and open the run, check the recorded run against the locked values, report the
version identifiers. Step 3 compares the run's VALIDATION and TEST PR-AUC and
ROC-AUC against the locked raw metrics (VALIDATION 0.474352 and 0.884513, TEST
0.449254 and 0.865318) within `1e-6` and raises `AssertionError` on drift, so a
run is only ever recorded for the model that is actually the locked one.

It does not define a second training pipeline. The modeling frame comes from
`ml.data.modeling.build_modeling_frame`, the split from
`ml.split.temporal.temporal_split`, the XGBoost configuration from
`ml.models.xgboost_model.fit_xgboost_model`, the scores from
`predict_fraud_probability` and the metrics from
`ml.models.evaluation.evaluate_predictions`.

Hashing the inputs takes a few seconds (the transaction CSV is about 683 MB),
and that digest is what makes the run's dataset identity real rather than
asserted.

---

## 4. Parameters logged

Twenty-eight parameters, all of them derived. The twelve locked hyperparameters
are read back off the fitted classifier's own `get_params()` and logged as
`hyperparameter_<name>`:

`objective`, `eval_metric`, `n_estimators`, `max_depth`, `learning_rate`,
`min_child_weight`, `subsample`, `colsample_bytree`, `reg_lambda`,
`scale_pos_weight`, `random_state`, `tree_method`.

Reading them off the fitted object rather than restating them is the point: a
parameter copied from the source would still be recorded faithfully after
someone changed the source and forgot the record.

The rest describe what was fitted and on what:

- `model_type` (`xgboost`, taken from `type(classifier).__module__`) and
  `classifier_class`;
- `evaluation_threshold`, read from `evaluate_predictions`'s signature, because
  precision, recall, F1, FPR and FNR are threshold-dependent and a run that does
  not record the threshold records metrics nobody can reproduce;
- `calibration_method`, which is `none` for a tracked run, as section 9
  explains;
- `model_feature_count`, `categorical_feature_count`, `numeric_feature_count`;
- `split_strategy`, `split_column`, `train_fraction`, `validation_fraction`,
  where the fractions come from `temporal_split`'s signature rather than from
  constants restated in the tracking layer;
- `n_train_rows`, `n_validation_rows`, `n_test_rows`;
- `dataset_fingerprint` and `dataset_file_count`.

Python and library versions are logged as the `environment.json` artifact rather
than as parameters: `python_version`, `mlflow_version`, and the version of each
of pandas, numpy, scikit-learn, scipy, xgboost, pyarrow, joblib and shap,
resolved through `importlib.metadata`. A missing distribution raises
`LineageError` instead of recording a blank.

No secret is logged. `tests/test_tracking_run.py` puts a sentinel into
`DATABASE_URL` and `AWS_SECRET_ACCESS_KEY` before a real logging call and then
asserts that the sentinel, `riskforge:` and `DATABASE_URL` appear in no
parameter, no tag and no downloaded artifact.

---

## 5. Metrics logged

Validation and test are logged separately, under `validation/` and `test/`
prefixes, from `metrics_to_dict`:

`pr_auc`, `roc_auc`, `precision`, `recall`, `f1`, `false_positive_rate`,
`false_negative_rate`, `true_negatives`, `false_positives`, `false_negatives`,
`true_positives` — 22 metrics.

Plus, for each of train, validation and test: `<partition>/fraud_count`,
`<partition>/positive_rate` and `<partition>/row_count` — 9 more, 31 total.

**Accuracy is not logged at all.** With a fraud rate near three percent it would
be high for a model that catches nothing, and the locked primary metric is
PR-AUC. `test/pr_auc` is recorded as the `primary_metric` tag so a reader of the
run knows which number is primary without inferring it, and a test asserts that
no logged metric key contains `accuracy`.

These are raw XGBoost scores. The API returns isotonic-calibrated probabilities,
which are fitted downstream on the VALIDATION partition by
`scripts/build_serving_artifact.py`. The `calibration_note` in
`model_metadata.json` says so in those words, because a reader comparing a
tracked run's `test/pr_auc` of 0.449254 against the serving artifact's
calibrated 0.438919 needs to know that the difference is by design.

---

## 6. Feature-contract lineage

The run records the exact 28 features and their ordering, as
`feature_contract.json` and, for a human, as `feature_contract.txt`: a numbered
line per feature.

Both are built by `tracking/lineage.py` from `ml.data.modeling`, which owns the
contract: `MODEL_FEATURES`, `CATEGORICAL_FEATURES`, `NUMERIC_FEATURES`,
`BEHAVIORAL_FEATURES`, `CD_FEATURES`, `TARGET_COLUMN` and `SPLIT_COLUMN`, plus
`source_module`, `feature_version`, `model_version` and `feature_count`.

The list is never retyped in the tracking layer. Two tests enforce that:
`tests/test_serving_discipline.py` asserts that no module under `tracking/`
contains a feature-name literal (`TransactionAmt`, `amount_log1p`, `card6`,
`P_emaildomain`, `D5`) and that `tracking/lineage.py` imports `ml.data.modeling`;
`tests/test_tracking_run.py` asserts the recorded list equals `MODEL_FEATURES`
element-for-element, in order.

Someone reading the run can therefore determine exactly which features produced
the model, and can confirm from `source_module` that the answer came from the
module that owns the contract rather than from a copy.

---

## 7. Model artifact lineage

The fitted `sklearn.pipeline.Pipeline` (preprocessor plus `XGBClassifier`) is
logged with `mlflow.sklearn.log_model` under the artifact name `model`, so it is
addressable as `runs:/<run_id>/model`.

It is serialized with `serialization_format="cloudpickle"`. MLflow 3.x defaults
the sklearn flavor to `skops`, which is not installed here, and the default
raises `ModuleNotFoundError: No module named 'skops'`. cloudpickle is the format
MLflow documents as the alternative and is already what the Phase 6 joblib
bundle uses, so the logged artifact and the served artifact agree on how a
fitted pipeline is represented. It emits a `WARNING mlflow.sklearn` about pickle
safety, which is cosmetic: the artifact stays inside a gitignored local store.

`tests/test_tracking_run.py` proves the logged artifact is sufficient to
reproduce the prediction pipeline rather than merely asserting it. It loads the
model back out of the store with `mlflow.sklearn.load_model`, checks the
pipeline's step names and that `feature_names_in_` equals `MODEL_FEATURES`, and
then checks that `predict_proba` reproduces
`predict_fraud_probability(model, frame)` exactly (maximum absolute difference
`0.0`) on both the validation and the test partition.

Four further artifacts carry the run's identity:

| Artifact | Contents |
| --- | --- |
| `feature_contract.json` and `feature_contract.txt` | The 28 features in order, by family (section 6) |
| `dataset_identity.json` | SHA-256 of each input file, its size, and the combined fingerprint |
| `environment.json` | Python, MLflow and library versions, serialization format |
| `model_metadata.json` | Version identifiers, calibration and serving notes, lineage, split, partition rows and fraud counts |

Dataset identity is computed, never invented. `tracking/dataset.py` hashes the
actual local bytes with SHA-256 in 8 MiB chunks, records each file's digest and
size, sorts the entries by their path relative to the repository root, and folds
`path`, size and digest per file into one fingerprint. The raw IEEE-CIS dataset
is gitignored and is never committed or uploaded; the fingerprint is what stands
in for it. Tests cover the properties that make it a real identity:
reproducible, order-independent, changed by a single byte, changed by moving a
file to a different path even though that file's own digest is unchanged, and
refusing an empty file list, a missing file, a directory, an empty file, a file
outside the dataset root and a non-directory root.

---

## 8. Model-version metadata

Three identifiers are recorded as run tags, as `log_model` metadata and in
`model_metadata.json`:

- `model_version = xgb-001`
- `feature_version = behavioral-v1`
- `policy_version = policy-001`

They are imported from `ml.models.artifact` and `policy.locked`, the modules
that own them, so the tracking layer cannot drift from the approved values.

The run also carries `git_commit_sha`, `git_working_tree` (`clean`, `dirty` or
`unknown`), `dataset_fingerprint`, `split_strategy`, `primary_metric` and
`evaluation_threshold` as tags. Between them, the tags and the artifacts are the
reproducibility record: commit, model and feature version, dataset identity,
split strategy, random seed (as `hyperparameter_random_state`), dependency
versions and training parameters.

### The local registry

If `MLFLOW_REGISTERED_MODEL_NAME` is set, the logged model is registered under
that name and the resulting version is tagged with the three version
identifiers, the commit SHA, the dataset fingerprint, the serving-artifact path
and `promotion = none`.

That tagging is the entire extent of it. No alias is set and no stage is
promoted. `tests/test_tracking_run.py` asserts the registered version's
`aliases` is empty, that `promotion` is `none`, and that the string `production`
appears in no tag key or value. `tests/test_serving_discipline.py` bans the
three calls that would decide what gets served (`set_registered_model_alias`,
`transition_model_version_stage` and `delete_registered_model`) anywhere under
`tracking/`.

Setting `MLFLOW_REGISTERED_MODEL_NAME` to a blank value disables the registry
entirely; a run is then logged with no registered version, and a test covers
that path.

One MLflow 3.x behaviour is worth knowing when reading a registry entry. A
`log_model` call creates a *logged model* entity as well as the run artifacts,
and `register_model` resolves through it, emitting
`Run with id ... has no artifacts at artifact path 'model', registering model
based on models:/m-... instead`. The registered version's `source` is therefore
a `models:/m-...` URI rather than the `runs:/.../model` URI. Both address the
same artifact: `mlflow.sklearn.load_model("runs:/<run_id>/model")` loads it, and
so does the registered version.

---

## 9. MLflow and the Phase 6 serving artifact

These are two different things with two different jobs, and conflating them is
the one mistake this phase is built to prevent.

| | MLflow model artifact | Phase 6 serving artifact |
| --- | --- | --- |
| Path | `runs:/<run_id>/model` in the tracking store | `artifacts/serving/riskforge_serving_artifact.joblib` |
| Contents | The raw fitted pipeline: preprocessor plus `XGBClassifier` | That pipeline plus the isotonic calibrator, the locked policy configuration, the feature contract and version-guarded library versions |
| Written by | `scripts/train_tracked_model.py` | `scripts/build_serving_artifact.py` |
| Read by | A person inspecting lineage | `app/service.py`, through `ml.models.artifact.load_artifact` |
| Purpose | Records what training produced | Provides what the API serves |

A tracked run records `calibration_method = none` and tags itself
`produces_serving_artifact = false` with `serving_artifact` naming the joblib
path. `model_metadata.json` carries `serving_artifact_note` in full: the MLflow
artifact records what training produced, it is not the runtime artifact, and the
API never reads from MLflow.

The inference path has no MLflow dependency. `POST /v1/risk/score` does not
create a run, download a model, query a registry or log an inference request.
The API stays deterministic and offline-capable once its serving artifact
exists, and the existing offline and API parity tests pass unchanged.

This is enforced, not promised:

- `test_mlflow_has_exactly_one_home_in_the_product_code`: `tracking/run.py` is
  the only product module that imports `mlflow`.
- `test_no_serving_or_model_layer_imports_mlflow` and
  `test_no_serving_or_model_layer_imports_the_tracking_layer`, parametrized over
  `app/`, `db/`, `ml/`, `policy/`, `explain/` and `alembic/`.
- `test_no_app_module_makes_an_mlflow_call` and
  `test_the_score_endpoint_never_touches_mlflow`; the second walks the AST of the
  score endpoint itself.
- `test_the_tracking_layer_never_writes_or_reads_the_serving_artifact`: no
  `save_artifact`, `load_artifact` or `fit_calibrator` call under `tracking/`.
- `test_only_the_tracked_training_script_reaches_the_tracking_layer`: the sole
  importer of `tracking` outside its own tests is
  `scripts/train_tracked_model.py`.
- and empirically: one test hashes the serving artifact before and after a real
  logging call and asserts the SHA-256 did not change.

---

## 10. What is intentionally not automated

- **No promotion.** A registered version is tagged, never aliased to
  `production` and never transitioned to a stage. Deciding that a model should
  be served is a human decision.
- **No serving-artifact replacement.** Nothing under `tracking/` can write
  `artifacts/serving/riskforge_serving_artifact.joblib`. Rebuilding it stays an
  explicit run of `scripts/build_serving_artifact.py`, which fits isotonic
  calibration on VALIDATION and guards the locked configuration.
- **No retraining from the API.** The request path cannot trigger a run.
- **No model download at inference time.** The API does not resolve a model from
  MLflow, so a prediction cannot depend on a tracking store being reachable.
- **No remote backend.** Only `file:` and absolute `sqlite:` are accepted; a
  cloud registry or a hosted tracking server is rejected at configuration time.
- **No threshold, policy or contract change.** `policy-001`, `T_REVIEW = 0.075`,
  `T_BLOCK = 0.110` and the 28 features are read from their owners and recorded.
  `test_no_tracking_module_restates_the_frozen_thresholds` asserts that no module
  under `tracking/` contains the literals `0.075` or `0.110`, so a second copy of
  a frozen threshold cannot appear here.
- **No substitute data.** A missing input file raises rather than falling back to
  a sample dataset.
- **No orchestration, no infrastructure.** No scheduler, container orchestration,
  message queue, feature store or distributed training. Local MLflow is the whole
  footprint.

---

## Tests

| File | Covers |
| --- | --- |
| `tests/test_tracking_config.py` | Tracking-URI configuration, local-only validation, secret-free errors, opt-out, path resolution |
| `tests/test_tracking_dataset.py` | Deterministic dataset fingerprinting and its rejection cases |
| `tests/test_tracking_run.py` | Run creation, parameters, metrics, feature contract, reproducibility metadata, model artifact round-trip, registry versioning, serving-artifact immutability, secret scanning |
| `tests/test_serving_discipline.py` | MLflow confined to `tracking/run.py`, and the tracking/serving boundary |
