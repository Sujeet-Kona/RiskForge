"""Recording a training run in MLflow without changing what the run does.

Everything here is downstream of the training. The model is fitted by
:func:`ml.models.xgboost_model.fit_xgboost_model`, the split comes from
:func:`ml.split.temporal.temporal_split`, the metrics come from
:func:`ml.models.evaluation.evaluate_predictions`, and the feature contract is
read from :mod:`ml.data.modeling`. This module adds a record of those facts; it
defines none of them and alters none of them.

Two boundaries are enforced by construction rather than by convention:

*The recorded values come from the fitted objects.* Hyperparameters are read back
off the classifier that was actually fitted, and the split fractions off the
split function's own signature. A parameter restated from the source would still
be recorded faithfully after someone changed the source and forgot the record.

*Nothing here publishes a model.* The run logs the pipeline as an MLflow
artifact and, if configured, registers a version tagged with the approved model
version. It never sets an alias, never promotes anything to production, and never
writes or replaces ``artifacts/serving/riskforge_serving_artifact.joblib``, which
remains the only artifact the API loads.
"""

from __future__ import annotations

import inspect
from dataclasses import dataclass
from pathlib import Path

import mlflow
import pandas as pd

from ml.data.modeling import (
    CATEGORICAL_FEATURES,
    CD_FEATURES,
    TARGET_COLUMN,
    build_modeling_frame,
)
from ml.models.artifact import FEATURE_VERSION, MODEL_VERSION
from ml.models.evaluation import (
    ClassificationMetrics,
    evaluate_predictions,
    metrics_to_dict,
)
from ml.models.xgboost_model import (
    CLASSIFIER_STEP_NAME,
    XGBoostFraudModel,
    fit_xgboost_model,
    predict_fraud_probability,
)
from ml.split.temporal import TemporalSplit, temporal_split
from policy.locked import LOCKED_POLICY_VERSION
from tracking.config import TrackingConfig, apply_file_store_opt_out
from tracking.dataset import DatasetFingerprint, fingerprint_files
from tracking.lineage import (
    collect_lineage,
    feature_contract,
    feature_contract_text,
    split_strategy,
)


MODEL_ARTIFACT_NAME = "model"
FEATURE_CONTRACT_ARTIFACT = "feature_contract.json"
FEATURE_CONTRACT_TEXT_ARTIFACT = "feature_contract.txt"
MODEL_METADATA_ARTIFACT = "model_metadata.json"
ENVIRONMENT_ARTIFACT = "environment.json"
DATASET_IDENTITY_ARTIFACT = "dataset_identity.json"

# MLflow's sklearn flavor serializes with skops by default, and skops is not
# installed here. cloudpickle is the format MLflow documents as the alternative,
# and it is already what the Phase 6 serving bundle uses via joblib, so the
# logged artifact and the served artifact agree on how a fitted pipeline is
# represented.
MODEL_SERIALIZATION_FORMAT = "cloudpickle"

# Fraud is rare, so the metric that matters is the one that does not reward
# predicting "not fraud" for everything. Recorded as a tag so a reader of the run
# knows which number is primary without having to infer it.
PRIMARY_METRIC = "test/pr_auc"

# The hyperparameters the brief requires, read back from the fitted classifier.
CLASSIFIER_PARAMETERS = (
    "objective",
    "eval_metric",
    "n_estimators",
    "max_depth",
    "learning_rate",
    "min_child_weight",
    "subsample",
    "colsample_bytree",
    "reg_lambda",
    "scale_pos_weight",
    "random_state",
    "tree_method",
)

CALIBRATION_IN_TRACKED_RUN = "none"
CALIBRATION_NOTE = (
    "This run trains and scores the raw XGBoost pipeline. Isotonic calibration is "
    "fitted downstream on the VALIDATION partition by "
    "scripts/build_serving_artifact.py, so the metrics recorded here are raw and "
    "are not the calibrated probabilities the API returns."
)

SERVING_ARTIFACT_NOTE = (
    "The MLflow model artifact records what training produced. It is not the "
    "runtime artifact: the API loads artifacts/serving/riskforge_serving_artifact.joblib "
    "and never reads from MLflow."
)

SERVING_ARTIFACT_PATH = "artifacts/serving/riskforge_serving_artifact.joblib"

DEFAULT_BEHAVIORAL_PATH = Path("data/processed/train_behavioral_v1.parquet")
DEFAULT_TRANSACTION_PATH = Path("data/raw/ieee_cis/train_transaction.csv")

TRANSACTION_COLUMNS = [
    "TransactionID",
    TARGET_COLUMN,
    *CATEGORICAL_FEATURES,
    *CD_FEATURES,
]


class TrackingError(RuntimeError):
    """A run could not be recorded, or was recorded without required lineage."""


@dataclass(frozen=True)
class TrackedTraining:
    """The results of one training run, before any of it is logged."""

    model: XGBoostFraudModel
    split: TemporalSplit
    validation_metrics: ClassificationMetrics
    test_metrics: ClassificationMetrics
    dataset: DatasetFingerprint


@dataclass(frozen=True)
class TrackedRun:
    """Where a training run was recorded, and what it claims to be."""

    run_id: str
    run_name: str
    experiment_name: str
    tracking_uri: str
    model_uri: str
    model_version: str
    feature_version: str
    policy_version: str
    dataset_fingerprint: str
    commit_sha: str
    validation_pr_auc: float
    validation_roc_auc: float
    test_pr_auc: float
    test_roc_auc: float
    registered_model_name: str | None
    registered_model_version: str | None
    file_store_opt_out_applied: bool


def default_evaluation_threshold() -> float:
    """The reporting threshold the evaluation utility applies by default.

    Read from the signature rather than restated: precision, recall, F1, FPR and
    FNR are all threshold-dependent, so a run that does not record the threshold
    records metrics nobody can reproduce.
    """
    parameters = inspect.signature(evaluate_predictions).parameters

    return float(parameters["threshold"].default)


def training_parameters(training: TrackedTraining) -> dict[str, object]:
    """Every parameter a reader needs to know what was fitted, and on what.

    Hyperparameters are taken from the fitted classifier's own ``get_params()``,
    so the record describes the model that exists rather than the model the
    source code intended to build.
    """
    classifier = training.model.model.named_steps[CLASSIFIER_STEP_NAME]
    fitted = classifier.get_params()

    missing = [name for name in CLASSIFIER_PARAMETERS if name not in fitted]

    if missing:
        raise TrackingError(
            f"The fitted classifier does not expose {missing}, so its "
            "configuration cannot be recorded faithfully."
        )

    strategy = split_strategy()

    parameters: dict[str, object] = {
        "model_type": type(classifier).__module__.split(".")[0],
        "classifier_class": type(classifier).__name__,
        "evaluation_threshold": default_evaluation_threshold(),
        "calibration_method": CALIBRATION_IN_TRACKED_RUN,
        "model_feature_count": len(training.model.feature_columns),
        "categorical_feature_count": len(training.model.categorical_features),
        "numeric_feature_count": len(training.model.numeric_features),
        "split_strategy": strategy["strategy"],
        "split_column": strategy["split_column"],
        "train_fraction": strategy["train_fraction"],
        "validation_fraction": strategy["validation_fraction"],
        "n_train_rows": len(training.split.train),
        "n_validation_rows": len(training.split.validation),
        "n_test_rows": len(training.split.test),
        "dataset_fingerprint": training.dataset.fingerprint,
        "dataset_file_count": len(training.dataset.files),
    }

    parameters.update(
        {f"hyperparameter_{name}": fitted[name] for name in CLASSIFIER_PARAMETERS}
    )

    return parameters


def training_metrics(training: TrackedTraining) -> dict[str, float]:
    """Validation and test metrics separately, plus the class-rate context.

    Accuracy is deliberately absent: with a fraud rate near three percent it
    would be high for a model that catches nothing, and the locked primary metric
    is PR-AUC.
    """
    metrics: dict[str, float] = {}

    for partition, evaluated in (
        ("validation", training.validation_metrics),
        ("test", training.test_metrics),
    ):
        metrics.update(
            {
                f"{partition}/{name}": float(value)
                for name, value in metrics_to_dict(evaluated).items()
            }
        )

    for partition, frame in (
        ("train", training.split.train),
        ("validation", training.split.validation),
        ("test", training.split.test),
    ):
        labels = frame[TARGET_COLUMN]
        fraud_count = int(labels.sum())

        metrics[f"{partition}/fraud_count"] = float(fraud_count)
        metrics[f"{partition}/positive_rate"] = float(labels.mean())
        metrics[f"{partition}/row_count"] = float(len(labels))

    return metrics


def version_metadata() -> dict[str, str]:
    """The approved version identifiers, from the modules that own them."""
    return {
        "model_version": MODEL_VERSION,
        "feature_version": FEATURE_VERSION,
        "policy_version": LOCKED_POLICY_VERSION,
    }


def log_training_run(
    training: TrackedTraining,
    *,
    config: TrackingConfig,
    repo_root: Path,
    run_name: str | None = None,
) -> TrackedRun:
    """Record one already-completed training run in MLflow.

    This is the whole tracking surface: parameters, metrics, feature contract,
    dataset identity, environment versions, provenance, the fitted pipeline as a
    model artifact, and an explicitly tagged registry version when a registry
    name is configured. It returns where the record lives.
    """
    lineage = collect_lineage(repo_root)
    threshold = default_evaluation_threshold()
    resolved_name = run_name or MODEL_VERSION

    config = config.resolved_against(repo_root)
    opt_out_applied = apply_file_store_opt_out(config)

    mlflow.set_tracking_uri(config.tracking_uri)
    mlflow.set_experiment(config.experiment_name)

    parameters = training_parameters(training)
    metrics = training_metrics(training)
    contract = feature_contract()

    with mlflow.start_run(run_name=resolved_name) as run:
        mlflow.log_params(parameters)
        mlflow.log_metrics(metrics)

        mlflow.set_tags(
            {
                **version_metadata(),
                "git_commit_sha": lineage.commit_sha,
                "git_working_tree": lineage.working_tree,
                "dataset_fingerprint": training.dataset.fingerprint,
                "split_strategy": str(parameters["split_strategy"]),
                "primary_metric": PRIMARY_METRIC,
                "evaluation_threshold": str(threshold),
                # Machine-readable form of the boundary: this run does not
                # produce, replace or approve the artifact the API serves.
                "produces_serving_artifact": "false",
                "serving_artifact": SERVING_ARTIFACT_PATH,
            }
        )

        mlflow.log_dict(contract, FEATURE_CONTRACT_ARTIFACT)
        mlflow.log_text(feature_contract_text(), FEATURE_CONTRACT_TEXT_ARTIFACT)
        mlflow.log_dict(training.dataset.to_dict(), DATASET_IDENTITY_ARTIFACT)
        mlflow.log_dict(
            {
                "python_version": lineage.python_version,
                "mlflow_version": mlflow.__version__,
                "library_versions": lineage.library_versions,
                "model_serialization_format": MODEL_SERIALIZATION_FORMAT,
            },
            ENVIRONMENT_ARTIFACT,
        )
        mlflow.log_dict(
            {
                **version_metadata(),
                "model_artifact_name": MODEL_ARTIFACT_NAME,
                "model_artifact_flavor": "sklearn",
                "model_serialization_format": MODEL_SERIALIZATION_FORMAT,
                "feature_count": len(contract["features"]),
                "evaluation_threshold": threshold,
                "calibration_method": CALIBRATION_IN_TRACKED_RUN,
                "calibration_note": CALIBRATION_NOTE,
                "serving_artifact_note": SERVING_ARTIFACT_NOTE,
                "serving_artifact_path": SERVING_ARTIFACT_PATH,
                "dataset_fingerprint": training.dataset.fingerprint,
                "dataset": training.dataset.to_dict(),
                "lineage": lineage.to_dict(),
                "split": split_strategy(),
                "partition_rows": {
                    "train": len(training.split.train),
                    "validation": len(training.split.validation),
                    "test": len(training.split.test),
                },
                "partition_fraud_counts": {
                    "train": int(training.split.train[TARGET_COLUMN].sum()),
                    "validation": int(training.split.validation[TARGET_COLUMN].sum()),
                    "test": int(training.split.test[TARGET_COLUMN].sum()),
                },
            },
            MODEL_METADATA_ARTIFACT,
        )

        mlflow.sklearn.log_model(
            training.model.model,
            name=MODEL_ARTIFACT_NAME,
            serialization_format=MODEL_SERIALIZATION_FORMAT,
            metadata={
                "model_version": MODEL_VERSION,
                "feature_version": FEATURE_VERSION,
                "policy_version": LOCKED_POLICY_VERSION,
                "feature_count": str(len(contract["features"])),
                "scale_pos_weight": str(training.model.scale_pos_weight),
            },
        )

        model_uri = f"runs:/{run.info.run_id}/{MODEL_ARTIFACT_NAME}"

        registered = _register_model_version(
            config=config,
            model_uri=model_uri,
            dataset=training.dataset,
            commit_sha=lineage.commit_sha,
        )

    return TrackedRun(
        run_id=run.info.run_id,
        run_name=resolved_name,
        experiment_name=config.experiment_name,
        tracking_uri=config.tracking_uri,
        model_uri=model_uri,
        model_version=MODEL_VERSION,
        feature_version=FEATURE_VERSION,
        policy_version=LOCKED_POLICY_VERSION,
        dataset_fingerprint=training.dataset.fingerprint,
        commit_sha=lineage.commit_sha,
        validation_pr_auc=training.validation_metrics.pr_auc,
        validation_roc_auc=training.validation_metrics.roc_auc,
        test_pr_auc=training.test_metrics.pr_auc,
        test_roc_auc=training.test_metrics.roc_auc,
        registered_model_name=registered[0],
        registered_model_version=registered[1],
        file_store_opt_out_applied=opt_out_applied,
    )


def _register_model_version(
    *,
    config: TrackingConfig,
    model_uri: str,
    dataset: DatasetFingerprint,
    commit_sha: str,
) -> tuple[str | None, str | None]:
    """Register the logged model under an explicit name and tag its version.

    Tagging is the entire extent of it. No alias is set and no stage is
    promoted, because promoting a model is a decision a human makes about the
    serving artifact — and the serving artifact is not what this function
    touches. A registered version here is a labelled pointer for inspection.
    """
    if config.registered_model_name is None:
        return None, None

    version = mlflow.register_model(model_uri, config.registered_model_name)

    client = mlflow.MlflowClient()

    tags = {
        **version_metadata(),
        "git_commit_sha": commit_sha,
        "dataset_fingerprint": dataset.fingerprint,
        "serving_artifact": SERVING_ARTIFACT_PATH,
        # The approved artifact is built by a separate, deterministic script. A
        # reader who finds this version in a registry needs to know that being
        # here changed nothing about what the API serves.
        "promotion": "none",
    }

    for key, value in tags.items():
        client.set_model_version_tag(version.name, version.version, key, value)

    return version.name, str(version.version)


def run_tracked_training(
    config: TrackingConfig,
    *,
    repo_root: Path,
    behavioral_path: Path = DEFAULT_BEHAVIORAL_PATH,
    transaction_path: Path = DEFAULT_TRANSACTION_PATH,
    run_name: str | None = None,
) -> TrackedRun:
    """Train the locked model and record the run, reusing the existing pipeline.

    The steps are the approved ones: the modeling frame from
    :func:`build_modeling_frame`, the chronological split from
    :func:`temporal_split`, the XGBoost configuration from
    :func:`fit_xgboost_model`, the scores from :func:`predict_fraud_probability`
    and the metrics from :func:`evaluate_predictions`. Nothing is reimplemented
    and nothing about the model is altered; the only addition is that the results
    are recorded.

    The behavioral artifact is not absolute on purpose: it is resolved against
    ``repo_root`` so the same configuration fingerprints the same files
    regardless of the directory the run was launched from.
    """
    behavioral = _resolve(behavioral_path, repo_root)
    transactions = _resolve(transaction_path, repo_root)

    behavioral_frame = pd.read_parquet(behavioral)
    transaction_frame = pd.read_csv(transactions, usecols=TRANSACTION_COLUMNS)

    modeling = build_modeling_frame(
        behavioral=behavioral_frame,
        transaction_features=transaction_frame,
        labels=transaction_frame,
    )

    split = temporal_split(modeling.data)
    model = fit_xgboost_model(split.train)

    training = TrackedTraining(
        model=model,
        split=split,
        validation_metrics=evaluate_predictions(
            split.validation[TARGET_COLUMN],
            predict_fraud_probability(model, split.validation),
        ),
        test_metrics=evaluate_predictions(
            split.test[TARGET_COLUMN],
            predict_fraud_probability(model, split.test),
        ),
        dataset=fingerprint_files([behavioral, transactions], root=repo_root),
    )

    return log_training_run(
        training,
        config=config,
        repo_root=repo_root,
        run_name=run_name,
    )


def _resolve(path: Path, repo_root: Path) -> Path:
    candidate = Path(path)

    if not candidate.is_absolute():
        candidate = repo_root / candidate

    if not candidate.is_file():
        raise TrackingError(
            f"{candidate} does not exist. Training is not run against a "
            "substitute dataset, so nothing is recorded."
        )

    return candidate
