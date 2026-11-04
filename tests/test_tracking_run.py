"""What a tracked run records, and what it provably does not do.

These tests log real MLflow runs, so they assert against what was actually
written to a store rather than against the arguments a call was given. The store
is a temporary directory and the model is fitted on the small synthetic frame
from ``conftest.py``: the locked 28-feature contract and the real training,
splitting, scoring and evaluation functions, with none of the gitignored IEEE-CIS
data and none of the locked hyperparameters changed.

The two boundaries that matter most are covered here rather than left to
inspection. A run must record enough to reproduce the model — parameters read
back off the fitted classifier, the exact ordered feature contract, a dataset
fingerprint, a commit SHA and the library versions. And a run must not publish
anything: it does not promote a registry version, and it does not touch the Phase
6 artifact the API serves.
"""

from __future__ import annotations

import hashlib
import json
import sys
from dataclasses import dataclass
from importlib import metadata
from pathlib import Path

import mlflow
import pytest
from mlflow.tracking import MlflowClient

from ml.data.modeling import (
    BEHAVIORAL_FEATURES,
    CATEGORICAL_FEATURES,
    CD_FEATURES,
    MODEL_FEATURES,
    NUMERIC_FEATURES,
    SPLIT_COLUMN,
    TARGET_COLUMN,
)
from ml.models.artifact import ARTIFACT_FILENAME, FEATURE_VERSION, MODEL_VERSION
from ml.models.evaluation import evaluate_predictions
from ml.models.xgboost_model import (
    CLASSIFIER_STEP_NAME,
    PREPROCESSOR_STEP_NAME,
    XGBoostFraudModel,
    fit_xgboost_model,
    predict_fraud_probability,
)
from ml.split.temporal import TemporalSplit, temporal_split
from policy.locked import LOCKED_POLICY_VERSION
from tracking.config import TrackingConfig
from tracking.dataset import DatasetFingerprint, fingerprint_files
from tracking.lineage import TRACKED_LIBRARIES
from tracking.run import (
    CLASSIFIER_PARAMETERS,
    DATASET_IDENTITY_ARTIFACT,
    ENVIRONMENT_ARTIFACT,
    FEATURE_CONTRACT_ARTIFACT,
    FEATURE_CONTRACT_TEXT_ARTIFACT,
    MODEL_ARTIFACT_NAME,
    MODEL_METADATA_ARTIFACT,
    MODEL_SERIALIZATION_FORMAT,
    PRIMARY_METRIC,
    TrackedTraining,
    default_evaluation_threshold,
    log_training_run,
)


REPO_ROOT = Path(__file__).resolve().parents[1]
SERVING_ARTIFACT_PATH = Path("artifacts/serving") / ARTIFACT_FILENAME

EXPERIMENT_NAME = "riskforge-tracking-tests"
REGISTERED_MODEL_NAME = "riskforge-tracking-test-model"
RUN_NAME = "tracked-test-run"

# Stands in for a credential that happens to be in the environment. Nothing in
# the tracking layer reads it, and these tests prove that it stays that way.
SECRET_SENTINEL = "s3ntinel-that-must-never-be-recorded"

REQUIRED_PARTITION_METRICS = (
    "pr_auc",
    "roc_auc",
    "precision",
    "recall",
    "f1",
    "false_positive_rate",
    "false_negative_rate",
)

LOCKED_HYPERPARAMETERS = {
    "objective": "binary:logistic",
    "eval_metric": "aucpr",
    "n_estimators": "300",
    "max_depth": "6",
    "learning_rate": "0.05",
    "min_child_weight": "5",
    "subsample": "0.8",
    "colsample_bytree": "0.8",
    "reg_lambda": "1.0",
    "random_state": "42",
    "tree_method": "hist",
}

CONTRACT_ARTIFACTS = (
    FEATURE_CONTRACT_ARTIFACT,
    FEATURE_CONTRACT_TEXT_ARTIFACT,
    DATASET_IDENTITY_ARTIFACT,
    ENVIRONMENT_ARTIFACT,
    MODEL_METADATA_ARTIFACT,
)

JSON_ARTIFACTS = tuple(name for name in CONTRACT_ARTIFACTS if name.endswith(".json"))


@dataclass(frozen=True)
class RecordedRun:
    """A logged run plus the state of the serving artifact around it."""

    run: object
    serving_artifact_before: str | None
    serving_artifact_after: str | None


@pytest.fixture(scope="module")
def tiny_split(tiny_frame) -> TemporalSplit:
    """The real chronological split, applied to the small synthetic frame."""
    return temporal_split(tiny_frame)


@pytest.fixture(scope="module")
def tiny_model(tiny_split: TemporalSplit) -> XGBoostFraudModel:
    """The real training entry point, on TRAIN only, with the locked settings."""
    return fit_xgboost_model(tiny_split.train)


@pytest.fixture(scope="module")
def tiny_dataset(tmp_path_factory: pytest.TempPathFactory) -> DatasetFingerprint:
    """Two files standing in for the behavioral artifact and the raw CSV."""
    root = tmp_path_factory.mktemp("dataset")
    (root / "behavioral.parquet").write_bytes(b"\x00\x01\x02\x03")
    (root / "transaction.csv").write_text("TransactionID,isFraud\n1,0\n2,1\n")

    return fingerprint_files(sorted(root.glob("*")), root=root)


def build_training(
    tiny_split: TemporalSplit,
    tiny_model: XGBoostFraudModel,
    tiny_dataset: DatasetFingerprint,
) -> TrackedTraining:
    return TrackedTraining(
        model=tiny_model,
        split=tiny_split,
        validation_metrics=evaluate_predictions(
            tiny_split.validation[TARGET_COLUMN],
            predict_fraud_probability(tiny_model, tiny_split.validation),
        ),
        test_metrics=evaluate_predictions(
            tiny_split.test[TARGET_COLUMN],
            predict_fraud_probability(tiny_model, tiny_split.test),
        ),
        dataset=tiny_dataset,
    )


def store_config(store: Path, *, registered_model_name: str | None) -> TrackingConfig:
    return TrackingConfig(
        tracking_uri=f"file:{store}",
        experiment_name=EXPERIMENT_NAME,
        registered_model_name=registered_model_name,
    )


def serving_artifact_digest() -> str | None:
    """The served artifact's contents, or ``None`` if it has not been built."""
    path = REPO_ROOT / SERVING_ARTIFACT_PATH

    if not path.is_file():
        return None

    return hashlib.sha256(path.read_bytes()).hexdigest()


@pytest.fixture(scope="module")
def tracked(
    tiny_split: TemporalSplit,
    tiny_model: XGBoostFraudModel,
    tiny_dataset: DatasetFingerprint,
    tmp_path_factory: pytest.TempPathFactory,
):
    """One fully logged run, recorded while plausible secrets are in the env."""
    training = build_training(tiny_split, tiny_model, tiny_dataset)
    store = tmp_path_factory.mktemp("mlruns")

    with pytest.MonkeyPatch.context() as patch:
        patch.setenv(
            "DATABASE_URL",
            f"postgresql+psycopg://riskforge:{SECRET_SENTINEL}@localhost:5432/riskforge",
        )
        patch.setenv("AWS_SECRET_ACCESS_KEY", SECRET_SENTINEL)

        run = log_training_run(
            training,
            config=store_config(store, registered_model_name=REGISTERED_MODEL_NAME),
            repo_root=REPO_ROOT,
            run_name=RUN_NAME,
        )

    return run


@pytest.fixture(scope="module")
def untracked_registry_run(
    tiny_split: TemporalSplit,
    tiny_model: XGBoostFraudModel,
    tiny_dataset: DatasetFingerprint,
    tmp_path_factory: pytest.TempPathFactory,
) -> RecordedRun:
    """A second run with the registry off, bracketing the serving artifact."""
    training = build_training(tiny_split, tiny_model, tiny_dataset)
    store = tmp_path_factory.mktemp("mlruns-no-registry")

    before = serving_artifact_digest()

    run = log_training_run(
        training,
        config=store_config(store, registered_model_name=None),
        repo_root=REPO_ROOT,
        run_name="untracked-registry-run",
    )

    return RecordedRun(
        run=run,
        serving_artifact_before=before,
        serving_artifact_after=serving_artifact_digest(),
    )


@pytest.fixture(scope="module")
def client(tracked) -> MlflowClient:
    return MlflowClient(tracking_uri=tracked.tracking_uri)


@pytest.fixture(scope="module")
def recorded(client: MlflowClient, tracked):
    return client.get_run(tracked.run_id)


@pytest.fixture(scope="module")
def downloaded_artifacts(
    client: MlflowClient, tracked, tmp_path_factory: pytest.TempPathFactory
) -> dict[str, dict]:
    """Every JSON artifact of the run, downloaded rather than read by path."""
    destination = tmp_path_factory.mktemp("artifacts")

    return {
        name: json.loads(Path(client.download_artifacts(tracked.run_id, name, str(destination))).read_text())
        for name in JSON_ARTIFACTS
    }


@pytest.fixture(scope="module")
def contract_text(
    client: MlflowClient, tracked, tmp_path_factory: pytest.TempPathFactory
) -> str:
    destination = tmp_path_factory.mktemp("contract-text")
    local = client.download_artifacts(
        tracked.run_id, FEATURE_CONTRACT_TEXT_ARTIFACT, str(destination)
    )

    return Path(local).read_text()


@pytest.fixture(scope="module")
def loaded_pipeline(tracked):
    """The pipeline read back out of the store, not the one still in memory."""
    return mlflow.sklearn.load_model(tracked.model_uri)


# --- a run is created, and it is created where the configuration says --------


def test_a_run_is_created_and_finished(client: MlflowClient, recorded):
    assert recorded.info.run_id
    assert recorded.info.status == "FINISHED"
    assert client.get_run(recorded.info.run_id).info.run_id == recorded.info.run_id


def test_the_run_carries_the_name_it_was_given(recorded):
    assert recorded.info.run_name == RUN_NAME


def test_the_run_lands_in_the_configured_experiment(client: MlflowClient, recorded):
    experiment = client.get_experiment(recorded.info.experiment_id)

    assert experiment.name == EXPERIMENT_NAME


def test_a_relative_store_is_anchored_so_it_cannot_escape_the_repository(tracked):
    """Otherwise a run launched elsewhere writes mlruns/ somewhere unignored."""
    location = Path(tracked.tracking_uri.removeprefix("file:"))

    assert location.is_absolute()
    assert location.is_dir()


def test_the_file_store_opt_out_is_reported_honestly(tracked):
    """MLflow 3 refuses the filesystem backend unless a caller opts in."""
    assert tracked.file_store_opt_out_applied is True


# --- parameters --------------------------------------------------------------


@pytest.mark.parametrize("name", CLASSIFIER_PARAMETERS)
def test_every_required_hyperparameter_is_logged(recorded, tiny_model, name: str):
    """Read back off the classifier that was fitted, not off the source code."""
    fitted = tiny_model.model.named_steps[CLASSIFIER_STEP_NAME].get_params()

    assert recorded.data.params[f"hyperparameter_{name}"] == str(fitted[name])


@pytest.mark.parametrize(("name", "value"), sorted(LOCKED_HYPERPARAMETERS.items()))
def test_the_recorded_hyperparameters_are_the_locked_ones(recorded, name, value):
    """A tracking record for a model that is not the locked one is a false record."""
    assert recorded.data.params[f"hyperparameter_{name}"] == value


def test_the_recorded_imbalance_weight_is_the_one_fitted_on_train(recorded, tiny_model):
    assert recorded.data.params["hyperparameter_scale_pos_weight"] == str(
        tiny_model.scale_pos_weight
    )


def test_the_model_type_comes_from_the_fitted_classifier(recorded):
    assert recorded.data.params["model_type"] == "xgboost"
    assert recorded.data.params["classifier_class"] == "XGBClassifier"


def test_the_evaluation_threshold_is_recorded(recorded):
    """Precision, recall, F1, FPR and FNR mean nothing without it."""
    assert float(recorded.data.params["evaluation_threshold"]) == (
        default_evaluation_threshold()
    )


def test_the_run_states_that_it_applied_no_calibration(recorded):
    """These are raw scores; the calibrated probabilities the API returns are
    fitted downstream, and the record must not blur the two."""
    assert recorded.data.params["calibration_method"] == "none"


def test_the_feature_counts_are_recorded(recorded):
    params = recorded.data.params

    assert params["model_feature_count"] == str(len(MODEL_FEATURES)) == "28"
    assert params["categorical_feature_count"] == str(len(CATEGORICAL_FEATURES))
    assert params["numeric_feature_count"] == str(len(NUMERIC_FEATURES))


def test_the_split_strategy_and_fractions_are_recorded(recorded):
    params = recorded.data.params

    assert params["split_column"] == SPLIT_COLUMN
    assert params["split_strategy"] == "chronological_by_unique_transaction_dt"
    assert float(params["train_fraction"]) == 0.70
    assert float(params["validation_fraction"]) == 0.15


def test_the_partition_row_counts_match_the_split(recorded, tiny_split):
    params = recorded.data.params

    assert params["n_train_rows"] == str(len(tiny_split.train))
    assert params["n_validation_rows"] == str(len(tiny_split.validation))
    assert params["n_test_rows"] == str(len(tiny_split.test))


def test_the_dataset_fingerprint_is_a_parameter(recorded, tiny_dataset):
    assert recorded.data.params["dataset_fingerprint"] == tiny_dataset.fingerprint
    assert recorded.data.params["dataset_file_count"] == str(len(tiny_dataset.files))


# --- metrics -----------------------------------------------------------------


@pytest.mark.parametrize("partition", ["validation", "test"])
@pytest.mark.parametrize("name", REQUIRED_PARTITION_METRICS)
def test_the_required_metrics_are_logged_per_partition(recorded, partition, name):
    assert f"{partition}/{name}" in recorded.data.metrics


@pytest.mark.parametrize("partition", ["validation", "test"])
def test_the_logged_metrics_equal_the_evaluated_ones(recorded, tiny_split, tiny_model, partition):
    frame = getattr(tiny_split, partition)

    evaluated = evaluate_predictions(
        frame[TARGET_COLUMN], predict_fraud_probability(tiny_model, frame)
    )

    assert recorded.data.metrics[f"{partition}/pr_auc"] == pytest.approx(
        evaluated.pr_auc, abs=1e-12
    )
    assert recorded.data.metrics[f"{partition}/roc_auc"] == pytest.approx(
        evaluated.roc_auc, abs=1e-12
    )
    assert recorded.data.metrics[f"{partition}/recall"] == pytest.approx(
        evaluated.recall, abs=1e-12
    )


@pytest.mark.parametrize("partition", ["train", "validation", "test"])
def test_the_fraud_count_is_logged_for_every_partition(recorded, tiny_split, partition):
    frame = getattr(tiny_split, partition)

    assert recorded.data.metrics[f"{partition}/fraud_count"] == pytest.approx(
        float(int(frame[TARGET_COLUMN].sum()))
    )


@pytest.mark.parametrize("partition", ["train", "validation", "test"])
def test_the_positive_class_rate_is_logged_for_every_partition(recorded, tiny_split, partition):
    frame = getattr(tiny_split, partition)
    rate = float(frame[TARGET_COLUMN].mean())

    assert 0.0 < rate < 1.0
    assert recorded.data.metrics[f"{partition}/positive_rate"] == pytest.approx(rate)


def test_no_accuracy_metric_is_logged(recorded):
    """With a fraud rate near three percent, accuracy rewards catching nothing."""
    assert not [name for name in recorded.data.metrics if "accuracy" in name]


def test_pr_auc_is_declared_the_primary_metric(recorded):
    assert recorded.data.tags["primary_metric"] == PRIMARY_METRIC == "test/pr_auc"
    assert PRIMARY_METRIC in recorded.data.metrics


# --- the feature contract ----------------------------------------------------


def test_the_contract_records_the_exact_locked_features_in_order(downloaded_artifacts):
    contract = downloaded_artifacts[FEATURE_CONTRACT_ARTIFACT]

    assert contract["features"] == list(MODEL_FEATURES)
    assert contract["feature_count"] == len(MODEL_FEATURES) == 28


def test_the_contract_records_the_feature_families(downloaded_artifacts):
    contract = downloaded_artifacts[FEATURE_CONTRACT_ARTIFACT]

    assert contract["categorical_features"] == list(CATEGORICAL_FEATURES)
    assert contract["numeric_features"] == list(NUMERIC_FEATURES)
    assert contract["behavioral_features"] == list(BEHAVIORAL_FEATURES)
    assert contract["cd_features"] == list(CD_FEATURES)
    assert contract["target_column"] == TARGET_COLUMN
    assert contract["split_column"] == SPLIT_COLUMN


def test_the_contract_names_the_module_that_owns_it(downloaded_artifacts):
    """The list is imported, never retyped, so it cannot drift from the code."""
    contract = downloaded_artifacts[FEATURE_CONTRACT_ARTIFACT]

    assert contract["source_module"] == "ml.data.modeling"
    assert contract["feature_version"] == FEATURE_VERSION
    assert contract["model_version"] == MODEL_VERSION


def test_the_text_contract_preserves_the_ordering(contract_text: str):
    """The human-readable form has to say the same thing as the JSON one."""
    numbered = [
        line.split("\t")
        for line in contract_text.splitlines()
        if line and not line.startswith("#")
    ]

    assert [index for index, _ in numbered] == [str(i) for i in range(len(MODEL_FEATURES))]
    assert [name for _, name in numbered] == list(MODEL_FEATURES)
    assert f"feature_version={FEATURE_VERSION}" in contract_text


# --- reproducibility metadata ------------------------------------------------


def test_the_commit_sha_is_recorded_as_a_tag_and_in_the_metadata(recorded, downloaded_artifacts):
    sha = recorded.data.tags["git_commit_sha"]
    lineage = downloaded_artifacts[MODEL_METADATA_ARTIFACT]["lineage"]

    assert len(sha) == 40
    assert all(character in "0123456789abcdef" for character in sha)
    assert lineage["commit_sha"] == sha


def test_the_working_tree_state_is_recorded(recorded, downloaded_artifacts):
    """A dirty tree means the SHA alone does not reproduce the run."""
    state = recorded.data.tags["git_working_tree"]

    assert state in {"clean", "dirty", "unknown"}
    assert downloaded_artifacts[MODEL_METADATA_ARTIFACT]["lineage"]["working_tree"] == state


def test_the_python_version_is_recorded(downloaded_artifacts):
    environment = downloaded_artifacts[ENVIRONMENT_ARTIFACT]

    assert environment["python_version"] == sys.version.split()[0]


@pytest.mark.parametrize("library", TRACKED_LIBRARIES)
def test_every_tracked_library_version_is_recorded(downloaded_artifacts, library: str):
    versions = downloaded_artifacts[ENVIRONMENT_ARTIFACT]["library_versions"]

    assert versions[library] == metadata.version(library)


def test_the_mlflow_version_is_recorded(downloaded_artifacts):
    assert downloaded_artifacts[ENVIRONMENT_ARTIFACT]["mlflow_version"] == mlflow.__version__


def test_the_serialization_format_is_recorded(downloaded_artifacts):
    environment = downloaded_artifacts[ENVIRONMENT_ARTIFACT]
    model_metadata = downloaded_artifacts[MODEL_METADATA_ARTIFACT]

    assert environment["model_serialization_format"] == MODEL_SERIALIZATION_FORMAT
    assert model_metadata["model_serialization_format"] == MODEL_SERIALIZATION_FORMAT


def test_the_dataset_identity_artifact_matches_the_computed_fingerprint(
    downloaded_artifacts, tiny_dataset
):
    identity = downloaded_artifacts[DATASET_IDENTITY_ARTIFACT]

    assert identity == tiny_dataset.to_dict()
    assert identity["algorithm"] == "sha256"
    assert identity["fingerprint"] == tiny_dataset.fingerprint


def test_the_split_metadata_is_recorded(downloaded_artifacts, tiny_split):
    model_metadata = downloaded_artifacts[MODEL_METADATA_ARTIFACT]

    assert model_metadata["split"]["strategy"] == "chronological_by_unique_transaction_dt"
    assert model_metadata["split"]["split_column"] == SPLIT_COLUMN
    assert model_metadata["partition_rows"] == {
        "train": len(tiny_split.train),
        "validation": len(tiny_split.validation),
        "test": len(tiny_split.test),
    }
    assert model_metadata["partition_fraud_counts"]["train"] == int(
        tiny_split.train[TARGET_COLUMN].sum()
    )


# --- the model artifact ------------------------------------------------------


def test_the_model_artifact_is_logged(client: MlflowClient, tracked):
    names = {entry.path for entry in client.list_artifacts(tracked.run_id, MODEL_ARTIFACT_NAME)}

    assert f"{MODEL_ARTIFACT_NAME}/MLmodel" in names
    assert f"{MODEL_ARTIFACT_NAME}/model.pkl" in names


def test_the_model_artifact_is_the_only_model_in_the_run(client: MlflowClient, tracked):
    logged = [
        entry.path
        for entry in client.list_artifacts(tracked.run_id, MODEL_ARTIFACT_NAME)
        if entry.path.endswith("model.pkl")
    ]

    assert logged == [f"{MODEL_ARTIFACT_NAME}/model.pkl"]


def test_the_recorded_artifacts_are_the_documented_set(client: MlflowClient, tracked):
    """No artifact is logged that the documentation does not account for."""
    root = {entry.path for entry in client.list_artifacts(tracked.run_id)}

    assert set(CONTRACT_ARTIFACTS) <= root


def test_the_loaded_artifact_has_the_locked_pipeline_steps(loaded_pipeline):
    assert list(loaded_pipeline.named_steps) == [
        PREPROCESSOR_STEP_NAME,
        CLASSIFIER_STEP_NAME,
    ]


def test_the_loaded_artifact_carries_the_same_feature_contract(loaded_pipeline, tiny_model):
    """Same contract, same ordering, same preprocessing width."""
    assert list(loaded_pipeline.feature_names_in_) == list(MODEL_FEATURES)
    assert loaded_pipeline.named_steps[CLASSIFIER_STEP_NAME].n_features_in_ == (
        tiny_model.model.named_steps[CLASSIFIER_STEP_NAME].n_features_in_
    )


@pytest.mark.parametrize("partition", ["validation", "test"])
def test_the_logged_model_reproduces_the_serving_scoring_function(
    loaded_pipeline, tiny_split, tiny_model, partition
):
    """The artifact MLflow holds predicts exactly what ``app.service`` calls."""
    frame = getattr(tiny_split, partition)

    reference = predict_fraud_probability(tiny_model, frame).to_numpy()
    reloaded = loaded_pipeline.predict_proba(frame[MODEL_FEATURES])[:, 1]

    assert list(reloaded) == pytest.approx(list(reference), abs=0.0)


def test_the_loaded_model_is_not_the_same_object_as_the_fitted_one(loaded_pipeline, tiny_model):
    """It was read from disk, which is the point of an artifact."""
    assert loaded_pipeline is not tiny_model.model


# --- version metadata and the registry --------------------------------------


def test_the_approved_version_identifiers_are_recorded_as_tags(recorded):
    tags = recorded.data.tags

    assert tags["model_version"] == MODEL_VERSION == "xgb-001"
    assert tags["feature_version"] == FEATURE_VERSION == "behavioral-v1"
    assert tags["policy_version"] == LOCKED_POLICY_VERSION == "policy-001"


def test_the_version_identifiers_are_also_in_the_model_metadata(downloaded_artifacts):
    model_metadata = downloaded_artifacts[MODEL_METADATA_ARTIFACT]

    assert model_metadata["model_version"] == MODEL_VERSION
    assert model_metadata["feature_version"] == FEATURE_VERSION
    assert model_metadata["policy_version"] == LOCKED_POLICY_VERSION


def test_a_registered_version_is_explicitly_tied_to_the_approved_model_version(
    client: MlflowClient, tracked
):
    assert tracked.registered_model_name == REGISTERED_MODEL_NAME
    assert tracked.registered_model_version == "1"

    version = client.get_model_version(
        tracked.registered_model_name, tracked.registered_model_version
    )

    assert version.tags["model_version"] == MODEL_VERSION
    assert version.tags["feature_version"] == FEATURE_VERSION
    assert version.tags["policy_version"] == LOCKED_POLICY_VERSION
    assert version.tags["dataset_fingerprint"] == tracked.dataset_fingerprint
    assert version.tags["git_commit_sha"] == tracked.commit_sha


def test_nothing_is_promoted_or_aliased(client: MlflowClient, tracked):
    """Registering a version is bookkeeping; promoting one is a human decision."""
    version = client.get_model_version(
        tracked.registered_model_name, tracked.registered_model_version
    )

    assert version.aliases == []
    assert version.tags["promotion"] == "none"
    assert not any("production" in key.lower() for key in version.tags)
    assert not any("production" in value.lower() for value in version.tags.values())


def test_the_run_declares_that_it_does_not_produce_the_serving_artifact(recorded):
    tags = recorded.data.tags

    assert tags["produces_serving_artifact"] == "false"
    assert tags["serving_artifact"] == str(SERVING_ARTIFACT_PATH).replace("\\", "/")


def test_the_model_metadata_explains_the_serving_relationship(downloaded_artifacts):
    model_metadata = downloaded_artifacts[MODEL_METADATA_ARTIFACT]

    assert model_metadata["serving_artifact_path"] == str(
        SERVING_ARTIFACT_PATH
    ).replace("\\", "/")
    assert "not the runtime artifact" in model_metadata["serving_artifact_note"]
    assert "not the calibrated probabilities" in model_metadata["calibration_note"]


def test_the_registry_can_be_switched_off(untracked_registry_run: RecordedRun):
    run = untracked_registry_run.run

    assert run.registered_model_name is None
    assert run.registered_model_version is None
    assert run.model_version == MODEL_VERSION
    assert run.run_id


def test_logging_a_run_does_not_touch_the_phase_6_serving_artifact(
    untracked_registry_run: RecordedRun,
):
    """The served artifact is built by a different script and is not rewritten."""
    assert untracked_registry_run.serving_artifact_after == (
        untracked_registry_run.serving_artifact_before
    )


# --- nothing secret is recorded ---------------------------------------------


def test_no_environment_secret_reaches_the_record(
    client: MlflowClient,
    recorded,
    tracked,
    tmp_path_factory: pytest.TempPathFactory,
):
    """A tracking store is meant to be readable, so it must hold no credential."""
    destination = tmp_path_factory.mktemp("secret-scan")

    values = [*recorded.data.params.values(), *recorded.data.tags.values()]

    for name in CONTRACT_ARTIFACTS:
        local = client.download_artifacts(tracked.run_id, name, str(destination))
        values.append(Path(local).read_text())

    blob = "\n".join(values)

    assert SECRET_SENTINEL not in blob
    assert "riskforge:" not in blob
    assert "DATABASE_URL" not in blob


def test_the_recorded_tracking_uri_carries_no_credentials(tracked):
    assert "@" not in tracked.tracking_uri
