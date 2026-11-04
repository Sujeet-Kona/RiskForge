"""The serving artifact is the only state inference is allowed to have.

These tests cover the two promises the API depends on: a bundle written offline
comes back able to produce the same scores, and a bundle that could not have
come from the locked pipeline is refused rather than served.
"""

from __future__ import annotations

import dataclasses
import json

import joblib
import numpy as np
import pandas as pd
import pytest

from ml.data.modeling import MODEL_FEATURES
from ml.models.artifact import (
    ARTIFACT_FILENAME,
    ARTIFACT_SCHEMA_VERSION,
    ArtifactError,
    ArtifactNotFoundError,
    RiskForgeArtifact,
    current_library_versions,
    load_artifact,
    save_artifact,
    utc_now_iso,
)
from ml.models.calibration import FraudCalibrator, apply_calibrator


def _raw_probabilities(artifact: RiskForgeArtifact, frame: pd.DataFrame) -> np.ndarray:
    return artifact.pipeline.predict_proba(frame[MODEL_FEATURES])[:, 1]


def test_the_artifact_survives_a_write_and_read_cycle(tiny_artifact, tmp_path):
    path = save_artifact(tiny_artifact, tmp_path / ARTIFACT_FILENAME)

    assert path.is_file()

    loaded = load_artifact(path)

    assert loaded.model_version == tiny_artifact.model_version
    assert loaded.feature_version == tiny_artifact.feature_version
    assert loaded.policy_version == tiny_artifact.policy_version
    assert loaded.calibration_method == tiny_artifact.calibration_method
    assert loaded.review_threshold == tiny_artifact.review_threshold
    assert loaded.block_threshold == tiny_artifact.block_threshold
    assert loaded.scale_pos_weight == tiny_artifact.scale_pos_weight
    assert loaded.n_transformed_columns == tiny_artifact.n_transformed_columns
    assert loaded.feature_columns == tiny_artifact.feature_columns
    assert loaded.library_versions == tiny_artifact.library_versions
    assert loaded.built_at_utc == tiny_artifact.built_at_utc


def test_the_reloaded_bundle_scores_identically_to_the_original(
    tiny_artifact, tiny_artifact_path, tiny_frame
):
    """Loading is not a rebuild: the same rows must get the same probabilities."""
    sample = tiny_frame.head(64)

    expected = _raw_probabilities(tiny_artifact, sample)
    actual = _raw_probabilities(load_artifact(tiny_artifact_path), sample)

    np.testing.assert_array_equal(actual, expected)


def test_the_reloaded_calibrator_maps_scores_identically(
    tiny_artifact, tiny_artifact_path, tiny_frame
):
    sample = tiny_frame.head(64)
    raw = pd.Series(_raw_probabilities(tiny_artifact, sample), index=sample.index)

    expected = apply_calibrator(tiny_artifact.calibrator, raw)
    actual = apply_calibrator(load_artifact(tiny_artifact_path).calibrator, raw)

    np.testing.assert_allclose(actual.to_numpy(), expected.to_numpy())


def test_save_artifact_creates_missing_parent_directories(tiny_artifact, tmp_path):
    path = tmp_path / "nested" / "deeper" / ARTIFACT_FILENAME

    assert not path.parent.exists()

    save_artifact(tiny_artifact, path)

    assert path.is_file()


def test_save_artifact_refuses_anything_that_is_not_a_bundle(tmp_path):
    with pytest.raises(TypeError):
        save_artifact({"model_version": "xgb-001"}, tmp_path / ARTIFACT_FILENAME)


def test_describe_is_json_serialisable_and_carries_no_fitted_object(loaded_artifact):
    described = loaded_artifact.describe()

    assert json.loads(json.dumps(described)) == described
    assert "pipeline" not in described
    assert "calibrator" not in described
    assert described["n_features"] == len(MODEL_FEATURES)
    assert described["artifact_schema_version"] == ARTIFACT_SCHEMA_VERSION


def test_the_classifier_property_addresses_the_fitted_step(loaded_artifact):
    from ml.models.xgboost_model import CLASSIFIER_STEP_NAME

    assert loaded_artifact.classifier is loaded_artifact.pipeline.named_steps[
        CLASSIFIER_STEP_NAME
    ]
    assert loaded_artifact.classifier.n_features_in_ == (
        loaded_artifact.n_transformed_columns
    )


def test_a_missing_artifact_fails_clearly_and_names_the_offline_builder(tmp_path):
    path = tmp_path / ARTIFACT_FILENAME

    with pytest.raises(ArtifactNotFoundError) as error:
        load_artifact(path)

    assert "never trains" in str(error.value)


def test_artifact_not_found_is_a_kind_of_artifact_error():
    assert issubclass(ArtifactNotFoundError, ArtifactError)


def test_an_unreadable_artifact_is_refused(tmp_path):
    path = tmp_path / ARTIFACT_FILENAME
    path.write_bytes(b"this is not a joblib bundle")

    with pytest.raises(ArtifactError, match="could not be read"):
        load_artifact(path)


def test_a_foreign_object_at_the_artifact_path_is_refused(tmp_path):
    path = tmp_path / ARTIFACT_FILENAME
    joblib.dump({"model_version": "xgb-001"}, path)

    with pytest.raises(ArtifactError, match="not a RiskForgeArtifact"):
        load_artifact(path)


def test_a_bundle_built_under_other_library_versions_is_refused(
    tiny_artifact, tmp_path
):
    """A pickled estimator is only meaningful under the versions that made it."""
    doctored = dataclasses.replace(
        tiny_artifact,
        library_versions={"scikit-learn": "0.0.1", "xgboost": "0.0.1"},
    )
    path = save_artifact(doctored, tmp_path / ARTIFACT_FILENAME)

    with pytest.raises(ArtifactError, match="different library versions"):
        load_artifact(path)


def test_the_recorded_library_versions_are_the_installed_ones(tiny_artifact):
    assert tiny_artifact.library_versions == current_library_versions()
    assert set(tiny_artifact.library_versions) == {"scikit-learn", "xgboost"}


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("artifact_schema_version", 999, "Unsupported artifact schema version"),
        ("feature_columns", tuple(MODEL_FEATURES[:-1]), "different feature contract"),
        ("n_transformed_columns", 3, "transformed"),
        ("review_threshold", 0.5, "strictly less than block_threshold"),
        ("block_threshold", 1.5, "between 0 and 1"),
        ("model_version", "", "non-blank string"),
        ("feature_version", "   ", "non-blank string"),
        ("policy_version", "", "non-blank string"),
        ("scale_pos_weight", 0.0, "must be positive"),
        ("n_train_rows", 0, "empty training or calibration partition"),
        ("n_calibration_rows", 0, "empty training or calibration partition"),
        ("calibration_method", "sigmoid", "declares calibration method"),
    ],
)
def test_a_bundle_that_contradicts_the_locked_contract_is_never_constructed(
    tiny_artifact, field, value, message
):
    with pytest.raises(ArtifactError, match=message):
        dataclasses.replace(tiny_artifact, **{field: value})


def test_an_unfitted_calibrator_is_refused(tiny_artifact):
    with pytest.raises(ArtifactError, match="not fitted"):
        dataclasses.replace(
            tiny_artifact, calibrator=FraudCalibrator(method="isotonic")
        )


def test_a_non_calibrator_object_is_refused(tiny_artifact):
    with pytest.raises(TypeError):
        dataclasses.replace(tiny_artifact, calibrator="isotonic")


def test_a_non_pipeline_model_is_refused(tiny_artifact):
    with pytest.raises(TypeError):
        dataclasses.replace(tiny_artifact, pipeline="not a pipeline")


def test_a_pipeline_missing_the_classifier_step_is_refused(tiny_artifact):
    from sklearn.pipeline import Pipeline

    from ml.models.xgboost_model import PREPROCESSOR_STEP_NAME

    truncated = Pipeline(
        steps=[
            (
                PREPROCESSOR_STEP_NAME,
                tiny_artifact.pipeline.named_steps[PREPROCESSOR_STEP_NAME],
            )
        ]
    )

    with pytest.raises(ArtifactError, match="missing the locked classifier step"):
        dataclasses.replace(tiny_artifact, pipeline=truncated)


def test_timestamps_are_utc_and_second_precision():
    stamp = utc_now_iso()

    assert stamp.endswith("+00:00")
    assert "." not in stamp
