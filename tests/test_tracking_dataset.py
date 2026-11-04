"""The dataset identity is computed from the real files, and only from them.

The raw IEEE-CIS data is gitignored, so a run cannot cite it the way it cites a
commit. The fingerprint is what stands in for that citation, which makes two
properties load-bearing: it must be reproducible from the same bytes, and it must
change when the bytes change. A digest that survived a regenerated dataset would
be worse than no digest at all.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from tracking.dataset import (
    CHUNK_BYTES,
    HASH_ALGORITHM,
    DatasetFingerprint,
    DatasetIdentityError,
    digest_file,
    fingerprint_files,
)


@pytest.fixture
def dataset_root(tmp_path: Path) -> Path:
    """Two files under a root, standing in for the behavioral and raw inputs."""
    root = tmp_path / "data"
    (root / "raw").mkdir(parents=True)
    (root / "processed").mkdir()

    (root / "raw" / "train_transaction.csv").write_text(
        "TransactionID,isFraud\n1,0\n2,1\n"
    )
    (root / "processed" / "train_behavioral_v1.parquet").write_bytes(b"\x00\x01\x02")

    return root


def dataset_files(root: Path) -> list[Path]:
    return sorted(path for path in root.rglob("*") if path.is_file())


def test_the_digest_matches_a_direct_hash_of_the_file_bytes(dataset_root: Path):
    path = dataset_root / "processed" / "train_behavioral_v1.parquet"

    recorded = digest_file(path)

    assert recorded.algorithm == HASH_ALGORITHM == "sha256"
    assert recorded.digest == hashlib.sha256(path.read_bytes()).hexdigest()
    assert recorded.size_bytes == path.stat().st_size


def test_a_file_larger_than_one_chunk_is_hashed_completely(tmp_path: Path):
    """Chunked reading must not truncate: a partial digest is a false identity."""
    path = tmp_path / "big.bin"
    payload = bytes(range(256)) * (CHUNK_BYTES // 256 + 97)
    path.write_bytes(payload)

    assert len(payload) > CHUNK_BYTES
    assert digest_file(path).digest == hashlib.sha256(payload).hexdigest()
    assert digest_file(path).size_bytes == len(payload)


def test_the_fingerprint_is_reproducible(dataset_root: Path):
    files = dataset_files(dataset_root)

    first = fingerprint_files(files, root=dataset_root)
    second = fingerprint_files(files, root=dataset_root)

    assert first.fingerprint == second.fingerprint
    assert first.to_dict() == second.to_dict()


def test_the_fingerprint_does_not_depend_on_argument_order(dataset_root: Path):
    files = dataset_files(dataset_root)

    forward = fingerprint_files(files, root=dataset_root)
    backward = fingerprint_files(list(reversed(files)), root=dataset_root)

    assert forward.fingerprint == backward.fingerprint
    assert forward.paths == backward.paths


def test_changing_one_byte_changes_the_fingerprint(dataset_root: Path):
    files = dataset_files(dataset_root)
    before = fingerprint_files(files, root=dataset_root)

    csv_path = dataset_root / "raw" / "train_transaction.csv"
    csv_path.write_text(csv_path.read_text() + "3,0\n")

    after = fingerprint_files(files, root=dataset_root)

    assert after.fingerprint != before.fingerprint
    assert after.total_bytes > before.total_bytes


def test_moving_a_file_to_a_different_path_changes_the_fingerprint(tmp_path: Path):
    """Location is part of a dataset's identity, not just its contents."""
    root = tmp_path / "data"
    root.mkdir()
    (root / "a.csv").write_text("x\n1\n")
    original = fingerprint_files([root / "a.csv"], root=root)

    (root / "b.csv").write_text("x\n1\n")
    (root / "a.csv").unlink()
    renamed = fingerprint_files([root / "b.csv"], root=root)

    assert original.fingerprint != renamed.fingerprint
    assert original.files[0].digest == renamed.files[0].digest


def test_paths_are_recorded_relative_to_the_root(dataset_root: Path):
    fingerprint = fingerprint_files(dataset_files(dataset_root), root=dataset_root)

    assert fingerprint.paths == (
        "processed/train_behavioral_v1.parquet",
        "raw/train_transaction.csv",
    )


def test_the_recorded_fingerprint_covers_every_input_file(dataset_root: Path):
    fingerprint = fingerprint_files(dataset_files(dataset_root), root=dataset_root)

    assert len(fingerprint.files) == 2
    assert fingerprint.total_bytes == sum(entry.size_bytes for entry in fingerprint.files)
    assert fingerprint.fingerprint != fingerprint.files[0].digest


def test_the_fingerprint_serialises_to_json(dataset_root: Path):
    fingerprint = fingerprint_files(dataset_files(dataset_root), root=dataset_root)

    payload = json.loads(json.dumps(fingerprint.to_dict()))

    assert payload["algorithm"] == HASH_ALGORITHM
    assert payload["fingerprint"] == fingerprint.fingerprint
    assert payload["file_count"] == 2
    assert payload["total_bytes"] == fingerprint.total_bytes
    assert [entry["path"] for entry in payload["files"]] == list(fingerprint.paths)
    assert {entry["algorithm"] for entry in payload["files"]} == {HASH_ALGORITHM}


def test_the_fingerprint_is_immutable(dataset_root: Path):
    fingerprint = fingerprint_files(dataset_files(dataset_root), root=dataset_root)

    assert isinstance(fingerprint, DatasetFingerprint)

    with pytest.raises(AttributeError):
        fingerprint.fingerprint = "0" * 64  # type: ignore[misc]


def test_an_empty_input_list_is_rejected(dataset_root: Path):
    with pytest.raises(DatasetIdentityError, match="No dataset files"):
        fingerprint_files([], root=dataset_root)


def test_a_missing_file_is_rejected(tmp_path: Path):
    with pytest.raises(DatasetIdentityError, match="not an existing file"):
        fingerprint_files([tmp_path / "absent.csv"], root=tmp_path)


def test_a_directory_is_rejected(dataset_root: Path):
    with pytest.raises(DatasetIdentityError, match="not an existing file"):
        fingerprint_files([dataset_root / "raw"], root=dataset_root)


def test_an_empty_file_is_rejected(tmp_path: Path):
    (tmp_path / "empty.csv").write_bytes(b"")

    with pytest.raises(DatasetIdentityError, match="is empty"):
        fingerprint_files([tmp_path / "empty.csv"], root=tmp_path)


def test_a_file_outside_the_root_is_rejected(tmp_path: Path):
    """It could not be recorded under a path that identifies the dataset."""
    root = tmp_path / "data"
    root.mkdir()
    outside = tmp_path / "outside.csv"
    outside.write_text("x\n")

    with pytest.raises(DatasetIdentityError, match="outside the dataset root"):
        fingerprint_files([outside], root=root)


def test_a_non_directory_root_is_rejected(tmp_path: Path):
    a_file = tmp_path / "not-a-directory"
    a_file.write_text("x\n")

    with pytest.raises(DatasetIdentityError, match="is not a directory"):
        fingerprint_files([a_file], root=a_file)


def test_a_relative_path_is_resolved_against_the_root(
    dataset_root: Path, monkeypatch: pytest.MonkeyPatch
):
    """Callers pass repository-relative paths; the root is what anchors them."""
    absolute = fingerprint_files(dataset_files(dataset_root), root=dataset_root)

    monkeypatch.chdir(dataset_root)

    relative = fingerprint_files(
        [
            Path("raw/train_transaction.csv"),
            Path("processed/train_behavioral_v1.parquet"),
        ],
        root=Path("."),
    )

    assert relative.fingerprint == absolute.fingerprint
    assert relative.paths == absolute.paths
