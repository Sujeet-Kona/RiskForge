"""A deterministic identity for the data a model was trained on.

The raw IEEE-CIS files are deliberately not committed, so a run cannot point at
them the way it points at a commit. What it can carry is a fingerprint: the
SHA-256 of every input file plus one combined digest derived from them. Two runs
over the same bytes produce the same fingerprint, and a run whose inputs were
regenerated produces a different one — which is exactly the question lineage has
to answer.

The digests are computed here from the actual local files. Nothing is inferred
from a filename, a size alone, or a constant: a fabricated dataset hash would be
worse than none, because it would look like evidence.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from pathlib import Path


HASH_ALGORITHM = "sha256"

# Large enough that the 683 MB transaction CSV is hashed in a bounded number of
# reads, small enough that memory use stays flat regardless of file size.
CHUNK_BYTES = 8 * 1024 * 1024


class DatasetIdentityError(RuntimeError):
    """The input files cannot be fingerprinted, so no identity can be claimed."""


@dataclass(frozen=True)
class DatasetFileDigest:
    """One input file, identified by where it sits under the dataset root."""

    path: str
    algorithm: str
    digest: str
    size_bytes: int

    def to_dict(self) -> dict[str, object]:
        return {
            "path": self.path,
            "algorithm": self.algorithm,
            "digest": self.digest,
            "size_bytes": self.size_bytes,
        }


@dataclass(frozen=True)
class DatasetFingerprint:
    """The identity of a whole set of inputs.

    ``fingerprint`` is a digest of the per-file digests, not of the concatenated
    file bytes, so it is stable under any reordering of the input list and cheap
    to compare.
    """

    algorithm: str
    files: tuple[DatasetFileDigest, ...]
    fingerprint: str
    total_bytes: int

    @property
    def paths(self) -> tuple[str, ...]:
        return tuple(entry.path for entry in self.files)

    def to_dict(self) -> dict[str, object]:
        return {
            "algorithm": self.algorithm,
            "fingerprint": self.fingerprint,
            "total_bytes": self.total_bytes,
            "file_count": len(self.files),
            "files": [entry.to_dict() for entry in self.files],
        }


def digest_file(path: Path) -> DatasetFileDigest:
    """Hash one file in chunks and report its size alongside."""
    hasher = hashlib.new(HASH_ALGORITHM)
    size = 0

    with path.open("rb") as handle:
        while chunk := handle.read(CHUNK_BYTES):
            hasher.update(chunk)
            size += len(chunk)

    return DatasetFileDigest(
        path=path.name,
        algorithm=HASH_ALGORITHM,
        digest=hasher.hexdigest(),
        size_bytes=size,
    )


def fingerprint_files(
    paths: list[Path],
    *,
    root: Path,
) -> DatasetFingerprint:
    """Fingerprint every file in ``paths``, recording each relative to ``root``.

    ``root`` is what makes the recorded paths meaningful: a digest of
    ``data/raw/ieee_cis/train_transaction.csv`` says which dataset was used,
    while a digest of ``/Users/someone/.../train_transaction.csv`` says whose
    machine it ran on. Files outside ``root`` are rejected rather than recorded
    under a misleading relative path.
    """
    if not paths:
        raise DatasetIdentityError("No dataset files were supplied.")

    resolved_root = root.resolve()

    if not resolved_root.is_dir():
        raise DatasetIdentityError(
            f"The dataset root {resolved_root} is not a directory."
        )

    digests: list[DatasetFileDigest] = []

    for path in paths:
        resolved = path.resolve()

        try:
            relative = resolved.relative_to(resolved_root)
        except ValueError:
            raise DatasetIdentityError(
                f"{resolved} is outside the dataset root {resolved_root}, so it "
                "cannot be recorded under a path that identifies the dataset."
            ) from None

        if not resolved.is_file():
            raise DatasetIdentityError(f"{resolved} is not an existing file.")

        digest = digest_file(resolved)

        if digest.size_bytes == 0:
            raise DatasetIdentityError(
                f"{relative.as_posix()} is empty. An empty input has no dataset "
                "identity worth recording."
            )

        digests.append(
            DatasetFileDigest(
                path=relative.as_posix(),
                algorithm=digest.algorithm,
                digest=digest.digest,
                size_bytes=digest.size_bytes,
            )
        )

    digests.sort(key=lambda entry: entry.path)

    combined = hashlib.new(HASH_ALGORITHM)

    for entry in digests:
        combined.update(
            f"{entry.path}\t{entry.size_bytes}\t{entry.digest}\n".encode("utf-8")
        )

    return DatasetFingerprint(
        algorithm=HASH_ALGORITHM,
        files=tuple(digests),
        fingerprint=combined.hexdigest(),
        total_bytes=sum(entry.size_bytes for entry in digests),
    )
