"""Run the Phase 3 depth-boundary experiment (depths 6, 7, and 8).

This entrypoint reuses the verified Phase 2 fitting, selection, confirmation,
and protection protocol. It replaces only the expensive bootstrap sampler with
the already-tested equivalent that precomputes timestamp groups once.
"""

from __future__ import annotations

from pathlib import Path

from scripts import resume_tree_depth_phase2 as resume
from scripts import run_tree_depth_phase2 as phase2

PHASE3_DEPTHS = (6, 7, 8)


def optimized_bootstrap_comparisons(output: Path, selection: dict) -> dict:
    """Run the identical paired bootstrap using precomputed timestamp groups."""
    return resume.run_bootstrap(output, output, selection)


def main() -> None:
    # The Phase 2 runner reads DEPTHS at runtime throughout fitting, selection,
    # bootstrap reference selection, and protocol metadata.
    phase2.DEPTHS = PHASE3_DEPTHS
    # Use the optimized implementation: same seeded draws, but avoid repeated
    # timestamp factorization and row-index scans inside each replicate.
    phase2.bootstrap_comparisons = optimized_bootstrap_comparisons
    # Make run metadata hash this Phase 3 entrypoint, not the imported module.
    phase2.__file__ = __file__
    phase2.main()


if __name__ == "__main__":
    main()
