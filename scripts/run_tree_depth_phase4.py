"""Run the Phase 4 XGBoost depth-boundary experiment (8, 9, and 10).

Reuses the locked Phase 2 chronological fitting, selection, confirmation, and
protection protocol. The optimized paired bootstrap compares the selected
candidate with the shared Phase 3 depth-8 anchor and the Phase 4 runner-up.
"""

from __future__ import annotations

from pathlib import Path

from scripts import resume_tree_depth_phase2 as resume
from scripts import run_tree_depth_phase2 as phase2

PHASE4_DEPTHS = (8, 9, 10)
PHASE4_BASELINE_DEPTH = 8


def optimized_bootstrap_comparisons(output: Path, selection: dict) -> dict:
    """Use the same paired sampler with Phase 3 depth 8 as the anchor."""
    return resume.run_bootstrap(
        output,
        output,
        selection,
        baseline_depth=PHASE4_BASELINE_DEPTH,
    )


def main() -> None:
    phase2.DEPTHS = PHASE4_DEPTHS
    phase2.bootstrap_comparisons = optimized_bootstrap_comparisons
    phase2.__file__ = __file__
    phase2.main()


if __name__ == "__main__":
    main()
