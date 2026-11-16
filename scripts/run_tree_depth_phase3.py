"""Run the Phase 3 depth-boundary experiment (depths 6, 7, and 8).

This is a thin, auditable entrypoint over the verified Phase 2 protocol. It
reuses the same locked partitions, feature pipeline, expanding folds, metrics,
bootstrap, protection snapshots, and retrospective confirmation. Only the
candidate depth set changes.
"""

from __future__ import annotations

from scripts import run_tree_depth_phase2 as phase2

PHASE3_DEPTHS = (6, 7, 8)


def main() -> None:
    # The Phase 2 runner reads DEPTHS at runtime throughout selection, fitting,
    # bootstrap reference selection, and protocol metadata. Override only this
    # candidate set; all other protocol constants and implementation stay fixed.
    phase2.DEPTHS = PHASE3_DEPTHS
    # Make run metadata hash this Phase 3 entrypoint, not the imported module.
    phase2.__file__ = __file__
    phase2.main()


if __name__ == "__main__":
    main()
