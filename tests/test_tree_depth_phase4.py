from scripts import resume_tree_depth_phase2 as resume
from scripts import run_tree_depth_phase2 as phase2
from scripts import run_tree_depth_phase4 as phase4


def test_phase4_compares_depth_eight_nine_and_ten():
    assert phase4.PHASE4_DEPTHS == (8, 9, 10)
    assert phase4.PHASE4_BASELINE_DEPTH == 8


def test_phase4_delegates_to_locked_phase2_protocol(monkeypatch):
    called = []
    monkeypatch.setattr(phase2, "main", lambda: called.append(True))
    monkeypatch.setattr(phase2, "DEPTHS", (3, 4, 5, 6))
    monkeypatch.setattr(phase2, "__file__", "phase2_runner.py")
    monkeypatch.setattr(phase2, "bootstrap_comparisons", lambda *_: None)

    phase4.main()

    assert called == [True]
    assert phase2.DEPTHS == (8, 9, 10)
    assert phase2.__file__ == phase4.__file__
    assert phase2.bootstrap_comparisons is phase4.optimized_bootstrap_comparisons


def test_phase4_bootstrap_uses_depth_eight_as_anchor(monkeypatch, tmp_path):
    expected = {"comparisons": []}
    calls = []

    def fake_run_bootstrap(source, output, selection, *, baseline_depth):
        calls.append((source, output, selection, baseline_depth))
        return expected

    monkeypatch.setattr(resume, "run_bootstrap", fake_run_bootstrap)
    selection = {"selected_depth": 10}
    result = phase4.optimized_bootstrap_comparisons(tmp_path, selection)

    assert result == expected
    assert calls == [(tmp_path, tmp_path, selection, 8)]
