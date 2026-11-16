from scripts import resume_tree_depth_phase2 as resume
from scripts import run_tree_depth_phase2 as phase2
from scripts import run_tree_depth_phase3 as phase3


def test_phase3_compares_depth_six_seven_and_eight():
    assert phase3.PHASE3_DEPTHS == (6, 7, 8)


def test_phase3_delegates_to_locked_phase2_protocol(monkeypatch):
    called = []

    def fake_main():
        called.append(True)

    monkeypatch.setattr(phase2, "main", fake_main)
    monkeypatch.setattr(phase2, "DEPTHS", (3, 4, 5))
    monkeypatch.setattr(phase2, "__file__", "phase2_runner.py")
    monkeypatch.setattr(phase2, "bootstrap_comparisons", lambda *_: None)

    phase3.main()

    assert called == [True]
    assert phase2.DEPTHS == (6, 7, 8)
    assert phase2.__file__ == phase3.__file__
    assert phase2.bootstrap_comparisons is phase3.optimized_bootstrap_comparisons


def test_phase3_bootstrap_uses_verified_optimized_resume_helper(monkeypatch, tmp_path):
    expected = {"comparisons": [], "does_not_change_selection": True}
    calls = []

    def fake_run_bootstrap(source, output, selection):
        calls.append((source, output, selection))
        return expected

    monkeypatch.setattr(resume, "run_bootstrap", fake_run_bootstrap)
    selection = {"selected_depth": 6}
    result = phase3.optimized_bootstrap_comparisons(tmp_path, selection)

    assert result == expected
    assert calls == [(tmp_path, tmp_path, selection)]
