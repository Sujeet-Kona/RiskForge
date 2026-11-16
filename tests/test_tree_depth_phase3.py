from scripts import run_tree_depth_phase2 as phase2
from scripts import run_tree_depth_phase3 as phase3


def test_phase3_compares_depth_six_seven_and_eight():
    assert phase3.PHASE3_DEPTHS == (6, 7, 8)


def test_phase3_delegates_to_locked_phase2_protocol(monkeypatch):
    called = []

    def fake_main():
        called.append(True)

    monkeypatch.setattr(phase2, "main", fake_main)
    phase3.main()

    assert called == [True]
    assert phase2.DEPTHS == (6, 7, 8)
