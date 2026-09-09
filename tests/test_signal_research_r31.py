from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
SNAPSHOT = ROOT / "reports/signal-audit-v5-review-20260904"


def test_r31_filter_is_strict_b_only_and_preserves_jf_collision(monkeypatch):
    import gcn.backtest.signal_research_r31 as research

    index = pd.date_range("2024-01-01", periods=6)
    original = pd.DataFrame(
        {
            "OPEN": [9.0, 10.0, 11.0, np.nan, np.inf, 9.0],
            "MID": [10.0, 10.0, 10.0, 10.0, 10.0, -1.0],
            "B_SIGNAL": [True, True, True, True, True, True],
            "ICON_JUEFAN": [False, False, False, False, False, True],
            "S_SIGNAL": [False, True, False, False, False, False],
            "ENTRY_STOP": [0.0] * 6,
            "ENTRY_LIMIT": [0.0] * 6,
            "USE_EXTRA": [False] * 6,
            "EXTRA_EXIT": [False] * 6,
        },
        index=index,
    )
    before = original.copy(deep=True)
    monkeypatch.setattr(research, "prior_signals", lambda frame: {"v5": original.copy(deep=True)})

    rules = research.candidate_signals(original)
    pd.testing.assert_frame_equal(original, before)
    pd.testing.assert_frame_equal(rules["v5"], before)
    candidate = rules[research.CHALLENGERS[0]]
    assert candidate.B_SIGNAL.tolist() == [False, True, True, True, True, True]
    assert candidate.ICON_JUEFAN.tolist() == before.ICON_JUEFAN.tolist()
    assert candidate.S_SIGNAL.tolist() == before.S_SIGNAL.tolist()
    assert bool(candidate.ICON_JUEFAN.iloc[5])


def test_r31_real_training_screen_has_exact_four_cross_source_losers():
    import gcn.backtest.signal_research_r31 as research

    rows = research.training_screen(
        ROOT / "reports/gcn-historical-r28-20260906/training"
    )
    assert rows[["symbol", "confirmation_date"]].to_records(index=False).tolist() == [
        ("AAOI", "2024-07-10"),
        ("SNOW", "2021-12-08"),
        ("SNOW", "2024-06-27"),
        ("TQQQ", "2023-10-09"),
    ]
    assert len(rows) == 4
    assert rows.symbol.nunique() == 3
    assert not rows.trade_win.any()
    assert set(rows.source_component) == {
        "B_BASE_BULL",
        "B_STAGE_COMPONENT",
        "B_CRASH_RECOVER",
    }


def test_r31_training_wrapper_binds_single_candidate(tmp_path):
    import gcn.backtest.signal_research_r31 as research

    decision = research.run_training(SNAPSHOT, tmp_path)
    metrics = pd.read_csv(tmp_path / "training.csv")
    assert metrics.rule.tolist() == ["v5", "B-confirm-open-mid-filter"]
    assert decision["research_version"] == "gcn-historical-r31"
    assert decision["selected"] is None
    assert decision["validation_status"] == "not_run_no_eligible_candidate"
    assert decision["failures"] == {"B-confirm-open-mid-filter": ["buy_covered"]}
    assert decision["recommended"] == "v5"
    assert not decision["production_changed"]
    control, candidate = metrics.to_dict("records")
    assert candidate["trades"] == control["trades"] - 4
    assert candidate["buy_covered"] == 9
    assert control["buy_covered"] == 11


def test_r31_formal_training_archive_reproduces_byte_for_byte(tmp_path):
    import gcn.backtest.signal_research_r31 as research

    formal = ROOT / "reports/gcn-historical-r31-20260908/training"
    research.run_training(SNAPSHOT, tmp_path)
    expected = sorted(path.relative_to(formal) for path in formal.rglob("*") if path.is_file())
    actual = sorted(path.relative_to(tmp_path) for path in tmp_path.rglob("*") if path.is_file())
    assert actual == expected
    for relative in expected:
        assert (tmp_path / relative).read_bytes() == (formal / relative).read_bytes()
