"""r34过期Setup后空仓再准入时钟的训练筛查。"""

from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]


def test_r34_real_training_screen_rejects_all_fixed_reentry_clocks():
    from gcn.backtest.signal_research_r34 import training_screen

    events, summary = training_screen(
        ROOT / "reports/signal-audit-v5-review-20260904",
        ROOT / "reports/gcn-historical-r27-20260905/training",
    )
    rows = summary.set_index("mechanism")

    assert len(events) == 119
    assert rows.loc["expired-both-level", "mature_events"] == 14
    assert rows.loc["expired-both-level", "wins"] == 7
    assert np.isclose(rows.loc["expired-both-level", "median_net_pct"], -2.262, atol=0.001)
    assert rows.loc["upper-breakout", "mature_events"] == 12
    assert rows.loc["upper-breakout", "wins"] == 7
    assert rows.loc["upper-breakout", "interference"] == 5
    assert np.isclose(rows.loc["upper-breakout", "control_overlap_pct"], 91.667, atol=0.001)
    assert rows.loc["suppressed-raw-b", "mature_events"] == 4
    assert rows.loc["suppressed-raw-b", "wins"] == 0
    macd = events[events.mechanism.eq("macd-zero-reclaim")][
        ["episode_id", "trigger_date"]
    ].reset_index(drop=True)
    dif = events[events.mechanism.eq("dif-dea-reclaim")][
        ["episode_id", "trigger_date"]
    ].reset_index(drop=True)
    assert macd.equals(dif)
    assert set(rows["status"]) == {"rejected"}


def test_r34_runner_writes_no_candidate_archive(tmp_path):
    from gcn.backtest.signal_research_r34 import run_training_screen

    output = tmp_path / "r34"
    result = run_training_screen(
        ROOT / "reports/signal-audit-v5-review-20260904",
        ROOT / "reports/gcn-historical-r27-20260905/training",
        output,
    )
    assert result["decision"]["status"] == "no_candidate"
    assert result["decision"]["candidate"] is None
    assert result["decision"]["candidate_backtest_run"] is False
    assert result["decision"]["validation_run"] is False
    assert result["decision"]["production_changed"] is False
    assert set(result["manifest"]["outputs"]) == {
        "events.csv", "summary.csv", "decision.json", "protocol.md"
    }


def test_r34_formal_archive_reproduces_byte_for_byte(tmp_path):
    from gcn.backtest.signal_research_r34 import run_training_screen

    output = tmp_path / "reproduced"
    result = run_training_screen(
        ROOT / "reports/signal-audit-v5-review-20260904",
        ROOT / "reports/gcn-historical-r27-20260905/training",
        output,
    )
    formal = ROOT / "reports/gcn-historical-r34-20260908/training"
    for name in (*result["manifest"]["outputs"], "manifest.json"):
        assert (output / name).read_bytes() == (formal / name).read_bytes()
