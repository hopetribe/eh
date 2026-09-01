# -*- coding: utf-8 -*-
"""回测 CLI 启动与参数边界测试。"""
import contextlib
import io
import subprocess
import sys

from gcn.backtest.cli import print_report


def _run(*args):
    return subprocess.run(
        [sys.executable, "kk2_backtest.py", *args],
        capture_output=True, text=True, timeout=15,
    )


def test_cli_help_imports_cleanly():
    proc = _run("--help")
    assert proc.returncode == 0, proc.stderr
    assert "--version" in proc.stdout


def test_cli_rejects_invalid_cost_and_hold():
    assert _run("--cost", "-0.1").returncode == 2
    assert _run("--cost", "nan").returncode == 2
    assert _run("--max-hold", "0").returncode == 2


def test_cli_report_uses_excess_q_and_intraday_units():
    stats = {"n": 12, "win": 60.0, "mean": 2.5, "excess": 1.25,
             "t": 2.1, "p": 0.03, "q": 0.04}
    split = {"horizon": 5, "in_sample": stats, "out_sample": stats}
    report = {
        "timeframe": {"interval": "15m", "period_label": "15分钟",
                      "periods_per_year": 6552},
        "events": [
            {"signal": "B_SIGNAL", "label": "B买", "count": 12,
             "horizons": {"5": stats}, "split": split, "split5": split},
            {"signal": "_BASE", "label": "基线", "count": 100,
             "horizons": {"5": {"n": 90, "win": 52.0, "mean": 1.25,
                                    "t": None, "p": None, "q": None}},
             "split": None, "split5": None},
        ],
        "strategies": [],
    }
    output = io.StringIO()
    with contextlib.redirect_stdout(output):
        print_report("TEST", {}, 0.0, report, 100)
    text = output.getvalue()
    assert "5×15分钟" in text
    assert "超额 +1.25%" in text
    assert "t= 2.10" in text and "q=0.0400" in text
