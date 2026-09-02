# -*- coding: utf-8 -*-
"""EHOPT10 配方输入与因果性回归测试。"""
import numpy as np
import pandas as pd

from gcn.data.sample import make_sample_data
from gcn.recipes.gcn_main import VERSIONS, _stage_confirmation, compute_ehopt10


def _rising(n):
    close = np.arange(10.0, 10.0 + n)
    return pd.DataFrame({
        "open": close - 0.1,
        "high": close + 0.2,
        "low": close - 0.2,
        "close": close,
        "volume": np.full(n, 1000.0),
    }, index=pd.date_range("2025-01-01", periods=n, freq="D"))


def test_recipe_rejects_unknown_version_and_nonpositive_n():
    df = _rising(30)
    for kwargs in ({"version": "v6"}, {"N": 0}, {"N": -1}):
        try:
            compute_ehopt10(df, **kwargs)
        except ValueError:
            pass
        else:
            raise AssertionError(f"应拒绝参数: {kwargs}")


def test_nine_turn_labels_match_completed_and_live_reference_sequences():
    incomplete = compute_ehopt10(_rising(5), N=1)
    assert incomplete["NINE2_UP_LABEL"].tolist() == [0.0] * 5

    live = compute_ehopt10(_rising(7), N=1)
    assert live["NINE2_UP_LABEL"].tolist() == [0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0]

    completed = compute_ehopt10(_rising(12), N=1)
    assert completed["NINE2_UP_LABEL"].tolist() == [
        0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 0.0, 0.0, 0.0,
    ]
    assert completed["NINE2_UP_9"].tolist() == [False] * 9 + [True, False, False]


def test_stage_confirmation_requires_breakout_and_ma60_within_window():
    idx = pd.date_range("2025-01-01", periods=8, freq="D")
    setup = pd.Series([True] + [False] * 7, index=idx)
    high = pd.Series([10.0] * 8, index=idx)
    close = pd.Series([9.8, 10.5, 11.1, 11.3, 11.4, 11.5, 11.6, 11.7], index=idx)
    ma60 = pd.Series([9.0, 11.0, 10.8, 10.8, 10.8, 10.8, 10.8, 10.8], index=idx)

    entry, expired = _stage_confirmation(setup, high, close, ma60, window=5)

    assert entry.tolist() == [False, False, True, False, False, False, False, False]
    assert not expired.any()


def test_stage_confirmation_expires_on_last_valid_bar_unless_confirmed():
    idx = pd.date_range("2025-01-01", periods=7, freq="D")
    setup = pd.Series([True] + [False] * 6, index=idx)
    high = pd.Series([10.0] * 7, index=idx)
    ma60 = pd.Series([9.0] * 7, index=idx)

    late_close = pd.Series([9.0] * 6 + [11.0], index=idx)
    entry, expired = _stage_confirmation(setup, high, late_close, ma60, window=5)
    assert not entry.any()
    assert expired.tolist() == [False, False, False, False, False, True, False]

    final_bar_close = pd.Series([9.0] * 5 + [11.0, 11.5], index=idx)
    entry, expired = _stage_confirmation(setup, high, final_bar_close, ma60, window=5)
    assert entry.tolist() == [False, False, False, False, False, True, False]
    assert not expired.any()


def test_experimental_version_is_isolated_from_v4_schema():
    df = _rising(250)
    stable = compute_ehopt10(df, version="v4")
    experiment = compute_ehopt10(df, version="v4-exp")

    assert "v4-exp" in VERSIONS
    for col in ("B_STAGE_SETUP", "B_STAGE_ENTRY_SIGNAL", "B_STAGE_EXPIRED"):
        assert col not in stable
        assert col in experiment
    assert experiment["B_STAGE_SETUP"].equals(experiment["B_STAGE_SIGNAL"])


def test_v5_confirms_the_full_v4_b_signal_against_ma20_within_five_bars():
    df = make_sample_data(900, seed=11)
    stable = compute_ehopt10(df, version="v4")
    v5 = compute_ehopt10(df, version="v5")
    expected_entry, expected_expired = _stage_confirmation(
        stable["B_SIGNAL"], stable["HIGH"], stable["CLOSE"], stable["MID"], window=5,
    )

    assert "v5" in VERSIONS
    assert v5["B_SETUP"].equals(stable["B_SIGNAL"])
    assert v5["B_ENTRY_SIGNAL"].equals(expected_entry)
    assert v5["B_SIGNAL"].equals(expected_entry)
    assert v5["B_SETUP_EXPIRED"].equals(expected_expired)
    assert v5["ICON_JUEFAN"].equals(stable["ICON_JUEFAN"])
