# -*- coding: utf-8 -*-
"""Provider-adjusted K-line revision/rebase contract tests."""
import json

import pandas as pd

from gcn.backtest import shadow_runner
from gcn.backtest.shadow_validation import (
    canonical_bar_hash, merge_accepted_bars, rebase_adjusted_incoming,
)


_OHLCV = ["open", "high", "low", "close", "volume"]


def _frame(dates, rows):
    return pd.DataFrame(
        rows, index=pd.to_datetime(dates), columns=_OHLCV, dtype=float,
    )


def test_registration_serialization_freezes_rebase_determinism_policy():
    assert shadow_runner._SERIALIZATION_PROTOCOL == {
        "adjusted_rebase_addition_uniqueness":
            "required_fail_closed_data_blocked",
        "adjusted_rebase_canonical_significant_digits": 12,
        "bar_hash": "sha256-v1",
        "canonical_json": "sorted_compact_utf8_lf",
        "float": "python_float_hex",
        "generation": "one_core_common_session_v1",
    }


def test_split_like_uniform_overlap_scales_price_and_volume_to_accepted_basis():
    accepted = _frame(
        ["2026-09-01", "2026-09-02", "2026-09-03"],
        [
            [100, 110, 90, 105, 1000],
            [102, 112, 92, 107, 1200],
            [104, 114, 94, 109, 1400],
        ],
    )
    incoming = _frame(
        ["2026-09-02", "2026-09-03", "2026-09-04"],
        [
            [51, 56, 46, 53.5, 2400],
            [52, 57, 47, 54.5, 2800],
            [53, 58, 48, 55.5, 3200],
        ],
    )

    rebased, metadata = rebase_adjusted_incoming(
        accepted, incoming, tolerance_ppm=1,
    )

    expected = incoming.copy().astype(float)
    expected.loc[:, ["open", "high", "low", "close"]] *= 2.0
    expected.loc[:, "volume"] *= 0.5
    pd.testing.assert_frame_equal(rebased, expected)
    assert metadata == {
        "overlap_rows": 2,
        "price_factor": 2.0,
        "volume_factor": 0.5,
        "price_rebased": True,
        "volume_rebased": True,
        "rebase_applied": True,
    }


def test_single_nonuniform_price_revision_is_rejected():
    accepted = _frame(
        ["2026-09-01", "2026-09-02"],
        [
            [100, 110, 90, 105, 1000],
            [102, 112, 92, 107, 1200],
        ],
    )
    incoming = accepted.copy()
    incoming.loc[incoming.index[-1], "open"] += 0.001

    try:
        rebase_adjusted_incoming(accepted, incoming, tolerance_ppm=1)
    except ValueError as error:
        assert "价格修订非统一缩放" in str(error)
        assert "2026-09-02 open" in str(error)
        assert "max_deviation_ppm=" in str(error)
    else:
        raise AssertionError("a nonuniform price rewrite must be rejected")


def test_price_ratio_jitter_blocks_an_ambiguously_scaled_addition():
    accepted = _frame(
        ["2026-09-01", "2026-09-02"],
        [
            [100, 110, 90, 105, 1000],
            [102, 112, 92, 107, 1200],
        ],
    )
    incoming = _frame(
        ["2026-09-01", "2026-09-02", "2026-09-03"],
        [
            [100 / 2.0000018, 55, 45, 52.5, 1000],
            [51, 56, 46, 53.5, 1200],
            [52, 57, 47, 54.5, 1400],
        ],
    )

    try:
        rebase_adjusted_incoming(accepted, incoming, tolerance_ppm=1)
    except ValueError as error:
        assert "价格缩放在容差内但新增行基准不唯一" in str(error)
        assert "DATA_BLOCKED" in str(error)
    else:
        raise AssertionError("ambiguous tolerated price jitter must fail closed")


def test_price_ratio_jitter_within_one_ppm_can_verify_overlap_only():
    accepted = _frame(
        ["2026-09-01", "2026-09-02"],
        [
            [100, 110, 90, 105, 1000],
            [102, 112, 92, 107, 1200],
        ],
    )
    incoming = accepted / 2.0
    incoming.loc[incoming.index[0], "open"] = 100 / 2.0000018
    incoming.loc[:, "volume"] = accepted["volume"]

    rebased, metadata = rebase_adjusted_incoming(
        accepted, incoming, tolerance_ppm=1,
    )

    pd.testing.assert_frame_equal(rebased, accepted, check_exact=True)
    assert metadata["overlap_rows"] == 2


def test_volume_zero_must_be_paired_on_every_overlap_row():
    accepted = _frame(
        ["2026-09-01", "2026-09-02"],
        [
            [100, 110, 90, 105, 0],
            [102, 112, 92, 107, 1000],
        ],
    )
    incoming = accepted.copy()
    incoming.loc[incoming.index[0], "volume"] = 10
    incoming.loc[incoming.index[1], "volume"] = 2000

    try:
        rebase_adjusted_incoming(accepted, incoming, tolerance_ppm=1)
    except ValueError as error:
        assert "成交量零值未成对" in str(error)
    else:
        raise AssertionError("unpaired zero volume must be rejected")


def test_single_nonuniform_positive_volume_revision_is_rejected():
    accepted = _frame(
        ["2026-09-01", "2026-09-02", "2026-09-03"],
        [
            [100, 110, 90, 105, 1000],
            [102, 112, 92, 107, 1200],
            [104, 114, 94, 109, 1400],
        ],
    )
    incoming = accepted.copy()
    incoming.loc[:, "volume"] = [2000, 2400, 2801]

    try:
        rebase_adjusted_incoming(accepted, incoming, tolerance_ppm=1)
    except ValueError as error:
        assert "成交量修订非统一缩放" in str(error)
        assert "2026-09-03 volume" in str(error)
        assert "max_deviation_ppm=" in str(error)
    else:
        raise AssertionError("a nonuniform volume rewrite must be rejected")


def test_all_zero_overlap_volume_uses_identity_factor():
    accepted = _frame(
        ["2026-09-01", "2026-09-02"],
        [
            [100, 110, 90, 105, 0],
            [102, 112, 92, 107, 0],
        ],
    )
    incoming = _frame(
        ["2026-09-01", "2026-09-02", "2026-09-03"],
        [
            [100, 110, 90, 105, 0],
            [102, 112, 92, 107, 0],
            [104, 114, 94, 109, 0],
        ],
    )

    rebased, metadata = rebase_adjusted_incoming(
        accepted, incoming, tolerance_ppm=1,
    )

    pd.testing.assert_frame_equal(rebased, incoming)
    assert metadata["volume_factor"] == 1.0
    assert metadata["volume_rebased"] is False
    assert metadata["rebase_applied"] is False


def test_existing_series_requires_overlap_to_verify_rebase_factors():
    accepted = _frame(
        ["2026-09-01", "2026-09-02"],
        [
            [100, 110, 90, 105, 1000],
            [102, 112, 92, 107, 1200],
        ],
    )
    incoming = _frame(
        ["2026-09-03", "2026-09-04"],
        [
            [104, 114, 94, 109, 1400],
            [106, 116, 96, 111, 1600],
        ],
    )

    try:
        rebase_adjusted_incoming(accepted, incoming, tolerance_ppm=1)
    except ValueError as error:
        assert "没有可验证的重叠日期" in str(error)
        assert "DATA_BLOCKED" in str(error)
    else:
        raise AssertionError("an existing series needs overlap for scale proof")


def test_no_positive_volume_pairs_rejects_nonzero_incoming_addition():
    accepted = _frame(
        ["2026-09-01", "2026-09-02"],
        [
            [100, 110, 90, 105, 0],
            [102, 112, 92, 107, 0],
        ],
    )
    incoming = _frame(
        ["2026-09-01", "2026-09-02", "2026-09-03"],
        [
            [100, 110, 90, 105, 0],
            [102, 112, 92, 107, 0],
            [104, 114, 94, 109, 1500],
        ],
    )

    try:
        rebase_adjusted_incoming(accepted, incoming, tolerance_ppm=1)
    except ValueError as error:
        assert "无正成交量对时必须全零" in str(error)
    else:
        raise AssertionError("unverified nonzero volume scale must be rejected")


def test_tolerance_must_be_a_nonnegative_plain_integer():
    bars = _frame(
        ["2026-09-01"],
        [[100, 110, 90, 105, 1000]],
    )

    for invalid in (True, 1.0, -1):
        try:
            rebase_adjusted_incoming(bars, bars, tolerance_ppm=invalid)
        except ValueError as error:
            assert "tolerance_ppm必须是非负整数" in str(error)
        else:
            raise AssertionError(f"invalid tolerance accepted: {invalid!r}")


def test_initial_registration_is_identity_with_json_safe_metadata():
    incoming = _frame(
        ["2026-09-01", "2026-09-02"],
        [
            [100, 110, 90, 105, 1000],
            [102, 112, 92, 107, 1200],
        ],
    )

    rebased, metadata = rebase_adjusted_incoming(
        None, incoming, tolerance_ppm=1,
    )

    pd.testing.assert_frame_equal(rebased, incoming)
    assert metadata == {
        "overlap_rows": 0,
        "price_factor": 1.0,
        "volume_factor": 1.0,
        "price_rebased": False,
        "volume_rebased": False,
        "rebase_applied": False,
    }
    assert json.loads(json.dumps(metadata, allow_nan=False)) == metadata


def test_paired_zero_volume_is_ignored_when_positive_pairs_prove_scale():
    accepted = _frame(
        ["2026-09-01", "2026-09-02", "2026-09-03"],
        [
            [100, 110, 90, 105, 0],
            [102, 112, 92, 107, 1000],
            [104, 114, 94, 109, 1200],
        ],
    )
    incoming = _frame(
        ["2026-09-01", "2026-09-02", "2026-09-03", "2026-09-04"],
        [
            [100, 110, 90, 105, 0],
            [102, 112, 92, 107, 2000],
            [104, 114, 94, 109, 2400],
            [106, 116, 96, 111, 3000],
        ],
    )

    rebased, metadata = rebase_adjusted_incoming(
        accepted, incoming, tolerance_ppm=1,
    )

    assert metadata["volume_factor"] == 0.5
    assert rebased.loc[pd.Timestamp("2026-09-04"), "volume"] == 1500.0


def test_positive_volume_ratio_jitter_blocks_an_ambiguously_scaled_addition():
    accepted = _frame(
        ["2026-09-01", "2026-09-02", "2026-09-03"],
        [
            [100, 110, 90, 105, 1000],
            [102, 112, 92, 107, 1200],
            [104, 114, 94, 109, 1400],
        ],
    )
    incoming = _frame(
        ["2026-09-01", "2026-09-02", "2026-09-03", "2026-09-04"],
        [
            [100, 110, 90, 105, 2000],
            [102, 112, 92, 107, 2400],
            [104, 114, 94, 109, 1400 / 0.50000045],
            [106, 116, 96, 111, 3000],
        ],
    )

    try:
        rebase_adjusted_incoming(accepted, incoming, tolerance_ppm=1)
    except ValueError as error:
        assert "成交量缩放在容差内但新增行基准不唯一" in str(error)
        assert "DATA_BLOCKED" in str(error)
    else:
        raise AssertionError("ambiguous tolerated volume jitter must fail closed")


def test_dividend_like_price_rebase_merges_without_rescaling_volume():
    accepted = _frame(
        ["2026-09-01", "2026-09-02"],
        [
            [100, 110, 90, 105, 1000],
            [102, 112, 92, 107, 1200],
        ],
    )
    price_factor = 1.025
    incoming = _frame(
        ["2026-09-01", "2026-09-02", "2026-09-03"],
        [
            [100 / price_factor, 110 / price_factor,
             90 / price_factor, 105 / price_factor, 1000],
            [102 / price_factor, 112 / price_factor,
             92 / price_factor, 107 / price_factor, 1200],
            [104 / price_factor, 114 / price_factor,
             94 / price_factor, 109 / price_factor, 1400],
        ],
    )

    rebased, metadata = rebase_adjusted_incoming(
        accepted, incoming, tolerance_ppm=1,
    )
    merged = merge_accepted_bars(accepted, rebased)

    assert abs(metadata["price_factor"] - price_factor) < 1e-12
    assert metadata["volume_factor"] == 1.0
    assert abs(
        merged.loc[pd.Timestamp("2026-09-03"), "close"] - 109.0
    ) < 1e-12
    pd.testing.assert_frame_equal(
        merged.loc[accepted.index], accepted, check_exact=True,
    )


def test_uniform_rebase_is_canonical_across_different_observed_scale_factors():
    accepted = _frame(
        ["2026-09-01"],
        [[100, 110, 90, 105, 1000]],
    )
    economic_addition = [100, 110, 90, 105]
    rebased = []
    for factor in (1.025, 1.037):
        incoming = _frame(
            ["2026-09-01", "2026-09-02"],
            [
                [100 / factor, 110 / factor, 90 / factor,
                 105 / factor, 1000],
                [value / factor for value in economic_addition] + [1200],
            ],
        )
        actual, _metadata = rebase_adjusted_incoming(
            accepted, incoming, tolerance_ppm=1,
        )
        rebased.append(actual)

    assert canonical_bar_hash(rebased[0]) == canonical_bar_hash(rebased[1])
    pd.testing.assert_frame_equal(rebased[0], rebased[1], check_exact=True)


def test_uniform_volume_rebase_is_canonical_across_observed_scale_factors():
    accepted = _frame(
        ["2026-09-01"],
        [[100, 110, 90, 105, 1000]],
    )
    rebased = []
    for factor in (1.025, 1.037):
        incoming = _frame(
            ["2026-09-01", "2026-09-02"],
            [
                [100, 110, 90, 105, 1000 / factor],
                [102, 112, 92, 107, 1200 / factor],
            ],
        )
        actual, _metadata = rebase_adjusted_incoming(
            accepted, incoming, tolerance_ppm=1,
        )
        rebased.append(actual)

    assert canonical_bar_hash(rebased[0]) == canonical_bar_hash(rebased[1])
    pd.testing.assert_frame_equal(rebased[0], rebased[1], check_exact=True)


def test_canonicalized_overlap_can_accept_the_next_high_precision_row():
    raw_base = _frame(
        ["2026-09-01"],
        [[
            100.123456789123, 110.234567891234,
            90.0123456789123, 105.345678912345, 1000.12345678912,
        ]],
    )
    accepted, _metadata = rebase_adjusted_incoming(
        None, raw_base, tolerance_ppm=1,
    )
    incoming = pd.concat([
        raw_base,
        _frame(
            ["2026-09-02"],
            [[
                102.123456789123, 112.234567891234,
                92.0123456789123, 107.345678912345, 1200.12345678912,
            ]],
        ),
    ])

    rebased, _metadata = rebase_adjusted_incoming(
        accepted, incoming, tolerance_ppm=1,
    )

    assert len(rebased) == 2


def test_progressive_and_batch_rebases_produce_identical_canonical_bars():
    dates = ["2026-09-01", "2026-09-02", "2026-09-03"]
    economic = [
        [100, 110, 90, 105, 1000],
        [102, 112, 92, 107, 1200],
        [104, 114, 94, 109, 1400],
    ]
    base = _frame(dates[:1], economic[:1])

    first_scale = 1.025
    first_snapshot = _frame(dates[:2], [
        [value / first_scale for value in row[:4]] + [row[4]]
        for row in economic[:2]
    ])
    first_rebased, _metadata = rebase_adjusted_incoming(
        base, first_snapshot, tolerance_ppm=1,
    )
    progressive = merge_accepted_bars(base, first_rebased)

    final_scale = 1.037
    final_snapshot = _frame(dates, [
        [value / final_scale for value in row[:4]] + [row[4]]
        for row in economic
    ])
    next_rebased, _metadata = rebase_adjusted_incoming(
        progressive, final_snapshot, tolerance_ppm=1,
    )
    progressive = merge_accepted_bars(progressive, next_rebased)

    batch_rebased, _metadata = rebase_adjusted_incoming(
        base, final_snapshot, tolerance_ppm=1,
    )
    batch = merge_accepted_bars(base, batch_rebased)

    pd.testing.assert_frame_equal(progressive, batch, check_exact=True)
    assert canonical_bar_hash(progressive) == canonical_bar_hash(batch)


def test_nonfinite_price_factor_is_data_blocked_before_canonicalization():
    accepted = _frame(
        ["2026-09-01"],
        [[1.0e308, 1.1e308, 0.9e308, 1.05e308, 1000]],
    )
    incoming = _frame(
        ["2026-09-01"],
        [[1.0e-308, 1.1e-308, 0.9e-308, 1.05e-308, 1000]],
    )

    try:
        rebase_adjusted_incoming(accepted, incoming, tolerance_ppm=1)
    except ValueError as error:
        assert "价格缩放因子无效" in str(error)
        assert "DATA_BLOCKED" in str(error)
    else:
        raise AssertionError("nonfinite rebase factor must be rejected")
