"""r27过期门槛、可知时点和后续自然恢复，不生成候选订单。"""
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]


def _fixture():
    from gcn.backtest.signal_research_r14 import COMPONENTS
    index = pd.bdate_range('2024-01-01', periods=16)
    frame = pd.DataFrame({'OPEN': 100., 'HIGH': 110., 'LOW': 80., 'CLOSE': 90., 'MID': 100.}, index=index)
    for col in ('B_ALL_RAW', 'JF_RAW', 'B_SETUP', 'B_ENTRY_SIGNAL', 'B_SETUP_EXPIRED',
                'B_SIGNAL', 'ICON_JUEFAN', 'S_SIGNAL', *COMPONENTS):
        frame[col] = False
    frame.loc[index[1], ['B_ALL_RAW', 'B_SETUP', 'B_BEAR_RECOVER', 'B_CRASH_RECOVER']] = True
    frame.loc[index[6], 'B_SETUP_EXPIRED'] = True
    orders = pd.DataFrame(columns=['symbol', 'entry_date', 'trade_id'])
    return frame, orders


def test_r27_expiry_keeps_five_known_waiting_bars_and_strict_three_way_failure_with_cross_window_setup():
    from gcn.backtest.signal_research_r27 import audit_setups
    for close, mid, failure, above_high, above_mid in (
        (110., 100., 'high', False, True), (111., 111., 'mid', True, False),
        (110., 110., 'both', False, False)):
        frame, orders = _fixture(); idx = frame.index; dates = idx.strftime('%Y-%m-%d')
        frame.loc[idx[6], ['CLOSE', 'MID', 'HIGH']] = [close, mid, max(110., close)]
        episodes, waiting, _ = audit_setups('TEST', frame, orders, idx[4], idx[-1], source_trusted=True)
        assert len(episodes) == 1 and len(waiting) == 5
        row = episodes.iloc[0]
        assert row.setup_date == dates[1] and row.expiry_date == dates[6] and row.setup_before_window
        assert row.failure == failure and row.expiry_position == 'flat' and row.eligible_at_expiry
        assert row.setup_open == 100. and row.setup_high == 110. and row.setup_low == 80.
        assert row.setup_close == 90. and row.setup_mid == 100.
        assert row.B_BEAR_RECOVER and row.B_CRASH_RECOVER and row.source_trusted
        assert waiting.date.tolist() == dates[2:7].tolist() and waiting.age.tolist() == [1, 2, 3, 4, 5]
        assert waiting.available_at.eq(dates[6]).all()
        assert waiting.position.iloc[:2].eq('outside_window').all()
        assert waiting.position.iloc[2:].eq('flat').all()
        assert bool(waiting.above_high.iloc[-1]) == above_high
        assert bool(waiting.above_mid.iloc[-1]) == above_mid and not waiting.both.any()
        assert audit_setups('TEST', frame.loc[:idx[5]], orders, idx[4], idx[-1])[0].empty


def test_r27_recovery_is_first_observed_close_only_and_new_setup_ends_old_observation_before_its_day():
    from gcn.backtest.signal_research_r27 import audit_setups
    from gcn.backtest.signal_research_r14 import COMPONENTS
    frame, _ = _fixture(); idx = frame.index; dates = idx.strftime('%Y-%m-%d')
    frame.loc[idx[7], ['CLOSE', 'HIGH']] = [111., 111.]
    frame.loc[idx[9], ['B_ALL_RAW', 'B_SETUP', 'B_STAGE_COMPONENT']] = True
    frame.loc[idx[9], ['CLOSE', 'HIGH']] = [111., 112.]
    frame.loc[idx[10], ['CLOSE', 'HIGH']] = [113., 113.]
    frame.loc[idx[10], ['B_ENTRY_SIGNAL', 'B_SIGNAL']] = True
    orders = pd.DataFrame([dict(symbol='TEST', trade_id='TEST:' + dates[11], entry_date=dates[11],
        entry_signal_date=dates[10], entry_open=100., entry_b=True, entry_jf=False, entry_kind='B',
        setup_date=dates[9], **{c: c == 'B_STAGE_COMPONENT' for c in COMPONENTS},
        exit_date=dates[-1], exit_reason='terminal', exit_price=90., hold_bars=5)])
    episodes, waiting, observations = audit_setups('TEST', frame, orders, idx[0], idx[-1])
    row = episodes.iloc[0]
    assert row.first_recovery_date == dates[7] and row.recovery_wait_bars == 1
    assert row.recovered and row.recovery_position == 'flat' and row.recovery_uncovered and not row.recovery_buy
    assert row.end_kind == 'next_setup' and row.next_setup_date == dates[9]
    assert row.observation_end_date == dates[8] and row.post_expiry_bars == 2 and not row.right_censored
    assert observations.date.tolist() == dates[6:9].tolist()
    assert observations.first_recovery_today.tolist() == [False, True, False]
    assert observations.recovered.tolist() == [False, True, True]
    assert observations.both.tolist() == [False, True, False]
    assert pd.isna(observations.first_recovery_date.iloc[0])
    assert observations.first_recovery_date.iloc[1:].eq(dates[7]).all()
    for cut in idx:
        ep, wait, obs = audit_setups('TEST', frame.loc[:cut], orders, idx[0], idx[-1]); date = str(cut.date())
        pd.testing.assert_frame_equal(wait, waiting[waiting.available_at.le(date)], check_exact=True)
        pd.testing.assert_frame_equal(obs, observations[observations.date.le(date)], check_exact=True)
        assert ep.first_recovery_date.dropna().le(date).all() and ep.next_setup_date.dropna().le(date).all()
    before = audit_setups('TEST', frame.loc[:idx[6]], orders, idx[0], idx[-1])[0].iloc[0]
    assert not before.recovered and before.right_censored and before.end_kind == 'asof'
    assert pd.isna(before.first_recovery_date) and pd.isna(before.recovery_position)


def test_r27_expiry_and_recovery_classification_respect_actual_jf_entry_and_terminal_close():
    from gcn.backtest.signal_research_r27 import audit_setups
    from gcn.backtest.signal_research_r14 import COMPONENTS
    for mode, signal_i, recovery_i, expected_expiry, expiry_buy, recovery_position, recovery_buy in (
        ('already_held', 5, 8, 'held', False, 'held', False),
        ('buy_at_expiry', 6, 8, 'flat', True, 'held', False),
        ('buy_at_recovery', 8, 8, 'flat', False, 'flat', True),
        ('terminal_day', 5, 15, 'held', False, 'held', False)):
        frame, _ = _fixture(); idx = frame.index; dates = idx.strftime('%Y-%m-%d')
        frame.loc[idx[signal_i], ['JF_RAW', 'ICON_JUEFAN']] = True
        frame.loc[idx[recovery_i], ['CLOSE', 'HIGH']] = [111., 111.]
        i = signal_i+1
        orders = pd.DataFrame([dict(symbol='TEST', trade_id='TEST:' + dates[i], entry_date=dates[i],
            entry_signal_date=dates[signal_i], entry_open=100., entry_b=False, entry_jf=True, entry_kind='JF',
            setup_date=None, **{c: False for c in COMPONENTS}, exit_date=dates[-1], exit_reason='terminal',
            exit_price=float(frame.CLOSE.iloc[-1]), hold_bars=len(frame)-i)])
        episodes, _, observations = audit_setups('TEST', frame, orders, idx[0], idx[-1])
        row = episodes.iloc[0]
        assert row.expiry_position == expected_expiry and bool(row.expiry_buy) == expiry_buy
        assert bool(row.eligible_at_expiry) == (expected_expiry == 'flat' and not expiry_buy)
        assert row.recovery_position == recovery_position and bool(row.recovery_buy) == recovery_buy
        assert not row.recovery_uncovered and row.first_recovery_date == dates[recovery_i]
        assert observations[observations.first_recovery_today].position.iloc[0] == recovery_position
        if mode == 'terminal_day':
            assert observations.iloc[-1].terminal_today and observations.iloc[-1].position == 'held'


def test_r27_next_setup_censors_old_recovery_even_when_its_close_crosses_old_high():
    from gcn.backtest.signal_research_r27 import audit_setups
    for new_i in (7, 9):
        frame, orders = _fixture(); idx = frame.index; dates = idx.strftime('%Y-%m-%d')
        frame.loc[idx[new_i], ['B_ALL_RAW', 'B_SETUP', 'B_STAGE_COMPONENT']] = True
        frame.loc[idx[new_i], ['CLOSE', 'HIGH']] = [111., 111.]
        frame.loc[idx[new_i+5], 'B_SETUP_EXPIRED'] = True
        episodes, waiting, observations = audit_setups('TEST', frame, orders, idx[0], idx[-1])
        row = episodes.iloc[0]
        assert len(episodes) == 2 and len(waiting) == 10
        assert not row.recovered and row.right_censored and row.end_kind == 'next_setup'
        assert row.next_setup_date == dates[new_i] and row.observation_end_date == dates[new_i-1]
        assert row.post_expiry_bars == new_i-7
        assert observations[observations.episode_id.eq(row.episode_id)].date.lt(dates[new_i]).all()
        assert observations.first_recovery_date.isna().all()


def test_r27_three_event_clocks_have_separate_twenty_bar_labels_and_known_at_boundaries():
    from gcn.backtest.signal_research_r27 import audit_setups, event_labels
    frame, orders = _fixture()
    frame = frame.reindex(pd.bdate_range(frame.index[0], periods=45)).ffill().astype(frame.dtypes.to_dict())
    idx = frame.index; dates = idx.strftime('%Y-%m-%d')
    for pos, close in ((8, 111.), (21, 80.), (26, 120.), (28, 95.)):
        frame.loc[idx[pos], ['CLOSE', 'HIGH']] = [close, max(110., close)]
    episodes, waiting, observations = audit_setups('TEST', frame, orders, idx[4], idx[-1])
    events = event_labels(frame, episodes, end=idx[-1]).set_index('clock')
    assert events.index.tolist() == ['setup', 'expiry', 'recovery']
    assert events.date.tolist() == dates[[1, 6, 8]].tolist()
    assert events.available_at.tolist() == dates[[6, 6, 8]].tolist()
    assert events.reference_open_date.tolist() == dates[[2, 7, 9]].tolist()
    assert events.outcome_date.tolist() == dates[[21, 26, 28]].tolist()
    assert np.allclose(events.ret20_pct, [-20., 20., -5.])
    assert events.win.tolist() == [False, True, False] and events.interference.tolist() == [True, False, True]
    for cut, complete_count in ((5, 0), (6, 0), (7, 0), (8, 0), (20, 0), (21, 1), (25, 1), (26, 2), (27, 2), (28, 3)):
        early = event_labels(frame, episodes, end=idx[cut])
        assert early.outcome_complete.sum() == complete_count
        assert early.available_at.le(dates[cut]).all() and early.date.le(dates[cut]).all()
        assert early.loc[~early.outcome_complete, ['outcome_date', 'ret20_pct', 'win', 'interference']].isna().all().all()
        for row in early[early.outcome_complete].itertuples():
            assert row.ret20_pct == events.loc[row.clock, 'ret20_pct']
    no_next = event_labels(frame, episodes, end=idx[8]).set_index('clock').loc['recovery']
    assert pd.isna(no_next.reference_open_date) and pd.isna(no_next.reference_open)
    for table in (episodes, waiting, observations):
        assert not set(('ret20_pct', 'win', 'interference', 'outcome_date')) & set(table.columns)


def test_r27_summary_keeps_eligible_cohort_missing_recovery_clock_and_complete_denominators():
    from gcn.backtest.signal_research_r27 import audit_setups, event_labels, summarize_events
    from gcn.backtest.historical_research import CORE
    frame, orders = _fixture(); idx = frame.index
    frame.loc[idx[8], ['CLOSE', 'HIGH']] = [111., 111.]
    episodes, _, _ = audit_setups('TEST', frame, orders, idx[0], idx[-1])
    events = event_labels(frame, episodes, end=idx[-1])
    expiry = events.clock.eq('expiry')
    events.loc[expiry, ['outcome_complete', 'win', 'interference', 'ret20_pct']] = [True, True, False, 10.]
    held = episodes.copy(); held['episode_id'] = 'held'
    held['expiry_position'] = held['recovery_position'] = 'held'
    held['eligible_at_expiry'] = held['recovery_uncovered'] = False
    held_events = events.copy(); held_events['episode_id'] = 'held'
    held_events['expiry_position'] = 'held'; held_events['eligible_at_expiry'] = False
    held_events['outcome_complete'] = False; held_events[['win', 'interference', 'ret20_pct']] = pd.NA
    censored = episodes.copy(); censored['episode_id'] = 'censored'
    censored['first_recovery_date'] = censored['recovery_position'] = pd.NA
    censored['recovered'] = censored['recovery_uncovered'] = False; censored['right_censored'] = True
    censored_events = held_events[held_events.clock.ne('recovery')].copy()
    censored_events['episode_id'] = 'censored'; censored_events['eligible_at_expiry'] = True
    censored_events['expiry_position'] = 'flat'
    ep = pd.concat([episodes, held, censored], ignore_index=True)
    ev = pd.concat([events, held_events, censored_events], ignore_index=True)
    summary = summarize_events(ep, ev)
    assert len(summary) == 198
    key = summary.set_index(['scope', 'clock', 'group_by', 'group'])
    row = key.loc[('all', 'expiry', 'all', 'all')]
    assert row.cohort_episodes == row.events == 3 and row.complete == row.wins == 1 and row.win_rate_pct == 100.
    recovery = key.loc[('eligible', 'recovery', 'all', 'all')]
    assert recovery.cohort_episodes == 2 and recovery.events == recovery.without_clock == 1
    assert recovery.complete == 0 and pd.isna(recovery.win_rate_pct)
    assert key.loc[('all', 'setup', 'component', 'multiple')].cohort_episodes == 3
    for symbol in CORE:
        zero = key.loc[('eligible', 'recovery', 'symbol', symbol)]
        assert zero.cohort_episodes == zero.events == zero.wins == 0 and pd.isna(zero.win_rate_pct)
    empty = summarize_events(ep.iloc[:0], ev.iloc[:0])
    assert len(empty) == 198 and empty.events.eq(0).all() and empty.win_rate_pct.isna().all()


def test_r27_training_archive_binds_frozen_inputs_and_reconciles_all_native_expiries_without_simulation(tmp_path, monkeypatch):
    import hashlib
    import json
    from gcn.backtest.signal_research_r27 import run_diagnostic
    from gcn.backtest.historical_research import CORE, load_snapshot
    from gcn.recipes.gcn_main import compute_ehopt10
    import gcn.backtest.engine as engine
    import gcn.backtest.historical_research as historical
    def forbidden(*args, **kwargs):
        raise AssertionError('r27不得生成新订单')
    monkeypatch.setattr(engine, '_one_strategy', forbidden)
    monkeypatch.setattr(historical, '_one_strategy', forbidden)
    snapshot = ROOT / 'reports/signal-audit-v5-review-20260904'
    prior = ROOT / 'reports/gcn-historical-r22-20260905'
    decision = run_diagnostic(snapshot, prior, tmp_path)
    assert decision['stage'] == 'diagnostic_only' and decision['recommended'] == 'v5'
    assert not decision['production_changed'] and decision['window'][0] == 'training'
    manifest = json.loads((tmp_path / 'manifest.json').read_bytes())
    frames, quality = load_snapshot(snapshot)
    assert manifest['source_quality'] == quality and manifest['environment'] == manifest['input_environment']
    for key, prefix, count in (('outputs', '', 8), ('algorithm_sources', 'source_snapshot', 10),
                               ('input_files', 'input_snapshot', 7), ('input_algorithm_sources', 'input_source_snapshot', 9),
                               ('parent_files', 'parent_snapshot', 23), ('mechanism_files', 'mechanism_snapshot', 2)):
        assert len(manifest[key]) == count
        for name, expected in manifest[key].items():
            assert hashlib.sha256((tmp_path / prefix / name).read_bytes()).hexdigest() == expected
    ep = pd.read_csv(tmp_path / 'episodes.csv'); wait = pd.read_csv(tmp_path / 'waiting.csv')
    obs = pd.read_csv(tmp_path / 'observations.csv'); events = pd.read_csv(tmp_path / 'events.csv')
    checks = pd.read_csv(tmp_path / 'reconciliation.csv').set_index('symbol')
    assert len(pd.read_csv(tmp_path / 'summary.csv')) == 198 and len(checks) == 10 and checks.reconciled.all()
    assert checks.actual_entries.sum() == 50 and len(wait) == 5*len(ep)
    assert len(events) == 2*len(ep) + ep.recovered.sum()
    for symbol in CORE:
        frame = compute_ehopt10(frames[symbol].loc[:'2024-08-26'], version='v5', diagnostics=True)
        own = ep[ep.symbol.eq(symbol)]
        assert checks.loc[symbol, 'expired_setups'] == len(own) == frame.loc['2021-08-27':].B_SETUP_EXPIRED.sum()
        for row in own.itertuples():
            bars = wait[wait.episode_id.eq(row.episode_id)]
            path = obs[obs.episode_id.eq(row.episode_id)]
            assert len(bars) == 5 and len(path) == row.post_expiry_bars+1
            assert not bars.both.any() and bars.available_at.eq(row.expiry_date).all()
            assert path.date.iloc[0] == row.expiry_date and path.date.iloc[-1] == row.observation_end_date
            if row.recovered:
                first = path[path.first_recovery_today].iloc[0]
                assert first.date == row.first_recovery_date and first.both and first.bars_after_expiry > 0


def test_r27_rejects_changed_inputs_mechanism_source_midrun_and_nonempty_outputs(tmp_path, monkeypatch):
    import shutil
    import gcn.backtest.signal_research_r27 as research
    prior = tmp_path / 'r22'; snapshot = tmp_path / 'parent'; output = tmp_path / 'result'
    shutil.copytree(ROOT / 'reports/gcn-historical-r22-20260905/training', prior / 'training')
    shutil.copytree(ROOT / 'reports/gcn-historical-r26-20260905/training/parent_snapshot', snapshot)
    for path in (prior / 'training/manifest.json', prior / 'training/trades.csv', prior / 'training/paths.csv',
                 prior / 'training/source_snapshot/gcn/backtest/engine.py', snapshot / 'manifest.json',
                 snapshot / 'input_snapshot/NVDA_1d.csv'):
        raw = path.read_bytes(); path.write_bytes(raw+b'\n')
        with pytest.raises(ValueError):
            research.run_diagnostic(snapshot, prior, output)
        assert not output.exists(); path.write_bytes(raw)
    original_read = Path.read_bytes
    for target, after_first in ((ROOT / 'reports/gcn-historical-r26-20260905/training/manifest.json', False),
                                (ROOT / 'reports/gcn-historical-r26-20260905/training-mechanism.md', False),
                                (ROOT / 'gcn/backtest/signal_research_r26.py', False),
                                (ROOT / 'reports/gcn-historical-r27-20260905/protocol.md', False),
                                (ROOT / 'gcn/backtest/signal_research_r27.py', True),
                                (ROOT / 'reports/gcn-historical-r27-20260905/protocol.md', True)):
        reads = 0
        def altered(path):
            nonlocal reads
            raw = original_read(path)
            if path == target:
                reads += 1
                if not after_first or reads > 1:
                    return raw+b'\n'
            return raw
        with monkeypatch.context() as context:
            context.setattr(Path, 'read_bytes', altered)
            with pytest.raises(ValueError, match='机制|源码|协议'):
                research.run_diagnostic(snapshot, prior, output)
        assert not output.exists()
    with monkeypatch.context() as context:
        context.setattr(research.platform, 'python_version', lambda: '0.0.0')
        with pytest.raises(ValueError, match='环境'):
            research.run_diagnostic(snapshot, prior, output)
    saved = research.audit_setups
    changed = snapshot / 'input_snapshot/AAOI_1d.csv'; before = changed.read_bytes()
    def mutate(*args, **kwargs):
        changed.write_bytes(before+b'\n')
        return saved(*args, **kwargs)
    with monkeypatch.context() as context:
        context.setattr(research, 'audit_setups', mutate)
        with pytest.raises(ValueError, match='计算期间'):
            research.run_diagnostic(snapshot, prior, output)
    assert not output.exists(); changed.write_bytes(before)
    with pytest.raises(ValueError, match='固定'):
        research.run_diagnostic(snapshot, prior, output, window='unknown')
    output.mkdir(); marker = output / 'keep.txt'; marker.write_text('keep')
    with pytest.raises(FileExistsError):
        research.run_diagnostic(snapshot, prior, output)
    assert marker.read_text() == 'keep' and len(list(output.iterdir())) == 1


@pytest.mark.parametrize('window,start,end', [
    ('training', '2021-08-27', '2024-08-26'),
    ('validation', '2024-08-27', '2025-08-26'),
    ('recent', '2025-08-27', '2026-08-26'),
    ('full', '2021-08-27', '2026-08-26')])
def test_r27_real_fixed_prefixes_preserve_available_history_observations_and_matured_clocks(window, start, end):
    from gcn.backtest.historical_research import CORE, load_snapshot
    from gcn.backtest.signal_research_r27 import audit_setups, event_labels, EVENT_CONTEXT
    from gcn.recipes.gcn_main import compute_ehopt10
    frames, quality = load_snapshot(ROOT / 'reports/signal-audit-v5-review-20260904')
    orders = pd.read_csv(ROOT / 'reports/gcn-historical-r22-20260905' / window / 'trades.csv', float_precision='round_trip')
    start, end = pd.Timestamp(start), pd.Timestamp(end)
    checked = 0
    for symbol in CORE:
        raw = frames[symbol].loc[:end]
        def audit(cut):
            frame = compute_ehopt10(raw.loc[:cut], version='v5', diagnostics=True)
            ep, wait, obs = audit_setups(symbol, frame, orders, start, end, source_trusted=quality[symbol])
            return ep, wait, obs, event_labels(frame, ep, end=end)
        episodes, waiting, observations, events = audit(raw.index[-1])
        cuts = {raw.index[0], raw.index[-1], raw.loc[start:].index[0]}
        for row in episodes.itertuples():
            for date in (row.expiry_date, row.first_recovery_date, row.next_setup_date):
                if pd.notna(date):
                    pos = raw.index.get_loc(pd.Timestamp(date)); cuts.update(raw.index[max(0, pos-1):pos+1])
        for clock in ('setup', 'expiry', 'recovery'):
            complete = events[events.clock.eq(clock) & events.outcome_complete]
            if len(complete):
                cuts.add(pd.Timestamp(complete.outcome_date.iloc[0]))
        for cut in sorted(cuts):
            ep, wait, obs, early = audit(cut); date = str(cut.date())
            pd.testing.assert_frame_equal(wait, waiting[waiting.available_at.le(date)], check_exact=True)
            pd.testing.assert_frame_equal(obs, observations[observations.date.le(date)], check_exact=True)
            pd.testing.assert_frame_equal(ep[list(EVENT_CONTEXT)], episodes[episodes.expiry_date.le(date)][list(EVENT_CONTEXT)], check_exact=True)
            expected = events[events.available_at.le(date)]
            columns = [*EVENT_CONTEXT, 'clock', 'date', 'available_at']
            pd.testing.assert_frame_equal(early[columns], expected[columns], check_exact=True)
            pd.testing.assert_frame_equal(early[early.outcome_complete], expected.loc[early[early.outcome_complete].index], check_exact=True)
            assert ep.first_recovery_date.dropna().le(date).all() and ep.next_setup_date.dropna().le(date).all()
            assert early.reference_open_date.dropna().le(date).all()
            assert early.loc[~early.outcome_complete, ['ret20_pct', 'win', 'interference', 'outcome_date']].isna().all().all()
            checked += 1
    assert checked >= len(CORE)*3


@pytest.mark.parametrize('window,order_count', [('training', 50), ('validation', 17), ('recent', 17), ('full', 82)])
def test_r27_fixed_window_archive_preserves_original_cohorts_and_reproduces_formal_outputs(tmp_path, window, order_count):
    import json
    from gcn.backtest.historical_research import CORE, load_snapshot
    from gcn.backtest.signal_research_r27 import run_diagnostic
    from gcn.recipes.gcn_main import compute_ehopt10
    snapshot = ROOT / 'reports/signal-audit-v5-review-20260904'
    prior = ROOT / 'reports/gcn-historical-r22-20260905'
    decision = run_diagnostic(snapshot, prior, tmp_path, window=window)
    _, start, end = decision['window']
    frames, _ = load_snapshot(snapshot)
    ep = pd.read_csv(tmp_path / 'episodes.csv')
    ev = pd.read_csv(tmp_path / 'events.csv')
    wait = pd.read_csv(tmp_path / 'waiting.csv')
    obs = pd.read_csv(tmp_path / 'observations.csv')
    checks = pd.read_csv(tmp_path / 'reconciliation.csv').set_index('symbol')
    assert checks.actual_entries.sum() == order_count and checks.reconciled.all()
    assert len(ev) == 2*len(ep) + ep.recovered.sum() and len(wait) == 5*len(ep)
    assert not wait.both.any() and len(pd.read_csv(tmp_path / 'summary.csv')) == 198
    for symbol in CORE:
        frame = compute_ehopt10(frames[symbol].loc[:end], version='v5', diagnostics=True)
        assert checks.loc[symbol, 'expired_setups'] == frame.loc[start:end].B_SETUP_EXPIRED.sum()
    for row in ep.itertuples():
        path = obs[obs.episode_id.eq(row.episode_id)]
        assert len(path) == row.post_expiry_bars+1
        assert path.date.iloc[0] == row.expiry_date and path.date.iloc[-1] == row.observation_end_date
        assert bool(row.eligible_at_expiry) == (row.expiry_position == 'flat' and not row.expiry_buy)
        assert row.right_censored == (not row.recovered)
        if pd.notna(row.next_setup_date):
            assert path.date.lt(row.next_setup_date).all()
        if row.recovered:
            first = path[path.first_recovery_today]
            assert len(first) == 1 and first.both.iloc[0] and first.date.iloc[0] == row.first_recovery_date
            assert bool(row.recovery_uncovered) == (row.recovery_position == 'flat' and not row.recovery_buy)
        else:
            assert ev[ev.episode_id.eq(row.episode_id)].clock.ne('recovery').all()
    formal = ROOT / 'reports/gcn-historical-r27-20260905' / window
    if formal.exists():
        manifest = json.loads((formal / 'manifest.json').read_bytes())
        for name in manifest['outputs']:
            assert (tmp_path / name).read_bytes() == (formal / name).read_bytes(), name
        assert (tmp_path / 'manifest.json').read_bytes() == (formal / 'manifest.json').read_bytes()


@pytest.mark.parametrize('window,symbol,setup,expiry,recovery,expiry_position,recovery_position,raw_b,expected_return', [
    ('validation', 'TQQQ', '2025-04-09', '2025-04-16', '2025-04-25', 'held', 'flat', True, 22.769336),
    ('validation', 'SNOW', '2025-04-09', '2025-04-16', '2025-04-24', 'flat', 'flat', False, 28.757914),
    ('recent', 'AAOI', '2026-01-28', '2026-02-04', '2026-02-09', 'flat', 'flat', False, 153.529709),
    ('recent', 'MRNA', '2025-09-22', '2025-09-29', '2025-10-01', 'flat', 'flat', True, -11.848678),
    ('recent', 'NFLX', '2026-07-02', '2026-07-10', '2026-08-19', 'flat', 'held', False, None)])
def test_r27_real_recovery_examples_keep_missed_held_losing_and_immature_cases_separate(
        window, symbol, setup, expiry, recovery, expiry_position, recovery_position, raw_b, expected_return):
    from gcn.backtest.historical_research import load_snapshot
    from gcn.backtest.signal_research_r27 import audit_setups, event_labels, WINDOWS
    from gcn.recipes.gcn_main import compute_ehopt10
    frames, quality = load_snapshot(ROOT / 'reports/signal-audit-v5-review-20260904')
    orders = pd.read_csv(ROOT / 'reports/gcn-historical-r22-20260905' / window / 'trades.csv', float_precision='round_trip')
    _, start, end = next(spec for spec in WINDOWS if spec[0] == window)
    start, end = pd.Timestamp(start), pd.Timestamp(end)
    frame = compute_ehopt10(frames[symbol].loc[:end], version='v5', diagnostics=True)
    ep, _, obs = audit_setups(symbol, frame, orders, start, end, source_trusted=quality[symbol])
    own = ep[ep.setup_date.eq(setup)]; row = own.iloc[0]
    assert row.expiry_date == expiry and row.first_recovery_date == recovery
    assert row.expiry_position == expiry_position and row.recovery_position == recovery_position
    assert not row.expiry_buy and not row.recovery_buy
    assert bool(row.eligible_at_expiry) == (expiry_position == 'flat')
    assert bool(row.recovery_uncovered) == (recovery_position == 'flat')
    first = obs[obs.episode_id.eq(row.episode_id) & obs.first_recovery_today].iloc[0]
    assert bool(first.B_ALL_RAW) == raw_b and bool(first.raw_b_suppressed) == raw_b
    assert not first.B_SIGNAL and not first.ICON_JUEFAN
    events = event_labels(frame, own, end=end).set_index('clock')
    result = events.loc['recovery']
    if expected_return is None:
        assert not result.outcome_complete and pd.isna(result.ret20_pct)
        assert obs.iloc[-1].terminal_today and obs.iloc[-1].position == 'held'
    else:
        assert result.outcome_complete and result.ret20_pct == pytest.approx(expected_return, abs=1e-6)
