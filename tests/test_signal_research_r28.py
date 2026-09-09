"""r28实际B原Setup依据、真实持仓边界和独立事后标签。"""
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]


def _fixture(length=18):
    from gcn.backtest.signal_research_r14 import COMPONENTS
    idx = pd.bdate_range('2024-01-01', periods=length)
    frame = pd.DataFrame({'OPEN': 100., 'HIGH': 120., 'LOW': 90., 'CLOSE': 105., 'MID': 100.}, index=idx)
    for c in ('B_ALL_RAW', 'JF_RAW', 'B_SETUP', 'B_ENTRY_SIGNAL', 'B_SETUP_EXPIRED',
              'B_SIGNAL', 'ICON_JUEFAN', 'S_SIGNAL', *COMPONENTS):
        frame[c] = False
    frame.loc[idx[1], ['B_ALL_RAW', 'B_SETUP', 'B_BASE_BULL', 'B_STAGE_COMPONENT']] = True
    frame.loc[idx[1], 'HIGH'] = 110.
    frame.loc[idx[3], ['B_ENTRY_SIGNAL', 'B_SIGNAL', 'JF_RAW', 'ICON_JUEFAN']] = True
    frame.loc[idx[[3, 4, 7]], 'CLOSE'] = 115.
    frame.loc[idx[4], 'OPEN'] = 112.
    frame.loc[idx[5], 'CLOSE'] = 95.
    frame.loc[idx[8], 'CLOSE'] = 100.
    frame.loc[idx[10], 'OPEN'] = 108.
    frame.loc[idx[12], ['JF_RAW', 'ICON_JUEFAN']] = True
    orders = []
    for i, j, kind, reason in ((4, 10, 'B', 'trail'), (13, length-1, 'JF', 'terminal')):
        price = float(frame.CLOSE.iloc[j] if reason == 'terminal' else frame.OPEN.iloc[j])
        orders.append(dict(symbol='TEST', trade_id='TEST:' + str(idx[i].date()),
            entry_date=str(idx[i].date()), entry_signal_date=str(idx[i-1].date()), entry_open=float(frame.OPEN.iloc[i]),
            entry_b=kind == 'B', entry_jf=True, entry_kind=kind,
            setup_date=str(idx[1].date()) if kind == 'B' else None,
            **{c: kind == 'B' and c in ('B_BASE_BULL', 'B_STAGE_COMPONENT') for c in COMPONENTS},
            exit_date=str(idx[j].date()), exit_reason=reason, exit_price=price,
            hold_bars=j-i+int(reason == 'terminal'), return_pct=(price/frame.OPEN.iloc[i]*.999**2-1)*100))
    return frame, pd.DataFrame(orders)


def test_r28_observations_use_actual_b_entry_original_setup_and_exclude_exit_open_and_pure_jf():
    from gcn.backtest.signal_research_r28 import audit_positions
    frame, orders = _fixture(); idx = frame.index; dates = idx.strftime('%Y-%m-%d')
    ep, obs = audit_positions('TEST', frame, orders, idx[0], idx[-1], source_trusted=True)
    assert len(ep) == 1 and len(obs) == 6
    row = ep.iloc[0]
    assert row.trade_id == orders.trade_id.iloc[0] and row.setup_date == dates[1]
    assert row.confirmation_date == dates[3] and row.entry_date == dates[4]
    assert row.setup_high == 110. and row.entry_open == 112. and row.entry_jf and row.source_trusted
    assert row.B_BASE_BULL and row.B_STAGE_COMPONENT and not row.B_CRASH_RECOVER
    assert obs.date.tolist() == dates[4:10].tolist() and obs.position.eq('held').all()
    assert obs.held_trade_id.eq(row.trade_id).all() and obs.setup_high.eq(110.).all()
    assert obs.bars_from_entry.tolist() == list(range(6))
    assert row.observed_bars == 6 and row.observation_end_date == dates[9]
    assert audit_positions('TEST', frame.loc[:idx[3]], orders, idx[0], idx[-1])[0].empty


def test_r28_first_failure_and_recovery_are_causal_strict_and_keep_original_high_across_new_setup():
    from gcn.backtest.signal_research_r28 import audit_positions, CONTEXT_SCHEMA
    frame, orders = _fixture(); idx = frame.index; dates = idx.strftime('%Y-%m-%d')
    frame.loc[idx[6], ['B_ALL_RAW', 'B_SETUP', 'B_CRASH_RECOVER']] = True
    frame.loc[idx[6], 'HIGH'] = 200.
    frame.loc[idx[11], 'B_SETUP_EXPIRED'] = True
    ep, obs = audit_positions('TEST', frame, orders, idx[0], idx[-1])
    row = ep.iloc[0]
    assert row.failed and not row.failure_on_entry and row.first_failure_date == dates[5]
    assert row.bars_to_failure == 1 and row.recovered and row.first_recovery_date == dates[7]
    assert row.bars_to_recovery == 3 and row.failure_to_recovery_bars == 2
    assert obs.first_failure_today.tolist() == [False, True, False, False, False, False]
    assert obs.first_recovery_today.tolist() == [False, False, False, True, False, False]
    assert obs.both_failed.tolist() == [False, True, False, False, True, False]
    assert obs.both_above.tolist() == [True, False, False, True, False, False]
    assert obs.setup_high.eq(110.).all() and obs.setup_date.eq(dates[1]).all()
    assert obs.loc[2, 'pending_setup_date'] == dates[6]
    for cut in idx:
        early_ep, early_obs = audit_positions('TEST', frame.loc[:cut], orders, idx[0], idx[-1])
        date = str(cut.date())
        pd.testing.assert_frame_equal(early_obs, obs[obs.date.le(date)], check_exact=True)
        pd.testing.assert_frame_equal(early_ep[list(CONTEXT_SCHEMA)], ep[ep.entry_date.le(date)][list(CONTEXT_SCHEMA)], check_exact=True)
        assert early_ep.first_failure_date.dropna().le(date).all() and early_ep.first_recovery_date.dropna().le(date).all()
    first = audit_positions('TEST', frame.loc[:idx[4]], orders, idx[0], idx[-1])[0].iloc[0]
    assert not first.failed and not first.recovered and pd.isna(first.first_failure_date)
    assert pd.isna(first.bars_to_failure) and pd.isna(first.first_recovery_date)


def test_r28_entry_equality_terminal_recovery_unknown_values_and_original_order_rejections():
    from gcn.backtest.signal_research_r28 import audit_positions
    from gcn.backtest.signal_research_r14 import COMPONENTS
    frame, orders = _fixture(); idx = frame.index; dates = idx.strftime('%Y-%m-%d')
    frame.loc[idx[4], ['CLOSE', 'MID']] = 110.
    frame.loc[idx[10], ['B_ALL_RAW', 'B_SETUP', 'B_BEAR_RECOVER']] = True
    frame.loc[idx[10], 'HIGH'] = 110.
    frame.loc[idx[12], ['B_ENTRY_SIGNAL', 'B_SIGNAL']] = True
    frame.loc[idx[12], 'CLOSE'] = 115.
    frame.loc[idx[13], 'CLOSE'] = 95.
    frame.loc[idx[-1], 'CLOSE'] = 115.
    orders.loc[1, ['entry_b', 'entry_kind', 'setup_date', 'exit_price', 'return_pct']] = [True, 'B', dates[10], 115., (1.15*.999**2-1)*100]
    for c in COMPONENTS:
        orders.loc[1, c] = c == 'B_BEAR_RECOVER'
    ep, obs = audit_positions('TEST', frame, orders, idx[0], idx[-1])
    assert len(ep) == 2 and len(obs) == 11 and ep.failure_on_entry.all()
    assert ep.bars_to_failure.eq(0).all() and ep.recovered.all()
    assert ep.first_recovery_date.tolist() == dates[[7, 17]].tolist()
    last = obs.iloc[-1]
    assert last.terminal_today and last.position == 'held' and last.first_recovery_today
    assert not obs[obs.trade_id.eq(orders.trade_id.iloc[0])].date.eq(dates[10]).any()
    for cut in idx:
        _, early = audit_positions('TEST', frame.loc[:cut], orders, idx[0], idx[-1])
        pd.testing.assert_frame_equal(early, obs[obs.date.le(str(cut.date()))], check_exact=True)
    unknown, original = _fixture()
    unknown.loc[idx[4], ['CLOSE', 'MID']] = [95., np.nan]
    earlier, path = audit_positions('TEST', unknown, original, idx[0], idx[-1])
    assert not path.both_failed.iloc[0] and earlier.bars_to_failure.iloc[0] == 1
    for col, value in [('setup_date', dates[0]), ('entry_open', 99.), ('B_BASE_BULL', False), ('hold_bars', 1)]:
        changed = orders.copy(); changed.loc[0, col] = value
        with pytest.raises(ValueError, match='订单|来源|持仓'):
            audit_positions('TEST', frame, changed, idx[0], idx[-1])
    with pytest.raises(ValueError, match='订单'):
        audit_positions('TEST', frame, orders.iloc[1:], idx[0], idx[-1])


def test_r28_confirmation_entry_close_and_failure_use_three_independent_causal_twenty_bar_clocks():
    from gcn.backtest.signal_research_r28 import audit_positions, event_labels
    frame, orders = _fixture(55); idx = frame.index; dates = idx.strftime('%Y-%m-%d')
    for pos, close in ((23, 89.6), (24, 120.), (25, 95.)):
        frame.loc[idx[pos], ['CLOSE', 'LOW', 'HIGH']] = [close, min(90., close), max(120., close)]
    ep, obs = audit_positions('TEST', frame, orders, idx[0], idx[-1])
    events = event_labels(frame, ep, end=idx[-1]).set_index('clock')
    assert events.index.tolist() == ['confirmation', 'entry', 'failure']
    assert events.date.tolist() == dates[[3, 4, 5]].tolist()
    assert events.available_at.tolist() == dates[[4, 4, 5]].tolist()
    assert events.reference_open_date.tolist() == dates[[4, 5, 6]].tolist()
    assert events.outcome_date.tolist() == dates[[23, 24, 25]].tolist()
    assert np.allclose(events.ret20_pct, [-20., 20., -5.])
    assert events.win.tolist() == [False, True, False] and events.interference.tolist() == [True, False, True]
    for pos, n, complete in ((3, 0, 0), (4, 2, 0), (5, 3, 0), (22, 3, 0), (23, 3, 1), (24, 3, 2), (25, 3, 3)):
        early = event_labels(frame, ep, end=idx[pos])
        assert len(early) == n and early.outcome_complete.sum() == complete
        assert early.available_at.le(dates[pos]).all() and early.reference_open_date.dropna().le(dates[pos]).all()
        assert early.loc[~early.outcome_complete, ['ret20_pct', 'outcome_date', 'win', 'interference']].isna().all().all()
    first = event_labels(frame, ep, end=idx[4]).set_index('clock')
    assert first.loc['confirmation', 'reference_open'] == 112. and pd.isna(first.loc['entry', 'reference_open'])
    for table in (ep, obs):
        assert not {'outcome_date', 'ret20_pct', 'return_pct', 'exit_reason', 'win'} & set(table.columns)


def test_r28_original_net_outcomes_appear_only_after_actual_exit_and_stay_separate_from_price_clocks():
    from gcn.backtest.signal_research_r28 import audit_positions, trade_outcomes
    frame, orders = _fixture(); idx = frame.index; dates = idx.strftime('%Y-%m-%d')
    ep, _ = audit_positions('TEST', frame, orders, idx[0], idx[-1])
    closed = trade_outcomes(ep, orders, asof=idx[-1])
    assert len(closed) == 1 and closed.exit_observed.all()
    assert closed.exit_date.iloc[0] == dates[10] and closed.exit_reason.iloc[0] == 'trail'
    assert closed.return_pct.iloc[0] == orders.return_pct.iloc[0] and not closed.trade_win.iloc[0]
    assert not {'ret20_pct', 'first_failure_date', 'first_recovery_date'} & set(closed.columns)
    assert trade_outcomes(ep, orders, asof=idx[3]).empty
    early = trade_outcomes(ep, orders, asof=idx[9])
    assert len(early) == 1 and not early.exit_observed.iloc[0]
    fields = ['exit_date', 'exit_reason', 'exit_price', 'hold_bars', 'return_pct', 'trade_win']
    assert early[fields].isna().all().all()
    future = orders.copy(); future.loc[0, ['exit_price', 'hold_bars', 'return_pct']] = 98765
    pd.testing.assert_frame_equal(trade_outcomes(ep, future, asof=idx[9]), early, check_exact=True)
    pd.testing.assert_frame_equal(trade_outcomes(ep, orders, asof=idx[10]), closed, check_exact=True)
    terminal_frame = frame.copy(); terminal_frame.loc[idx[12], ['JF_RAW', 'ICON_JUEFAN']] = False
    terminal = orders.iloc[:1].copy()
    terminal.loc[0, ['exit_date', 'exit_reason', 'exit_price', 'hold_bars', 'return_pct']] = [
        dates[-1], 'terminal', 105., len(frame)-4, (105./112.*.999**2-1)*100]
    terminal_ep, _ = audit_positions('TEST', terminal_frame, terminal, idx[0], idx[-1])
    assert not trade_outcomes(terminal_ep, terminal, asof=idx[-2]).exit_observed.iloc[0]
    result = trade_outcomes(terminal_ep, terminal, asof=idx[-1]).iloc[0]
    assert result.exit_observed and result.exit_reason == 'terminal' and result.exit_price == 105.


def test_r28_summary_separates_original_trade_outcomes_from_event_clocks_and_keeps_zero_cohorts():
    from gcn.backtest.signal_research_r28 import audit_positions, event_labels, trade_outcomes, summarize
    from gcn.backtest.historical_research import CORE
    frame, orders = _fixture(); idx = frame.index
    ep, _ = audit_positions('TEST', frame, orders, idx[0], idx[-1])
    ev = event_labels(frame, ep, end=idx[-1]); trades = trade_outcomes(ep, orders, asof=idx[-1])
    ev.loc[ev.clock.eq('failure'), ['outcome_complete', 'win', 'interference', 'ret20_pct']] = [True, False, True, -10.]
    clean = ep.copy(); clean['trade_id'] = 'clean'; clean['entry_jf'] = False
    clean[['failed', 'failure_on_entry', 'recovered']] = False
    clean[['first_failure_date', 'first_recovery_date', 'bars_to_failure', 'bars_to_recovery', 'failure_to_recovery_bars']] = pd.NA
    clean_events = ev[ev.clock.ne('failure')].copy(); clean_events['trade_id'] = 'clean'; clean_events['entry_jf'] = False
    clean_trades = trades.copy(); clean_trades['trade_id'] = 'clean'; clean_trades['entry_jf'] = False
    clean_trades['return_pct'] = 10.; clean_trades['trade_win'] = True
    episodes = pd.concat([ep, clean], ignore_index=True)
    events = pd.concat([ev, clean_events], ignore_index=True)
    outcomes = pd.concat([trades, clean_trades], ignore_index=True)
    summary, clocks = summarize(episodes, events, outcomes)
    assert len(summary) == 31 and len(clocks) == 93
    row = summary.set_index(['group_by', 'group']).loc[('all', 'all')]
    assert row.trades == row.observed_exits == 2 and row.original_wins == 1
    assert row.failed_trades == row.recovered_trades == 1 and row.no_failure == 1
    assert row.mean_original_return_pct == outcomes.return_pct.mean()
    key = clocks.set_index(['clock', 'group_by', 'group'])
    failure = key.loc[('failure', 'all', 'all')]
    assert failure.cohort_trades == 2 and failure.events == failure.without_clock == failure.complete == 1
    assert failure.wins == 0 and failure.interference == 1 and failure.mean_ret20_pct == -10.
    confirmation = key.loc[('confirmation', 'all', 'all')]
    assert confirmation.events == 2 and confirmation.complete == 0 and pd.isna(confirmation.win_rate_pct)
    assert summary.set_index(['group_by', 'group']).loc[('component', 'multiple'), 'trades'] == 2
    for symbol in CORE:
        assert key.loc[('failure', 'symbol', symbol), 'events'] == 0
        assert pd.isna(key.loc[('failure', 'symbol', symbol), 'win_rate_pct'])
    empty, empty_clocks = summarize(episodes.iloc[:0], events.iloc[:0], outcomes.iloc[:0])
    assert len(empty) == 31 and len(empty_clocks) == 93
    assert empty.trades.eq(0).all() and empty.mean_original_return_pct.isna().all()


def test_r28_training_archive_binds_original_orders_and_reconciles_all_actual_b_without_new_simulation(tmp_path, monkeypatch):
    import hashlib
    import json
    from gcn.backtest.signal_research_r28 import run_diagnostic
    import gcn.backtest.engine as engine
    import gcn.backtest.historical_research as historical
    def forbidden(*args, **kwargs):
        raise AssertionError('r28不得重模拟订单')
    monkeypatch.setattr(engine, '_one_strategy', forbidden)
    monkeypatch.setattr(historical, '_one_strategy', forbidden)
    snapshot = ROOT / 'reports/signal-audit-v5-review-20260904'
    r22 = ROOT / 'reports/gcn-historical-r22-20260905'
    decision = run_diagnostic(snapshot, r22, tmp_path)
    assert decision['stage'] == 'diagnostic_only' and decision['recommended'] == 'v5'
    assert not decision['production_changed'] and decision['window'][0] == 'training'
    manifest = json.loads((tmp_path / 'manifest.json').read_bytes())
    for key, prefix, count in (('outputs', '', 9), ('algorithm_sources', 'source_snapshot', 11),
                               ('input_files', 'input_snapshot', 7), ('input_algorithm_sources', 'input_source_snapshot', 9),
                               ('parent_files', 'parent_snapshot', 23), ('mechanism_files', 'mechanism_snapshot', 2)):
        assert len(manifest[key]) == count
        for name, expected in manifest[key].items():
            assert hashlib.sha256((tmp_path / prefix / name).read_bytes()).hexdigest() == expected
    ep = pd.read_csv(tmp_path / 'episodes.csv'); obs = pd.read_csv(tmp_path / 'observations.csv')
    events = pd.read_csv(tmp_path / 'events.csv'); outcomes = pd.read_csv(tmp_path / 'trades.csv', float_precision='round_trip')
    original = pd.read_csv(r22 / 'training/trades.csv', float_precision='round_trip')
    selected = original[original.entry_b].set_index('trade_id')
    assert set(ep.trade_id) == set(selected.index) and len(ep) == len(outcomes) == len(selected)
    assert len(obs) == selected.hold_bars.sum() and len(events) == 2*len(ep)+ep.failed.sum()
    assert len(pd.read_csv(tmp_path / 'summary.csv')) == 31 and len(pd.read_csv(tmp_path / 'event_summary.csv')) == 93
    checks = pd.read_csv(tmp_path / 'reconciliation.csv')
    assert len(checks) == 10 and checks.actual_entries.sum() == 50 and checks.reconciled.all()
    assert outcomes.exit_observed.all()
    for trade in outcomes.itertuples():
        assert trade.return_pct == selected.loc[trade.trade_id, 'return_pct']
        own = obs[obs.trade_id.eq(trade.trade_id)]
        assert len(own) == trade.hold_bars and own.position.eq('held').all()
        assert own.date.iloc[0] == trade.entry_date
        if trade.exit_reason == 'terminal':
            assert own.date.iloc[-1] == trade.exit_date and own.terminal_today.iloc[-1]
        else:
            assert own.date.lt(trade.exit_date).all()
    for row in ep.itertuples():
        path = obs[obs.trade_id.eq(row.trade_id)]
        assert path.setup_high.eq(row.setup_high).all()
        if row.failed:
            assert path[path.first_failure_today].date.iloc[0] == row.first_failure_date
        if row.recovered:
            first = path[path.first_recovery_today].iloc[0]
            assert first.date == row.first_recovery_date and first.date > row.first_failure_date and first.both_above


def test_r28_rejects_changed_inputs_sources_mechanism_environment_midrun_and_nonempty_output(tmp_path, monkeypatch):
    import shutil
    import gcn.backtest.signal_research_r28 as research
    prior = tmp_path / 'r22'; snapshot = tmp_path / 'parent'; output = tmp_path / 'result'
    shutil.copytree(ROOT / 'reports/gcn-historical-r22-20260905/training', prior / 'training')
    shutil.copytree(ROOT / 'reports/gcn-historical-r27-20260905/training/parent_snapshot', snapshot)
    for path in (prior / 'training/manifest.json', prior / 'training/trades.csv', prior / 'training/paths.csv',
                 prior / 'training/source_snapshot/gcn/backtest/engine.py', snapshot / 'manifest.json',
                 snapshot / 'input_snapshot/NVDA_1d.csv', snapshot / 'input_snapshot/AAOI_1d.csv.meta.json'):
        raw = path.read_bytes(); path.write_bytes(raw+b'\n')
        with pytest.raises(ValueError):
            research.run_diagnostic(snapshot, prior, output)
        assert not output.exists(); path.write_bytes(raw)
    original_read = Path.read_bytes
    for target, after_first in ((ROOT / 'reports/gcn-historical-r27-20260905/training/manifest.json', False),
                                (ROOT / 'reports/gcn-historical-r27-20260905/training-mechanism.md', False),
                                (ROOT / 'gcn/backtest/signal_research_r26.py', False),
                                (ROOT / 'reports/gcn-historical-r28-20260906/protocol.md', False),
                                (ROOT / 'gcn/backtest/signal_research_r28.py', True),
                                (ROOT / 'reports/gcn-historical-r28-20260906/protocol.md', True)):
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
    saved = research.audit_positions
    changed = snapshot / 'input_snapshot/AAOI_1d.csv'; before = changed.read_bytes()
    def mutate(*args, **kwargs):
        changed.write_bytes(before+b'\n')
        return saved(*args, **kwargs)
    with monkeypatch.context() as context:
        context.setattr(research, 'audit_positions', mutate)
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
    ('training', '2021-08-27', '2024-08-26'), ('validation', '2024-08-27', '2025-08-26'),
    ('recent', '2025-08-27', '2026-08-26'), ('full', '2021-08-27', '2026-08-26')])
def test_r28_real_fixed_price_prefixes_preserve_known_states_matured_clocks_and_original_exit_outcomes(window, start, end):
    from gcn.backtest.historical_research import CORE, load_snapshot
    from gcn.backtest.signal_research_r28 import audit_positions, event_labels, trade_outcomes, CONTEXT_SCHEMA
    from gcn.recipes.gcn_main import compute_ehopt10
    frames, quality = load_snapshot(ROOT / 'reports/signal-audit-v5-review-20260904')
    orders = pd.read_csv(ROOT / 'reports/gcn-historical-r22-20260905' / window / 'trades.csv', float_precision='round_trip')
    start, end = pd.Timestamp(start), pd.Timestamp(end)
    checked = 0
    for symbol in CORE:
        raw = frames[symbol].loc[:end]
        def audit(cut):
            frame = compute_ehopt10(raw.loc[:cut], version='v5', diagnostics=True)
            ep, obs = audit_positions(symbol, frame, orders, start, end, source_trusted=quality[symbol])
            return ep, obs, event_labels(frame, ep, end=end), trade_outcomes(ep, orders, asof=cut)
        episodes, observations, events, actual = audit(raw.index[-1])
        cuts = {raw.index[0], raw.index[-1], raw.loc[start:].index[0]}
        for row in episodes.itertuples():
            for date in (row.entry_date, row.first_failure_date, row.first_recovery_date, row.observation_end_date):
                if pd.notna(date):
                    pos = raw.index.get_loc(pd.Timestamp(date)); cuts.update(raw.index[max(0, pos-1):pos+1])
        for date in actual.exit_date.dropna():
            pos = raw.index.get_loc(pd.Timestamp(date)); cuts.update(raw.index[max(0, pos-1):pos+1])
        for clock in ('confirmation', 'entry', 'failure'):
            complete = events[events.clock.eq(clock) & events.outcome_complete]
            if len(complete):
                cuts.add(pd.Timestamp(complete.outcome_date.iloc[0]))
        for cut in sorted(cuts):
            ep, obs, early, outcomes = audit(cut); date = str(cut.date())
            pd.testing.assert_frame_equal(obs, observations[observations.date.le(date)], check_exact=True)
            pd.testing.assert_frame_equal(ep[list(CONTEXT_SCHEMA)], episodes[episodes.entry_date.le(date)][list(CONTEXT_SCHEMA)], check_exact=True)
            assert ep.first_failure_date.dropna().le(date).all() and ep.first_recovery_date.dropna().le(date).all()
            expected = events[events.available_at.le(date)]
            columns = [*CONTEXT_SCHEMA, 'clock', 'date', 'available_at']
            pd.testing.assert_frame_equal(early[columns], expected[columns], check_exact=True)
            pd.testing.assert_frame_equal(early[early.outcome_complete], expected.loc[early[early.outcome_complete].index], check_exact=True)
            assert early.reference_open_date.dropna().le(date).all()
            assert early.loc[~early.outcome_complete, ['ret20_pct', 'win', 'interference', 'outcome_date']].isna().all().all()
            known = actual[actual.entry_date.le(date)].copy(); pending = known.exit_date.gt(date)
            known.loc[pending, 'exit_observed'] = False
            known.loc[pending, ['exit_date', 'exit_reason', 'exit_price', 'hold_bars', 'return_pct', 'trade_win']] = pd.NA
            pd.testing.assert_frame_equal(outcomes, known, check_exact=True)
            checked += 1
    assert checked >= len(CORE)*3


def test_r28_nonfinite_close_cannot_create_a_recovery_after_known_failure():
    from gcn.backtest.signal_research_r28 import audit_positions
    frame, orders = _fixture(); idx = frame.index
    frame.loc[idx[7], ['CLOSE', 'HIGH']] = np.inf
    ep, obs = audit_positions('TEST', frame, orders, idx[0], idx[-1])
    row = obs[obs.date.eq(str(idx[7].date()))].iloc[0]
    assert not row.both_above and not row.first_recovery_today
    assert ep.failed.iloc[0] and not ep.recovered.iloc[0]


@pytest.mark.parametrize('window,order_count', [('training', 50), ('validation', 17), ('recent', 17), ('full', 82)])
def test_r28_fixed_archives_keep_original_b_outcomes_and_reproduce_all_frozen_outputs(tmp_path, window, order_count):
    import json
    from gcn.backtest.signal_research_r28 import run_diagnostic
    snapshot = ROOT / 'reports/signal-audit-v5-review-20260904'
    prior = ROOT / 'reports/gcn-historical-r22-20260905'
    run_diagnostic(snapshot, prior, tmp_path, window=window)
    orders = pd.read_csv(prior / window / 'trades.csv', float_precision='round_trip')
    b = orders[orders.entry_b].set_index('trade_id')
    ep = pd.read_csv(tmp_path / 'episodes.csv', float_precision='round_trip')
    obs = pd.read_csv(tmp_path / 'observations.csv', float_precision='round_trip')
    outcomes = pd.read_csv(tmp_path / 'trades.csv', float_precision='round_trip')
    events = pd.read_csv(tmp_path / 'events.csv'); checks = pd.read_csv(tmp_path / 'reconciliation.csv')
    assert len(orders) == order_count == checks.actual_entries.sum()
    assert set(ep.trade_id) == set(outcomes.trade_id) == set(b.index)
    assert len(obs) == b.hold_bars.sum() and len(events) == len(ep)*2+ep.failed.sum()
    assert checks.reconciled.all() and checks.b_trades.sum() == len(ep)
    assert outcomes.exit_observed.all()
    for row in outcomes.itertuples():
        assert row.return_pct == b.loc[row.trade_id, 'return_pct']
        path = obs[obs.trade_id.eq(row.trade_id)]
        assert path.setup_date.eq(row.setup_date).all() and path.setup_high.eq(row.setup_high).all()
        assert path.held_trade_id.eq(row.trade_id).all() and len(path) == row.hold_bars
        assert path.date.iloc[0] == row.entry_date
        assert path.date.iloc[-1] == row.exit_date if row.exit_reason == 'terminal' else path.date.lt(row.exit_date).all()
    formal = ROOT / 'reports/gcn-historical-r28-20260906' / window
    if formal.exists():
        manifest = json.loads((formal / 'manifest.json').read_bytes())
        for name in manifest['outputs']:
            assert (tmp_path / name).read_bytes() == (formal / name).read_bytes(), name
        assert (tmp_path / 'manifest.json').read_bytes() == (formal / 'manifest.json').read_bytes()


@pytest.mark.parametrize('window,symbol,entry,failure,recovery,day_zero,net_return', [
    ('training', 'YINN', '2022-11-08', '2022-11-09', '2022-11-10', False, 68.811923),
    ('validation', 'MSFT', '2025-03-25', '2025-03-28', '2025-04-25', False, 27.192428),
    ('validation', 'NVDA', '2025-01-07', '2025-01-10', '2025-01-22', False, -23.723331),
    ('recent', 'MRNA', '2025-11-13', '2025-11-13', '2025-12-05', True, 57.402070),
    ('recent', 'GOOGL', '2026-03-12', '2026-03-12', '2026-03-17', True, 19.241632),
    ('recent', 'TSLA', '2026-02-12', '2026-02-12', '2026-05-08', True, -8.078612)])
def test_r28_real_counterexamples_keep_day_zero_and_later_failures_with_winners_and_losers(
        window, symbol, entry, failure, recovery, day_zero, net_return):
    from gcn.backtest.historical_research import load_snapshot
    from gcn.backtest.signal_research_r28 import audit_positions, trade_outcomes, WINDOWS
    from gcn.recipes.gcn_main import compute_ehopt10
    frames, quality = load_snapshot(ROOT / 'reports/signal-audit-v5-review-20260904')
    orders = pd.read_csv(ROOT / 'reports/gcn-historical-r22-20260905' / window / 'trades.csv', float_precision='round_trip')
    _, start, end = next(spec for spec in WINDOWS if spec[0] == window)
    start, end = pd.Timestamp(start), pd.Timestamp(end)
    frame = compute_ehopt10(frames[symbol].loc[:end], version='v5', diagnostics=True)
    ep, obs = audit_positions(symbol, frame, orders, start, end, source_trusted=quality[symbol])
    own = ep[ep.entry_date.eq(entry)]; row = own.iloc[0]
    assert row.first_failure_date == failure and row.first_recovery_date == recovery
    assert bool(row.failure_on_entry) == day_zero and bool(row.bars_to_failure == 0) == day_zero
    assert row.failed and row.recovered
    path = obs[obs.trade_id.eq(row.trade_id)]
    assert path[path.first_failure_today].both_failed.iloc[0]
    assert path[path.first_recovery_today].both_above.iloc[0]
    actual = trade_outcomes(own, orders, asof=end).iloc[0]
    assert actual.exit_observed and actual.return_pct == pytest.approx(net_return, abs=1e-6)
    assert bool(actual.trade_win) == (net_return > 0)
