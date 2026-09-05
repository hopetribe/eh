"""r26原v5真实空仓状态及事后事件标签；不生成新候选订单。"""
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]


def _fixture():
    from gcn.backtest.signal_research_r14 import COMPONENTS
    index = pd.bdate_range('2024-01-01', periods=16)
    frame = pd.DataFrame({'OPEN': 100., 'HIGH': 110., 'LOW': 80., 'CLOSE': 90., 'MID': 100.}, index=index)
    for column in ('B_ALL_RAW', 'JF_RAW', 'B_SETUP', 'B_ENTRY_SIGNAL', 'B_SETUP_EXPIRED',
                   'B_SIGNAL', 'ICON_JUEFAN', 'S_SIGNAL', *COMPONENTS):
        frame[column] = False
    frame.loc[index[[3, 12]], ['JF_RAW', 'ICON_JUEFAN']] = True
    frame.loc[index[5], ['B_ALL_RAW', 'B_SETUP', 'B_CRASH_RECOVER', 'S_SIGNAL']] = True
    frame.loc[index[7], ['B_ALL_RAW', 'B_ENTRY_SIGNAL', 'B_SIGNAL']] = True
    frame.loc[index[7], 'CLOSE'] = 111.
    frame.loc[index[10], 'CLOSE'] = 80.
    rows = []
    for i, j, reason, kind in ((4, 6, 'signal', 'JF'), (8, 11, 'trail', 'B'), (13, 15, 'terminal', 'JF')):
        terminal = reason == 'terminal'
        rows.append(dict(symbol='TEST', trade_id='TEST:' + str(index[i].date()),
                         entry_date=str(index[i].date()), entry_signal_date=str(index[i-1].date()),
                         entry_open=float(frame.OPEN.iloc[i]), entry_b=kind == 'B', entry_jf=kind == 'JF',
                         entry_kind=kind, setup_date=str(index[5].date()) if kind == 'B' else None,
                         **{c: kind == 'B' and c == 'B_CRASH_RECOVER' for c in COMPONENTS},
                         exit_date=str(index[j].date()), exit_reason=reason,
                         exit_price=float(frame.CLOSE.iloc[j] if terminal else frame.OPEN.iloc[j]),
                         hold_bars=j-i+int(terminal)))
    return frame, pd.DataFrame(rows)


def test_r26_positions_use_initial_flat_and_actual_open_boundaries_not_terminal_close_as_flat():
    from gcn.backtest.signal_research_r26 import position_states
    frame, trades = _fixture(); dates = frame.index.strftime('%Y-%m-%d')
    states = position_states('TEST', frame, trades, frame.index[2], frame.index[-1])
    assert states.index.equals(frame.index[2:])
    flat = states[states.position.eq('flat')]
    assert flat.index.tolist() == frame.index[[2, 3, 6, 7, 11, 12]].tolist()
    assert flat.flat_origin.tolist() == ['initial', 'initial', 'S', 'S', 'trail', 'trail']
    assert flat.flat_since.tolist() == [dates[2], dates[2], dates[6], dates[6], dates[11], dates[11]]
    assert flat.flat_bar.tolist() == [1, 2, 1, 2, 1, 2]
    assert states[states.entry_today].index.tolist() == frame.index[[4, 8, 13]].tolist()
    assert states[states.exit_today].index.tolist() == frame.index[[6, 11]].tolist()
    assert states.iloc[-1].position == 'held' and states.iloc[-1].terminal_today
    assert not states.iloc[-1].pending_buy
    assert states[states.pending_buy].index.tolist() == frame.index[[3, 7, 12]].tolist()
    assert states.loc[frame.index[6], 'pending_setup_date'] == dates[5]
    assert states.loc[frame.index[6], 'pending_B_CRASH_RECOVER']
    assert states.loc[frame.index[7], 'raw_b_count20'] == 2
    assert states.loc[frame.index[7], 'raw_b_suppressed'] and states.loc[frame.index[7], 'B_SIGNAL']
    assert states.loc[frame.index[7], 'resolved_setup_date'] == dates[5]
    assert states.loc[frame.index[7], 'resolved_B_CRASH_RECOVER']


def test_r26_rejects_missing_duplicate_overlapping_or_wrong_source_and_price_orders():
    from gcn.backtest.signal_research_r26 import position_states
    frame, original = _fixture()
    corruptions = [original.iloc[1:], pd.concat([original, original.iloc[:1]])]
    for row, column, value in (
        (0, 'entry_date', str(frame.index[2].date())),
        (0, 'entry_signal_date', str(frame.index[2].date())),
        (0, 'entry_open', 99.), (0, 'entry_b', True), (0, 'entry_kind', 'B'),
        (1, 'setup_date', str(frame.index[6].date())), (1, 'B_CRASH_RECOVER', False),
        (1, 'entry_date', str(frame.index[6].date())),
        (0, 'exit_reason', 'unsupported'), (0, 'exit_price', 99.), (0, 'hold_bars', 3),
        (2, 'exit_date', str(frame.index[14].date())),
    ):
        changed = original.copy(); changed.loc[row, column] = value; corruptions.append(changed)
    for trades in corruptions:
        with pytest.raises(ValueError, match='订单|入场|来源|持仓|退出'):
            position_states('TEST', frame, trades, frame.index[2], frame.index[-1])


def test_r26_states_preserve_full_history_collision_prefixes_and_pending_without_next_open():
    from gcn.backtest.signal_research_r26 import position_states
    frame, trades = _fixture(); index = frame.index
    frame.loc[index[0], ['B_ALL_RAW', 'B_SETUP', 'B_BEAR_RECOVER']] = True
    frame.loc[index[7], ['JF_RAW', 'ICON_JUEFAN']] = True
    trades.loc[1, 'entry_jf'] = True
    full = position_states('TEST', frame, trades, index[2], index[-1])
    assert full.iloc[0].pending_setup_date == str(index[0].date()) and full.iloc[0].pending_setup_age == 2
    assert full.loc[index[7], 'raw_b_count20'] == 3
    assert full.loc[index[7], 'B_SIGNAL'] and full.loc[index[7], 'ICON_JUEFAN']
    assert full.loc[index[8], 'held_trade_id'] == trades.trade_id.iloc[1]
    for cut in index:
        early = position_states('TEST', frame.loc[:cut], trades, index[2], index[-1])
        pd.testing.assert_frame_equal(early, full.loc[:cut], check_exact=True)
    early = position_states('TEST', frame.loc[:index[12]], trades, index[2], index[-1])
    assert early.iloc[-1].pending_buy and not early.iloc[-1].entry_today
    changed = trades.copy(); changed.loc[2, ['exit_price', 'hold_bars']] = [98765., 98765]
    pd.testing.assert_frame_equal(position_states('TEST', frame.loc[:index[14]], changed, index[2], index[-1]),
                                  full.loc[:index[14]], check_exact=True)
    empty = frame.copy()
    for col in empty.columns.difference(['OPEN', 'HIGH', 'LOW', 'CLOSE', 'MID']):
        empty[col] = False
    states = position_states('TEST', empty, trades.iloc[:0], index[2], index[-1])
    assert states.position.eq('flat').all() and states.flat_origin.eq('initial').all()
    assert not states.pending_buy.any()


def test_r26_event_clocks_sources_collisions_and_twenty_bar_labels_stay_out_of_observations():
    from gcn.backtest.signal_research_r26 import audit_frame
    frame, trades = _fixture()
    frame = frame.reindex(pd.bdate_range(frame.index[0], periods=40)).ffill().astype(frame.dtypes.to_dict())
    idx = frame.index; dates = idx.strftime('%Y-%m-%d')
    trades.loc[2, ['exit_date', 'hold_bars']] = [dates[-1], 27]
    frame.loc[idx[7], ['JF_RAW', 'ICON_JUEFAN', 'B_BEAR_RECOVER']] = True
    trades.loc[1, 'entry_jf'] = True
    frame.loc[idx[23], ['CLOSE', 'HIGH']] = [120., 130.]
    frame.loc[idx[36], ['B_ALL_RAW', 'B_SETUP', 'B_STAGE_COMPONENT']] = True
    observations, events, check = audit_frame('TEST', frame, trades, idx[2], idx[-1], source_trusted=True)
    assert len(observations) == 38 and check['actual_entries'] == 3 and check['reconciled']
    assert observations.source_trusted.all()
    assert not set(('ret20_pct', 'win', 'outcome_date', 'actual_entry_date')) & set(observations.columns)
    collision = events[events.date.eq(dates[7])].set_index('event_type')
    assert set(collision.index) == {'raw_b', 'confirmed_b', 'raw_jf', 'tradable_jf'}
    assert collision.position.eq('flat').all() and collision.actual_entry_date.eq(dates[8]).all()
    assert collision.actual_entry_kind.eq('B').all()
    assert collision.drives_entry.tolist() == [False, True, False, False]
    assert collision.loc['raw_b', 'raw_b_state'] == 'suppressed'
    assert collision.loc['raw_b', 'B_BEAR_RECOVER'] and not collision.loc['raw_b', 'B_CRASH_RECOVER']
    assert collision.loc['confirmed_b', 'B_CRASH_RECOVER'] and not collision.loc['confirmed_b', 'B_BEAR_RECOVER']
    assert collision.outcome_date.eq(dates[27]).all() and collision.outcome_complete.all()
    assert np.allclose(collision.ret20_pct, -10.) and collision.interference.all()
    jf = events[events.date.eq(dates[3]) & events.event_type.eq('raw_jf')].iloc[0]
    assert jf.outcome_date == dates[23] and np.isclose(jf.ret20_pct, 20.) and jf.win and not jf.interference
    assert jf.reference_open_date == dates[4] and jf.reference_open == 100.
    late = events[events.date.eq(dates[36])].iloc[0]
    assert late.raw_b_state == 'accepted' and late.position == 'held'
    assert not late.outcome_complete and pd.isna(late.win) and pd.isna(late.interference)
    assert pd.isna(late.ret20_pct) and pd.isna(late.outcome_date)


def test_r26_event_labels_mature_at_twenty_only_and_realized_entries_never_appear_in_past_states():
    from gcn.backtest.signal_research_r26 import audit_frame, EVENT_SCHEMA
    from gcn.backtest.signal_research_r14 import COMPONENTS
    frame, original = _fixture()
    frame = frame.reindex(pd.bdate_range(frame.index[0], periods=42)).ffill().astype(frame.dtypes.to_dict())
    for col in frame.columns.difference(['OPEN', 'HIGH', 'LOW', 'CLOSE', 'MID']):
        frame[col] = False
    idx = frame.index
    frame.loc[idx[[0, 40]], ['JF_RAW', 'ICON_JUEFAN']] = True
    order = original.iloc[[0]].copy()
    order.loc[0, ['entry_date', 'entry_signal_date', 'exit_date', 'exit_reason', 'exit_price', 'hold_bars']] = [
        str(idx[1].date()), str(idx[0].date()), str(idx[-1].date()), 'terminal', 90., 41]
    observed, events, _ = audit_frame('TEST', frame, order, idx[0], idx[-1])
    causal = list(observed.columns) + ['event_type', 'raw_b_state', *COMPONENTS]
    for cut in idx:
        early_obs, early_events, _ = audit_frame('TEST', frame.loc[:cut], order, idx[0], idx[-1])
        selected = events[events.date.le(str(cut.date()))]
        pd.testing.assert_frame_equal(early_obs, observed[observed.date.le(str(cut.date()))], check_exact=True)
        pd.testing.assert_frame_equal(early_events[causal], selected[causal], check_exact=True)
        complete = early_events.outcome_complete
        pd.testing.assert_frame_equal(early_events[complete], selected.loc[early_events[complete].index], check_exact=True)
        assert early_events.loc[~complete, ['win', 'interference', 'ret20_pct', 'outcome_date']].isna().all().all()
        assert early_events.actual_entry_date.dropna().le(str(cut.date())).all()
    at_signal = audit_frame('TEST', frame.loc[:idx[0]], order, idx[0], idx[-1])[1]
    assert at_signal.reference_open_date.isna().all() and not at_signal.next_entry_observed.any()
    assert not audit_frame('TEST', frame, order, idx[0], idx[19])[1].outcome_complete.any()
    assert audit_frame('TEST', frame, order, idx[0], idx[20])[1].outcome_complete.all()
    assert not set(EVENT_SCHEMA) & set(observed.columns)


def test_r26_summary_keeps_event_clocks_complete_denominators_core_zeros_and_overlapping_sources():
    from gcn.backtest.signal_research_r26 import audit_frame, summarize_events
    from gcn.backtest.historical_research import CORE
    frame, trades = _fixture()
    _, events, _ = audit_frame('TEST', frame, trades, frame.index[2], frame.index[-1])
    first = events.event_type.eq('raw_jf') & events.date.eq(str(frame.index[3].date()))
    events.loc[first, ['outcome_complete', 'win', 'interference', 'ret20_pct']] = [True, True, False, 10.]
    accepted = events.event_type.eq('raw_b') & events.raw_b_state.eq('accepted')
    events.loc[accepted, 'B_BEAR_RECOVER'] = True
    summary = summarize_events(events)
    assert len(summary) == 210
    key = summary.set_index(['position', 'event_type', 'group_by', 'group'])
    raw_jf = key.loc[('flat', 'raw_jf', 'all', 'all')]
    assert raw_jf.events == 2 and raw_jf.complete == 1 and raw_jf.incomplete == 1
    assert raw_jf.wins == 1 and raw_jf.win_rate_pct == 100. and raw_jf.actual_entries == 2
    assert raw_jf.direct_entries == 0
    raw_b = key.loc[('flat', 'raw_b', 'raw_b_state', 'suppressed')]
    assert raw_b.events == 1 and raw_b.suppressed_b_with_buy == 1 and raw_b.actual_entries == 1
    assert raw_b.direct_entries == 0 and pd.isna(raw_b.win_rate_pct)
    assert key.loc[('held', 'raw_b', 'component', 'multiple')].events == 1
    for event in ('raw_b', 'confirmed_b', 'raw_jf', 'tradable_jf'):
        for symbol in CORE:
            row = key.loc[('flat', event, 'symbol', symbol)]
            assert row.events == row.complete == row.wins == 0 and pd.isna(row.win_rate_pct)
    empty = summarize_events(events.iloc[:0])
    assert len(empty) == 210 and empty.events.eq(0).all() and empty.win_rate_pct.isna().all()


def test_r26_training_archive_reconciles_frozen_original_orders_without_calling_simulator(tmp_path, monkeypatch):
    import hashlib
    import json
    from gcn.backtest.signal_research_r26 import run_diagnostic
    from gcn.backtest.historical_research import CORE, load_snapshot
    from gcn.recipes.gcn_main import compute_ehopt10
    import gcn.backtest.engine as engine
    import gcn.backtest.historical_research as historical
    def forbidden(*args, **kwargs):
        raise AssertionError('r26不得重新生成订单')
    monkeypatch.setattr(engine, '_one_strategy', forbidden)
    monkeypatch.setattr(historical, '_one_strategy', forbidden)
    snapshot = ROOT / 'reports/signal-audit-v5-review-20260904'
    prior = ROOT / 'reports/gcn-historical-r22-20260905'
    decision = run_diagnostic(snapshot, prior, tmp_path)
    assert decision['recommended'] == 'v5' and not decision['production_changed']
    assert decision['stage'] == 'diagnostic_only' and decision['window'][0] == 'training'
    manifest = json.loads((tmp_path / 'manifest.json').read_bytes())
    archived = ROOT / 'reports/gcn-historical-r26-20260905/training'
    for name in manifest['outputs']:
        assert (tmp_path / name).read_bytes() == (archived / name).read_bytes()
    frames, quality = load_snapshot(snapshot)
    assert manifest['source_quality'] == quality
    assert manifest['environment'] == manifest['input_environment']
    for key, prefix, count in (('outputs', '', 6), ('algorithm_sources', 'source_snapshot', 9),
                               ('input_files', 'input_snapshot', 7), ('input_algorithm_sources', 'input_source_snapshot', 9),
                               ('parent_files', 'parent_snapshot', 23)):
        assert len(manifest[key]) == count
        for name, expected in manifest[key].items():
            assert hashlib.sha256((tmp_path / prefix / name).read_bytes()).hexdigest() == expected
    observations = pd.read_csv(tmp_path / 'observations.csv')
    events = pd.read_csv(tmp_path / 'events.csv')
    checks = pd.read_csv(tmp_path / 'reconciliation.csv').set_index('symbol')
    assert len(pd.read_csv(tmp_path / 'summary.csv')) == 210 and len(checks) == 10
    trades = pd.read_csv(prior / 'training/trades.csv', float_precision='round_trip')
    assert checks.actual_entries.sum() == len(trades) == events.drives_entry.sum() == 50
    for symbol in CORE:
        f = compute_ehopt10(frames[symbol].loc[:'2024-08-26'], version='v5', diagnostics=True).loc['2021-08-27':]
        own = trades[trades.symbol.eq(symbol)]
        obs = observations[observations.symbol.eq(symbol)]
        assert checks.loc[symbol, 'reconciled'] and len(obs) == len(f)
        assert checks.loc[symbol, 'held_closes'] == own.hold_bars.sum()
        for column in ('B_ALL_RAW', 'B_SIGNAL', 'JF_RAW', 'ICON_JUEFAN', 'S_SIGNAL'):
            assert obs[column].tolist() == f[column].tolist()
        assert obs.date.tolist() == f.index.strftime('%Y-%m-%d').tolist()


def test_r26_rejects_corruption_midrun_changes_wrong_environment_and_nonempty_output(tmp_path, monkeypatch):
    import shutil
    import gcn.backtest.signal_research_r26 as research
    prior = tmp_path / 'r22'
    shutil.copytree(ROOT / 'reports/gcn-historical-r22-20260905/training', prior / 'training')
    snapshot = tmp_path / 'parent'
    shutil.copytree(ROOT / 'reports/gcn-historical-r25-20260905/training/parent_snapshot', snapshot)
    output = tmp_path / 'result'
    for path in (prior / 'training/manifest.json', prior / 'training/trades.csv', prior / 'training/paths.csv',
                 prior / 'training/source_snapshot/gcn/backtest/engine.py', snapshot / 'manifest.json',
                 snapshot / 'input_snapshot/TQQQ_1d.csv'):
        raw = path.read_bytes(); path.write_bytes(raw+b'\n')
        with pytest.raises(ValueError):
            research.run_diagnostic(snapshot, prior, output)
        assert not output.exists(); path.write_bytes(raw)
    original_read = Path.read_bytes
    for target, after_first in ((ROOT / 'gcn/recipes/gcn_main.py', False),
                                (ROOT / 'reports/gcn-historical-r26-20260905/protocol.md', False),
                                (ROOT / 'gcn/backtest/signal_research_r26.py', True),
                                (ROOT / 'reports/gcn-historical-r26-20260905/protocol.md', True)):
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
            with pytest.raises(ValueError, match='计算期间|源码|协议'):
                research.run_diagnostic(snapshot, prior, output)
        assert not output.exists()
    with monkeypatch.context() as context:
        context.setattr(research.platform, 'python_version', lambda: '0.0.0')
        with pytest.raises(ValueError, match='环境'):
            research.run_diagnostic(snapshot, prior, output)
    changed = prior / 'training/trades.csv'; raw = changed.read_bytes()
    saved = research.audit_frame
    def mutate(*args, **kwargs):
        changed.write_bytes(raw+b'\n')
        return saved(*args, **kwargs)
    with monkeypatch.context() as context:
        context.setattr(research, 'audit_frame', mutate)
        with pytest.raises(ValueError, match='计算期间'):
            research.run_diagnostic(snapshot, prior, output)
    assert not output.exists(); changed.write_bytes(raw)
    with pytest.raises(ValueError, match='固定'):
        research.run_diagnostic(snapshot, prior, output, window='not-a-window')
    output.mkdir(); marker = output / 'keep.txt'; marker.write_text('keep')
    with pytest.raises(FileExistsError):
        research.run_diagnostic(snapshot, prior, output)
    assert marker.read_text() == 'keep' and len(list(output.iterdir())) == 1


@pytest.mark.parametrize('window,start,end', [
    ('training', '2021-08-27', '2024-08-26'), ('validation', '2024-08-27', '2025-08-26'),
    ('recent', '2025-08-27', '2026-08-26'), ('full', '2021-08-27', '2026-08-26')])
def test_r26_real_fixed_price_prefixes_keep_past_states_and_matured_event_labels(window, start, end):
    from gcn.backtest.historical_research import CORE, load_snapshot
    from gcn.backtest.signal_research_r14 import COMPONENTS
    from gcn.backtest.signal_research_r26 import audit_frame
    from gcn.recipes.gcn_main import compute_ehopt10
    frames, quality = load_snapshot(ROOT / 'reports/signal-audit-v5-review-20260904')
    trades = pd.read_csv(ROOT / 'reports/gcn-historical-r22-20260905' / window / 'trades.csv', float_precision='round_trip')
    start, end = pd.Timestamp(start), pd.Timestamp(end)
    checked = 0
    for symbol in CORE:
        raw = frames[symbol].loc[:end]
        def audit(cut):
            frame = compute_ehopt10(raw.loc[:cut], version='v5', diagnostics=True)
            return audit_frame(symbol, frame, trades, start, end, source_trusted=quality[symbol])
        observations, events, _ = audit(raw.index[-1])
        cuts = {raw.index[0], raw.index[-1], pd.Timestamp(observations.date.iloc[0])}
        own = trades[trades.symbol.eq(symbol)]
        for row in own.itertuples():
            for date in (row.entry_date, row.exit_date):
                pos = raw.index.get_loc(pd.Timestamp(date))
                cuts.update(raw.index[pos-1:pos+1])
        for mask in (observations.raw_b_suppressed, observations.pending_setup_date.notna(),
                     observations.resolved_setup_status.notna()):
            if mask.any():
                cuts.add(pd.Timestamp(observations[mask].date.iloc[0]))
        if events.outcome_complete.any():
            cuts.add(pd.Timestamp(events[events.outcome_complete].outcome_date.iloc[0]))
        causal = list(observations.columns) + ['event_type', 'raw_b_state', *COMPONENTS]
        for cut in sorted(cuts):
            obs, early, _ = audit(cut); date = str(cut.date())
            pd.testing.assert_frame_equal(obs, observations[observations.date.le(date)], check_exact=True)
            expected = events[events.date.le(date)]
            pd.testing.assert_frame_equal(early[causal], expected[causal], check_exact=True)
            complete = early.outcome_complete
            pd.testing.assert_frame_equal(early[complete], expected.loc[early[complete].index], check_exact=True)
            assert early.actual_entry_date.dropna().le(date).all()
            assert early.loc[~complete, ['ret20_pct', 'win', 'interference', 'outcome_date']].isna().all().all()
            checked += 1
    assert checked >= (150 if window in ('training', 'full') else 50)


@pytest.mark.parametrize('window,count', [('validation', 17), ('recent', 17), ('full', 82)])
def test_r26_other_fixed_original_windows_keep_native_orders_terminal_boundaries_and_zero_stocks(tmp_path, window, count):
    import json
    from gcn.backtest.signal_research_r26 import run_diagnostic
    from gcn.backtest.historical_research import CORE
    prior = ROOT / 'reports/gcn-historical-r22-20260905'
    decision = run_diagnostic(ROOT / 'reports/signal-audit-v5-review-20260904', prior, tmp_path, window=window)
    assert decision['window'][0] == window and not decision['production_changed']
    manifest = json.loads((tmp_path / 'manifest.json').read_bytes())
    train = json.loads((ROOT / 'reports/gcn-historical-r26-20260905/training/manifest.json').read_bytes())
    archived = ROOT / 'reports/gcn-historical-r26-20260905' / window
    for name in manifest['outputs']:
        assert (tmp_path / name).read_bytes() == (archived / name).read_bytes()
    assert manifest['algorithm_sources'] == train['algorithm_sources']
    assert manifest['environment'] == train['environment']
    obs = pd.read_csv(tmp_path / 'observations.csv')
    ev = pd.read_csv(tmp_path / 'events.csv')
    checks = pd.read_csv(tmp_path / 'reconciliation.csv').set_index('symbol')
    assert checks.reconciled.all() and checks.actual_entries.sum() == count == ev.drives_entry.sum()
    trades = pd.read_csv(prior / window / 'trades.csv')
    assert obs.terminal_today.sum() == trades.exit_reason.eq('terminal').sum()
    assert obs[obs.terminal_today].position.eq('held').all()
    assert obs[obs.exit_today].position.eq('flat').all()
    for symbol in CORE:
        own = trades[trades.symbol.eq(symbol)]
        assert checks.loc[symbol, 'actual_entries'] == len(own)
        assert checks.loc[symbol, 'held_closes'] == own.hold_bars.sum()
        if own.empty:
            assert checks.loc[symbol, 'flat_closes'] == checks.loc[symbol, 'close_rows']
