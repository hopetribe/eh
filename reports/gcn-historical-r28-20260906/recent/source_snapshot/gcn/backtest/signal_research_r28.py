"""r28：原生实际B入场后的固定Setup依据失效，仅诊断。"""
from __future__ import annotations

import hashlib
import io
import json
import platform
from pathlib import Path

import numpy as np
import pandas as pd

from gcn.backtest.signal_research_r14 import COMPONENTS
from gcn.backtest.signal_research_r26 import position_states, WINDOWS, R22_MANIFESTS
from gcn.backtest.signal_audit import _forward_path
from gcn.backtest.historical_research import CORE, SNAPSHOT_SHA, load_snapshot
from gcn.recipes.gcn_main import compute_ehopt10

CONTEXT_SCHEMA = {
    **{c: 'string' for c in ('symbol', 'trade_id', 'setup_date', 'confirmation_date', 'entry_date')},
    'setup_high': 'float64', 'entry_open': 'float64',
    **{c: 'boolean' for c in ('source_trusted', 'entry_jf', *COMPONENTS)},
}
STATE_SCHEMA = {
    'first_failure_date': 'string', 'first_recovery_date': 'string',
    **{c: 'boolean' for c in ('failed', 'failure_on_entry', 'recovered')},
    **{c: 'Int64' for c in ('bars_to_failure', 'bars_to_recovery', 'failure_to_recovery_bars')},
}
EPISODE_SCHEMA = {**CONTEXT_SCHEMA, **STATE_SCHEMA, 'observed_bars': 'Int64', 'observation_end_date': 'string'}
EVENT_SCHEMA = {
    **CONTEXT_SCHEMA,
    **{c: 'string' for c in ('clock', 'date', 'available_at', 'reference_open_date', 'outcome_date')},
    **{c: 'float64' for c in ('reference_open', 'ret20_pct', 'mfe20_pct', 'mae20_pct')},
    **{c: 'boolean' for c in ('outcome_complete', 'win', 'interference')},
}
TRADE_SCHEMA = {
    **CONTEXT_SCHEMA, 'exit_observed': 'boolean', 'exit_date': 'string', 'exit_reason': 'string',
    'exit_price': 'float64', 'hold_bars': 'Int64', 'return_pct': 'float64', 'trade_win': 'boolean',
}


def audit_positions(symbol: str, frame: pd.DataFrame, orders: pd.DataFrame,
                    start: pd.Timestamp, end: pd.Timestamp, *, source_trusted: bool = False
                    ) -> tuple[pd.DataFrame, pd.DataFrame]:
    """队列只有实际B入场后才出现，固定其原Setup，不以持仓期新信号换参照。"""
    frame = frame.loc[:end]
    states = position_states(symbol, frame, orders, start, end)
    last = str(states.index[-1].date()) if len(states) else ''
    own = orders[orders.symbol.eq(symbol) & orders.entry_b.eq(True) & orders.entry_date.le(last)].sort_values('entry_date')
    episodes, observations = [], []
    obs_schema = {**CONTEXT_SCHEMA, 'date': 'string', **states.dtypes.to_dict(), 'bars_from_entry': 'Int64',
                  **STATE_SCHEMA, **{c: 'boolean' for c in ('above_high', 'above_mid', 'both_above', 'both_failed',
                                                           'first_failure_today', 'first_recovery_today')}}
    for trade in own.itertuples():
        source = frame.loc[pd.Timestamp(trade.setup_date)]
        context = {'symbol': symbol, 'trade_id': trade.trade_id, 'setup_date': trade.setup_date,
                   'confirmation_date': trade.entry_signal_date, 'entry_date': trade.entry_date,
                   'setup_high': float(source.HIGH), 'entry_open': float(trade.entry_open),
                   'source_trusted': source_trusted, 'entry_jf': bool(trade.entry_jf),
                   **{c: bool(getattr(trade, c)) for c in COMPONENTS}}
        held = states[states.held_trade_id.eq(trade.trade_id)]
        first, recovery = None, None
        for offset, (date, state) in enumerate(held.iterrows()):
            above_high, above_mid = bool(state.CLOSE > source.HIGH), bool(state.CLOSE > state.MID)
            known = bool(np.isfinite([state.CLOSE, source.HIGH, state.MID]).all())
            both_above = known and above_high and above_mid
            invalid = known and not above_high and not above_mid
            failure_today = first is None and invalid
            if failure_today:
                first = offset
            recovery_today = first is not None and recovery is None and offset > first and both_above
            if recovery_today:
                recovery = offset
            tracking = {'first_failure_date': str(held.index[first].date()) if first is not None else None,
                        'first_recovery_date': str(held.index[recovery].date()) if recovery is not None else None,
                        'failed': first is not None, 'failure_on_entry': first == 0,
                        'recovered': recovery is not None, 'bars_to_failure': first,
                        'bars_to_recovery': recovery, 'failure_to_recovery_bars': recovery-first if recovery is not None else None}
            observations.append({**context, 'date': str(date.date()), **state.to_dict(), 'bars_from_entry': offset,
                                 **tracking, 'above_high': above_high, 'above_mid': above_mid,
                                 'both_above': both_above, 'both_failed': invalid,
                                 'first_failure_today': failure_today, 'first_recovery_today': recovery_today})
        episodes.append({**context, **tracking, 'observed_bars': len(held),
                         'observation_end_date': str(held.index[-1].date())})
    return (pd.DataFrame(episodes, columns=EPISODE_SCHEMA).astype(EPISODE_SCHEMA),
            pd.DataFrame(observations, columns=obs_schema).astype(obs_schema))


def event_labels(frame: pd.DataFrame, episodes: pd.DataFrame, *, end: pd.Timestamp) -> pd.DataFrame:
    """实际入场队列在入场时才可知；三个价格时钟分别等到20根成熟。"""
    frame = frame.loc[:end]
    last = str(frame.index[-1].date()) if len(frame) else ''
    rows = []
    for ep in episodes[episodes.entry_date.le(last)].to_dict('records'):
        for clock, column in (('confirmation', 'confirmation_date'), ('entry', 'entry_date'), ('failure', 'first_failure_date')):
            date = ep[column]
            if pd.isna(date) or date > last:
                continue
            pos = frame.index.get_loc(pd.Timestamp(date))
            path = _forward_path(frame, pos, 20, outcome_end=end)
            complete = bool(np.isfinite(path['return']))
            rows.append({**{c: ep[c] for c in CONTEXT_SCHEMA}, 'clock': clock, 'date': date,
                         'available_at': date if clock == 'failure' else ep['entry_date'],
                         'reference_open_date': str(frame.index[pos+1].date()) if pos+1 < len(frame) else None,
                         'reference_open': float(frame.OPEN.iloc[pos+1]) if pos+1 < len(frame) else np.nan,
                         'outcome_complete': complete,
                         'outcome_date': str(frame.index[pos+20].date()) if complete else None,
                         'ret20_pct': path['return']*100, 'mfe20_pct': path['mfe']*100, 'mae20_pct': path['mae']*100,
                         'win': bool(path['return'] > 0) if complete else None,
                         'interference': bool(path['return'] < 0 and path['mae'] <= -.08) if complete else None})
    return pd.DataFrame(rows, columns=EVENT_SCHEMA).astype(EVENT_SCHEMA)


def trade_outcomes(episodes: pd.DataFrame, orders: pd.DataFrame, *, asof: pd.Timestamp) -> pd.DataFrame:
    """只转录已观察到退出的原订单净结果，不把事后损益放入实时观察。"""
    last = str(asof.date())
    rows = []
    for ep in episodes[episodes.entry_date.le(last)].to_dict('records'):
        own = orders[orders.trade_id.eq(ep['trade_id'])]
        if len(own) != 1 or own.symbol.iloc[0] != ep['symbol'] or own.entry_date.iloc[0] != ep['entry_date']:
            raise ValueError('原订单不能与B队列唯一配对')
        order = own.iloc[0]
        observed = order.exit_date <= last
        rows.append({**{c: ep[c] for c in CONTEXT_SCHEMA}, 'exit_observed': observed,
                     **{c: order[c] if observed else None for c in ('exit_date', 'exit_reason', 'exit_price', 'hold_bars', 'return_pct')},
                     'trade_win': bool(order.return_pct > 0) if observed else None})
    return pd.DataFrame(rows, columns=TRADE_SCHEMA).astype(TRADE_SCHEMA)


def summarize(episodes: pd.DataFrame, events: pd.DataFrame, trades: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """实际交易和三个价格时钟用独立表及分母，结局分层仅为事后归因。"""
    groups = [('all', 'all', episodes)]
    groups.extend(('symbol', symbol, episodes[episodes.symbol.eq(symbol)]) for symbol in CORE)
    for field in ('source_trusted', 'entry_jf'):
        groups.extend((field, str(value).lower(), episodes[episodes[field].eq(value)]) for value in (False, True))
    groups.extend(('component', c, episodes[episodes[c]]) for c in COMPONENTS)
    counts = episodes[list(COMPONENTS)].sum(axis=1)
    groups.extend([('component', 'multiple', episodes[counts.gt(1)]), ('component', 'none', episodes[counts.eq(0)])])
    for group, mask in (('none', ~episodes.failed), ('entry', episodes.failure_on_entry),
                        ('later', episodes.failed & ~episodes.failure_on_entry)):
        groups.append(('first_failure', group, episodes[mask]))
    for group, mask in (('no_failure', ~episodes.failed), ('unrecovered', episodes.failed & ~episodes.recovered),
                        ('recovered', episodes.recovered)):
        groups.append(('recovery', group, episodes[mask]))
    for reason in ('signal', 'trail', 'terminal', 'unobserved'):
        selected = trades[~trades.exit_observed if reason == 'unobserved' else trades.exit_reason.eq(reason)]
        groups.append(('exit_reason', reason, episodes[episodes.trade_id.isin(selected.trade_id)]))
    rows, clock_rows = [], []
    for field, group, cohort in groups:
        actual = trades[trades.trade_id.isin(cohort.trade_id) & trades.exit_observed]
        failed, recovered = cohort[cohort.failed], cohort[cohort.recovered]
        rows.append({'group_by': field, 'group': group, 'trades': len(cohort), 'observed_exits': len(actual),
                     'unobserved_exits': len(cohort)-len(actual), 'original_wins': int(actual.trade_win.sum()),
                     'original_win_rate_pct': actual.trade_win.mean()*100 if len(actual) else np.nan,
                     'mean_original_return_pct': actual.return_pct.mean(), 'median_original_return_pct': actual.return_pct.median(),
                     'failed_trades': len(failed), 'no_failure': len(cohort)-len(failed),
                     'entry_failures': int(cohort.failure_on_entry.sum()), 'recovered_trades': len(recovered),
                     'unrecovered_failed': len(failed)-len(recovered),
                     'median_bars_to_failure': failed.bars_to_failure.median() if len(failed) else np.nan,
                     'median_failure_to_recovery_bars': recovered.failure_to_recovery_bars.median() if len(recovered) else np.nan})
        for clock in ('confirmation', 'entry', 'failure'):
            selected = events[events.trade_id.isin(cohort.trade_id) & events.clock.eq(clock)]
            complete = selected[selected.outcome_complete]
            clock_rows.append({'clock': clock, 'group_by': field, 'group': group, 'cohort_trades': len(cohort),
                               'events': len(selected), 'without_clock': len(cohort)-len(selected),
                               'complete': len(complete), 'incomplete': len(selected)-len(complete),
                               'wins': int(complete.win.sum()), 'interference': int(complete.interference.sum()),
                               'win_rate_pct': complete.win.mean()*100 if len(complete) else np.nan,
                               'mean_ret20_pct': complete.ret20_pct.mean(), 'median_ret20_pct': complete.ret20_pct.median(),
                               'mean_mfe20_pct': complete.mfe20_pct.mean(), 'mean_mae20_pct': complete.mae20_pct.mean()})
    schema = {'group_by': 'string', 'group': 'string'}
    return pd.DataFrame(rows).astype(schema), pd.DataFrame(clock_rows).astype({**schema, 'clock': 'string'})


def run_diagnostic(snapshot: Path, r22: Path, output: Path, *, window: str = 'training') -> dict:
    """绑定原订单、原生源码和r27训练机制，每次生成一个固定窗口诊断。"""
    selected = next((list(spec) for spec in WINDOWS if spec[0] == window), None)
    if selected is None:
        raise ValueError('仅允许r28固定窗口')
    if output.exists() and any(output.iterdir()):
        raise FileExistsError('诊断目录非空，请使用新的输出目录')
    root = Path(__file__).resolve().parents[2]
    captured = {}
    def capture(path):
        raw = path.read_bytes(); captured[path] = raw
        return raw
    digest = lambda raw: hashlib.sha256(raw).hexdigest()
    native = ('gcn/backtest/signal_research_r14.py', 'gcn/backtest/signal_audit.py',
              'gcn/recipes/gcn_main.py', 'gcn/core/tdx.py', 'gcn/core/indicators.py')
    names = ('gcn/backtest/signal_research_r28.py', 'gcn/backtest/signal_research_r27.py',
             'gcn/backtest/signal_research_r26.py', 'gcn/backtest/signal_research_r25.py',
             'gcn/backtest/historical_research.py', 'gcn/backtest/engine.py', *native)
    sources = {name: capture(root / name) for name in names}
    protocol = capture(root / 'reports/gcn-historical-r28-20260906/protocol.md')
    if digest(protocol) != '4e39aa222ba4b24ad7115d85b7ec1b514757ef48530d1c77e76c4f2d45f11a22':
        raise ValueError('r28冻结协议变化')
    mechanism = {}
    for name, expected in (
        ('reports/gcn-historical-r27-20260905/training/manifest.json', '201c115c61db4f610ed41893be8ba2a9ba2763a939b90089708c7d57d45c4956'),
        ('reports/gcn-historical-r27-20260905/training-mechanism.md', '4ef6688cb47c4456c719f741c89bf1694fbc1906140ee1577eb201c90314bb13')):
        raw = capture(root / name)
        if digest(raw) != expected:
            raise ValueError('r27冻结训练机制变化')
        mechanism[name] = raw
    origin = json.loads(mechanism['reports/gcn-historical-r27-20260905/training/manifest.json'])
    for name, expected in origin['algorithm_sources'].items():
        if digest(sources[name]) != expected:
            raise ValueError(f'原生状态源码与r27不一致：{name}')
    folder = r22 / window
    inputs = {'manifest.json': capture(folder / 'manifest.json')}
    if digest(inputs['manifest.json']) != R22_MANIFESTS[window]:
        raise ValueError('r22冻结manifest不匹配')
    parent = json.loads(inputs['manifest.json'])
    if (parent['research_version'] != 'gcn-historical-r22' or parent['window'] != selected
            or parent['parent_manifest_sha256'] != SNAPSHOT_SHA):
        raise ValueError('r22固定窗口或父输入不一致')
    for name, expected in parent['outputs'].items():
        raw = capture(folder / name)
        if digest(raw) != expected:
            raise ValueError(f'r22输入内容变化：{name}')
        inputs[name] = raw
    input_sources = {}
    for name, expected in parent['algorithm_sources'].items():
        raw = capture(folder / 'source_snapshot' / name)
        if digest(raw) != expected or (name in native and sources[name] != raw):
            raise ValueError(f'r22冻结或原生源码变化：{name}')
        input_sources[name] = raw
    parent_files = {'manifest.json': capture(snapshot / 'manifest.json')}
    if digest(parent_files['manifest.json']) != SNAPSHOT_SHA:
        raise ValueError('父manifest摘要不匹配')
    for spec in json.loads(parent_files['manifest.json'])['inputs'].values():
        for key, sha in (('snapshot_path', 'sha256'), ('metadata_snapshot_path', 'metadata_sha256')):
            name = spec[key]; raw = capture(snapshot / name)
            if digest(raw) != spec[sha]:
                raise ValueError(f'父输入快照变化：{name}')
            parent_files[name] = raw
    frames, quality = load_snapshot(snapshot)
    environment = {'python': platform.python_version(), 'numpy': np.__version__, 'pandas': pd.__version__}
    if quality != parent['source_quality'] or environment != parent['environment'] or environment != origin['environment']:
        raise ValueError('r22/r27来源质量或运行环境不一致')
    orders = pd.read_csv(io.BytesIO(inputs['trades.csv']), float_precision='round_trip')
    prior_checks = pd.read_csv(io.BytesIO(inputs['reconciliation.csv'])).set_index('symbol')
    if (not orders.window.eq(window).all() or not set(orders.symbol).issubset(CORE)
            or set(prior_checks.index) != set(CORE)):
        raise ValueError('r22原订单窗口或标的集合不一致')
    start, end = pd.Timestamp(selected[1]), pd.Timestamp(selected[2])
    episode_parts, observation_parts, event_parts, trade_parts, checks = [], [], [], [], []
    for symbol in CORE:
        frame = compute_ehopt10(frames[symbol].loc[:end], version='v5', diagnostics=True)
        ep, obs = audit_positions(symbol, frame, orders, start, end, source_trusted=quality[symbol])
        states = position_states(symbol, frame, orders, start, end)
        own = orders[orders.symbol.eq(symbol)]; b = own[own.entry_b]; prior = prior_checks.loc[symbol]
        if (int(states.entry_today.sum()) != prior.trades or int(states.position.eq('held').sum()) != own.hold_bars.sum()
                or int(states.B_SIGNAL.sum()) != prior.b_signals or int(states.ICON_JUEFAN.sum()) != prior.jf_signals
                or int(states.S_SIGNAL.sum()) != prior.s_signals or len(ep) != len(b) or len(ep) != prior.b_trades
                or len(obs) != b.hold_bars.sum() or len(obs) != int(ep.observed_bars.sum())):
            raise ValueError(f'{symbol}: 原订单/信号/B队列或持仓根数对账失败')
        events = event_labels(frame, ep, end=end)
        trades = trade_outcomes(ep, orders, asof=frame.index[-1])
        if len(events) != 2*len(ep)+int(ep.failed.sum()) or not trades.exit_observed.all():
            raise ValueError('原交易结局或三个事件时钟数量不一致')
        checks.append({'symbol': symbol, 'actual_entries': len(own), 'b_trades': len(b),
                       'collision_trades': int(ep.entry_jf.sum()), 'observed_b_closes': len(obs),
                       'first_failures': int(ep.failed.sum()), 'entry_failures': int(ep.failure_on_entry.sum()),
                       'recoveries': int(ep.recovered.sum()), 'reconciled': True})
        episode_parts.append(ep); observation_parts.append(obs); event_parts.append(events); trade_parts.append(trades)
    episodes = pd.concat(episode_parts, ignore_index=True)
    observations = pd.concat(observation_parts, ignore_index=True)
    events = pd.concat(event_parts, ignore_index=True)
    trades = pd.concat(trade_parts, ignore_index=True)
    summary, event_summary = summarize(episodes, events, trades)
    tables = {'episodes': episodes, 'observations': observations, 'events': events, 'trades': trades,
              'summary': summary, 'event_summary': event_summary, 'reconciliation': pd.DataFrame(checks)}
    decision = {'research_version': 'gcn-historical-r28', 'stage': 'diagnostic_only', 'recommended': 'v5',
                'production_changed': False, 'window': selected, 'core': CORE,
                'input': 'frozen r22 original orders and r27 training mechanism; no order simulation',
                'cohort': 'actual original entry_b trades; B/JF collision retained; pure JF excluded',
                'observation': 'actual entry CLOSE through pre-exit CLOSE; terminal last CLOSE included',
                'failure': 'first finite CLOSE <= original Setup HIGH and current MID; later first both-above recovery',
                'clocks': 'confirmation, entry CLOSE, first failure use separate next OPEN / 20th CLOSE gross labels',
                'trade_outcomes': 'original net outcomes in separate table, only after actual exit observed',
                'stopping': 'no new exit rule, parameter scan or production promotion'}
    for path, raw in captured.items():
        if path.read_bytes() != raw:
            raise ValueError(f'计算期间输入或源码变化：{path.name}')
    if output.exists() and any(output.iterdir()):
        raise FileExistsError('诊断目录非空，请使用新的输出目录')
    output.mkdir(parents=True, exist_ok=True)
    for name, table in tables.items():
        table.to_csv(output / (name + '.csv'), index=False)
    (output / 'decision.json').write_text(json.dumps(decision, indent=2, ensure_ascii=False) + '\n')
    (output / 'protocol.md').write_bytes(protocol)
    for prefix, files in (('source_snapshot', sources), ('input_snapshot', inputs), ('input_source_snapshot', input_sources),
                          ('parent_snapshot', parent_files), ('mechanism_snapshot', mechanism)):
        for name, raw in files.items():
            target = output / prefix / name
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(raw)
    manifest = {'research_version': 'gcn-historical-r28', 'window': selected,
                'parent_manifest_sha256': SNAPSHOT_SHA, 'input_manifest_sha256': R22_MANIFESTS[window],
                'source_quality': quality, 'input_environment': parent['environment'], 'environment': environment,
                'protocol_sha256': digest(protocol),
                **{key: {name: digest(raw) for name, raw in files.items()} for key, files in
                   (('algorithm_sources', sources), ('input_files', inputs), ('input_algorithm_sources', input_sources),
                    ('parent_files', parent_files), ('mechanism_files', mechanism))},
                'outputs': {p.name: digest(p.read_bytes()) for p in sorted(output.iterdir()) if p.is_file()}}
    (output / 'manifest.json').write_text(json.dumps(manifest, indent=2) + '\n')
    return decision


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--snapshot', type=Path, default=Path('reports/signal-audit-v5-review-20260904'))
    parser.add_argument('--r22', type=Path, default=Path('reports/gcn-historical-r22-20260905'))
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--window', choices=[spec[0] for spec in WINDOWS], default='training')
    args = parser.parse_args()
    print(json.dumps(run_diagnostic(args.snapshot, args.r22, args.output, window=args.window), indent=2, ensure_ascii=False))
