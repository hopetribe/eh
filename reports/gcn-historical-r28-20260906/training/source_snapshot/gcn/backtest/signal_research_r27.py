"""r27：原v5过期Setup的价格门槛与后续自然恢复，仅诊断。"""
from __future__ import annotations

import hashlib
import io
import json
import platform
from pathlib import Path

import numpy as np
import pandas as pd

from gcn.backtest.signal_research_r14 import COMPONENTS, trace_setups
from gcn.backtest.signal_research_r26 import position_states, WINDOWS, R22_MANIFESTS
from gcn.backtest.signal_audit import _forward_path
from gcn.backtest.historical_research import CORE, SNAPSHOT_SHA, load_snapshot
from gcn.recipes.gcn_main import compute_ehopt10

EPISODE_SCHEMA = {
    **{c: 'string' for c in ('symbol', 'episode_id', 'setup_date', 'expiry_date', 'failure', 'expiry_position',
                             'first_recovery_date', 'recovery_position', 'next_setup_date', 'observation_end_date', 'end_kind')},
    **{c: 'float64' for c in ('setup_open', 'setup_high', 'setup_low', 'setup_close', 'setup_mid')},
    **{c: 'boolean' for c in ('source_trusted', 'setup_before_window', 'expiry_buy', 'eligible_at_expiry',
                              'recovered', 'recovery_buy', 'recovery_uncovered', 'right_censored', *COMPONENTS)},
    **{c: 'Int64' for c in ('recovery_wait_bars', 'post_expiry_bars')},
}
WAIT_SCHEMA = {
    **{c: 'string' for c in ('symbol', 'episode_id', 'date', 'available_at', 'position')},
    'age': 'Int64', **{c: 'float64' for c in ('close', 'mid', 'setup_high')},
    **{c: 'boolean' for c in ('above_high', 'above_mid', 'both')},
}
EVENT_CONTEXT = ('symbol', 'episode_id', 'setup_date', 'expiry_date', 'failure', 'expiry_position',
                 'expiry_buy', 'eligible_at_expiry', 'source_trusted', 'setup_before_window', *COMPONENTS)
EVENT_SCHEMA = {
    **{c: EPISODE_SCHEMA[c] for c in EVENT_CONTEXT},
    **{c: 'string' for c in ('clock', 'date', 'available_at', 'reference_open_date', 'outcome_date')},
    **{c: 'float64' for c in ('reference_open', 'ret20_pct', 'mfe20_pct', 'mae20_pct')},
    **{c: 'boolean' for c in ('outcome_complete', 'win', 'interference')},
}


def audit_setups(symbol: str, frame: pd.DataFrame, orders: pd.DataFrame,
                 start: pd.Timestamp, end: pd.Timestamp, *, source_trusted: bool = False
                 ) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """过期当天才建立诊断事件；五根等待历史显式以过期日为可知时间。"""
    frame = frame.loc[:end]
    trace = trace_setups(frame)
    states = position_states(symbol, frame, orders, start, end)
    expired = trace[trace.status.eq('expired') & trace.resolution_date.ge(str(start.date()))
                    & trace.resolution_date.le(str(end.date()))]
    episodes, waiting, observations = [], [], []
    obs_schema = {
        **{c: 'string' for c in ('symbol', 'episode_id', 'date', 'expiry_date', 'first_recovery_date')},
        **states.dtypes.to_dict(), 'setup_high': 'float64', 'bars_after_expiry': 'Int64',
        **{c: 'boolean' for c in ('above_high', 'above_mid', 'both', 'recovered', 'first_recovery_today')},
    }
    for setup in expired.itertuples():
        i, j = int(setup.setup_i), int(setup.resolution_i)
        state = states.loc[frame.index[j]]
        source = frame.iloc[i]
        episode_id = symbol + ':' + setup.setup_date
        wait = frame.iloc[i+1:j+1]
        if len(wait) != 5:
            raise ValueError('过期必须经过五根等待K线')
        for age, (date, bar) in enumerate(wait.iterrows(), 1):
            above_high, above_mid = bool(bar.CLOSE > source.HIGH), bool(bar.CLOSE > bar.MID)
            waiting.append({'symbol': symbol, 'episode_id': episode_id, 'date': str(date.date()),
                            'available_at': setup.resolution_date, 'age': age,
                            'position': states.loc[date, 'position'] if date in states.index else 'outside_window',
                            'close': float(bar.CLOSE), 'mid': float(bar.MID), 'setup_high': float(source.HIGH),
                            'above_high': above_high, 'above_mid': above_mid, 'both': above_high and above_mid})
        last = waiting[-1]
        failure = 'both' if not last['above_high'] and not last['above_mid'] else 'mid' if last['above_high'] else 'high'
        buy = bool(state.B_SIGNAL or state.ICON_JUEFAN)
        later = trace.loc[trace.setup_i.gt(j), 'setup_i']
        stop = int(later.iloc[0]) if len(later) else len(frame)
        first = None
        for k in range(j, stop):
            date = frame.index[k]; current = states.loc[date]
            above_high = bool(current.CLOSE > source.HIGH)
            above_mid = bool(current.CLOSE > current.MID)
            recovered_today = first is None and k > j and above_high and above_mid
            if recovered_today:
                first = k
            observations.append({'symbol': symbol, 'episode_id': episode_id, 'date': str(date.date()),
                                 'expiry_date': setup.resolution_date, **current.to_dict(),
                                 'setup_high': float(source.HIGH), 'bars_after_expiry': k-j,
                                 'above_high': above_high, 'above_mid': above_mid, 'both': above_high and above_mid,
                                 'first_recovery_date': str(frame.index[first].date()) if first is not None else None,
                                 'recovered': first is not None, 'first_recovery_today': recovered_today})
        recovery = states.loc[frame.index[first]] if first is not None else None
        recovery_buy = bool(recovery.B_SIGNAL or recovery.ICON_JUEFAN) if recovery is not None else False
        episodes.append({'symbol': symbol, 'episode_id': episode_id, 'setup_date': setup.setup_date,
                         'expiry_date': setup.resolution_date, 'failure': failure,
                         'expiry_position': state.position, 'expiry_buy': buy,
                         'eligible_at_expiry': state.position == 'flat' and not buy,
                         'source_trusted': source_trusted, 'setup_before_window': frame.index[i] < start,
                         **{'setup_' + c.lower(): float(source[c]) for c in ('OPEN', 'HIGH', 'LOW', 'CLOSE', 'MID')},
                         **{c: bool(getattr(setup, c)) for c in COMPONENTS},
                         'first_recovery_date': str(frame.index[first].date()) if first is not None else None,
                         'recovery_wait_bars': first-j if first is not None else None,
                         'recovery_position': recovery.position if recovery is not None else None,
                         'recovery_buy': recovery_buy,
                         'recovery_uncovered': recovery is not None and recovery.position == 'flat' and not recovery_buy,
                         'recovered': first is not None, 'right_censored': first is None,
                         'next_setup_date': str(frame.index[stop].date()) if stop < len(frame) else None,
                         'observation_end_date': str(frame.index[stop-1].date()), 'post_expiry_bars': stop-j-1,
                         'end_kind': 'next_setup' if len(later) else 'window_end' if frame.index[-1] == end else 'asof'})
    return (pd.DataFrame(episodes, columns=EPISODE_SCHEMA).astype(EPISODE_SCHEMA),
            pd.DataFrame(waiting, columns=WAIT_SCHEMA).astype(WAIT_SCHEMA),
            pd.DataFrame(observations, columns=obs_schema).astype(obs_schema))


def event_labels(frame: pd.DataFrame, episodes: pd.DataFrame, *, end: pd.Timestamp) -> pd.DataFrame:
    """过期队列的三个独立事后时钟；选择已知时点和20根成熟时点均受当前前缀约束。"""
    frame = frame.loc[:end]
    last = str(frame.index[-1].date()) if len(frame) else ''
    rows = []
    for episode in episodes[episodes.expiry_date.le(last)].to_dict('records'):
        for clock, column in (('setup', 'setup_date'), ('expiry', 'expiry_date'), ('recovery', 'first_recovery_date')):
            date = episode[column]
            if pd.isna(date) or date > last:
                continue
            pos = frame.index.get_loc(pd.Timestamp(date))
            path = _forward_path(frame, pos, 20, outcome_end=end)
            complete = bool(np.isfinite(path['return']))
            rows.append({**{c: episode[c] for c in EVENT_CONTEXT}, 'clock': clock, 'date': date,
                         'available_at': episode['expiry_date'] if clock != 'recovery' else date,
                         'reference_open_date': str(frame.index[pos+1].date()) if pos+1 < len(frame) else None,
                         'reference_open': float(frame.OPEN.iloc[pos+1]) if pos+1 < len(frame) else np.nan,
                         'outcome_complete': complete,
                         'outcome_date': str(frame.index[pos+20].date()) if complete else None,
                         'ret20_pct': path['return']*100, 'mfe20_pct': path['mfe']*100, 'mae20_pct': path['mae']*100,
                         'win': bool(path['return'] > 0) if complete else None,
                         'interference': bool(path['return'] < 0 and path['mae'] <= -.08) if complete else None})
    return pd.DataFrame(rows, columns=EVENT_SCHEMA).astype(EVENT_SCHEMA)


def summarize_events(episodes: pd.DataFrame, events: pd.DataFrame) -> pd.DataFrame:
    """过期队列、恢复缺失及成熟20根分别计数；不合并三个事件时钟。"""
    rows = []
    def add(scope, clock, group_by, group, subset):
        selected = events[events.episode_id.isin(subset.episode_id) & events.clock.eq(clock)]
        complete = selected[selected.outcome_complete]
        recovered = subset[subset.recovered]
        n = len(complete)
        rows.append({'scope': scope, 'clock': clock, 'group_by': group_by, 'group': group,
                     'cohort_episodes': len(subset), 'events': len(selected), 'without_clock': len(subset)-len(selected),
                     'complete': n, 'incomplete': len(selected)-n, 'symbols': subset.symbol.nunique(),
                     'wins': int(complete.win.sum()), 'interference': int(complete.interference.sum()),
                     'win_rate_pct': complete.win.mean()*100 if n else np.nan,
                     'interference_rate_pct': complete.interference.mean()*100 if n else np.nan,
                     'mean_ret20_pct': complete.ret20_pct.mean() if n else np.nan,
                     'median_ret20_pct': complete.ret20_pct.median() if n else np.nan,
                     'mean_mfe20_pct': complete.mfe20_pct.mean() if n else np.nan,
                     'mean_mae20_pct': complete.mae20_pct.mean() if n else np.nan,
                     'recovered_episodes': len(recovered), 'uncovered_recoveries': int(subset.recovery_uncovered.sum()),
                     'right_censored': int(subset.right_censored.sum()),
                     'median_recovery_bars': recovered.recovery_wait_bars.median() if len(recovered) else np.nan})
    for scope in ('all', 'eligible'):
        selected = episodes if scope == 'all' else episodes[episodes.eligible_at_expiry]
        expiry_groups = {'held': selected.expiry_position.eq('held'),
                         'flat_native_buy': selected.expiry_position.eq('flat') & selected.expiry_buy,
                         'flat_uncovered': selected.eligible_at_expiry}
        recovery_groups = {'unobserved': ~selected.recovered,
                           'held': selected.recovered & selected.recovery_position.eq('held'),
                           'flat_native_buy': selected.recovered & selected.recovery_position.eq('flat') & selected.recovery_buy,
                           'flat_uncovered': selected.recovery_uncovered}
        for clock in ('setup', 'expiry', 'recovery'):
            add(scope, clock, 'all', 'all', selected)
            for symbol in CORE:
                add(scope, clock, 'symbol', symbol, selected[selected.symbol.eq(symbol)])
            for field in ('source_trusted', 'right_censored', 'setup_before_window'):
                for value in (False, True):
                    add(scope, clock, field, str(value).lower(), selected[selected[field].eq(value)])
            for group, mask in expiry_groups.items():
                add(scope, clock, 'expiry_state', group, selected[mask])
            for group, mask in recovery_groups.items():
                add(scope, clock, 'recovery_state', group, selected[mask])
            for failure in ('high', 'mid', 'both'):
                add(scope, clock, 'failure', failure, selected[selected.failure.eq(failure)])
            for component in COMPONENTS:
                add(scope, clock, 'component', component, selected[selected[component]])
            counts = selected[list(COMPONENTS)].sum(axis=1)
            add(scope, clock, 'component', 'multiple', selected[counts.gt(1)])
            add(scope, clock, 'component', 'none', selected[counts.eq(0)])
    return pd.DataFrame(rows).astype({c: 'string' for c in ('scope', 'clock', 'group_by', 'group')})


def run_diagnostic(snapshot: Path, r22: Path, output: Path, *, window: str = 'training') -> dict:
    """绑定旧订单、五根原生状态与训练机制，每次只生成一个固定窗口诊断。"""
    selected = next((list(spec) for spec in WINDOWS if spec[0] == window), None)
    if selected is None:
        raise ValueError('仅允许r27固定窗口')
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
    names = ('gcn/backtest/signal_research_r27.py', 'gcn/backtest/signal_research_r26.py',
             'gcn/backtest/signal_research_r25.py', 'gcn/backtest/historical_research.py',
             'gcn/backtest/engine.py', *native)
    sources = {name: capture(root / name) for name in names}
    protocol = capture(root / 'reports/gcn-historical-r27-20260905/protocol.md')
    if digest(protocol) != 'f085ced96ccf00213cf99e7a2c47f9789d2c368891b942f82a5be21c855d2b75':
        raise ValueError('r27冻结协议变化')
    mechanism = {}
    for name, expected in (
        ('reports/gcn-historical-r26-20260905/training/manifest.json', 'c52b74900530db0f267f04a98602b00dc7b0474c73ef4f70e804d3f6efc7665a'),
        ('reports/gcn-historical-r26-20260905/training-mechanism.md', 'caabcb33a76613528ed2bd6be98e73ea3088dfce12383bd8440905cf9c5eb518')):
        raw = capture(root / name)
        if digest(raw) != expected:
            raise ValueError('r26冻结训练机制变化')
        mechanism[name] = raw
    origin = json.loads(mechanism['reports/gcn-historical-r26-20260905/training/manifest.json'])
    for name, expected in origin['algorithm_sources'].items():
        if digest(sources[name]) != expected:
            raise ValueError(f'原生状态源码与r26不一致：{name}')
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
        raise ValueError('r22/r26来源质量或运行环境不一致')
    orders = pd.read_csv(io.BytesIO(inputs['trades.csv']), float_precision='round_trip')
    prior_checks = pd.read_csv(io.BytesIO(inputs['reconciliation.csv'])).set_index('symbol')
    if (not orders.window.eq(window).all() or not set(orders.symbol).issubset(CORE)
            or set(prior_checks.index) != set(CORE)):
        raise ValueError('r22原订单窗口或标的集合不一致')
    start, end = pd.Timestamp(selected[1]), pd.Timestamp(selected[2])
    episode_parts, waiting_parts, observation_parts, checks = [], [], [], []
    for symbol in CORE:
        frame = compute_ehopt10(frames[symbol].loc[:end], version='v5', diagnostics=True)
        ep, wait, obs = audit_setups(symbol, frame, orders, start, end, source_trusted=quality[symbol])
        states = position_states(symbol, frame, orders, start, end)
        own = orders[orders.symbol.eq(symbol)]; prior = prior_checks.loc[symbol]
        if (int(states.entry_today.sum()) != prior.trades or int(states.position.eq('held').sum()) != own.hold_bars.sum()
                or int(states.B_SIGNAL.sum()) != prior.b_signals or int(states.ICON_JUEFAN.sum()) != prior.jf_signals
                or int(states.S_SIGNAL.sum()) != prior.s_signals or len(ep) != int(frame.loc[start:end].B_SETUP_EXPIRED.sum())
                or len(wait) != 5*len(ep) or wait.both.any() or len(obs) != int((ep.post_expiry_bars+1).sum())):
            raise ValueError(f'{symbol}: 原订单/信号/过期或观察根数对账失败')
        checks.append({'symbol': symbol, 'actual_entries': len(own), 'expired_setups': len(ep),
                       'eligible_at_expiry': int(ep.eligible_at_expiry.sum()), 'waiting_rows': len(wait),
                       'observation_rows': len(obs), 'recovered': int(ep.recovered.sum()), 'reconciled': True})
        episode_parts.append(ep); waiting_parts.append(wait); observation_parts.append(obs)
    episodes = pd.concat(episode_parts, ignore_index=True)
    waiting = pd.concat(waiting_parts, ignore_index=True)
    observations = pd.concat(observation_parts, ignore_index=True)
    event_parts = []
    for symbol in CORE:
        frame = frames[symbol].loc[:end].rename(columns=str.upper)
        event_parts.append(event_labels(frame, episodes[episodes.symbol.eq(symbol)], end=end))
    events = pd.concat(event_parts, ignore_index=True)
    if len(events) != 2*len(episodes) + int(episodes.recovered.sum()):
        raise ValueError('三个事件时钟数量与已知过期/恢复不一致')
    tables = {'episodes': episodes, 'waiting': waiting, 'observations': observations, 'events': events,
              'summary': summarize_events(episodes, events), 'reconciliation': pd.DataFrame(checks)}
    decision = {'research_version': 'gcn-historical-r27', 'stage': 'diagnostic_only', 'recommended': 'v5',
                'production_changed': False, 'window': selected, 'core': CORE,
                'input': 'frozen r22 original v5 orders and r26 training mechanism; no order simulation',
                'waiting': 'five historical bars become available only at expiry; pre-window position is unknown',
                'observations': 'expiry CLOSE onward; first later CLOSE above old Setup HIGH and current MID',
                'termination': 'next native Setup day excluded, otherwise window end; no recovery is right-censored',
                'event_clocks': 'Setup, expiry and first recovery each use their own next OPEN / 20th CLOSE gross labels',
                'stopping': 'no confirmation-window search, buyback strategy or production promotion'}
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
    manifest = {'research_version': 'gcn-historical-r27', 'window': selected,
                'parent_manifest_sha256': SNAPSHOT_SHA, 'input_manifest_sha256': R22_MANIFESTS[window],
                'source_quality': quality, 'input_environment': parent['environment'],
                'protocol_sha256': digest(protocol), 'environment': environment,
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
