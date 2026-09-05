"""r26：原v5真实空仓买信号诊断；复用冻结订单，不生成候选交易。"""
from __future__ import annotations

import hashlib
import io
import json
import platform
from pathlib import Path

import numpy as np
import pandas as pd

from gcn.backtest.signal_research_r14 import COMPONENTS
from gcn.backtest.signal_research_r25 import signal_states
from gcn.backtest.signal_audit import _forward_path
from gcn.backtest.historical_research import CORE, SNAPSHOT_SHA, load_snapshot
from gcn.recipes.gcn_main import compute_ehopt10

WINDOWS = (('training', '2021-08-27', '2024-08-26'), ('validation', '2024-08-27', '2025-08-26'),
           ('recent', '2025-08-27', '2026-08-26'), ('full', '2021-08-27', '2026-08-26'))
R22_MANIFESTS = {
    'training': 'aee1b37ad0becc5ade4e66bb8a249b502715febb37b5dc01efb20e6b3510b6c4',
    'validation': 'abd706a2a5eeee260c5cde656980d95fb79c0760c3033dbfed24333cf0db28ef',
    'recent': '747e3ab581e0181a01152858a956c30a53219b5a3958a71e262da6c0cad18840',
    'full': 'c80a0c27b46af61f088ae2fec672a39f0cb98da6b72599a3e5940acd83d39b2f',
}

EVENTS = (('raw_b', 'B_ALL_RAW'), ('confirmed_b', 'B_SIGNAL'),
          ('raw_jf', 'JF_RAW'), ('tradable_jf', 'ICON_JUEFAN'))
EVENT_SCHEMA = {
    **{c: 'string' for c in ('event_type', 'raw_b_state', 'reference_open_date', 'outcome_date',
                             'actual_entry_date', 'actual_entry_kind')},
    **{c: 'float64' for c in ('reference_open', 'ret20_pct', 'mfe20_pct', 'mae20_pct')},
    **{c: 'boolean' for c in ('outcome_complete', 'win', 'interference', 'next_entry_observed',
                              'drives_entry', *COMPONENTS)},
}


def position_states(symbol: str, frame: pd.DataFrame, trades: pd.DataFrame,
                    start: pd.Timestamp, end: pd.Timestamp) -> pd.DataFrame:
    """从已发生的真实OPEN成交还原CLOSE状态，terminal末CLOSE仍作持仓观察。"""
    frame = frame.loc[:end]
    states = signal_states(frame).join(frame[list(COMPONENTS)].add_prefix('raw_')).loc[start:end].copy()
    states['position'] = pd.Series('flat', index=states.index, dtype='string')
    states['flat_origin'] = pd.Series('initial', index=states.index, dtype='string')
    states['flat_since'] = pd.Series(str(states.index[0].date()) if len(states) else pd.NA,
                                     index=states.index, dtype='string')
    states['flat_bar'] = pd.Series(np.arange(1, len(states)+1), index=states.index, dtype='Int64')
    states['held_trade_id'] = pd.Series(pd.NA, index=states.index, dtype='string')
    for col in ('entry_today', 'exit_today', 'terminal_today'):
        states[col] = False
    last = str(states.index[-1].date()) if len(states) else ''
    own = trades[trades.symbol.eq(symbol)].sort_values('entry_date')
    observed = own[own.entry_date.le(last)]
    if observed.entry_date.duplicated().any() or observed.trade_id.duplicated().any():
        raise ValueError('重复原订单不能唯一还原持仓')
    previous_exit = None
    for trade in observed.itertuples():
        entered, exited = pd.Timestamp(trade.entry_date), pd.Timestamp(trade.exit_date)
        terminal = trade.exit_reason == 'terminal'
        if (entered not in states.index or entered == states.index[0] or exited < entered
                or (not terminal and exited == entered) or trade.exit_reason not in ('signal', 'trail', 'terminal')
                or (previous_exit is not None and entered <= previous_exit)):
            raise ValueError('原订单入场/退出边界或持仓重叠')
        i = states.index.get_loc(entered)
        signal = states.iloc[i-1]
        if (trade.entry_signal_date != str(states.index[i-1].date())
                or not (signal.B_SIGNAL or signal.ICON_JUEFAN)
                or bool(trade.entry_b) != bool(signal.B_SIGNAL) or bool(trade.entry_jf) != bool(signal.ICON_JUEFAN)
                or trade.entry_kind != ('B' if signal.B_SIGNAL else 'JF')
                or not np.isclose(trade.entry_open, states.OPEN.iloc[i], rtol=1e-12, atol=1e-12)):
            raise ValueError('原订单入场日期/价格或B/JF来源不一致')
        if signal.B_SIGNAL:
            if (trade.setup_date != signal.resolved_setup_date
                    or any(bool(getattr(trade, c)) != bool(signal['resolved_' + c]) for c in COMPONENTS)):
                raise ValueError('原订单B入场Setup来源不一致')
        elif pd.notna(trade.setup_date) or any(bool(getattr(trade, c)) for c in COMPONENTS):
            raise ValueError('原订单JF入场不应有B来源')
        if trade.exit_date <= last:
            if exited not in states.index or (terminal and exited != states.index[-1]):
                raise ValueError('原订单退出日期或terminal边界不一致')
            j = states.index.get_loc(exited)
            price = states.CLOSE.iloc[j] if terminal else states.OPEN.iloc[j]
            if (trade.hold_bars != j-i+int(terminal)
                    or not np.isclose(trade.exit_price, price, rtol=1e-12, atol=1e-12)):
                raise ValueError('原订单退出价格或持仓根数不一致')
        held = (states.index >= entered) & ((states.index <= exited) if terminal else (states.index < exited))
        states.loc[held, 'position'] = 'held'
        states.loc[held, 'held_trade_id'] = trade.trade_id
        states.loc[held, ['flat_origin', 'flat_since', 'flat_bar']] = pd.NA
        states.loc[entered, 'entry_today'] = True
        if trade.exit_date <= last:
            states.loc[exited, 'terminal_today' if terminal else 'exit_today'] = True
            if not terminal:
                flat = states.index[states.index >= exited]
                states.loc[flat, 'flat_origin'] = 'S' if trade.exit_reason == 'signal' else 'trail'
                states.loc[flat, 'flat_since'] = trade.exit_date
                states.loc[flat, 'flat_bar'] = np.arange(1, len(flat)+1)
        previous_exit = exited
    states['pending_buy'] = states.position.eq('flat') & (states.B_SIGNAL | states.ICON_JUEFAN)
    if len(states) and not np.array_equal(states.pending_buy.iloc[:-1], states.entry_today.iloc[1:]):
        raise ValueError('原订单与真实空仓可交易信号的次OPEN入场不一致')
    return states


def audit_frame(symbol: str, frame: pd.DataFrame, trades: pd.DataFrame,
                start: pd.Timestamp, end: pd.Timestamp, *, source_trusted: bool = False
                ) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    """每种触发用自己的当日时钟；20根结果和后续成交只放事件表，不回填观察状态。"""
    frame = frame.loc[:end]
    states = position_states(symbol, frame, trades, start, end)
    observations = states.reset_index(names='date')
    observations['date'] = observations.date.dt.strftime('%Y-%m-%d').astype('string')
    observations.insert(0, 'symbol', pd.Series(symbol, index=observations.index, dtype='string'))
    observations['source_trusted'] = pd.Series(source_trusted, index=observations.index, dtype='boolean')
    rows = []
    for offset, (date, state) in enumerate(states.iterrows()):
        pos = frame.index.get_loc(date)
        for event, column in EVENTS:
            if not state[column]:
                continue
            path = _forward_path(frame, pos, 20, outcome_end=end)
            complete = bool(np.isfinite(path['return']))
            next_open = pos + 1 < len(frame)
            entered = (bool(state.pending_buy) and offset+1 < len(states)
                       and bool(states.entry_today.iloc[offset+1]))
            kind = ('B' if state.B_SIGNAL else 'JF') if entered else None
            prefix = 'raw_' if event == 'raw_b' else 'resolved_'
            rows.append({**observations.iloc[offset].to_dict(), 'event_type': event,
                         'raw_b_state': ('accepted' if state.B_SETUP else 'suppressed') if event == 'raw_b' else None,
                         **{c: bool(state[prefix+c]) if event in ('raw_b', 'confirmed_b') else False for c in COMPONENTS},
                         'reference_open_date': str(frame.index[pos+1].date()) if next_open else None,
                         'reference_open': float(frame.OPEN.iloc[pos+1]) if next_open else np.nan,
                         'actual_entry_date': str(frame.index[pos+1].date()) if entered else None,
                         'actual_entry_kind': kind, 'next_entry_observed': entered,
                         'drives_entry': (event == 'confirmed_b' and kind == 'B') or (event == 'tradable_jf' and kind == 'JF'),
                         'outcome_complete': complete,
                         'outcome_date': str(frame.index[pos+20].date()) if complete else None,
                         'ret20_pct': path['return']*100, 'mfe20_pct': path['mfe']*100, 'mae20_pct': path['mae']*100,
                         'win': bool(path['return'] > 0) if complete else None,
                         'interference': bool(path['return'] < 0 and path['mae'] <= -.08) if complete else None})
    schema = {**observations.dtypes.to_dict(), **EVENT_SCHEMA}
    events = pd.DataFrame(rows, columns=schema).astype(schema)
    own = trades[trades.symbol.eq(symbol)]
    observed_orders = own[own.entry_date.le(str(states.index[-1].date()))] if len(states) else own.iloc[:0]
    if len(observed_orders) != int(states.entry_today.sum()) or events.drives_entry.sum() != len(observed_orders):
        raise ValueError('原订单与逐事件实际入场数量不一致')
    check = {'symbol': symbol, 'close_rows': len(states), 'flat_closes': int(states.position.eq('flat').sum()),
             'held_closes': int(states.position.eq('held').sum()), 'events': len(events),
             'actual_entries': len(observed_orders), 'actual_open_exits': int(states.exit_today.sum()),
             'terminal_exits': int(states.terminal_today.sum()),
             'pending_at_cutoff': bool(states.pending_buy.iloc[-1]) if len(states) else False,
             'reconciled': True}
    return observations, events, check


def summarize_events(events: pd.DataFrame) -> pd.DataFrame:
    """各事件时钟独立分母，完整20根才进入方向/干扰率；组件和同日事件有重叠。"""
    rows = []
    def add(position, event, group_by, group, subset):
        complete = subset[subset.outcome_complete]
        n = len(complete)
        rows.append({'position': position, 'event_type': event, 'group_by': group_by, 'group': group,
                     'events': len(subset), 'complete': n, 'incomplete': len(subset)-n,
                     'symbols': subset.symbol.nunique(), 'wins': int(complete.win.sum()),
                     'interference': int(complete.interference.sum()),
                     'win_rate_pct': complete.win.mean()*100 if n else np.nan,
                     'interference_rate_pct': complete.interference.mean()*100 if n else np.nan,
                     'mean_ret20_pct': complete.ret20_pct.mean(), 'median_ret20_pct': complete.ret20_pct.median(),
                     'mean_mfe20_pct': complete.mfe20_pct.mean(), 'mean_mae20_pct': complete.mae20_pct.mean(),
                     'actual_entries': int(subset.next_entry_observed.sum()), 'direct_entries': int(subset.drives_entry.sum()),
                     'raw_b_jf_overlap': int((subset.B_ALL_RAW & subset.JF_RAW).sum()),
                     'tradable_b_jf_overlap': int((subset.B_SIGNAL & subset.ICON_JUEFAN).sum()),
                     'suppressed_b_with_buy': int((subset.raw_b_suppressed & (subset.B_SIGNAL | subset.ICON_JUEFAN)).sum()),
                     'pending_setup_rows': int(subset.pending_setup_date.notna().sum()),
                     'confirmed_setup_rows': int(subset.resolved_setup_status.eq('confirmed').sum())})
    for position in ('all', 'flat', 'held'):
        selected = events if position == 'all' else events[events.position.eq(position)]
        for event, _ in EVENTS:
            subset = selected[selected.event_type.eq(event)]
            add(position, event, 'all', 'all', subset)
            for symbol in CORE:
                add(position, event, 'symbol', symbol, subset[subset.symbol.eq(symbol)])
            for quality in (False, True):
                add(position, event, 'source_trusted', str(quality).lower(), subset[subset.source_trusted.eq(quality)])
            if position == 'flat':
                for origin in ('initial', 'S', 'trail'):
                    add(position, event, 'flat_origin', origin, subset[subset.flat_origin.eq(origin)])
            if event == 'raw_b':
                for status in ('accepted', 'suppressed'):
                    add(position, event, 'raw_b_state', status, subset[subset.raw_b_state.eq(status)])
            if event in ('raw_b', 'confirmed_b'):
                for component in COMPONENTS:
                    add(position, event, 'component', component, subset[subset[component]])
                component_count = subset[list(COMPONENTS)].sum(axis=1)
                add(position, event, 'component', 'multiple', subset[component_count.gt(1)])
                add(position, event, 'component', 'none', subset[component_count.eq(0)])
    return pd.DataFrame(rows).astype({c: 'string' for c in ('position', 'event_type', 'group_by', 'group')})


def run_diagnostic(snapshot: Path, r22: Path, output: Path, *, window: str = 'training') -> dict:
    """每次固定一窗原v5输入；捕获全部依赖字节，拒绝篡改和非空输出。"""
    selected = next((list(spec) for spec in WINDOWS if spec[0] == window), None)
    if selected is None:
        raise ValueError('仅允许r26固定窗口')
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
    names = ('gcn/backtest/signal_research_r26.py', 'gcn/backtest/signal_research_r25.py',
             'gcn/backtest/historical_research.py', 'gcn/backtest/engine.py', *native)
    sources = {name: capture(root / name) for name in names}
    protocol = capture(root / 'reports/gcn-historical-r26-20260905/protocol.md')
    if digest(protocol) != '0e834ee45377bc81372d999b7a829a939afed58a8300bcc85b8a458a261a2f57':
        raise ValueError('r26冻结协议变化')
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
        if digest(raw) != expected:
            raise ValueError(f'r22冻结源码变化：{name}')
        if name in native and sources[name] != raw:
            raise ValueError(f'原生信号源码与r22不一致：{name}')
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
    if quality != parent['source_quality'] or environment != parent['environment']:
        raise ValueError('r22来源质量或运行环境不一致')
    trades = pd.read_csv(io.BytesIO(inputs['trades.csv']), float_precision='round_trip')
    prior_checks = pd.read_csv(io.BytesIO(inputs['reconciliation.csv'])).set_index('symbol')
    if (not trades.window.eq(window).all() or not set(trades.symbol).issubset(CORE)
            or set(prior_checks.index) != set(CORE)):
        raise ValueError('r22原订单窗口或标的集合不一致')
    start, end = pd.Timestamp(selected[1]), pd.Timestamp(selected[2])
    observation_parts, event_parts, checks = [], [], []
    for symbol in CORE:
        frame = compute_ehopt10(frames[symbol].loc[:end], version='v5', diagnostics=True)
        obs, events, check = audit_frame(symbol, frame, trades, start, end, source_trusted=quality[symbol])
        own = trades[trades.symbol.eq(symbol)]
        prior = prior_checks.loc[symbol]
        if (check['actual_entries'] != prior.trades or check['held_closes'] != own.hold_bars.sum()
                or int(obs.B_SIGNAL.sum()) != prior.b_signals or int(obs.ICON_JUEFAN.sum()) != prior.jf_signals
                or int(obs.S_SIGNAL.sum()) != prior.s_signals):
            raise ValueError(f'{symbol}: r22原订单、持仓根数或原生信号对账失败')
        observation_parts.append(obs); event_parts.append(events); checks.append(check)
    observations = pd.concat(observation_parts, ignore_index=True)
    events = pd.concat(event_parts, ignore_index=True)
    tables = {'observations': observations, 'events': events, 'summary': summarize_events(events),
              'reconciliation': pd.DataFrame(checks)}
    decision = {'research_version': 'gcn-historical-r26', 'stage': 'diagnostic_only', 'recommended': 'v5',
                'production_changed': False, 'window': selected, 'core': CORE,
                'input': 'frozen original r22 v5 orders; no order simulation or new candidate returns',
                'state': 'full-history signals; initial/actual OPEN exits are flat; terminal CLOSE remains held',
                'event_clocks': 'raw B, confirmed B, raw JF and tradable JF have separate signal-date denominators',
                'labels': 'next OPEN reference to 20th CLOSE, gross and posthoc, complete within window only',
                'overlap': 'raw suppression may coincide with an old Setup confirmation; B wins B/JF entry collisions',
                'stopping': 'diagnostics are not a promoted strategy; no cooldown search or later r24 evaluation'}
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
    for prefix, files in (('source_snapshot', sources), ('input_snapshot', inputs),
                          ('input_source_snapshot', input_sources), ('parent_snapshot', parent_files)):
        for name, raw in files.items():
            target = output / prefix / name
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(raw)
    manifest = {'research_version': 'gcn-historical-r26', 'window': selected,
                'parent_manifest_sha256': SNAPSHOT_SHA, 'input_manifest_sha256': R22_MANIFESTS[window],
                'source_quality': quality, 'input_environment': parent['environment'],
                'protocol_sha256': digest(protocol), 'environment': environment,
                **{key: {name: digest(raw) for name, raw in files.items()} for key, files in
                   (('algorithm_sources', sources), ('input_files', inputs), ('input_algorithm_sources', input_sources),
                    ('parent_files', parent_files))},
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
