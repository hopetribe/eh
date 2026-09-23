"""Offline regression cases from the September code review."""
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from gcn.data import service
from gcn.backtest.engine import run_backtest
from gcn.radar import emailer, engine as radar
from gcn.radar.config import normalize_config
from gcn.radar.universe import UNIVERSE_CACHE_SCHEMA
from gcn.screener import engine, fundamentals
from gcn.screener.sec import parse_companyfacts
from gcn.server.app import _dumps_json
from tests.test_server import _serve_request
from tests.test_screener_sec import payload


def frame(dates, prices):
    out = pd.DataFrame({col: prices for col in ('open', 'high', 'low', 'close', 'volume')},
                       index=pd.to_datetime(dates))
    out.attrs['adjustment'] = 'split-adjusted'
    return out


def test_remote_admin_requires_token_for_reads_and_writes(tmp_path, monkeypatch):
    monkeypatch.setattr(emailer, 'DATA_DIR', tmp_path)
    monkeypatch.setenv('GCN_ADMIN_TOKEN', 'test-secret')
    headers = {'Host': 'example.test', 'Content-Type': 'application/json'}
    kwargs = dict(bound_host='172.17.0.1', allowed_hosts={'example.test'})
    for method, path in [('GET', '/api/radar/email'), ('POST', '/api/radar/email'),
                         ('POST', '/api/radar/scan')]:
        status, _, _ = _serve_request(method, path, b'{}', headers=headers, **kwargs)
        assert status == 401
    assert not list(tmp_path.iterdir())
    headers['Authorization'] = 'Bearer test-secret'
    status, _, raw = _serve_request('POST', '/api/radar/email',
        json.dumps({'action': 'add', 'email': 'review@example.com'}).encode(), headers=headers, **kwargs)
    assert status == 200 and 'review@example.com' in json.loads(raw)['recipients']
    monkeypatch.delenv('GCN_ADMIN_TOKEN')
    assert _serve_request('GET', '/api/radar/email', headers=headers, **kwargs)[0] == 401


def test_cache_identity_keeps_market_and_aliases():
    assert service._cache_path('SH.000001', '1d') != service._cache_path('SZ.000001', '1d')
    assert service._cache_path('00700', '1d') == service._cache_path('0700.HK', '1d')
    assert service._cache_path('US.AAPL', '1d') == service._cache_path('AAPL', '1d')


def test_cache_requires_observation_after_session_close():
    for interval in ('1d', '1wk'):
        cached = frame(['2026-09-21'], [100])
        cached.attrs['observed_at'] = pd.Timestamp('2026-09-21 12:00', tz='America/New_York').timestamp()
        now = pd.Timestamp('2026-09-21 17:00', tz='America/New_York')
        assert not service._cache_is_fresh(cached, interval, None, 'AAPL', now=now)
        cached.attrs['observed_at'] = now.timestamp()
        assert service._cache_is_fresh(cached, interval, None, 'AAPL', now=now)
        if interval == '1wk':
            assert not service._cache_is_fresh(cached, interval, None, 'AAPL',
                now=pd.Timestamp('2026-09-25 17:00', tz='America/New_York'))


def test_changed_adjustment_basis_never_splices_old_prefix():
    old = frame(['2026-09-17', '2026-09-18', '2026-09-21'], [100, 100, 100])
    fresh = frame(['2026-09-18', '2026-09-21', '2026-09-22'], [50, 50, 50])
    assert service._merge_market_data(old, fresh).close.tolist() == [50, 50, 50]
    fresh.loc[:, :] = 100
    assert len(service._merge_market_data(old, fresh)) == 4


def test_successful_but_old_quote_is_marked_stale(tmp_path, monkeypatch):
    monkeypatch.setattr(service, 'DATA_DIR', tmp_path)
    monkeypatch.setattr(service, '_fetch_tradingview', lambda *a: frame(['2020-01-02'], [100]))
    result = service.fetch_quote('AAPL')
    assert result['stale'] and result['refresh_failed']


def test_break_even_trade_does_not_break_report_json():
    res = pd.DataFrame({'OPEN': [100, 100, 110, 110, 110, 110],
        'CLOSE': [100, 100, 110, 110, 110, 110],
        'B_SIGNAL': [True, False, False, True, False, False],
        'S_SIGNAL': [False, True, False, False, True, False]})
    report = run_backtest(res, cost=0)
    assert report['strategies'][0]['pf'] is None
    _dumps_json(report)


def test_snapshot_ages_signals_without_mutating_stored_cache(monkeypatch):
    config = normalize_config()
    cache = {'config': config, 'universe_schema': UNIVERSE_CACHE_SCHEMA,
        'generated_at': 0, 'results': [{'code': 'TEST', 'signals': [
            {'date': '2020-01-02', 'days_ago': 0}]}]}
    monkeypatch.setattr(radar, 'load_cache', lambda market: cache)
    snap = radar.RadarService().snapshot(['us'], config)
    assert snap['markets']['us']['cache']['results'][0]['signals'][0]['days_ago'] > 10
    assert cache['results'][0]['signals'][0]['days_ago'] == 0
    assert not emailer._market_rows(snap)


def metrics(monkeypatch, info=None, edit=None):
    data = parse_companyfacts(payload(), 'TEST', as_of='2026-09-20')
    if edit:
        edit(data)
    ticker = SimpleNamespace(info=info or {'currency': 'USD', 'marketCap': 1e12},
        get_income_stmt=lambda: data['income'], get_balance_sheet=lambda: data['balance'],
        get_cash_flow=lambda: data['cashflow'],
        dividends=pd.Series(dtype=float, index=pd.DatetimeIndex([])))
    monkeypatch.setattr('yfinance.Ticker', lambda symbol: ticker)
    monkeypatch.setattr(fundamentals, 'fetch_quote', lambda *a, **k: {
        'rows': [['2026-09-22', 10, 10, 10, 10, 100]]})
    return fundamentals.compute_metrics('0700.HK')


def test_yahoo_missing_latest_report_is_not_backfilled(monkeypatch):
    def edit(data):
        data['income'].loc['NetIncome', data['income'].columns[0]] = np.nan
    m = metrics(monkeypatch, edit=edit)
    assert np.isnan(m['roe_yearly'][0])
    assert np.isnan(m['roe_avg_3y'])
    assert not engine._eval_condition({'field': 'roe_yearly', 'op': 'yearly_gt',
        'need': 3, 'value': .1, 'text': 'ROE'}, m)['passed']


def test_missing_whole_year_is_preserved_in_annual_series():
    stmt = pd.DataFrame([[20, 30]], index=['NetIncome'],
                        columns=pd.to_datetime(['2025-12-31', '2023-12-31']))
    row = fundamentals._series_all(stmt, 'NetIncome')
    assert row.index.year.tolist() == [2025, 2024, 2023]
    assert np.isnan(row.iloc[1])


def test_mixed_currencies_do_not_generate_unconverted_valuations(monkeypatch):
    m = metrics(monkeypatch, {'currency': 'HKD', 'financialCurrency': 'CNY', 'marketCap': 1000})
    assert m['trailing_pe'] is None and m['pb_mrq'] is None
    assert np.isnan(m['net_cash_to_mktcap']) and np.isnan(m['roic'])
    assert m['market_cap'] == 1000


def test_debt_uses_total_or_complete_short_and_long_components(monkeypatch):
    def edit(data):
        for key, value in [('LongTermDebt', 10), ('CurrentDebt', 90),
                           ('TotalDebt', 100), ('CashAndCashEquivalents', 50)]:
            data['balance'].loc[key] = value
    m = metrics(monkeypatch, edit=edit)
    assert m['total_debt'] == 100 and m['net_cash'] == -50
    def components(data):
        edit(data)
        data['balance'].drop(index='TotalDebt', inplace=True)
    assert metrics(monkeypatch, edit=components)['total_debt'] == 100
    assert metrics(monkeypatch)['total_debt'] is None


def test_cagr_short_coverage_disclosed_for_yahoo(monkeypatch):
    from tests.test_screener import _passing_metrics
    from gcn.screener.strategies import STRATEGIES
    m = _passing_metrics(STRATEGIES['growth'])
    m.update(_source={'source': 'Yahoo'}, _coverage={'eps_cagr_10y': '2年(数据上限)'})
    monkeypatch.setattr(fundamentals, 'compute_metrics', lambda *a, **k: m)
    result = engine.evaluate_symbol('TEST', 'growth')
    assert result['data_status'] == 'incomplete'
    assert '2年' in next(c for c in result['conditions'] if '10年' in c['text'])['note']
    m['_coverage'] = {'eps_cagr_10y': '10年(数据上限)', 'eps_cagr_10y_years': 10}
    assert engine.evaluate_symbol('TEST', 'growth')['data_status'] == 'complete'


def test_bad_symbol_isolated_from_rest_of_screen(monkeypatch):
    monkeypatch.setattr(engine.time, 'sleep', lambda *a: None)
    monkeypatch.setattr(fundamentals, 'compute_metrics', lambda *a, **k: {})
    results = engine.run_screen(['AAPL', 'INVALID/SYM', 'MSFT'], 'graham', log=False)
    assert len(results) == 3
    assert next(r for r in results if r['symbol'] == 'INVALID/SYM')['data_status'] == 'error'


def test_runner_collects_new_modules_with_pytest(monkeypatch):
    from tests import run_all
    captured = []
    monkeypatch.setattr('pytest.main', lambda args: captured.extend(args) or 7)
    monkeypatch.setattr('sys.argv', ['run_all.py'])
    assert run_all.main() == 7
    assert Path(captured[0]).name == 'tests'


def test_live_research_still_rejects_source_drift(tmp_path, monkeypatch):
    from gcn.backtest.signal_research_r25 import run_diagnostic
    root = Path(__file__).resolve().parents[1]
    original_read = Path.read_bytes
    target = root / 'gcn/backtest/historical_research.py'
    monkeypatch.setattr(Path, 'read_bytes', lambda path:
        original_read(path) + (b'\n# source drift' if path == target else b''))
    with pytest.raises(ValueError, match='原生信号源码与r24不一致'):
        run_diagnostic(root / 'reports/signal-audit-v5-review-20260904',
                       root / 'reports/gcn-historical-r24-20260905', tmp_path)
    assert not list(tmp_path.iterdir())
