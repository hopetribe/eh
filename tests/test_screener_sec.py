"""SEC statement normalization, source routing and all-strategy regression."""
import json
from unittest.mock import patch

import numpy as np
import pandas as pd
import pytest

from gcn.screener import sec, fundamentals, engine
from gcn.screener.strategies import STRATEGIES


def fact(value, end="2025-12-31", start="2025-01-01", filed="2026-02-01", **extra):
    result = dict(val=value, end=end, filed=filed, accn="latest", form="10-K", **extra)
    if start is not None:
        result["start"] = start
    return result


def test_annual_period_units_restatements_and_future_filings():
    facts = {"Revenue": {"units": {"USD": [fact(100),
        fact(110, filed="2026-03-01"), fact(900, filed="2027-01-01"),
        fact(20, start="2025-10-01"), fact(float("inf"))], "EUR": [fact(888)]}}}
    result = sec._facts_series(facts, ("Revenue",), as_of="2026-09-20")
    assert result.to_list() == [110]
    assert result.index[0] == pd.Timestamp("2025-12-31")


def test_tag_migration_and_comparable_eps_do_not_mix_split_bases():
    old = fact(100, "2024-12-31", "2024-01-01", "2025-02-01")
    old["accn"] = "older"
    facts = {"Old": {"units": {"USD": [old]}},
             "New": {"units": {"USD": [fact(120)]}}}
    assert sec._facts_series(facts, ("New", "Old")).to_list() == [120, 100]
    facts = {"EPS": {"units": {"USD/shares": [old, fact(5),
        fact(4, "2024-12-31", "2024-01-01"),
        {**old, "end": "2023-12-31", "start": "2023-01-01"}]}}}
    assert sec._facts_series(facts, ("EPS",), "USD/shares", comparable=True).to_list() == [5, 4]


def payload():
    facts = {}
    values = {"NetIncomeLoss": 20, "RevenueFromContractWithCustomerExcludingAssessedTax": 100,
              "GrossProfit": 60, "EarningsPerShareDiluted": 2,
              "StockholdersEquity": 100, "Assets": 200, "Liabilities": 100,
              "AssetsCurrent": 80, "LiabilitiesCurrent": 30, "LongTermDebtNoncurrent": 20,
              "Goodwill": 10, "CommonStockSharesOutstanding": 10,
              "CashAndCashEquivalentsAtCarryingValue": 30, "InventoryNet": 10,
              "NetCashProvidedByUsedInOperatingActivities": 30,
              "PaymentsToAcquirePropertyPlantAndEquipment": 5,
              "PaymentsOfDividendsCommonStock": 8, "CommonStockDividendsPerShareDeclared": 0.8}
    instant = {tag for tags in sec.BALANCE.values() for tag in tags}
    for key, value in values.items():
        unit = "USD/shares" if "PerShare" in key else "shares" if key == "CommonStockSharesOutstanding" else "USD"
        facts[key] = {"units": {unit: [fact(value * (1 - i * 0.1), f"{2025-i}-12-31",
            None if key in instant else f"{2025-i}-01-01") for i in range(5)]}}
    return {"cik": 123, "entityName": "TEST COMPANY", "facts": {"us-gaap": facts}}


def test_statement_normalization_and_capex_sign():
    data = sec.parse_companyfacts(payload(), "TEST", as_of="2026-09-20")
    assert data["metadata"]["annual_years"] == 5
    assert data["income"].iloc[0, 0] == 20
    assert data["cashflow"].loc["CapitalExpenditure"].iloc[0] == -5
    assert data["cashflow"].loc["CashDividendsPaid"].iloc[0] == -8
    with pytest.raises(ValueError, match="550"):
        sec.parse_companyfacts(payload(), "TEST", as_of="2028-01-01")


def test_all_eight_strategies_use_sec_when_yahoo_empty_and_strict_json():
    from types import SimpleNamespace
    data = sec.parse_companyfacts(payload(), "TEST", as_of="2026-09-20")
    quote = {"rows": [["2026-09-18", 10, 10, 10, 10, 100]], "source": "tradingview"}
    with patch.object(sec, "fetch_statements", return_value=data), \
         patch.object(fundamentals, "fetch_quote", return_value=quote), \
         patch("yfinance.Ticker", return_value=SimpleNamespace(info={},
               get_income_stmt=pd.DataFrame, get_balance_sheet=pd.DataFrame, get_cash_flow=pd.DataFrame)):
        metrics = fundamentals.compute_metrics("TEST")
        assert metrics["market_cap"] == 100
        assert metrics["trailing_pe"] == 5
        assert metrics["fcf_to_net_income"] == 1.25
        assert metrics["static_div_yield"] == 0.08
        assert metrics["div_yield_yearly"] == []  # unavailable, not fabricated zeroes
        assert np.isnan(metrics["pe_avg_3y"])
        for strategy in STRATEGIES:
            result = engine.evaluate_symbol("TEST", strategy)
            json.dumps(result, allow_nan=False)
            assert result["data_source"]["source"] == "SEC EDGAR"
            assert result["market_cap_cny"] == 720
            assert any(c["value"] is not None for c in result["conditions"])
            assert all("TTM" not in c["text"] for c in result["conditions"])


def test_yahoo_snapshot_failure_does_not_block_statements_and_hk_skips_sec():
    data = sec.parse_companyfacts(payload(), "TEST", as_of="2026-09-20")
    class Ticker:
        @property
        def info(self):
            raise RuntimeError("429")
        dividends = pd.Series(dtype=float, index=pd.DatetimeIndex([]))
        get_income_stmt = lambda self: data["income"]
        get_balance_sheet = lambda self: data["balance"]
        get_cash_flow = lambda self: data["cashflow"]
    with patch.object(sec, "fetch_statements") as fetch, \
         patch("yfinance.Ticker", return_value=Ticker()), \
         patch.object(fundamentals, "fetch_quote", return_value={"rows": []}):
        result = fundamentals.compute_metrics("0700.HK")
        fetch.assert_not_called()
        assert result["fcf_to_net_income"] == 1.25
        assert result["div_yield_yearly"] == []
        assert result["currency"] == "HKD"
        result_us = fundamentals.compute_metrics("MSFT")
        assert result_us["_source"]["source"] == "Yahoo"
        fetch.assert_not_called()


def test_all_sources_empty_is_fetch_error_not_incomplete_financials():
    from types import SimpleNamespace
    empty = lambda: pd.DataFrame()
    ticker = SimpleNamespace(info={}, get_income_stmt=empty, get_balance_sheet=empty, get_cash_flow=empty)
    with patch.object(sec, "fetch_statements", side_effect=RuntimeError("offline")), \
         patch("yfinance.Ticker", return_value=ticker):
        result = engine.evaluate_symbol("MSFT", "neff")
        assert result["data_status"] == "error"
        assert "财报源未返回数据" in result["note"]


def test_sec_cache_and_failed_response_not_cached(tmp_path, monkeypatch):
    monkeypatch.setenv("SEC_USER_AGENT", "test contact@example.com")
    monkeypatch.setattr(sec, "CACHE_DIR", tmp_path)
    with patch.object(sec.requests, "get") as get:
        get.return_value.json.return_value = payload()
        assert sec._json("https://data.sec.gov/test", "facts_test")["cik"] == 123
        assert sec._json("https://data.sec.gov/test", "facts_test")["cik"] == 123
        assert get.call_count == 1
        get.return_value.raise_for_status.side_effect = RuntimeError("429")
        with pytest.raises(RuntimeError):
            sec._json("https://data.sec.gov/test", "facts_error")
        assert not (tmp_path / "facts_error.json").exists()


def test_sec_requires_declared_contact(monkeypatch):
    monkeypatch.delenv("SEC_USER_AGENT", raising=False)
    with patch.object(sec.requests, "get") as get, pytest.raises(ValueError, match="SEC_USER_AGENT"):
        sec._json("https://data.sec.gov/test", "test")
    get.assert_not_called()


def test_sec_missing_latest_year_is_not_backfilled_or_passed():
    from types import SimpleNamespace
    data = sec.parse_companyfacts(payload(), "TEST", as_of="2026-09-20")
    data["balance"].iloc[0, 0] = np.nan
    data["balance"].loc["OrdinarySharesNumber", data["balance"].columns[0]] = np.nan
    with patch.object(sec, "fetch_statements", return_value=data), \
         patch.object(fundamentals, "fetch_quote", return_value={"rows": [["2026-09-18", 10, 10, 10, 10, 1]]}), \
         patch("yfinance.Ticker", return_value=SimpleNamespace(info={},
               get_income_stmt=pd.DataFrame, get_balance_sheet=pd.DataFrame, get_cash_flow=pd.DataFrame)):
        m = fundamentals.compute_metrics("TEST")
        assert np.isnan(m["market_cap"])
        assert np.isnan(m["roe_yearly"][0])
        assert np.isnan(m["roe_avg_3y"])
        condition = {"field": "roe_yearly", "op": "yearly_gt", "need": 3, "value": 0.1, "text": "ROE"}
        assert not engine._eval_condition(condition, m)["passed"]


def test_sec_stale_price_cannot_produce_current_valuation():
    from types import SimpleNamespace
    data = sec.parse_companyfacts(payload(), "TEST", as_of="2026-09-20")
    with patch.object(sec, "fetch_statements", return_value=data), \
         patch.object(fundamentals, "fetch_quote", return_value={"stale": True,
             "rows": [["2026-09-18", 10, 10, 10, 10, 1]]}), \
         patch("yfinance.Ticker", return_value=SimpleNamespace(info={},
               get_income_stmt=pd.DataFrame, get_balance_sheet=pd.DataFrame, get_cash_flow=pd.DataFrame)):
        m = fundamentals.compute_metrics("TEST")
        assert m["market_cap"] is None
        assert m["trailing_pe"] is None
        assert m["fcf_to_net_income"] == 1.25
