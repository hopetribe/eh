"""SEC Company Facts adapter for US-GAAP annual statements (not a backtest feed).

Only published, full-year, entity-wide USD facts are used. Quarterly/YTD values
must never be mistaken for annual values. Latest restatements win per period.
Per-share histories are restricted to one filing's comparable periods, avoiding
splicing pre/post-split EPS from unrelated filings into a ten-year growth rate.
"""
from __future__ import annotations

import json
import math
import os
import re
import tempfile
import threading
import time
from datetime import date
from pathlib import Path

import pandas as pd
import requests

from gcn.data.service import DATA_DIR

CACHE_DIR = DATA_DIR / "fundamentals_sec"
_lock = threading.Lock()
_last_request = 0.0

# Output names match the existing statement adapter; alternatives are resolved
# per period, not by choosing the first tag with any (possibly obsolete) data.
INCOME = {
    "NetIncome": ("NetIncomeLoss", "ProfitLoss"),
    "TotalRevenue": ("RevenueFromContractWithCustomerExcludingAssessedTax",
                     "RevenueFromContractWithCustomerIncludingAssessedTax",
                     "Revenues", "SalesRevenueNet"),
    "GrossProfit": ("GrossProfit",),
    "DilutedEPS": ("EarningsPerShareDiluted",),
}
BALANCE = {
    "StockholdersEquity": ("StockholdersEquity",),
    "TotalAssets": ("Assets",),
    "TotalLiabilitiesNetMinorityInterest": ("Liabilities",),
    "CurrentAssets": ("AssetsCurrent",),
    "CurrentLiabilities": ("LiabilitiesCurrent",),
    "LongTermDebt": ("LongTermDebtNoncurrent", "LongTermDebt"),
    "Goodwill": ("Goodwill",),
    "OrdinarySharesNumber": ("CommonStockSharesOutstanding",),
    "CashAndCashEquivalents": ("CashAndCashEquivalentsAtCarryingValue",),
    "Inventory": ("InventoryNet",),
}
CASHFLOW = {
    "OperatingCashFlow": ("NetCashProvidedByUsedInOperatingActivities",),
    "CapitalExpenditure": ("PaymentsToAcquirePropertyPlantAndEquipment",),
    "CashDividendsPaid": ("PaymentsOfDividendsCommonStock", "PaymentsOfOrdinaryDividends", "PaymentsOfDividends"),
}


def _json(url: str, key: str) -> dict:
    """24-hour successful-response cache; no stale or failed responses cached."""
    global _last_request
    user_agent = os.environ.get("SEC_USER_AGENT", "").strip()
    if "@" not in user_agent:
        raise ValueError("请配置 SEC_USER_AGENT（应用名称及联系邮箱）")
    target = CACHE_DIR / (key + ".json")
    with _lock:  # coalesce concurrent callers; <=5 requests/s for this process
        try:
            if 0 <= time.time() - target.stat().st_mtime < 86400:
                cached = json.loads(target.read_text())
                if isinstance(cached, dict) and cached:
                    return cached
        except (OSError, ValueError):
            pass
        time.sleep(max(0, 0.2 - (time.monotonic() - _last_request)))
        _last_request = time.monotonic()
        response = requests.get(url, timeout=(5, 25), headers={
            "User-Agent": user_agent,
            "Accept-Encoding": "gzip, deflate",
        })
        response.raise_for_status()
        payload = response.json()
        if not isinstance(payload, dict) or not payload:
            raise ValueError("SEC returned an empty/invalid response")
        if key.startswith("facts_") and not payload.get("facts", {}).get("us-gaap"):
            raise ValueError("SEC 暂无支持的 US-GAAP 财报（不将 IFRS 混入 USD）")
        CACHE_DIR.mkdir(parents=True, exist_ok=True)
        tmp = None
        try:
            with tempfile.NamedTemporaryFile(mode="w", dir=CACHE_DIR, delete=False) as f:
                tmp = Path(f.name)
                json.dump(payload, f, allow_nan=False)
            os.replace(tmp, target)
        finally:
            if tmp is not None:
                tmp.unlink(missing_ok=True)
        return payload


def _facts_series(facts: dict, tags: tuple, unit="USD", *, instant=False,
                  as_of=None, comparable=False) -> pd.Series:
    cutoff = as_of or date.today().isoformat()
    candidates = []
    for priority, tag in enumerate(tags):
        for fact in facts.get(tag, {}).get("units", {}).get(unit, []):
            try:
                end = date.fromisoformat(fact["end"])
                filed = date.fromisoformat(fact["filed"])
                value = fact["val"]
                if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
                    continue
                if filed.isoformat() > cutoff or end.isoformat() > cutoff:
                    continue
                if fact.get("form") not in ("10-K", "10-K/A"):
                    continue
                if not instant:
                    duration = (end - date.fromisoformat(fact["start"])).days
                    if not 330 <= duration <= 380:
                        continue
                candidates.append((end, filed, -priority, fact.get("accn", ""), float(value)))
            except (KeyError, TypeError, ValueError):
                continue
    if comparable and candidates:
        # Use the filing containing the most recent annual observation.
        accession = max(candidates, key=lambda row: row[:3])[3]
        candidates = [row for row in candidates if row[3] == accession]
    by_end = {}
    for end, filed, priority, accession, value in sorted(candidates, key=lambda row: row[:3]):
        by_end[end] = value
    return pd.Series({pd.Timestamp(k): by_end[k] for k in sorted(by_end, reverse=True)}, dtype=float)


def parse_companyfacts(payload: dict, symbol: str, *, as_of=None) -> dict:
    facts = payload.get("facts", {}).get("us-gaap", {})
    inc = {key: _facts_series(facts, tags, unit="USD/shares" if key == "DilutedEPS" else "USD",
                              comparable=key == "DilutedEPS", as_of=as_of)
           for key, tags in INCOME.items()}
    # Anchor balance sheet instants to actual annual income statement ends.
    ends = inc["NetIncome"].index.union(inc["TotalRevenue"].index).sort_values(ascending=False)[:11]
    if ends.empty:
        raise ValueError("SEC 缺少可用的 USD 年报利润表")
    cutoff = pd.Timestamp(as_of or date.today())
    if (cutoff - ends[0]).days > 550:
        raise ValueError("SEC 最新年报超过 550 天，拒绝作为当前基本面")
    bs = {key: _facts_series(facts, tags, unit="shares" if key == "OrdinarySharesNumber" else "USD",
                             instant=True, as_of=as_of).reindex(ends)
          for key, tags in BALANCE.items()}
    cf = {key: _facts_series(facts, tags, as_of=as_of).reindex(ends)
          for key, tags in CASHFLOW.items()}
    cf["CapitalExpenditure"] = -cf["CapitalExpenditure"].abs()
    cf["CashDividendsPaid"] = -cf["CashDividendsPaid"].abs()
    dps = _facts_series(facts, ("CommonStockDividendsPerShareCashPaid",
                               "CommonStockDividendsPerShareDeclared"),
                        "USD/shares", as_of=as_of, comparable=True)
    info = {"currency": "USD", "shortName": payload.get("entityName") or symbol}
    metadata = {"source": "SEC EDGAR", "annual_period": ends[0].date().isoformat(),
                "annual_years": len(ends), "eps_years": len(inc["DilutedEPS"]),
                "valuation_basis": "年报估算，非 TTM；市值=年报股本×行情价，未包含年报后股本变化",
                "limitations": "历史股息率、历史 PE 与双击乘数未验证复权一致性，保持缺失；不代表不分红"}
    def frame(rows):
        return pd.DataFrame(rows).T.reindex(columns=ends)
    # No forward-fill: the latest year's absent fields stay absent.
    return {"info": info, "income": frame(inc), "balance": frame(bs),
            "cashflow": frame(cf), "dividends_per_share": dps.reindex(ends),
            "metadata": metadata}


def fetch_statements(symbol: str) -> dict:
    if not re.fullmatch(r"[A-Z][A-Z0-9.-]{0,14}", symbol) or symbol.endswith((".HK", ".SS", ".SZ")):
        raise ValueError("SEC 仅适用于已登记的美股代码")
    tickers = _json("https://www.sec.gov/files/company_tickers.json", "tickers")
    match = next((row for row in tickers.values() if isinstance(row, dict)
                  and str(row.get("ticker", "")).replace(".", "-") == symbol.replace(".", "-")), None)
    if match is None:
        raise ValueError(f"SEC 未登记股票代码 {symbol}")
    cik = f"{int(match['cik_str']):010d}"
    payload = _json(f"https://data.sec.gov/api/xbrl/companyfacts/CIK{cik}.json", f"facts_{cik}")
    if int(payload.get("cik", -1)) != int(cik):
        raise ValueError("SEC 公司标识与股票代码不匹配")
    return parse_companyfacts(payload, symbol)
