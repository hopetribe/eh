# -*- coding: utf-8 -*-
"""基本面指标适配层: 行情源基本面 + 三大报表 -> 选股结构化字段。

稳定性设计: yfinance 的 info 快照常被限流返回空, 因此所有快照字段均有
"报表 + 行情" 回填路径 (价格x股本=市值, 价格/EPS=PE 等), 保证限流时选股
依旧可用。数据不足的字段记为 None 并在 _coverage 标注实际覆盖年数。
"""
from __future__ import annotations

import time
import warnings

import numpy as np
import pandas as pd

from gcn.data.service import fetch_quote

warnings.filterwarnings("ignore")

K_NET_INCOME = ("NetIncome", "Net Income")
K_REVENUE = ("TotalRevenue", "Total Revenue")
K_EPS = ("DilutedEPS", "Diluted EPS", "Basic EPS")
K_EQUITY = ("StockholdersEquity", "Common Stock Equity", "Total Stockholder Equity")
K_GOODWILL = ("Goodwill", "Goodwill And Other Intangible Assets")
K_ASSETS = ("TotalAssets", "Total Assets")
K_LIAB = ("TotalLiabilitiesNetMinorityInterest", "Total Liabilities Net Minority Interest")
K_CA = ("CurrentAssets", "Current Assets")
K_CL = ("CurrentLiabilities", "Current Liabilities")
K_LTD = ("LongTermDebt", "Long Term Debt")
K_OCF = ("OperatingCashFlow", "Total Cash From Operating Activities")
K_CAPEX = ("CapitalExpenditure", "Capital Expenditures")
K_DIVPAID = ("CashDividendsPaid", "Cash Dividends Paid", "Common Stock Dividend Paid")
K_SHARES = ("OrdinarySharesNumber", "Share Issued")


def _div(a, b):
    """标量安全除法: 任一侧缺失/为零返回 NaN。"""
    try:
        if a is None or b is None or not np.isfinite(a) or not np.isfinite(b) or b == 0:
            return np.nan
        return float(a) / float(b)
    except (TypeError, ValueError):
        return np.nan


def _get(stmt: pd.DataFrame, *keys):
    """取最近一个可用年份的值, 返回 (值, 完整序列)。"""
    for k in keys:
        if k in stmt.index:
            s = stmt.loc[k].dropna()
            if len(s):
                return float(s.iloc[0]), s
    return np.nan, pd.Series(dtype=float)


def _cagr(first, last, years):
    if not first or not last or first <= 0 or last <= 0 or years <= 0:
        return np.nan
    return (last / first) ** (1.0 / years) - 1.0


def compute_metrics(symbol: str, count: int = 1300) -> dict:
    """计算某标的的全量基本面结构化字段 (见 docs/选股策略-条件清单.md)。"""
    import yfinance as yf

    t = yf.Ticker(symbol)
    info, inc, bs, cf = {}, pd.DataFrame(), pd.DataFrame(), pd.DataFrame()
    def _stmt(fn):
        df = fn()
        return df if isinstance(df, pd.DataFrame) and not df.empty else pd.DataFrame()

    for attempt in range(2):  # 限流重试
        try:
            info = t.info or {}
            inc = _stmt(t.get_income_stmt)
            bs = _stmt(t.get_balance_sheet)
            cf = _stmt(t.get_cash_flow)
            if info and (len(inc) or len(bs)):
                break
            time.sleep(2)
        except Exception:
            if attempt:
                raise
            time.sleep(2)

    m: dict = {"symbol": symbol, "currency": info.get("currency", "USD"),
               "name": info.get("shortName") or info.get("longName") or symbol}
    cov: dict = {}

    # ---------- 价格历史 (本地缓存优先): 年末收盘 / 5年分位 ----------
    price, px_df = None, pd.DataFrame()
    try:
        px = fetch_quote(symbol, "1d", count=count)
        px_df = pd.DataFrame([r[1:] for r in px["rows"]],
                             columns=["open", "high", "low", "close", "volume"],
                             index=pd.to_datetime([r[0] for r in px["rows"]], format="mixed"))
        if len(px_df):
            price = float(px_df["close"].iloc[-1])
            m["price_pct_5y"] = float((px_df["close"] < price).mean() * 100.0)
    except Exception:
        m["price_pct_5y"] = np.nan

    # ---------- 报表序列 ----------
    ni_all = _series_all(inc, *K_NET_INCOME)
    rev_all = _series_all(inc, *K_REVENUE)
    eps_all = _series_all(inc, *K_EPS)
    eq_all = _series_all(bs, *K_EQUITY)
    ni, _ = _get(inc, *K_NET_INCOME)
    rev, _ = _get(inc, *K_REVENUE)
    eps, _ = _get(inc, *K_EPS)
    equity, _ = _get(bs, *K_EQUITY)
    ocf, _ = _get(cf, *K_OCF)
    capex, _ = _get(cf, *K_CAPEX)
    ca, _ = _get(bs, *K_CA)
    cl, _ = _get(bs, *K_CL)
    ltd, _ = _get(bs, *K_LTD)
    goodwill, _ = _get(bs, *K_GOODWILL)
    assets, _ = _get(bs, *K_ASSETS)
    liab, _ = _get(bs, *K_LIAB)
    shares, _ = _get(bs, *K_SHARES)

    # ---------- 快照字段 (info 优先) ----------
    m["market_cap"] = info.get("marketCap")
    m["trailing_pe"] = info.get("trailingPE")
    m["pb_mrq"] = info.get("priceToBook")
    m["current_ratio"] = info.get("currentRatio")
    m["roe"] = info.get("returnOnEquity")
    m["gross_margin"] = info.get("grossMargins")
    m["net_margin"] = info.get("profitMargins")
    m["payout_ratio"] = info.get("payoutRatio")
    m["fcf"] = info.get("freeCashflow")
    m["total_cash"] = info.get("totalCash")
    m["total_debt"] = info.get("totalDebt")

    # ---------- 逐年序列 ----------
    def yearly_ratio(num_all, den_all, years):
        vals = []
        for col in num_all.index:
            if col in den_all.index:
                nv, dv = num_all[col], den_all[col]
                if nv and dv:
                    vals.append((col.year, float(_div(nv, dv))))
        vals.sort(reverse=True)
        return [y for y, _ in vals[:years]], [r for _, r in vals[:years]]

    _, roe_list = yearly_ratio(ni_all, eq_all, 5)
    m["roe_yearly"], cov["roe_yearly"] = roe_list, len(roe_list)
    _, nm_list = yearly_ratio(ni_all, rev_all, 3)
    m["net_margin_yearly"], cov["net_margin_yearly"] = nm_list, len(nm_list)
    gp_all = (rev_all - _series_all(inc, "GrossProfit")).dropna()
    _, gm_list = yearly_ratio(gp_all, rev_all, 3)
    m["gross_margin_yearly"], cov["gross_margin_yearly"] = gm_list, len(gm_list)
    div_paid = _series_all(cf, *K_DIVPAID)
    _, pr_list = yearly_ratio(div_paid.abs(), ni_all, 5)
    m["payout_ratio_yearly"], cov["payout_ratio_yearly"] = pr_list, len(pr_list)

    # 股息率逐年: 分红历史按自然年求和 / 该年年末收盘价
    try:
        div_hist = t.dividends
        div_year = div_hist.groupby(div_hist.index.year).sum()
        close_year = px_df["close"].groupby(px_df.index.year).last() if len(px_df) else pd.Series()
        dy = []
        for year in sorted(close_year.index, reverse=True):
            d = float(div_year.get(year, 0.0))
            p = float(close_year.get(year, np.nan))
            if np.isfinite(p) and p > 0:
                dy.append(_div(d, p))
        dy_list = dy[:7]
        m["div_yield_yearly"], cov["div_yield_yearly"] = dy_list, len(dy_list)
    except Exception:
        dy_list = []
        m["div_yield_yearly"], cov["div_yield_yearly"] = [], 0

    # ---------- info 缺失时的报表+行情回填 (限流兜底) ----------
    if price:
        if m["market_cap"] is None and shares:
            m["market_cap"] = shares * price
        if m["trailing_pe"] is None and eps:
            m["trailing_pe"] = _div(price, eps)
        if m["pb_mrq"] is None and shares and equity:
            m["pb_mrq"] = _div(price, _div(equity, shares))
    if m["roe"] is None and roe_list:
        m["roe"] = roe_list[0]
    if m["gross_margin"] is None and gm_list:
        m["gross_margin"] = gm_list[0]
    if m["net_margin"] is None and nm_list:
        m["net_margin"] = nm_list[0]
    if m["payout_ratio"] is None and pr_list:
        m["payout_ratio"] = pr_list[0]
    if m["current_ratio"] is None:
        m["current_ratio"] = _div(ca, cl)
    if m["fcf"] is None and np.isfinite(ocf) and np.isfinite(capex):
        m["fcf"] = ocf + capex  # capex 通常为负数
    if m["total_debt"] is None:
        m["total_debt"] = ltd if np.isfinite(ltd) else None
    if m["total_cash"] is None:
        cash_eq, _ = _get(bs, "CashAndCashEquivalents", "Cash And Cash Equivalents")
        m["total_cash"] = cash_eq if np.isfinite(cash_eq) else None

    # ---------- 派生字段 ----------
    m["graham_pe_pb"] = _div(m["trailing_pe"], 1) * m["pb_mrq"] \
        if m["trailing_pe"] and m["pb_mrq"] else np.nan
    m["fcf_to_net_income"] = _div(m["fcf"], ni)
    m["capex_to_revenue"] = _div(abs(capex), rev)
    m["net_cash"] = (m["total_cash"] - m["total_debt"]) \
        if m["total_cash"] is not None and m["total_debt"] is not None else np.nan
    m["net_cash_to_mktcap"] = _div(m["net_cash"], m["market_cap"])
    m["debt_to_assets"] = _div(liab, assets)
    m["goodwill_to_assets"] = _div(goodwill, assets)
    m["working_capital_to_ltd"] = _div(ca - cl, ltd)
    m["roic"] = _div(ni * 0.75, (m["total_debt"] or 0) + equity)
    m["roe_avg_3y"] = float(np.mean(roe_list[:3])) if len(roe_list) >= 3 else np.nan

    if len(eps_all) >= 2:
        cols = list(eps_all.index)
        m["eps_growth"] = _cagr(eps_all[cols[1]], eps_all[cols[0]], 1) \
            if eps_all[cols[1]] else np.nan
        span = cols[0].year - cols[-1].year
        m["eps_cagr_10y"] = _cagr(eps_all[cols[-1]], eps_all[cols[0]], span) \
            if span and eps_all[cols[-1]] else np.nan
        cov["eps_cagr_10y"] = f"{span}年(数据上限)"
    else:
        m["eps_growth"], m["eps_cagr_10y"] = np.nan, np.nan
        cov["eps_cagr_10y"] = "不足"
    if len(rev_all) >= 2:
        cols = list(rev_all.index)
        m["revenue_growth"] = _div(rev_all[cols[0]], rev_all[cols[1]]) - 1
    else:
        m["revenue_growth"] = np.nan
    # 市盈率(3年平均): 各财年年末收盘价 / 该年稀释 EPS (财年末与自然年尾近似对齐)
    if len(px_df) and len(eps_all):
        pe_years = []
        for col in list(eps_all.index)[:3]:
            yr_close = px_df["close"][px_df.index.year == col.year]
            if len(yr_close) and eps_all[col]:
                pe_years.append(_div(float(yr_close.iloc[-1]), float(eps_all[col])))
        m["pe_avg_3y"] = float(np.mean(pe_years)) if pe_years else np.nan
    else:
        m["pe_avg_3y"] = np.nan

    g = m["eps_growth"] * 100 if m["eps_growth"] and m["eps_growth"] > 0 else np.nan
    m["peg"] = _div(m["trailing_pe"], g)

    # ---- 第二批策略新增字段 ----
    # 静态股息率: 最近年度股息率
    m["static_div_yield"] = dy_list[0] if dy_list else np.nan
    # 营业总收入 CAGR(按可得年限, 通常~3年)
    if len(rev_all) >= 2:
        rcols = list(rev_all.index)
        span_r = rcols[0].year - rcols[-1].year
        m["rev_cagr_3y"] = _cagr(rev_all[rcols[-1]], rev_all[rcols[0]], span_r) \
            if span_r and rev_all[rcols[-1]] else np.nan
    else:
        m["rev_cagr_3y"] = np.nan
    # 销售增速 - 存货增速 (存货来自资产负债表 Inventory)
    inv_all = _series_all(bs, "Inventory")
    if len(rev_all) >= 2:
        rcols = list(rev_all.index)
        sales_g = _div(rev_all[rcols[0]], rev_all[rcols[1]]) - 1
        if len(inv_all) and rcols[1] in inv_all.index and inv_all[rcols[1]]:
            inv_g = _div(inv_all[rcols[0]], inv_all[rcols[1]]) - 1
            m["sales_minus_inventory_growth"] = sales_g - inv_g
        else:
            m["sales_minus_inventory_growth"] = np.nan
    else:
        m["sales_minus_inventory_growth"] = np.nan
    # 净利润增长率 (年报)
    if len(ni_all) >= 2:
        ncols = list(ni_all.index)
        m["net_income_growth"] = _div(ni_all[ncols[0]], ni_all[ncols[1]]) - 1 \
            if ni_all[ncols[1]] else np.nan
    else:
        m["net_income_growth"] = np.nan
    # 戴维斯双击乘数 = EPS同比倍数 x PE同比倍数 (等价于近一年股价涨幅倍数)
    if len(eps_all) >= 2 and len(px_df) >= 2:
        ecols = list(eps_all.index)
        eps_factor = _div(eps_all[ecols[0]], eps_all[ecols[1]])
        pe_factor = _div(float(px_df["close"].iloc[-1]), float(px_df["close"].iloc[-2]))
        m["double_play_multiplier"] = _div(eps_factor * pe_factor, 1.0)
    else:
        m["double_play_multiplier"] = np.nan
    # PE(TTM) 5年分位: 日频 PE = 收盘价 / TTM EPS (EPS 短期近似恒定)
    if len(px_df) and eps:
        daily_pe = px_df["close"] / eps
        m["pe_pct_5y"] = float((daily_pe < daily_pe.iloc[-1]).mean() * 100.0)
    else:
        m["pe_pct_5y"] = np.nan
    # 聂夫总报酬率 TRR = (EPS增速% + 股息率%) / PE(TTM)
    dy_pct = (dy_list[0] * 100) if dy_list else np.nan
    eg_pct = m["eps_growth"] * 100 if m["eps_growth"] else np.nan
    m["trr"] = _div(eg_pct + dy_pct, m["trailing_pe"]) if m["trailing_pe"] else np.nan

    m["_coverage"] = cov
    return m


def _series_all(stmt: pd.DataFrame, *keys) -> pd.Series:
    for k in keys:
        if k in stmt.index:
            return stmt.loc[k].dropna()
    return pd.Series(dtype=float)
