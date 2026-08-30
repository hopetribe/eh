# -*- coding: utf-8 -*-
"""选股策略定义: 策略图卡条件的结构化转写 (详见 docs/选股策略-条件清单.md)。

每个条件: {text 原文, field 结构化字段, op 运算符, value 阈值, need 需要的年数(逐年类)}
op 取值: "<" / ">" / "yearly_gt" (窗口内每年 > value) / "yearly_lt"
全局规则: 总市值 > 50 亿元 (增长双验 100 亿), 市值按汇率折算为人民币。
"""

FX_TO_CNY = {"USD": 7.2, "HKD": 0.92, "CNY": 1.0, "CNH": 1.0}

# 各市场演示候选池 (全市场扫描需列表数据源, 当前为候选池模式)
SAMPLE_UNIVERSE = {
    "us": ["MSFT", "AAPL", "NVDA", "GOOGL", "KO", "JNJ", "XOM", "T", "INTC", "PFE"],
    "hk": ["00700", "00941", "00005", "01299", "00388"],
    "cn": ["600519", "000858", "601318", "601398", "000333"],
}

STRATEGIES = {
    "graham": {
        "name": "格雷厄姆 TOP10 · 安全边界 防御型",
        "theme": "格雷厄姆数 + 存活保证 + 财务安全",
        "min_mktcap_cny": 50e8,
        "conditions": [
            {"text": "格雷厄姆数(PE*PB) < 22.5", "field": "graham_pe_pb", "op": "<", "value": 22.5},
            {"text": "市盈率(3年平均) < 15", "field": "pe_avg_3y", "op": "<", "value": 15},
            {"text": "流动比率 > 2", "field": "current_ratio", "op": ">", "value": 2},
            {"text": "近3年净资产收益率每年 > 0%", "field": "roe_yearly", "op": "yearly_gt",
             "value": 0.0, "need": 3},
            {"text": "近3年净利率每年 > 0%", "field": "net_margin_yearly", "op": "yearly_gt",
             "value": 0.0, "need": 3},
            {"text": "近7年股息率每年 > 1%", "field": "div_yield_yearly", "op": "yearly_gt",
             "value": 0.01, "need": 7},
            {"text": "营运资本/长期负债 > 1", "field": "working_capital_to_ltd", "op": ">", "value": 1},
        ],
    },
    "growth": {
        "name": "增长双验 · 跨周期 · 质量 · 价值护栏",
        "theme": "增长双验 + 跨周期 + 质量 + 价值护栏",
        "min_mktcap_cny": 100e8,
        "conditions": [
            {"text": "每股收益增长率 > 15%", "field": "eps_growth", "op": ">", "value": 0.15},
            {"text": "营业收入增长率 > 15%", "field": "revenue_growth", "op": ">", "value": 0.15},
            {"text": "每股收益CAGR(10年) > 8%", "field": "eps_cagr_10y", "op": ">", "value": 0.08},
            {"text": "净资产收益率(3年平均) > 10%", "field": "roe_avg_3y", "op": ">", "value": 0.10},
            {"text": "近3年净利率每年 > 8%", "field": "net_margin_yearly", "op": "yearly_gt",
             "value": 0.08, "need": 3},
            {"text": "PEG < 2", "field": "peg", "op": "<", "value": 2},
            {"text": "自由现金流净现比 > 0.6", "field": "fcf_to_net_income", "op": ">", "value": 0.6},
        ],
    },
    "schloss": {
        "name": "沃尔特·施洛斯 TOP10 · 破净买资产",
        "theme": "低PB灵魂 + 账面双验 + 分红支撑",
        "min_mktcap_cny": 50e8,
        "conditions": [
            {"text": "市净率(MRQ) < 1", "field": "pb_mrq", "op": "<", "value": 1},
            {"text": "股价5年分位数 < 30%", "field": "price_pct_5y", "op": "<", "value": 30},
            {"text": "资产负债率 < 50%", "field": "debt_to_assets", "op": "<", "value": 0.5},
            {"text": "商誉/总资产 < 10%", "field": "goodwill_to_assets", "op": "<", "value": 0.10},
            {"text": "净现金市值比 > 5%", "field": "net_cash_to_mktcap", "op": ">", "value": 0.05},
            {"text": "净资产收益率(3年平均) > 5%", "field": "roe_avg_3y", "op": ">", "value": 0.05},
            {"text": "股息率近3年每年 > 1%", "field": "div_yield_yearly", "op": "yearly_gt",
             "value": 0.01, "need": 3},
        ],
    },
    "buffett": {
        "name": "沃伦·巴菲特 TOP10 · 优质企业 合理价格",
        "theme": "护城河验证 + 盈利含金量 + 增长持续",
        "min_mktcap_cny": 50e8,
        "conditions": [
            {"text": "投入资本回报率ROIC > 15%", "field": "roic", "op": ">", "value": 0.15},
            {"text": "净资产收益率ROE > 15%", "field": "roe", "op": ">", "value": 0.15},
            {"text": "ROE近5年每年 > 10%", "field": "roe_yearly", "op": "yearly_gt",
             "value": 0.10, "need": 5},
            {"text": "毛利率近3年每年 > 20%", "field": "gross_margin_yearly", "op": "yearly_gt",
             "value": 0.20, "need": 3},
            {"text": "自由现金流净现比 > 0.5", "field": "fcf_to_net_income", "op": ">", "value": 0.5},
            {"text": "资本支出比率 < 20%", "field": "capex_to_revenue", "op": "<", "value": 0.20},
            {"text": "每股收益CAGR(10年) > 5%", "field": "eps_cagr_10y", "op": ">", "value": 0.05},
            {"text": "派息率近5年每年 > 30%", "field": "payout_ratio_yearly", "op": "yearly_gt",
             "value": 0.30, "need": 5},
        ],
    },
}


def get_strategy(sid: str) -> dict:
    if sid not in STRATEGIES:
        raise KeyError(f"未知策略: {sid} (可选: {', '.join(STRATEGIES)})")
    return STRATEGIES[sid]

# ---------------- 第二批: 聂夫 / 林奇 / 费雪 / 戴维斯双击 ----------------

STRATEGIES["neff"] = {
    "name": "约翰·聂夫 TOP10 · 低市盈率 总报酬率",
    "theme": "低PE + TRR灵魂 + 温和增长 + 股息保底",
    "min_mktcap_cny": 100e8,
    "conditions": [
        {"text": "约翰·聂夫总收益率(TRR) > 1", "field": "trr", "op": ">", "value": 1},
        {"text": "市盈率(TTM) < 15", "field": "trailing_pe", "op": "<", "value": 15},
        {"text": "每股收益增长率 7%~20%", "field": "eps_growth", "op": "between",
         "value": [0.07, 0.20]},
        {"text": "静态股息率 > 3%", "field": "static_div_yield", "op": ">", "value": 0.03},
        {"text": "净资产收益率(3年平均) > 10%", "field": "roe_avg_3y", "op": ">", "value": 0.10},
        {"text": "总市值 > 100 亿元", "field": "market_cap_cny", "op": ">", "value": 100e8},
        {"text": "自由现金流净现比 > 0.5", "field": "fcf_to_net_income", "op": ">", "value": 0.5},
        {"text": "近5年股息率每年 > 1%", "field": "div_yield_yearly", "op": "yearly_gt",
         "value": 0.01, "need": 5},
    ],
}

STRATEGIES["lynch"] = {
    "name": "彼得·林奇 TOP10 · 合理价格成长 (GARP)",
    "theme": "PEG + 销售与存货 + 质量双验",
    "min_mktcap_cny": 100e8,
    "conditions": [
        {"text": "PEG < 1.5", "field": "peg", "op": "<", "value": 1.5},
        {"text": "市盈率(TTM) < 25", "field": "trailing_pe", "op": "<", "value": 25},
        {"text": "每股收益增长率 8%~40%", "field": "eps_growth", "op": "between",
         "value": [0.08, 0.40]},
        {"text": "营业总收入CAGR(3年) > 8%", "field": "rev_cagr_3y", "op": ">", "value": 0.08},
        {"text": "销售增速减存货增速 > 0%", "field": "sales_minus_inventory_growth",
         "op": ">", "value": 0},
        {"text": "净资产收益率(3年平均) > 10%", "field": "roe_avg_3y", "op": ">", "value": 0.10},
        {"text": "自由现金流净现比 > 0.5", "field": "fcf_to_net_income", "op": ">", "value": 0.5},
    ],
}

STRATEGIES["fisher"] = {
    "name": "菲利普·费雪 TOP10 · 卓越成长",
    "theme": "增长双验 + 跨周期 + 质量 + 价值护栏",
    "min_mktcap_cny": 100e8,
    "conditions": [
        {"text": "每股收益增长率 > 15%", "field": "eps_growth", "op": ">", "value": 0.15},
        {"text": "营业收入增长率 > 15%", "field": "revenue_growth", "op": ">", "value": 0.15},
        {"text": "每股收益CAGR(10年) > 8%", "field": "eps_cagr_10y", "op": ">", "value": 0.08},
        {"text": "净资产收益率(3年平均) > 10%", "field": "roe_avg_3y", "op": ">", "value": 0.10},
        {"text": "近3年净利率每年 > 8%", "field": "net_margin_yearly", "op": "yearly_gt",
         "value": 0.08, "need": 3},
        {"text": "PEG < 2", "field": "peg", "op": "<", "value": 2},
        {"text": "自由现金流净现比 > 0.6", "field": "fcf_to_net_income", "op": ">", "value": 0.6},
    ],
}

STRATEGIES["davis"] = {
    "name": "戴维斯双击 TOP10 · 估值+增长 质量 含金量",
    "theme": "估值双锚 + 增长双确认 + 复利引擎",
    "min_mktcap_cny": 50e8,
    "conditions": [
        {"text": "双击乘数 > 2", "field": "double_play_multiplier", "op": ">", "value": 2},
        {"text": "市盈率(TTM)5年分位数 < 30%", "field": "pe_pct_5y", "op": "<", "value": 30},
        {"text": "每股收益增长率 > 15%", "field": "eps_growth", "op": ">", "value": 0.15},
        {"text": "市盈率(TTM) < 20", "field": "trailing_pe", "op": "<", "value": 20},
        {"text": "净利润增长率 > 10%", "field": "net_income_growth", "op": ">", "value": 0.10},
        {"text": "净资产收益率 > 12%", "field": "roe", "op": ">", "value": 0.12},
        {"text": "自由现金流净现比 > 0.6", "field": "fcf_to_net_income", "op": ">", "value": 0.6},
        {"text": "近5年派息率每年 > 30%", "field": "payout_ratio_yearly", "op": "yearly_gt",
         "value": 0.30, "need": 5},
    ],
}
