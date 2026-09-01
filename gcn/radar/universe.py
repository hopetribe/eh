# -*- coding: utf-8 -*-
"""机会雷达股票池: 各市场市值前100标的。

两级来源:
  1. 静态快照 (UNIVERSE, 按 2025 年中市值近似排序, 代码 -> 名称):
     无外部依赖, 始终可用; 排名随行情漂移, 仅作兜底;
  2. 动态快照: 本机 FutuOpenD 运行时通过 get_stock_filter 按总市值降序
     实时取前100 (A股 = SH+SZ 合并排序), 结果落盘缓存, 当日复用。

get_universe(market) 统一入口, 返回 ([(代码, 名称), ...], 来源) 按市值降序。
"""
from __future__ import annotations

import json
import time
from pathlib import Path

from gcn.data.service import DATA_DIR, _atomic_write_text, _opend_reachable

# 动态股票池缓存: data/radar_universe_<market>.json, 当日有效

# ---------------- A股 市值前100 (静态快照, 6位代码) ----------------
CN = {
    "600519": "贵州茅台", "601398": "工商银行", "601288": "农业银行",
    "600941": "中国移动", "601939": "建设银行", "300750": "宁德时代",
    "601857": "中国石油", "601988": "中国银行", "601628": "中国人寿",
    "002594": "比亚迪", "600938": "中国海油", "600036": "招商银行",
    "600028": "中国石化", "601088": "中国神华", "000858": "五粮液",
    "600900": "长江电力", "601318": "中国平安", "601899": "紫金矿业",
    "000333": "美的集团", "600030": "中信证券", "601166": "兴业银行",
    "601658": "邮储银行", "601816": "京沪高铁", "600050": "中国联通",
    "601319": "中国人保", "600276": "恒瑞医药", "601888": "中国中免",
    "000002": "万科A", "600011": "华能国际", "601328": "交通银行",
    "601898": "中煤能源", "601225": "陕西煤业", "601601": "中国太保",
    "601336": "新华保险", "688981": "中芯国际", "601138": "工业富联",
    "600690": "海尔智家", "000651": "格力电器", "000725": "京东方A",
    "601728": "中国电信", "000063": "中兴通讯", "002415": "海康威视",
    "600887": "伊利股份", "600809": "山西汾酒", "000568": "泸州老窖",
    "000596": "古井贡酒", "600436": "片仔癀", "300760": "迈瑞医疗",
    "603259": "药明康德", "600196": "复星医药", "002714": "牧原股份",
    "601766": "中国中车", "601390": "中国中铁", "601186": "中国铁建",
    "601668": "中国建筑", "601669": "中国电建", "601919": "中远海控",
    "601111": "中国国航", "600029": "南方航空", "600009": "上海机场",
    "601006": "大秦铁路", "600104": "上汽集团", "601633": "长城汽车",
    "000625": "长安汽车", "600745": "闻泰科技", "002475": "立讯精密",
    "603501": "韦尔股份", "002230": "科大讯飞", "300059": "东方财富",
    "688111": "金山办公", "300124": "汇川技术", "601012": "隆基绿能",
    "300274": "阳光电源", "688599": "天合光能", "600406": "国电南瑞",
    "601985": "中国核电", "003816": "中国广核", "600886": "国投电力",
    "600795": "国电电力", "600150": "中国船舶",
    "600893": "航发动力", "600760": "中航沈飞", "600019": "宝钢股份",
    "601998": "中信银行", "601600": "中国铝业", "603993": "洛阳钼业", "600016": "民生银行",
    "600000": "浦发银行", "601818": "光大银行", "601169": "北京银行",
    "600919": "江苏银行", "002142": "宁波银行", "601211": "国泰海通",
    "601688": "华泰证券", "600999": "招商证券", "000776": "广发证券",
    "601995": "中金公司", "600018": "上港集团", "002027": "分众传媒",
    "600584": "长电科技",
}

# ---------------- 港股 市值前100 (静态快照, 5位数字代码) ----------------
HK = {
    "00700": "腾讯控股", "09988": "阿里巴巴-W", "03690": "美团-W",
    "09618": "京东集团-SW", "01810": "小米集团-W", "01211": "比亚迪股份",
    "02318": "中国平安", "01299": "友邦保险", "00941": "中国移动",
    "01398": "工商银行", "00939": "建设银行", "03988": "中国银行",
    "03968": "招商银行", "03328": "交通银行", "01658": "邮储银行",
    "00005": "汇丰控股", "00241": "阿里健康", "00388": "香港交易所",
    "02388": "中银香港", "00386": "中国石油化工股份", "00857": "中国石油股份",
    "00883": "中国海洋石油", "01088": "中国神华", "01898": "中煤能源",
    "01171": "兖矿能源", "00836": "华润电力", "00902": "华能国际电力股份",
    "00762": "中国联通", "01766": "中国中车", "01919": "中远海控",
    "06030": "中信证券", "00688": "中国海外发展", "01109": "华润置地",
    "00016": "新鸿基地产", "00012": "恒基地产", "00001": "长和",
    "00002": "中电控股", "00003": "香港中华煤气", "00006": "电能实业",
    "00066": "港铁公司", "00101": "恒隆地产", "01113": "长实集团",
    "00019": "太古股份公司A", "00023": "东亚银行", "00027": "银河娱乐",
    "01928": "金沙中国", "00267": "中信股份", "00017": "新世界发展",
    "00823": "领展房产基金", "01099": "国药控股", "01177": "中国生物制药",
    "02269": "药明生物", "02319": "蒙牛乳业", "00291": "华润啤酒",
    "09633": "农夫山泉", "02313": "申洲国际", "02020": "安踏体育",
    "02331": "李宁", "02382": "舜宇光学科技", "02018": "瑞声科技",
    "00285": "比亚迪电子", "00981": "中芯国际", "01347": "华虹半导体",
    "00992": "联想集团", "01024": "快手-W", "09999": "网易-S",
    "09888": "百度集团-SW", "09961": "携程集团-S", "02015": "理想汽车-W",
    "09868": "小鹏汽车-W", "09866": "蔚来-SW", "02238": "广汽集团",
    "00175": "吉利汽车", "00914": "海螺水泥", "00960": "龙湖集团",
    "06618": "京东健康", "06690": "海尔智家", "06699": "时代电气",
    "01876": "百威亚太", "00322": "康师傅控股", "01044": "恒安国际",
    "03888": "金山软件", "06160": "百济神州", "06862": "海底捞",
    "01997": "九龙仓置业", "00083": "信和置业", "00268": "金蝶国际",
    "01800": "中国交通建设", "06060": "众安在线", "00868": "信义玻璃",
    "00881": "中升控股", "00728": "中国电信", "00390": "中国中铁",
    "09901": "新东方-S", "00669": "创科实业", "02618": "京东物流",
    "00316": "东方海外国际", "00966": "中国太平", "02359": "药明康德",
    "02333": "长城汽车",
}

# ---------------- 美股 市值前100 (静态快照) ----------------
US = {
    "AAPL": "苹果", "MSFT": "微软", "NVDA": "英伟达", "TSM": "台积电ADR",
    "GOOGL": "Alphabet",
    "AMZN": "亚马逊", "META": "Meta平台", "BRK-B": "伯克希尔B", "BABA": "阿里巴巴ADR",
    "TSLA": "特斯拉", "LLY": "礼来", "AVGO": "博通", "JPM": "摩根大通",
    "V": "维萨", "UNH": "联合健康", "XOM": "埃克森美孚", "MA": "万事达",
    "JNJ": "强生", "WMT": "沃尔玛", "PG": "宝洁", "ORCL": "甲骨文",
    "HD": "家得宝", "COST": "开市客", "MRK": "默沙东", "ABBV": "艾伯维",
    "CVX": "雪佛龙", "BAC": "美国银行", "CRM": "赛富时", "KO": "可口可乐",
    "AMD": "超威半导体", "PEP": "百事", "TMO": "赛默飞世尔", "LIN": "林德",
    "ACN": "埃森哲", "MCD": "麦当劳", "CSCO": "思科", "ABT": "雅培",
    "WFC": "富国银行", "DIS": "迪士尼", "DHR": "丹纳赫", "INTC": "英特尔",
    "VZ": "威瑞森", "QCOM": "高通", "TXN": "德州仪器", "AMGN": "安进",
    "NEE": "新纪元能源", "PM": "菲利普莫里斯", "CAT": "卡特彼勒",
    "AXP": "美国运通", "IBM": "国际商业机器", "GE": "通用电气", "GS": "高盛",
    "PGR": "前进保险", "UNP": "联合太平洋", "RTX": "雷神技术",
    "HON": "霍尼韦尔", "SPGI": "标普全球", "LOW": "劳氏", "ISRG": "直觉外科",
    "BKNG": "缤客", "ELV": "Elevance", "BLK": "贝莱德", "SYK": "史赛克",
    "TJX": "TJX公司", "ADP": "ADP", "GILD": "吉利德", "MDLZ": "亿滋国际",
    "LMT": "洛克希德马丁", "VRTX": "福泰制药", "CVS": "CVS健康",
    "AMAT": "应用材料", "MU": "美光科技", "SCHW": "嘉信理财",
    "LRCX": "拉姆研究", "ADI": "亚德诺", "ETN": "伊顿", "APH": "安费诺",
    "PFE": "辉瑞", "PLD": "安博", "C": "花旗集团", "BX": "黑石",
    "MMM": "3M", "MS": "摩根士丹利", "INTU": "财捷", "CB": "丘博",
    "PLTR": "Palantir", "T": "AT&T",
    "UBER": "优步", "NOW": "ServiceNow",
    "NFLX": "奈飞", "CMCSA": "康卡斯特", "TMUS": "T-Mobile",
    "ASML": "阿斯麦", "SNPS": "新思科技", "CDNS": "楷登电子",
    "KLAC": "科磊", "ROP": "罗珀技术", "GD": "通用动力",
    "DE": "迪尔", "BA": "波音",
}

UNIVERSE = {"cn": CN, "hk": HK, "us": US}
RADAR_MARKETS = [("us", "美股"), ("hk", "港股"), ("cn", "A股")]


def _static_universe(market: str) -> list[tuple[str, str]]:
    """静态快照 -> [(代码, 名称)], 按定义顺序 (近似市值降序)。"""
    return [(c, n) for c, n in UNIVERSE[market].items() if n]


def _universe_cache_path(market: str) -> Path:
    return DATA_DIR / f"radar_universe_{market}.json"


def _futu_market(market: str) -> list[str]:
    return {"cn": ["SH", "SZ"], "hk": ["HK"], "us": ["US"]}.get(market, [])


def _fetch_futu_top(market: str, n: int = 100) -> list[tuple[str, str]] | None:
    """FutuOpenD 在线时按总市值降序取前 n 只 (失败返回 None, 静默降级)。"""
    try:
        from futu import OpenQuoteContext, SimpleFilter, SortDir
        from futu.common.constant import StockField
    except ImportError:
        return None
    rows: list[tuple[float, str, str]] = []
    try:
        ctx = OpenQuoteContext(host="127.0.0.1", port=11111)
        try:
            for fmkt in _futu_market(market):
                flt = SimpleFilter()
                flt.stock_field = StockField.MARKET_VAL
                flt.is_no_filter = True
                flt.sort = SortDir.DESCEND
                begin = 0
                while begin < n:
                    ret, payload = ctx.get_stock_filter(
                        fmkt, filter_list=[flt], begin=begin, num=min(200, n - begin))
                    if ret != 0 or not isinstance(payload, tuple) or len(payload) != 3:
                        break
                    last_page, all_count, data = payload
                    if not data:
                        break
                    for rec in data:
                        code = str(getattr(rec, "stock_code", "")).split(".")[-1].strip()
                        name = str(getattr(rec, "stock_name", "") or "")
                        val = float(getattr(rec, "market_val", 0.0) or 0.0)
                        if code:
                            rows.append((val, code, name))
                    begin += len(data)
                    if last_page or begin >= int(all_count or 0):
                        break
        finally:
            ctx.close()
    except Exception:  # noqa: BLE001 - Futu 不可用/权限不足时静默降级
        return None
    if not rows:
        return None
    best: dict[str, tuple[float, str]] = {}
    for value, code, name in rows:
        if code not in best or value > best[code][0]:
            best[code] = (value, name)
    ranked = sorted(((value, code, name) for code, (value, name) in best.items()),
                    key=lambda item: (-item[0], item[1]))
    return [(code, name) for _, code, name in ranked[:n]]


def get_universe(market: str, n: int = 100, use_cache: bool = True) -> tuple[list[tuple[str, str]], str]:
    """返回 ([(代码, 名称)], 来源) — 来源: futu(当日动态快照) / static(静态快照)。

    FutuOpenD 在线时优先动态取市值前 n; 结果当日落盘缓存, 失败回退静态列表。
    """
    market = market if market in UNIVERSE else "us"
    cpath = _universe_cache_path(market)
    if use_cache and cpath.exists():
        try:
            blob = json.loads(cpath.read_text(encoding="utf-8"))
            if blob.get("day") == time.strftime("%Y-%m-%d") and blob.get("list"):
                return [(x[0], x[1]) for x in blob["list"]], "futu"
        except Exception:  # noqa: BLE001 - 缓存损坏时重新生成
            pass

    dyn = _fetch_futu_top(market, n=n) if _opend_reachable() else None
    if dyn:
        DATA_DIR.mkdir(exist_ok=True)
        _atomic_write_text(cpath, json.dumps(
            {"day": time.strftime("%Y-%m-%d"), "list": dyn}, ensure_ascii=False))
        return dyn, "futu"

    return _static_universe(market)[:n], "static"
