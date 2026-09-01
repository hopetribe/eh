# -*- coding: utf-8 -*-
"""机会雷达扫描引擎: 市值阈值股票池日K -> EHOPT10 指标 -> 近期信号。

设计
----
- 复用 gcn.data.service.fetch_quote 的 K线落盘缓存, 二次扫描基本零网络请求;
- 扫描结果按市场落盘 data/radar_<market>.json，由每日 09:00 调度或手动操作刷新;
- 每标的记录最近 SIGNAL_HISTORY 根K线内的全部信号及距今天数, 前端按
  近3日/近1周/近2周 本地过滤, 切换窗口无需重新扫描;
- 单只标的失败仅记账, 不阻断整体扫描。
"""
from __future__ import annotations

import json
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

import numpy as np
import pandas as pd

from gcn.data.service import (DATA_DIR, DEFAULT_COUNT, _cache_is_fresh,
                              _cache_path, _load_cache, _atomic_write_text,
                              fetch_quote)
from gcn.radar.universe import (MARKET_CAP_CURRENCIES, MARKET_CAP_THRESHOLDS,
                                RADAR_MARKETS, get_universe)
from gcn.recipes.gcn_main import compute_ehopt10

FETCH_COUNT = 300     # 每标的拉取的日K数量 (指标预热充足)
SCAN_WORKERS = 6      # 并发扫描线程数 (兼顾速度与数据源限流)
SIGNAL_HISTORY = 15   # 每标的记录最近 N 根K线内的信号 (覆盖 近3日/1周/2周)
CACHE_TTL = 26 * 3600  # 每日扫描结果跨过下一个 09:00 前保持有效

# 关注的信号列 -> 展示名 (与 webui SIG_TYPES 徽章一致)
SIGNAL_DEFS = [("B_SIGNAL", "B买"), ("ICON_JUEFAN", "绝反")]

MARKETS = [m for m, _ in RADAR_MARKETS]


# ============================ 单标的扫描 ============================

def _extract_recent(res: pd.DataFrame, max_days: int = SIGNAL_HISTORY,
                    as_of=None) -> list[dict]:
    """提取最近 max_days 根K线内的 B买/绝反 信号。

    返回 [{type, date, days_ago, close}], 按时间新 -> 旧。
    纯函数, 便于离线测试。
    """
    out = []
    n = len(res)
    reference = pd.Timestamp.now().normalize() if as_of is None else pd.Timestamp(as_of).normalize()
    if reference.tzinfo is not None:
        reference = reference.tz_localize(None)
    cols = {label: np.asarray(res[col], dtype=bool) for col, label in SIGNAL_DEFS
            if col in res.columns}
    for i in range(max(0, n - max_days), n):
        for label, col in cols.items():
            if col[i]:
                signal_day = pd.Timestamp(res.index[i])
                if signal_day.tzinfo is not None:
                    signal_day = signal_day.tz_localize(None)
                calendar_age = max(0, int(np.busday_count(
                    signal_day.normalize().date(), reference.date())))
                out.append({
                    "type": label,
                    "date": str(res.index[i])[:10],
                    "days_ago": max(int(n - 1 - i), calendar_age),
                    "close": round(float(res["CLOSE"].iloc[i]), 4),
                })
    out.sort(key=lambda s: s["days_ago"])
    return out


def scan_symbol(code: str, market: str, name: str = "",
                count: int = FETCH_COUNT) -> dict:
    """扫描单只标的: 拉日K(带缓存) -> 计算指标 -> 提取近期信号。

    拉取用与主图一致的 DEFAULT_COUNT (保证落盘缓存为全量历史, 主图
    不受影响), 计算只取最近 count 根 (指标预热充足)。
    """
    try:
        quote = fetch_quote(code, "1d", DEFAULT_COUNT)
        rows = quote["rows"][-count:]
        df = pd.DataFrame([r[1:] for r in rows],
                          columns=["open", "high", "low", "close", "volume"],
                          index=pd.to_datetime([r[0] for r in rows], format="mixed"))
        res = compute_ehopt10(df, version="v4")
        close = float(res["CLOSE"].iloc[-1])
        prev = float(res["CLOSE"].iloc[-2]) if len(res) > 1 else float("nan")
        return {
            "code": code, "market": market,
            "name": name or code,
            "date": str(res.index[-1])[:10],
            "close": round(close, 4),
            "chg_pct": (round((close / prev - 1) * 100, 2)
                        if np.isfinite(prev) and prev else None),
            "signals": _extract_recent(res),
            "error": None,
        }
    except Exception as exc:  # noqa: BLE001 - 单只失败不阻断整体扫描
        return {"code": code, "market": market, "name": name,
                "date": None, "close": None, "chg_pct": None,
                "signals": [], "error": f"{type(exc).__name__}: {exc}"}


def scan_market(market: str, progress=None,
                max_workers: int = SCAN_WORKERS) -> dict:
    """并发扫描一个市场全部标的, 返回结果块 (只保留有信号的命中项)。"""
    universe, src = get_universe(market)
    results: list[dict] = []
    with ThreadPoolExecutor(max_workers=max_workers) as pool:
        futs = [pool.submit(scan_symbol, c, market, n) for c, n in universe]
        for i, fut in enumerate(as_completed(futs), 1):
            results.append(fut.result())
            if progress:
                try:
                    progress(i, len(universe))
                except Exception:  # noqa: BLE001
                    pass
    hits = [r for r in results if r.get("signals")]
    hits.sort(key=lambda r: (min(s["days_ago"] for s in r["signals"]), r["code"]))
    return {
        "market": market,
        "universe_source": src,
        "universe_complete": src != "static-partial",
        "market_cap_threshold": MARKET_CAP_THRESHOLDS[market],
        "market_cap_currency": MARKET_CAP_CURRENCIES[market],
        "n_scanned": len(results),
        "n_errors": sum(1 for r in results if r.get("error")),
        "n_hits": len(hits),
        "results": hits,
        "failed_codes": [r["code"] for r in results if r.get("error")],
        "generated_at": time.time(),
    }


# ============================ 缓存 + 后台任务 ============================

def _result_cache_path(market: str):
    return DATA_DIR / f"radar_{market}.json"


def load_cache(market: str) -> dict | None:
    try:
        return json.loads(_result_cache_path(market).read_text(encoding="utf-8"))
    except Exception:  # noqa: BLE001 - 缺失/损坏都视为无缓存
        return None


def save_cache(market: str, block: dict):
    DATA_DIR.mkdir(exist_ok=True)
    _atomic_write_text(_result_cache_path(market),
                       json.dumps(block, ensure_ascii=False))


class RadarService:
    """雷达扫描调度: 后台线程 + 进度记账 + 缓存读写 (单例, 服务端共用)。"""

    def __init__(self):
        self._mu = threading.Lock()
        self.jobs: dict[str, dict] = {}  # market -> {status, done, total, error}

    # ---- 状态视图 ----
    def _block_view(self, market: str) -> dict:
        cache = load_cache(market)
        stale = (cache is None
                 or time.time() - cache.get("generated_at", 0) > CACHE_TTL)
        return {"cache": cache, "stale": stale}

    def snapshot(self, markets: list[str] | None = None) -> dict:
        """当前各市场状态: 扫描进度 + 缓存结果 (可能过期, 标记 stale)。"""
        markets = markets or MARKETS
        with self._mu:
            jobs = {m: dict(self.jobs.get(m) or {"status": "idle"})
                    for m in markets}
        out = {"markets": {}}
        for m in markets:
            view = self._block_view(m)
            job = jobs[m]
            scanning = job.get("status") == "scanning"
            if job.get("status") == "error":
                job["note"] = job.get("error") or "扫描失败"
            out["markets"][m] = {
                "job": job, "scanning": scanning,
                "cache": view["cache"], "stale": view["stale"],
            }
        return out

    # ---- 扫描调度 ----
    def _run(self, market: str):
        def progress(done, total):
            with self._mu:
                job = self.jobs.get(market)
                if job and job["status"] == "scanning":
                    job.update(done=done, total=total)

        try:
            block = scan_market(market, progress=progress)
            if block.get("n_scanned", 0) <= 0 \
                    or block.get("n_errors", 0) >= block.get("n_scanned", 0):
                raise RuntimeError("全市场扫描无成功结果，保留上一版缓存")
            previous = load_cache(market)
            failed = set(block.get("failed_codes") or [])
            current_codes = {item.get("code") for item in block.get("results", [])}
            if previous and failed:
                today = pd.Timestamp.now().normalize().date()
                for old in previous.get("results", []):
                    if old.get("code") not in failed or old.get("code") in current_codes:
                        continue
                    kept = dict(old)
                    kept["stale"] = True
                    kept["stale_reason"] = "本轮刷新失败，保留上一版命中"
                    kept["signals"] = [dict(sig) for sig in old.get("signals", [])]
                    for sig in kept["signals"]:
                        try:
                            age = int(np.busday_count(
                                pd.Timestamp(sig["date"]).date(), today))
                            sig["days_ago"] = max(int(sig.get("days_ago", 0)), age)
                        except Exception:  # noqa: BLE001 - 保留无法解析的旧展示字段
                            pass
                    block["results"].append(kept)
            block["n_hits"] = len(block.get("results", []))
            block["results"].sort(key=lambda r: (
                min((s.get("days_ago", 10**9) for s in r.get("signals", [])),
                    default=10**9), r.get("code", "")))
            save_cache(market, block)
            with self._mu:
                self.jobs[market] = {
                    "status": "done", "done": block["n_scanned"],
                    "total": block["n_scanned"],
                    "finished_at": block["generated_at"],
                }
        except Exception as exc:  # noqa: BLE001 - 后台失败记录后保持旧缓存可用
            with self._mu:
                self.jobs[market] = {"status": "error",
                                     "error": f"{type(exc).__name__}: {exc}"}

    def start_scan(self, markets: list[str]) -> list[str]:
        """对指定市场启动后台强制扫描, 返回实际启动的市场。"""
        started = []
        with self._mu:
            for m in markets:
                if m not in MARKETS:
                    continue
                job = self.jobs.get(m) or {}
                if job.get("status") == "scanning":
                    continue
                self.jobs[m] = {"status": "scanning", "done": 0, "total": 0}
                threading.Thread(target=self._run, args=(m,), daemon=True).start()
                started.append(m)
        return started

    def ensure_fresh(self, markets: list[str]) -> dict:
        """兼容显式续鲜调用：过期/缺失市场后台重扫并立即返回旧快照。"""
        need = [m for m in markets
                if (self.jobs.get(m) or {}).get("status") != "scanning"
                and self._block_view(m)["stale"]]
        self.start_scan(need)
        return self.snapshot(markets)


SERVICE = RadarService()


# ============================ 日K预热守护 ============================
# 目标: 三大市场超过市值阈值的全部标的日K数据常驻本地 (覆盖最近半年以上, 实际存
# 全量合并历史), 每天增量更新一次 —— 复用 fetch_quote 的合并落盘与新鲜度
# 规则 (当日已刷新过 / 已有今日K线 / 未错失交易日, 则零网络请求)。

WARM_TICK = 3600           # 巡检周期 (秒): 每小时检查一次
WARM_WORKERS = 4           # 并发刷新线程 (对数据源温和)
WARM_FAIL_BACKOFF = 6 * 3600  # 刷新失败标的的退避期 (秒后重试)

# 失败退避记账: code -> 上次失败时间戳 (跨巡检共享)
_warm_fails: dict[str, float] = {}


def _kline_cache_fresh(symbol: str) -> bool:
    """日K缓存是否足够新 (无需在线请求)。"""
    path = _cache_path(symbol, "1d")
    if not path.exists():
        return False
    cached = _load_cache(symbol, "1d")
    return (cached is not None and not cached.empty
            and _cache_is_fresh(cached, "1d", path, symbol=symbol))


def warm_market(market: str, force: bool = False,
                max_workers: int = WARM_WORKERS, log: bool = False) -> dict:
    """增量刷新一个市场的日K缓存: 只请求陈旧/缺失标的, 新K线合并落盘。

    force=True 忽略新鲜度与失败退避, 全部刷新 (手动预热用)。
    假日/周末所有标的均"新鲜", 自然零请求; A/港/美休市日历不同, 少数
    交易日错配日会有一次空刷新, 属可接受开销。
    """
    universe, src = get_universe(market)
    now = time.time()
    targets = []
    for code, name in universe:
        if force:
            targets.append((code, name))
        elif now - _warm_fails.get(code, 0) >= WARM_FAIL_BACKOFF \
                and not _kline_cache_fresh(code):
            targets.append((code, name))

    def _refresh(item):
        code, _ = item
        try:
            result = fetch_quote(code, "1d", DEFAULT_COUNT, force=force)
            if isinstance(result, dict) and result.get("refresh_failed"):
                raise RuntimeError(result.get("note") or "在线刷新失败")
            _warm_fails.pop(code, None)
            return True
        except Exception:  # noqa: BLE001 - 单只失败记账退避, 不阻断
            _warm_fails[code] = time.time()
            return False

    n_ok = n_fail = 0
    failed = []
    with ThreadPoolExecutor(max_workers=max_workers) as pool:
        for (code, _), ok in zip(targets, pool.map(_refresh, targets)):
            if ok:
                n_ok += 1
            else:
                n_fail += 1
                failed.append(code)
    stats = {"market": market, "universe_source": src,
             "universe_complete": src != "static-partial",
             "market_cap_threshold": MARKET_CAP_THRESHOLDS[market],
             "market_cap_currency": MARKET_CAP_CURRENCIES[market],
             "n_total": len(universe),
             "n_targets": len(targets), "n_ok": n_ok, "n_failed": n_fail,
             "failed": failed}
    if log:
        print(f"[radar-warm] {market}: 刷新 {n_ok}/{len(targets)} (池 {len(universe)}"
              f"{' · ' + src if src != 'futu' else ''}), 失败 {n_fail}"
              + (f": {', '.join(failed[:8])}{'…' if len(failed) > 8 else ''}" if failed else ""))
    return stats


def warm_loop(markets: list[str] | None = None, tick: int = WARM_TICK):
    """常驻守护: 每小时巡检各市场, 陈旧标的增量补数 (每日每标的至多一次)。"""
    markets = markets or MARKETS
    print(f"[radar-warm] 守护已启动: {'/'.join(markets)} 市值阈值股票池, "
          f"每 {tick // 60} 分钟巡检, 陈旧才增量请求")
    while True:
        for m in markets:
            try:
                warm_market(m, log=True)
            except Exception as exc:  # noqa: BLE001 - 守护循环不受单轮失败影响
                print(f"[radar-warm] {m} 巡检失败: {type(exc).__name__}: {exc}")
            time.sleep(2)  # 市场间隔, 分散请求
        time.sleep(tick)
