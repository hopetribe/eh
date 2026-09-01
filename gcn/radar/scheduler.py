# -*- coding: utf-8 -*-
"""机会雷达每日 09:00 扫描与邮件投递调度。"""
from __future__ import annotations

import threading
import time
from datetime import datetime, timedelta
from zoneinfo import ZoneInfo

from gcn.radar import engine
from gcn.radar.emailer import get_email_settings, record_delivery, send_radar_email

SCHEDULE_HOUR = 9
SCHEDULE_TIMEZONE = ZoneInfo("Asia/Shanghai")
SCAN_TIMEOUT = 8 * 3600


def next_run_at(now: datetime | None = None) -> datetime:
    """返回上海时区的下一次 09:00；精确 09:00 视为本轮应立即执行。"""
    current = now or datetime.now(SCHEDULE_TIMEZONE)
    if current.tzinfo is None:
        current = current.replace(tzinfo=SCHEDULE_TIMEZONE)
    else:
        current = current.astimezone(SCHEDULE_TIMEZONE)
    target = current.replace(hour=SCHEDULE_HOUR, minute=0, second=0, microsecond=0)
    return target if current <= target else target + timedelta(days=1)


def _wait_for_scans(service, markets: list[str], timeout: float = SCAN_TIMEOUT,
                    poll_interval: float = 1.0) -> dict:
    deadline = time.monotonic() + timeout
    while True:
        snapshot = service.snapshot(markets)
        if not any((snapshot["markets"].get(m) or {}).get("scanning") for m in markets):
            return snapshot
        if time.monotonic() >= deadline:
            raise TimeoutError("每日机会雷达扫描超时")
        time.sleep(poll_interval)


def run_daily_radar(markets: list[str] | None = None, service=None) -> dict:
    """预热阈值股票池、完成三市场扫描并投递一封汇总邮件。"""
    markets = markets or list(engine.MARKETS)
    service = service or engine.SERVICE
    for market in markets:
        try:
            engine.warm_market(market, log=True)
        except Exception as exc:  # noqa: BLE001 - 扫描仍可复用已有 K 线缓存
            print(f"[radar-daily] {market} 预热失败: {type(exc).__name__}: {exc}")
    service.start_scan(markets)
    snapshot = _wait_for_scans(service, markets)
    settings = get_email_settings()
    recipients = settings["recipients"]
    try:
        delivered = send_radar_email(snapshot, recipients=recipients)
        message = f"已发送至 {', '.join(delivered)}"
        record_delivery(True, message, delivered)
        print(f"[radar-daily] {message}")
    except Exception as exc:  # noqa: BLE001 - 邮件失败不能终止次日调度
        message = f"{type(exc).__name__}: {exc}"
        record_delivery(False, message, recipients)
        print(f"[radar-daily] 邮件发送失败: {message}")
    return snapshot


def daily_radar_loop(stop_event: threading.Event | None = None,
                     markets: list[str] | None = None):
    """常驻调度：每天 Asia/Shanghai 09:00 执行一次完整扫描和邮件投递。"""
    stop_event = stop_event or threading.Event()
    print("[radar-daily] 调度已启动: 每天 09:00 (Asia/Shanghai) 扫描并发送邮件")
    while not stop_event.is_set():
        target = next_run_at()
        wait_seconds = max(0.0, (target - datetime.now(SCHEDULE_TIMEZONE)).total_seconds())
        if stop_event.wait(wait_seconds):
            return
        try:
            run_daily_radar(markets=markets)
        except Exception as exc:  # noqa: BLE001 - 调度循环自愈
            message = f"{type(exc).__name__}: {exc}"
            record_delivery(False, message, get_email_settings()["recipients"])
            print(f"[radar-daily] 当日任务失败: {message}")
