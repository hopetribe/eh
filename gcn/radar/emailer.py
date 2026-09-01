# -*- coding: utf-8 -*-
"""机会雷达邮件配置、报告生成与 SMTP 投递。"""
from __future__ import annotations

import html
import json
import os
import re
import smtplib
import threading
import time
from datetime import datetime
from email.message import EmailMessage
from pathlib import Path
from zoneinfo import ZoneInfo

from gcn.data.service import DATA_DIR, _atomic_write_text

DEFAULT_RECIPIENT = "hopetribe@gmail.com"
SETTINGS_FILE = "radar_email_settings.json"
MARKET_NAMES = {"us": "美股", "hk": "港股", "cn": "A股"}
EMAIL_WINDOW = 5
REPORT_TIMEZONE = ZoneInfo("Asia/Shanghai")
_EMAIL_RE = re.compile(r"^[^\s@]{1,64}@[^\s@]{1,190}\.[^\s@]{2,63}$")
_settings_lock = threading.Lock()


def _settings_path() -> Path:
    return DATA_DIR / SETTINGS_FILE


def _normalize_email(value: str) -> str:
    email = str(value or "").strip().lower()
    if len(email) > 254 or not _EMAIL_RE.fullmatch(email):
        raise ValueError("邮箱地址格式无效")
    return email


def _read_local_settings() -> dict:
    try:
        value = json.loads(_settings_path().read_text(encoding="utf-8"))
        return value if isinstance(value, dict) else {}
    except Exception:  # noqa: BLE001 - 首次启动/损坏配置均回到安全默认值
        return {}


def _write_local_settings(value: dict):
    DATA_DIR.mkdir(exist_ok=True)
    _atomic_write_text(_settings_path(), json.dumps(value, ensure_ascii=False, indent=2))


def _smtp_config() -> dict:
    host = (os.getenv("GCN_SMTP_HOST") or "smtp.gmail.com").strip()
    user = (os.getenv("GCN_SMTP_USER") or "").strip()
    password = os.getenv("GCN_SMTP_PASSWORD") or ""
    sender = (os.getenv("GCN_SMTP_FROM") or user).strip()
    security = (os.getenv("GCN_SMTP_SECURITY") or "ssl").strip().lower()
    if security not in {"ssl", "starttls", "none"}:
        security = "ssl"
    try:
        port = int(os.getenv("GCN_SMTP_PORT") or (465 if security == "ssl" else 587))
    except ValueError:
        port = 465 if security == "ssl" else 587
    configured = bool(host and user and password and sender and 0 < port < 65536)
    return {"host": host, "port": port, "user": user, "password": password,
            "sender": sender, "security": security, "configured": configured}


def get_email_settings() -> dict:
    """返回可公开给本地 UI 的邮件设置，绝不暴露 SMTP 密码。"""
    with _settings_lock:
        local = _read_local_settings()
    extras = []
    for raw in local.get("additional_recipients", []):
        try:
            email = _normalize_email(raw)
        except ValueError:
            continue
        if email != DEFAULT_RECIPIENT and email not in extras:
            extras.append(email)
    smtp = _smtp_config()
    return {
        "default_recipient": DEFAULT_RECIPIENT,
        "recipients": [DEFAULT_RECIPIENT, *extras],
        "additional_recipients": extras,
        "schedule": "09:00",
        "timezone": "Asia/Shanghai",
        "smtp_configured": smtp["configured"],
        "smtp_sender": smtp["sender"] or None,
        "last_delivery": local.get("last_delivery"),
    }


def update_recipient(action: str, value: str) -> dict:
    """添加或移除附加收件人；默认收件人固定保留。"""
    email = _normalize_email(value)
    if action not in {"add", "remove"}:
        raise ValueError("未知邮箱配置操作")
    if action == "remove" and email == DEFAULT_RECIPIENT:
        raise ValueError("默认收件邮箱不能移除")
    with _settings_lock:
        local = _read_local_settings()
        extras = []
        for raw in local.get("additional_recipients", []):
            try:
                item = _normalize_email(raw)
            except ValueError:
                continue
            if item != DEFAULT_RECIPIENT and item not in extras:
                extras.append(item)
        if action == "add" and email != DEFAULT_RECIPIENT and email not in extras:
            extras.append(email)
        if action == "remove":
            extras = [item for item in extras if item != email]
        local["additional_recipients"] = extras
        _write_local_settings(local)
    return get_email_settings()


def record_delivery(ok: bool, message: str, recipients: list[str] | None = None):
    """持久化最近一次投递状态，便于 UI 明确展示成功或失败。"""
    with _settings_lock:
        local = _read_local_settings()
        local["last_delivery"] = {
            "ok": bool(ok), "message": str(message), "at": time.time(),
            "recipients": list(recipients or []),
        }
        _write_local_settings(local)


def _market_rows(snapshot: dict, window: int = EMAIL_WINDOW) -> list[dict]:
    rows = []
    for market in ("us", "hk", "cn"):
        state = (snapshot.get("markets") or {}).get(market) or {}
        cache = state.get("cache") or {}
        for item in cache.get("results") or []:
            signals = [sig for sig in item.get("signals") or []
                       if int(sig.get("days_ago", 10**9)) < window]
            if signals:
                rows.append({"market": market, **item, "signals": signals})
    rows.sort(key=lambda item: (
        min(int(sig.get("days_ago", 10**9)) for sig in item["signals"]),
        item["market"], item.get("code", "")))
    return rows


def build_report(snapshot: dict, now=None,
                 window: int = EMAIL_WINDOW) -> tuple[str, str, str]:
    """生成主题、纯文本和 HTML 双格式日报。"""
    stamp = datetime.fromtimestamp(
        now if now is not None else time.time(), tz=REPORT_TIMEZONE)
    day = stamp.strftime("%Y-%m-%d")
    rows = _market_rows(snapshot, window=window)
    subject = f"[GCN机会雷达] {day} 扫描结果 · {len(rows)} 个标的命中"
    lines = [f"GCN 机会雷达 · {day}", f"窗口：近 {window} 个交易日", ""]
    summaries = []
    for market in ("us", "hk", "cn"):
        state = (snapshot.get("markets") or {}).get(market) or {}
        cache = state.get("cache") or {}
        job = state.get("job") or {}
        summaries.append(
            f"{MARKET_NAMES[market]}：扫描 {cache.get('n_scanned', 0)}，"
            f"命中 {cache.get('n_hits', 0)}，失败 {cache.get('n_errors', 0)}，"
            f"状态 {job.get('status', 'unknown')}"
        )
    lines.extend(summaries)
    lines.extend(["", "近期信号："])
    if not rows:
        lines.append("无 B买 / 绝反 信号。")
    for item in rows:
        sigs = "；".join(f"{sig.get('type', '--')} {sig.get('date', '--')}"
                         for sig in item["signals"])
        lines.append(
            f"{MARKET_NAMES[item['market']]} {item.get('code', '--')} "
            f"{item.get('name', '')} | 收盘 {item.get('close', '--')} | {sigs}"
        )
    text_body = "\n".join(lines)

    summary_html = "".join(f"<li>{html.escape(line)}</li>" for line in summaries)
    if rows:
        table_rows = []
        for item in rows:
            sigs = "<br>".join(
                f"{html.escape(str(sig.get('type', '--')))} "
                f"<span style=\"color:#64748b\">{html.escape(str(sig.get('date', '--')))}</span>"
                for sig in item["signals"])
            table_rows.append(
                "<tr>"
                f"<td>{html.escape(MARKET_NAMES[item['market']])}</td>"
                f"<td>{html.escape(str(item.get('code', '--')))}</td>"
                f"<td>{html.escape(str(item.get('name', '') or '--'))}</td>"
                f"<td style=\"text-align:right\">{html.escape(str(item.get('close', '--')))}</td>"
                f"<td>{sigs}</td></tr>"
            )
        results_html = (
            "<table style=\"width:100%;border-collapse:collapse;font-size:13px\">"
            "<thead><tr><th>市场</th><th>代码</th><th>名称</th><th>收盘</th><th>信号</th></tr></thead>"
            f"<tbody>{''.join(table_rows)}</tbody></table>"
        )
    else:
        results_html = "<p>近期开窗内无 B买 / 绝反 信号。</p>"
    html_body = (
        "<!doctype html><html><body style=\"font-family:-apple-system,BlinkMacSystemFont,'Segoe UI',sans-serif;"
        "color:#17212d\">"
        f"<h2 style=\"font-size:18px\">GCN 机会雷达 · {day}</h2>"
        f"<p>窗口：近 {window} 个交易日</p><ul>{summary_html}</ul>{results_html}"
        "<p style=\"color:#64748b;font-size:12px\">信号为 EHOPT10 v4 口径，T 日收盘确认。</p>"
        "</body></html>"
    )
    return subject, text_body, html_body


def send_radar_email(snapshot: dict, recipients: list[str] | None = None,
                     smtp_factory=None) -> list[str]:
    """通过 SMTP 投递雷达日报；失败抛出异常，由调度器记录并重试到下一日。"""
    recipients = recipients or get_email_settings()["recipients"]
    recipients = list(dict.fromkeys(_normalize_email(item) for item in recipients))
    if not recipients:
        raise RuntimeError("没有可用的雷达邮件收件人")
    config = _smtp_config()
    if not config["configured"]:
        raise RuntimeError("SMTP 未配置：请设置 GCN_SMTP_USER 和 GCN_SMTP_PASSWORD")
    subject, text_body, html_body = build_report(snapshot)
    message = EmailMessage()
    message["Subject"] = subject
    message["From"] = config["sender"]
    message["To"] = ", ".join(recipients)
    message.set_content(text_body)
    message.add_alternative(html_body, subtype="html")

    factory = smtp_factory or (
        smtplib.SMTP_SSL if config["security"] == "ssl" else smtplib.SMTP)
    with factory(config["host"], config["port"], timeout=30) as client:
        if config["security"] == "starttls":
            client.starttls()
        client.login(config["user"], config["password"])
        client.send_message(message)
    return recipients
