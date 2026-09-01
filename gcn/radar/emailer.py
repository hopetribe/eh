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

    summary_cards = []
    for market in ("us", "hk", "cn"):
        state = (snapshot.get("markets") or {}).get(market) or {}
        cache = state.get("cache") or {}
        job = state.get("job") or {}
        summary_cards.append(
            "<td class=\"summary-cell\" style=\"width:33.333%;padding:4px;vertical-align:top\">"
            "<div class=\"summary-card\" style=\"padding:12px;background:#f8fafc;border:1px solid #e2e8f0;"
            "border-radius:8px\">"
            f"<div style=\"margin-bottom:8px;color:#334155;font-size:12px;font-weight:700\">{html.escape(MARKET_NAMES[market])}</div>"
            f"<div style=\"color:#64748b;font-size:11px;line-height:1.7\">扫描 {cache.get('n_scanned', 0)} · "
            f"命中 <strong style=\"color:#dc2626\">{cache.get('n_hits', 0)}</strong> · "
            f"失败 {cache.get('n_errors', 0)}<br>状态 {html.escape(str(job.get('status', 'unknown')))}</div>"
            "</div></td>"
        )

    if rows:
        table_rows = []
        for item in rows:
            signal_bits = []
            for sig in item["signals"]:
                signal_type = html.escape(str(sig.get("type", "--")))
                tone = "signal-buy" if signal_type == "B买" else "signal-reverse"
                signal_bits.append(
                    f"<span class=\"signal-pill {tone}\" style=\"display:inline-block;margin:0 4px 4px 0;"
                    "padding:3px 7px;border-radius:999px;font-size:11px;font-weight:700;line-height:1.3;"
                    f"{ 'background:#fef2f2;color:#b91c1c;border:1px solid #fecaca' if tone == 'signal-buy' else 'background:#fffbeb;color:#b45309;border:1px solid #fde68a' }\">"
                    f"{signal_type} <span style=\"font-weight:500\">{html.escape(str(sig.get('date', '--')))}</span></span>"
                )
            table_rows.append(
                "<tr class=\"signal-row\">"
                "<td class=\"market-cell\" style=\"padding:11px 10px;border-bottom:1px solid #e2e8f0;color:#475569;font-size:12px;white-space:nowrap\">"
                "<span class=\"cell-label\">市场</span>"
                f"{html.escape(MARKET_NAMES[item['market']])}</td>"
                "<td style=\"padding:11px 10px;border-bottom:1px solid #e2e8f0;color:#0f172a;font-size:13px;font-weight:700;white-space:nowrap\">"
                "<span class=\"cell-label\">代码</span>"
                f"{html.escape(str(item.get('code', '--')))}</td>"
                "<td style=\"padding:11px 10px;border-bottom:1px solid #e2e8f0;color:#334155;font-size:13px\">"
                "<span class=\"cell-label\">名称</span>"
                f"{html.escape(str(item.get('name', '') or '--'))}</td>"
                "<td style=\"padding:11px 10px;border-bottom:1px solid #e2e8f0;color:#0f172a;font-size:13px;text-align:right;white-space:nowrap\">"
                "<span class=\"cell-label\">收盘</span>"
                f"{html.escape(str(item.get('close', '--')))}</td>"
                "<td class=\"signal-cell\" style=\"padding:8px 10px;border-bottom:1px solid #e2e8f0;line-height:1.4\">"
                "<span class=\"cell-label\">信号</span>"
                f"{''.join(signal_bits)}</td></tr>"
            )
        results_html = (
            "<table class=\"signal-table\" role=\"presentation\" width=\"100%\" cellpadding=\"0\" cellspacing=\"0\" "
            "style=\"width:100%;border-collapse:separate;border-spacing:0;border:1px solid #e2e8f0;border-radius:10px;overflow:hidden\">"
            "<thead><tr style=\"background:#f8fafc\">"
            "<th align=\"left\" style=\"padding:10px;color:#64748b;font-size:11px;font-weight:700\">市场</th>"
            "<th align=\"left\" style=\"padding:10px;color:#64748b;font-size:11px;font-weight:700\">代码</th>"
            "<th align=\"left\" style=\"padding:10px;color:#64748b;font-size:11px;font-weight:700\">名称</th>"
            "<th align=\"right\" style=\"padding:10px;color:#64748b;font-size:11px;font-weight:700\">收盘</th>"
            "<th align=\"left\" style=\"padding:10px;color:#64748b;font-size:11px;font-weight:700\">信号</th>"
            "</tr></thead>"
            f"<tbody>{''.join(table_rows)}</tbody></table>"
        )
    else:
        results_html = (
            "<div style=\"padding:22px 16px;color:#64748b;font-size:13px;text-align:center;background:#f8fafc;"
            "border:1px solid #e2e8f0;border-radius:10px\">近期开窗内无 B买 / 绝反 信号。</div>"
        )
    html_body = (
        "<!doctype html><html lang=\"zh-CN\"><head>"
        "<meta charset=\"utf-8\"><meta name=\"viewport\" content=\"width=device-width,initial-scale=1\">"
        "<style>.cell-label{display:none!important}"
        "@media only screen and (max-width: 600px) {"
        ".email-shell{padding:0!important}.email-card{border-radius:0!important;border-left:0!important;border-right:0!important}"
        ".email-content{padding:20px 16px!important}.summary-cell{display:block!important;width:auto!important;padding:4px 0!important}"
        ".signal-table{border:0!important;border-radius:0!important}.signal-table thead{display:none!important}"
        ".signal-row{display:block!important;margin:0 0 10px!important;padding:12px!important;background:#ffffff!important;"
        "border:1px solid #e2e8f0!important;border-radius:8px!important}"
        ".signal-row td{display:block!important;width:auto!important;padding:3px 0!important;border:0!important;text-align:left!important;"
        "white-space:normal!important}.signal-row .signal-cell{padding-top:7px!important}.cell-label{display:inline-block!important;"
        "width:42px!important;color:#94a3b8!important;font-size:11px!important;font-weight:500!important}.market-cell{padding-top:0!important}"
        "}"
        "</style></head><body style=\"margin:0;padding:0;background:#f1f5f9;color:#17212d;font-family:-apple-system,BlinkMacSystemFont,'Segoe UI',sans-serif\">"
        "<table class=\"email-shell\" role=\"presentation\" width=\"100%\" cellpadding=\"0\" cellspacing=\"0\" style=\"width:100%;background:#f1f5f9\"><tr><td align=\"center\" style=\"padding:24px 12px\">"
        "<table class=\"email-card\" role=\"presentation\" width=\"680\" cellpadding=\"0\" cellspacing=\"0\" style=\"width:100%;max-width:680px;background:#ffffff;border:1px solid #dbe3ed;border-radius:12px;overflow:hidden\">"
        "<tr><td style=\"padding:22px 24px;background:#0f172a;color:#f8fafc\">"
        "<div style=\"margin-bottom:5px;font-size:11px;font-weight:700;letter-spacing:.08em;color:#93c5fd\">GCN · 机会雷达日报</div>"
        f"<div style=\"font-size:22px;font-weight:700;line-height:1.25\">{day} 扫描结果</div>"
        f"<div style=\"margin-top:7px;color:#cbd5e1;font-size:13px\">近 {window} 个交易日 · {len(rows)} 个标的命中</div>"
        "</td></tr><tr><td class=\"email-content\" style=\"padding:24px\">"
        "<div style=\"margin:0 0 10px;color:#334155;font-size:14px;font-weight:700\">市场概览</div>"
        "<table role=\"presentation\" width=\"100%\" cellpadding=\"0\" cellspacing=\"0\" style=\"width:100%;margin:0 0 24px\"><tr>"
        f"{''.join(summary_cards)}</tr></table>"
        "<div style=\"margin:0 0 10px;color:#334155;font-size:14px;font-weight:700\">近期信号</div>"
        f"{results_html}"
        "<div style=\"margin-top:20px;padding-top:14px;color:#64748b;font-size:11px;line-height:1.6;border-top:1px solid #e2e8f0\">"
        "信号为 EHOPT10 v4 口径，T 日收盘确认。该邮件为自动扫描摘要，请结合主图和风险管理规则使用。"
        "</div></td></tr></table></td></tr></table></body></html>"
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
