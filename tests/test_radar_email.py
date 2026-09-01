# -*- coding: utf-8 -*-
"""机会雷达邮件与每日调度离线测试。"""
import os
import tempfile
from contextlib import contextmanager
from datetime import datetime
from pathlib import Path

from gcn.radar import emailer, scheduler


@contextmanager
def _patched(target, name, value):
    old = getattr(target, name)
    setattr(target, name, value)
    try:
        yield value
    finally:
        setattr(target, name, old)


@contextmanager
def _env(**values):
    old = {key: os.environ.get(key) for key in values}
    try:
        for key, value in values.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value
        yield
    finally:
        for key, value in old.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value


def _snapshot():
    cache = {
        "n_scanned": 321, "n_hits": 1, "n_errors": 2,
        "results": [{"code": "AAPL", "market": "us", "name": "苹果",
                     "close": 123.4,
                     "signals": [{"type": "B买", "date": "2026-08-31",
                                  "days_ago": 1}]}],
    }
    return {"markets": {
        "us": {"cache": cache, "job": {"status": "done"}, "scanning": False},
        "hk": {"cache": {"n_scanned": 0, "n_hits": 0, "n_errors": 0,
                           "results": []}, "job": {"status": "done"}, "scanning": False},
        "cn": {"cache": {"n_scanned": 0, "n_hits": 0, "n_errors": 0,
                           "results": []}, "job": {"status": "done"}, "scanning": False},
    }}


def test_email_recipients_keep_default_and_persist_extras():
    with tempfile.TemporaryDirectory() as tmp, _patched(emailer, "DATA_DIR", Path(tmp)):
        settings = emailer.get_email_settings()
        assert settings["recipients"] == ["hopetribe@gmail.com"]
        settings = emailer.update_recipient("add", "Second@Example.com")
        assert settings["recipients"] == ["hopetribe@gmail.com", "second@example.com"]
        settings = emailer.update_recipient("remove", "second@example.com")
        assert settings["recipients"] == ["hopetribe@gmail.com"]
        try:
            emailer.update_recipient("remove", "hopetribe@gmail.com")
        except ValueError as exc:
            assert "不能移除" in str(exc)
        else:
            raise AssertionError("默认收件人不应允许移除")


def test_email_report_and_smtp_delivery():
    sent = []

    class FakeSMTP:
        def __init__(self, host, port, timeout):
            assert (host, port, timeout) == ("smtp.test", 465, 30)
        def __enter__(self): return self
        def __exit__(self, *args): pass
        def login(self, user, password):
            assert (user, password) == ("sender@test.com", "secret")
        def send_message(self, message): sent.append(message)

    subject, text, html = emailer.build_report(_snapshot(), now=1788192000)
    assert "GCN机会雷达" in subject and "AAPL" in text and "AAPL" in html
    with _env(GCN_SMTP_HOST="smtp.test", GCN_SMTP_PORT="465",
              GCN_SMTP_USER="sender@test.com", GCN_SMTP_PASSWORD="secret",
              GCN_SMTP_FROM="sender@test.com", GCN_SMTP_SECURITY="ssl"):
        delivered = emailer.send_radar_email(
            _snapshot(), recipients=["hopetribe@gmail.com"], smtp_factory=FakeSMTP)
    assert delivered == ["hopetribe@gmail.com"]
    assert sent[0]["To"] == "hopetribe@gmail.com"
    assert sent[0].is_multipart()


def test_email_report_uses_mobile_responsive_signal_rows():
    _, _, html = emailer.build_report(_snapshot(), now=1788192000)
    assert 'class="email-shell"' in html
    assert '@media only screen and (max-width: 600px)' in html
    assert 'class="signal-row"' in html
    assert 'class="cell-label">市场</span>' in html


def test_email_report_hides_mobile_field_labels_on_desktop():
    _, _, html = emailer.build_report(_snapshot(), now=1788192000)
    assert ".cell-label{display:none!important}" in html


def test_next_run_is_daily_nine_in_shanghai():
    before = datetime.fromisoformat("2026-09-01T08:30:00+08:00")
    after = datetime.fromisoformat("2026-09-01T09:00:01+08:00")
    assert scheduler.next_run_at(before).isoformat() == "2026-09-01T09:00:00+08:00"
    assert scheduler.next_run_at(after).isoformat() == "2026-09-02T09:00:00+08:00"


def test_daily_run_warms_scans_waits_and_records_delivery():
    calls = []

    class Service:
        def start_scan(self, markets): calls.append(("scan", tuple(markets)))
        def snapshot(self, markets): return _snapshot()

    with _patched(scheduler.engine, "warm_market",
                  lambda market, log=False: calls.append(("warm", market))), \
         _patched(scheduler, "send_radar_email",
                  lambda snapshot, recipients=None: list(recipients)), \
         _patched(scheduler, "record_delivery",
                  lambda ok, message, recipients=None: calls.append(("delivery", ok))), \
         _patched(scheduler, "get_email_settings",
                  lambda: {"recipients": ["hopetribe@gmail.com"]}):
        snapshot = scheduler.run_daily_radar(["us"], service=Service())
    assert snapshot["markets"]["us"]["cache"]["n_scanned"] == 321
    assert calls == [("warm", "us"), ("scan", ("us",)), ("delivery", True)]
