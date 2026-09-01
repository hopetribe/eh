# -*- coding: utf-8 -*-
"""HTTP API 边界和回测配置测试。"""
import http.client
import json
import math
import tempfile
import threading
from pathlib import Path

import gcn.server.app as server_app
from gcn.server.app import (
    MAX_BODY, MAX_SYMBOLS, AppServer, UiHandler, _dumps_json, _parse_markets,
)


def _serve_request(method, path, body=b"", headers=None, raw_length=None,
                   bound_host="127.0.0.1", allowed_hosts=None):
    server = AppServer(("127.0.0.1", 0), UiHandler)
    server.bound_host = bound_host
    server.allowed_hosts = allowed_hosts
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    conn = http.client.HTTPConnection("127.0.0.1", server.server_port, timeout=5)
    try:
        if raw_length is None:
            conn.request(method, path, body=body, headers=headers or {})
        else:
            conn.putrequest(method, path)
            for key, value in (headers or {}).items():
                conn.putheader(key, value)
            conn.putheader("Content-Length", str(raw_length))
            conn.endheaders(body)
        response = conn.getresponse()
        payload = response.read()
        return response.status, response.getheader("Content-Type"), payload
    finally:
        conn.close()
        server.shutdown()
        server.server_close()
        thread.join(timeout=2)


def test_server_accepts_explicit_proxy_host_on_private_binding():
    status, _, payload = _serve_request(
        "GET", "/",
        headers={
            "Host": "43.160.201.247:8443",
            "Origin": "https://43.160.201.247:8443",
        },
        bound_host="172.17.0.1",
        allowed_hosts={"43.160.201.247"},
    )
    assert status == 200, payload


def test_server_rejects_bad_length_type_origin_and_nonfinite_input():
    status, _, _ = _serve_request(
        "POST", "/api/compute", headers={"Content-Type": "application/json"},
        raw_length=-1,
    )
    assert status == 400

    status, _, _ = _serve_request(
        "POST", "/api/compute", headers={"Content-Type": "application/json"},
        raw_length=MAX_BODY + 1,
    )
    assert status == 413

    status, _, _ = _serve_request("POST", "/api/compute", body=b"{}")
    assert status == 415

    status, _, _ = _serve_request(
        "POST", "/api/compute", body=b"{}",
        headers={"Content-Type": "application/json", "Origin": "https://evil.example"},
    )
    assert status == 403

    body = json.dumps({"source": "sample", "cost": "NaN"}).encode()
    status, _, payload = _serve_request(
        "POST", "/api/backtest", body=body,
        headers={"Content-Type": "application/json"},
    )
    assert status == 400, payload

    for bad_row in (
        ["2025-01-02", "inf", 12, 9, 11, 1000],
        ["2025-01-02", 0, 12, 9, 11, 1000],
        ["2025-01-02", 10, 12, 9, 11, -1],
        ["2025-01-02", 10, 10, 9, 11, 1000],
    ):
        body = json.dumps({"rows": [bad_row]}).encode()
        status, _, payload = _serve_request(
            "POST", "/api/compute", body=body,
            headers={"Content-Type": "application/json"},
        )
        assert status == 400, payload


def test_server_backtest_preserves_zero_cost_and_interval():
    body = json.dumps({
        "source": "sample", "seed": 7, "cost": 0,
        "interval": "1wk", "years": 0.5,
    }).encode()
    status, ctype, payload = _serve_request(
        "POST", "/api/backtest", body=body,
        headers={"Content-Type": "application/json"},
    )
    assert status == 200, payload
    assert ctype.startswith("application/json")
    data = json.loads(payload)
    assert data["config"]["cost"] == 0
    assert data["config"]["interval"] == "1wk"
    assert data["config"]["years"] == 0.5
    assert data["timeframe"]["period_label"] == "周"


def test_server_rows_source_never_falls_back_to_sample():
    for path in ("/api/compute", "/api/backtest"):
        for body_obj in ({"source": "rows", "rows": []}, {"rows": []}):
            status, _, payload = _serve_request(
                "POST", path, body=json.dumps(body_obj).encode(),
                headers={"Content-Type": "application/json"},
            )
            assert status == 400, payload

    status, _, payload = _serve_request(
        "POST", "/api/compute",
        body=json.dumps({"source": "unknown"}).encode(),
        headers={"Content-Type": "application/json"},
    )
    assert status == 400, payload


def test_server_parse_csv_endpoint_serializes_rows():
    csv_text = ("date,open,high,low,close,volume\n"
                "2025-01-02,10,12,9,11,1000\n")
    body = json.dumps({"csv": csv_text}).encode()
    status, _, payload = _serve_request(
        "POST", "/api/parse_csv", body=body,
        headers={"Content-Type": "application/json"},
    )
    assert status == 200, payload
    data = json.loads(payload)
    assert data["rows"] == [["2025-01-02", 10.0, 12.0, 9.0, 11.0, 1000.0]]


def test_server_parses_comma_markets_and_enforces_symbol_limit():
    assert _parse_markets("market=us,hk") == ["us", "hk"]
    assert _parse_markets("market=cn&market=us") == ["us", "cn"]
    symbols = ",".join(f"S{i}" for i in range(MAX_SYMBOLS + 1))
    body = json.dumps({"strategy": "graham", "symbols": symbols}).encode()
    status, _, payload = _serve_request(
        "POST", "/api/screener", body=body,
        headers={"Content-Type": "application/json"},
    )
    assert status == 400, payload


def test_json_serialization_rejects_nonfinite_numbers():
    try:
        _dumps_json({"bad": math.nan})
    except ValueError:
        pass
    else:
        raise AssertionError("NaN 不应被编码为非标准 JSON")


def test_server_classifies_upstream_and_internal_errors_without_details():
    body = json.dumps({"symbol": "TQQQ", "interval": "1d"}).encode()
    original = server_app.fetch_quote
    try:
        server_app.fetch_quote = lambda *args, **kwargs: (_ for _ in ()).throw(
            RuntimeError("upstream secret"))
        status, _, payload = _serve_request(
            "POST", "/api/fetch", body=body,
            headers={"Content-Type": "application/json"},
        )
        assert status == 502 and b"upstream secret" not in payload

        server_app.fetch_quote = lambda *args, **kwargs: (_ for _ in ()).throw(
            KeyError("internal secret"))
        status, _, payload = _serve_request(
            "POST", "/api/fetch", body=body,
            headers={"Content-Type": "application/json"},
        )
        assert status == 500 and b"internal secret" not in payload
    finally:
        server_app.fetch_quote = original


def test_radar_email_settings_api_adds_recipient_without_exposing_credentials():
    original_dir = server_app.radar_emailer.DATA_DIR
    try:
        with tempfile.TemporaryDirectory() as tmp:
            server_app.radar_emailer.DATA_DIR = Path(tmp)
            status, _, payload = _serve_request("GET", "/api/radar/email")
            assert status == 200, payload
            data = json.loads(payload)
            assert data["recipients"] == ["hopetribe@gmail.com"]
            assert "password" not in data

            body = json.dumps({"action": "add", "email": "extra@example.com"}).encode()
            status, _, payload = _serve_request(
                "POST", "/api/radar/email", body=body,
                headers={"Content-Type": "application/json"})
            assert status == 200, payload
            assert json.loads(payload)["recipients"] == [
                "hopetribe@gmail.com", "extra@example.com"]
    finally:
        server_app.radar_emailer.DATA_DIR = original_dir


def test_radar_snapshot_get_does_not_start_background_scan():
    class SnapshotOnlyService:
        def snapshot(self, markets):
            return {"markets": {market: {"job": {"status": "idle"},
                                         "cache": None, "scanning": False,
                                         "stale": True}
                                for market in markets}}
        def ensure_fresh(self, markets):
            raise AssertionError("GET /api/radar 不应触发扫描")

    original = server_app.radar_engine.SERVICE
    try:
        server_app.radar_engine.SERVICE = SnapshotOnlyService()
        status, _, payload = _serve_request("GET", "/api/radar?market=us")
        assert status == 200, payload
        assert list(json.loads(payload)["markets"]) == ["us"]
    finally:
        server_app.radar_engine.SERVICE = original
