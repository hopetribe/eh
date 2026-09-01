# -*- coding: utf-8 -*-
"""Static regression checks for the dependency-free Web UI.

Rendered behavior is covered by browser smoke tests; these checks keep the
security, request-lifecycle, and accessibility contracts from regressing in the
plain HTML/CSS source.
"""
from __future__ import annotations

import re
import subprocess
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
INDEX = (ROOT / "webui" / "index.html").read_text(encoding="utf-8")
STYLES = (ROOT / "webui" / "styles.css").read_text(encoding="utf-8")


def _inline_script() -> str:
    scripts = re.findall(r"<script(?:\s[^>]*)?>(.*?)</script>", INDEX, re.S | re.I)
    return next(script for script in reversed(scripts) if script.strip())


def test_webui_inline_javascript_parses():
    proc = subprocess.run(
        ["node", "-e", "new Function(require('fs').readFileSync(0, 'utf8'))"],
        input=_inline_script(), text=True, capture_output=True, check=False,
    )
    assert proc.returncode == 0, proc.stderr


def test_webui_external_values_are_escaped_or_built_as_text_nodes():
    script = _inline_script()
    assert "drop.replaceChildren" in script
    assert "drop.innerHTML = list.length" not in script
    assert 'select.className = "s watch-select"' in script
    assert 'select.setAttribute("aria-label", `加载行情 ${symbol}`)' in script
    assert 'item.addEventListener("click"' not in script
    for expression in ("esc(r.note)", "esc(c.text)", "esc(c.value)",
                       "esc(c.threshold)", "esc(c.note)", "esc(s.type)",
                       "esc(s.date)", "esc(r.date)"):
        assert expression in script


def test_webui_requests_are_cancelled_and_zero_cost_is_preserved():
    script = _inline_script()
    assert "new AbortController()" in script
    assert "signal: request.signal" in script
    assert "readFiniteNumber(\"#btCost\", 0.1)" in script
    assert 'interval: state.dataInterval' in script
    assert 'parseFloat($("#btCost").value) || 0.1' not in script
    assert 'cancelRequest("backtest")' in script
    assert 'cancelRequest("screener")' in script
    compare = script[script.index("async function runCompare"):
                     script.index("function renderCompare")]
    assert "request.controller.abort()" in compare


def test_webui_mobile_drawer_and_controls_remain_accessible():
    script = _inline_script()
    assert 'aria-label="K线周期"' in INDEX
    assert 'side.toggleAttribute("inert"' in script
    assert 'main.toggleAttribute("inert"' in script
    assert 'side.setAttribute("aria-hidden"' in script
    assert "side.focus()" in script
    assert 'document.body.classList.contains("sidebar-open")' in script
    assert '!dialogReturnFocus.closest("[inert]")' in script
    for label in ("SD 数值", "SD 滑块", "WIDTH 数值", "WIDTH 滑块",
                  "N 数值", "N 滑块", "OFFSET 数值", "OFFSET 滑块",
                  "选股策略", "选股市场", "自定义候选池", "回测周期",
                  "单边成本百分比", "最长持有K线"):
        assert f'aria-label="{label}"' in INDEX
    assert 'id="scrStatus" role="status"' in INDEX
    assert 'id="sigList" class="signal-scroll" role="list"' not in INDEX
    sr_only = re.search(r"\.sr-only\s*\{([^}]+)\}", STYLES, re.S)
    assert sr_only and "display: none" not in sr_only.group(1)


def test_webui_legend_is_horizontally_scrollable():
    legend = re.search(r"\.legend-groups\s*\{([^}]+)\}", STYLES, re.S)
    assert legend
    rules = legend.group(1)
    assert "overflow-x: auto" in rules
    assert "overflow-y: hidden" in rules


def test_webui_compute_failure_clears_quote_loading_state():
    script = _inline_script()
    assert 'setDataStatus(`指标计算失败' in script
    assert 'setDataStatus("指标服务暂不可用", "error")' in script
    assert "clearTimeout(toastTimer)" in script
    assert 'toast("");' in script
    assert script.count('setDataStatus(`指标计算失败') == 1


def test_webui_nested_loading_restores_ready_state_without_duplicate_metrics():
    script = _inline_script()
    set_loading = script[script.index("function setLoading("):
                         script.index("function setDataStatus(")]
    assert '!status.classList.contains("is-error")' in set_loading
    assert 'status.innerHTML = "<i></i>数据就绪"' in set_loading

    metrics_start = script.index("const metrics = [")
    metrics = script[metrics_start:script.index("];", metrics_start)]
    assert metrics.count('["九转 上 / 下"') == 1


def test_webui_event_study_uses_excess_return_and_validates_async_payloads():
    script = _inline_script()
    assert "胜率/超额" in script
    assert "const excess = st.excess" in script
    assert "const formatHorizon" in script
    assert "row.split || row.split5" in script
    assert "st.excess" in script
    assert "n=${st.n}" in script
    assert 'if (data.error || !data.markets)' in script
    meta_parse = script.index("const data = await r.json()", script.index('fetch("/api/screener/meta"'))
    meta_freshness = script.index("if (!isCurrentRequest(request)) return", meta_parse)
    meta_schema = script.index("Array.isArray(data.strategies)", meta_freshness)
    meta_commit = script.index("scrMeta = data", meta_schema)
    assert meta_parse < meta_freshness < meta_schema < meta_commit


def test_webui_data_source_identity_is_committed_atomically():
    script = _inline_script()
    assert "dataInterval:" in script
    assert "let dataSourceEpoch" in script
    assert "function beginDataSourceChange" in script
    assert "function normalizeRowsPayload" in script
    assert "function commitDataSource" in script
    assert "function clearDerivedView" in script
    assert "function validateComputePayload" in script
    assert "activeCsvReader.abort()" in script
    assert "expectedEpoch !== dataSourceEpoch" in script
    assert 'INTERVAL_LABEL[state.dataInterval]' in script
    quote_fetch = script.index('fetch("/api/fetch"')
    quote_validation = script.index("normalizeRowsPayload(d.rows)", quote_fetch)
    quote_commit = script.index("commitDataSource(expectedEpoch", quote_validation)
    assert quote_validation < script.index("d.interval !== interval", quote_fetch) < quote_commit
    assert script.index("!r.ok", quote_fetch) < quote_validation

    csv_fetch = script.index('fetch("/api/parse_csv"')
    csv_validation = script.index("normalizeRowsPayload(d.rows)", csv_fetch)
    csv_commit = script.index("commitDataSource(expectedEpoch", csv_validation)
    assert script.index('d.source !== "csv"', csv_fetch) < csv_validation < csv_commit

    commit = script[script.index("function commitDataSource"):
                    script.index("function syncLegendButtons")]
    assert commit.index("invalidateReports") < commit.index("Object.assign")
    assert "clearDerivedView" in commit
    assert "updateBacktestTitle" in commit
    assert "if (!state.data) return" in script
    assert "hadCommittedSource" in script
    assert "const restored = await recompute(expectedEpoch)" in script

    compute_fetch = script.index('fetch("/api/compute"')
    compute_validation = script.index("validateComputePayload(data)", compute_fetch)
    compute_commit = script.index("state.data = candidateData", compute_validation)
    assert script.index("!r.ok", compute_fetch) < compute_validation < compute_commit
    assert "state.data = previousData" in script
    assert "clearTimeout(locateTimer)" in script
    assert "state.data !== locatedData" in script


def test_webui_dialogs_and_report_state_are_accessible_and_atomic():
    script = _inline_script()
    for panel, title in (("scrPanel", "scrDialogTitle"),
                         ("radPanel", "radDialogTitle"),
                         ("btPanel", "btDialogTitle")):
        assert f'id="{panel}" role="dialog" aria-modal="true" aria-labelledby="{title}"' in INDEX
    assert "function openDialog" in script and "function closeDialog" in script
    assert "activeDialog?.overlay === overlay && activeDialog.panel === panel" in script
    assert 'app.toggleAttribute("inert", true)' in script
    assert 'event.key === "Tab"' in script
    assert 'event.key === "Escape" && activeDialog' in script
    assert "const reportEpoch" in script or "let reportEpoch" in script
    assert "clearBacktestResults" in script
    assert 'setBacktestError("回测失败' in script
    assert 'class="rad-symbol" type="button"' in script
    assert "closeDialog(radOverlay, false);" in script
    assert "symbolInput.focus()" in script
    assert "focusedKey" in script
    assert 'id="radMarket" aria-label="雷达市场"' in INDEX
    assert 'id="radWindow" aria-label="信号时间窗口"' in INDEX
    assert 'id="radStatus" role="status"' in INDEX


def test_webui_radar_email_settings_are_accessible_and_escaped():
    script = _inline_script()
    assert 'id="radEmailForm"' in INDEX
    assert 'id="radEmailInput" type="email"' in INDEX
    assert 'for="radEmailInput"' in INDEX
    assert 'id="radDelivery" role="status"' in INDEX
    assert 'fetch("/api/radar/email"' in script
    assert "validateRadarEmailSettings" in script
    assert '${esc(email)}' in script
    assert 'input.reportValidity()' in script
    assert 'updateRadarRecipient("remove", button.dataset.email)' in script
    assert "A股/港股 &gt; 100亿" in INDEX


def test_webui_screener_meta_requires_semantically_valid_nonempty_strategies():
    script = _inline_script()
    meta = script[script.index('fetch("/api/screener/meta"'):
                  script.index("scrMeta = data")]
    assert "!data.strategies.length" in meta
    assert "strategy.id.trim()" in meta
    assert "strategy.name.trim()" in meta
    assert "!strategyIds.has(id)" in meta
    assert "id === strategy.id" in meta
    assert "name === strategy.name" in meta
    assert "Number.isInteger(strategy.n_conditions)" in meta
    assert "Number.isFinite(strategy.min_mktcap_cny)" in meta
