"use strict";

function bareSymbol(value) {
  const normalized = String(value).trim().toUpperCase();
  // 雷达使用五位港股代码；TradingView 搜索使用不带前导零的代码。
  // 六位 A 股代码（如 000001）必须保持原样。
  const hk = normalized.match(/^(\d{1,5})(?:\.HK)?$/);
  if (hk) return String(Number.parseInt(hk[1], 10));
  const match = normalized.match(/^(US|HK|SH|SZ)\.(.+)$/);
  if (!match) return normalized.replace(/\.(HK|SS|SZ)$/, "");
  const [, market, code] = match;
  return market === "HK" ? String(Number.parseInt(code, 10)) : code;
}

function matchesSymbol(candidate, value) {
  const expected = bareSymbol(value);
  const actual = String(candidate.symbol || "").toUpperCase();
  return actual === expected || (expected.match(/^\d+$/)
    && String(Number.parseInt(actual, 10)) === String(Number.parseInt(expected, 10)));
}

module.exports = { bareSymbol, matchesSymbol };
