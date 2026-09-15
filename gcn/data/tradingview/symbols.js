"use strict";

function bareSymbol(value) {
  const normalized = String(value).trim().toUpperCase();
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
