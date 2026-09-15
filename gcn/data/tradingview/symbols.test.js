"use strict";

const test = require("node:test");
const assert = require("node:assert/strict");
const { bareSymbol, matchesSymbol } = require("./symbols");

test("normalizes KK2 market aliases for TradingView lookup", () => {
  assert.equal(bareSymbol("US.TQQQ"), "TQQQ");
  assert.equal(bareSymbol("HK.00700"), "700");
  assert.equal(bareSymbol("600519.SS"), "600519");
  assert.equal(bareSymbol("000001.SZ"), "000001");
});

test("matches zero-padded exchange symbols", () => {
  assert.equal(matchesSymbol({ symbol: "700" }, "HK.00700"), true);
  assert.equal(matchesSymbol({ symbol: "9988" }, "HK.00700"), false);
});
