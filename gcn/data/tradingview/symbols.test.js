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
  assert.equal(matchesSymbol({ id: "HKEX:700", symbol: "700" }, "HK.00700"), true);
  assert.equal(matchesSymbol({ symbol: "9988" }, "HK.00700"), false);
});

test("never resolves the same ticker on a different exchange", () => {
  assert.equal(matchesSymbol({id: "MIL:AAPL", symbol: "AAPL"}, "US.AAPL"), false);
  assert.equal(matchesSymbol({id: "HKEX:1", symbol: "1"}, "000001.SZ"), false);
  assert.equal(matchesSymbol({id: "SSE:000700", symbol: "000700"}, "00700"), false);
  assert.equal(matchesSymbol({id: "NASDAQ:AAPL", symbol: "AAPL"}, "US.AAPL"), true);
  assert.equal(matchesSymbol({id: "SZSE:000001", symbol: "000001"}, "000001.SZ"), true);
});

test("radar Hong Kong codes search without padding while A-share codes retain it", () => {
  assert.equal(bareSymbol("00005"), "5");
  assert.equal(bareSymbol("00700"), "700");
  assert.equal(bareSymbol("09988"), "9988");
  assert.equal(bareSymbol("0005.HK"), "5");
  assert.equal(bareSymbol("000001"), "000001");
  assert.equal(bareSymbol("SZ.000001"), "000001");
});
