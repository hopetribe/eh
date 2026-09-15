#!/usr/bin/env node
"use strict";

const TradingView = require("@mathieuc/tradingview");
const { bareSymbol, matchesSymbol } = require("./symbols");

const TIMEFRAMES = {
  "1d": "D",
  "1wk": "W",
  "60m": "60",
  "15m": "15",
  "5m": "5",
};

const [symbol, interval, rawCount] = process.argv.slice(2);
const count = Number.parseInt(rawCount, 10);

if (!symbol || !TIMEFRAMES[interval] || !Number.isSafeInteger(count) || count < 1) {
  process.stderr.write("usage: fetch.js SYMBOL INTERVAL COUNT\n");
  process.exit(2);
}

async function resolveMarket(value) {
  if (String(value).includes(":")) return String(value).toUpperCase();
  const candidates = await TradingView.searchMarketV3(bareSymbol(value), "stock");
  const exact = candidates.find((candidate) => matchesSymbol(candidate, value));
  if (!exact) throw new Error(`TradingView 未找到股票代码 ${value}`);
  return exact.id;
}

let client;
let chart;
let settled = false;
let deadline;

function close() {
  if (chart) chart.delete();
  if (client) client.end();
}

function finish(error, payload) {
  if (settled) return;
  settled = true;
  clearTimeout(deadline);
  close();
  if (error) {
    process.stderr.write(`${error.message || error}\n`);
    setTimeout(() => process.exit(1), 0);
    return;
  }
  process.stdout.write(`${JSON.stringify(payload)}\n`, () => process.exit(0));
}

deadline = setTimeout(() => finish(new Error("TradingView 请求超时")), 18000);

(async () => {
  const market = await resolveMarket(symbol);
  client = new TradingView.Client();
  chart = new client.Session.Chart();
  client.onError((...messages) => finish(new Error(messages.join(" "))));
  chart.onError((...messages) => finish(new Error(messages.join(" "))));
  chart.onUpdate(() => {
    const periods = chart.periods
      .map((period) => [period.time, period.open, period.max, period.min,
        period.close, period.volume])
      .filter((period) => period.every((value) => Number.isFinite(value)))
      .sort((left, right) => left[0] - right[0])
      .slice(-count);
    if (!periods.length) return;
    finish(null, { market, timezone: chart.infos.timezone || "UTC", periods });
  });
  chart.setMarket(market, { timeframe: TIMEFRAMES[interval], range: count });
})().catch((error) => finish(error));
