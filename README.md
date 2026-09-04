# KK2 · 金筹九转 (GCN)

「金筹九转 GoldenChip NineTurn」——富途主图择时指标 **EHOPT10**（筹码×布林×九转）的
Python 工程化实现，配套信号回测、基本面选股、机会雷达与本地 Web 工作台。

- **主图指标**: 获利筹摆动 + 布林带框架 + MACD/RSI + 九转计数, 输出 B买/S卖/绝反 等图标信号
  (v3 = 原版, v4 = 绝反优化版, v4-exp = 阶段B实验版, v5 = 全量B五日MA20确认稳健版；Web UI 默认 v5，Python API 默认 v4 以保持兼容);
- **策略回测**: T+1 开盘成交口径, 信号组合对比 / 资金曲线 / 事件研究 / 分段一致性;
- **基本面选股**: 八套大师策略 (格雷厄姆/增长双验/施洛斯/巴菲特/聂夫/林奇/费雪/戴维斯双击),
  财报数据来自 Yahoo, 结构化条件逐条判定;
- **机会雷达**: 主动扫描 A股/港股**总市值 ≥ 100亿本币**、美股**总市值 ≥ 50亿美元**的全部标的,
  发现近期 **B买 / 绝反** 信号；每天 09:00 扫描并发送邮件日报;
- **Web UI**: 纯标准库 HTTP 服务 + 原生 JS + ECharts, 无需前端构建工具。

---

## 快速开始

```bash
# 依赖: Python 3.9+, numpy, pandas, yfinance
pip install -r requirements.txt

# 可选: FutuOpenD 行情源与动态股票池
pip install futu-api

# 启动 Web UI (默认端口 8642, 自动打开浏览器)
python3 -m gcn.server.app
python3 -m gcn.server.app --port 8642 --no-browser
# 仅查看/调试页面，不启动全市场缓存预热与定时雷达
python3 -m gcn.server.app --port 8642 --no-browser --no-background-jobs
```

默认仅监听回环地址。可信局域网中对外监听必须显式授权；通配地址还必须声明允许的 Host：

```bash
python3 -m gcn.server.app --host 0.0.0.0 --allow-remote \
  --allowed-host 192.168.1.20 --no-browser
```

生产环境由 Caddy 提供 HTTPS 反向代理，入口为
`https://43.160.201.247:8443/`；访问 `http://43.160.201.247/` 会保留路径并跳转至 HTTPS。
应用服务仅绑定 Docker 网桥 `172.17.0.1:8642`，不直接暴露在公网。对应配置见
`deploy/kk2.service` 和 `deploy/kk2.caddy`。由于主机标准 443 端口由既有 Xray 服务占用，
网关将公网 8443 映射到 Caddy 的 443 端口。

页面侧栏入口: **数据**(代码/周期/上传CSV) · **参数**(SD/WIDTH/N/OFFSET) · **回测** ·
**选股** · **雷达**。

离线运行全部测试:

```bash
python3 tests/run_all.py        # 全部离线测试
```

---

## 目录结构

```
gcn/
  core/         TDX 算子库 (MA/EMA/SMA/STDP/CROSS/BARSLAST…, 向量化实现) + 指标库
  recipes/      gcn_main.py — EHOPT10 配方 (v3/v4/v4-exp/v5 中间变量与信号)
  data/         数据服务: FutuOpenD→Yahoo 自动回退, 落盘缓存, 新鲜度判定, 每标的并发锁
  server/       纯标准库 HTTP 服务 (ThreadingHTTPServer) + JSON API
  backtest/     回测引擎: 信号预设组合, 资金曲线, 事件研究, 分段一致性
  screener/     基本面选股: 策略定义 / 财报指标计算 / 条件求值引擎
  radar/        机会雷达: 阈值股票池 / 扫描引擎 / 日K预热 / 每日调度 / 邮件日报
  plot.py       matplotlib 静态画图 (信号标注, 可选)
webui/          前端单页 (index.html + echarts.min.js)
tests/          离线测试 (兼容 tests/run_all.py 与 pytest)
data/           K线缓存与雷达结果缓存 (随仓库跟踪, 见下文"数据层")
docs/           选股策略条件清单 (策略图卡的结构化转写)
```

---

## 数据层

**数据源**: 本机 FutuOpenD (`127.0.0.1:11111`) 优先, 未运行或失败自动回退 Yahoo Finance。
Futu 按合法分页键拉取完整时间窗；两端 OHLC 均统一为复权口径并记录来源元数据，只有复权口径兼容
时才合并，随后过滤非有限/非法 bar、去重、按时间排序后落盘。

**落盘缓存** (`data/` 目录, 已纳入版本库, 预置三大市场数据):

| 文件 | 内容 |
|---|---|
| `<SYMBOL>_1d.csv` 等 | 各标的K线 (date, open, high, low, close, volume), 全量合并历史 |
| `<SYMBOL>_1d.csv.meta.json` | 本地来源/复权元数据与内容摘要（运行时生成，不纳入版本库） |
| `radar_<market>.json` | 雷达扫描结果缓存 (us/hk/cn, 每日或手动刷新) |
| `radar_universe_<market>.json` | Futu 市值阈值动态股票池快照 |
| `radar_email_settings.json` | 本机附加收件人与最近投递状态（不纳入版本库） |

**新鲜度规则** (避免重复在线请求): 按证券所属市场时区和常规收盘时间计算“最近已完成工作日”，
缓存最后一根日K覆盖该日才算新鲜；不再把文件 mtime 当作行情更新证据。请求历史长度大于缓存长度时
也会补拉。周线要求覆盖当前已完成交易周；日内周期 (60m/15m/5m) 始终刷新。

**并发安全**: Futu/Yahoo/裸代码别名归一到同一安全缓存键；同一标的的
读缓存→在线合并→落盘临界区同时使用进程内锁和跨进程文件锁，CSV/元数据/雷达 JSON 均以临时文件
原子替换。雷达扫描、预热守护、主图加载和 CLI 并行访问不会产生半文件或互相覆盖合并历史。

**自动维护**:

- 默认关注标的 (`DEFAULT_SYMBOLS`): 服务启动后每 6 小时检查，仅刷新陈旧或历史长度不足的缓存;
- **雷达预热守护**: 服务启动时拉起, 每小时巡检 A股/港股 ≥ 100亿本币、美股 ≥ 50亿美元的全部标的日K
  缓存, 只对陈旧/缺失标的发起增量请求；在线失败即使回退到旧缓存，也会进入 6 小时退避。
- **雷达日报**: 每天 09:00 (`Asia/Shanghai`) 先增量补齐日K、再扫描三市场，全部结束后投递一封汇总邮件。

---

## v6 前向 Shadow 运维

v6 目前只是冻结预注册的前向候选，**不是生产策略版本**；生产和 Web UI 继续使用 v5。
运维只允许使用 `gcn.backtest.shadow_cli`，不要直接调用底层 `shadow_runner`。状态目录必须位于仓库外、
只允许当前账号访问；服务器固定解释器为 `/home/eric/venvs/kk2/bin/python`。

```bash
# 1. 只读检查：校验10股相邻事务锁、CSV/meta摘要、adjusted来源、交易日与已有权威链
/home/eric/venvs/kk2/bin/python -m gcn.backtest.shadow_cli \
  --state-root /home/eric/.local/state/kk2-shadow preflight

# 2. 仅由操作员执行一次；首个cutoff后共同交易日到达前返回WAITING且不创建任何状态
/home/eric/venvs/kk2/bin/python -m gcn.backtest.shadow_cli \
  --state-root /home/eric/.local/state/kk2-shadow \
  --expected-python /home/eric/venvs/kk2/bin/python initialize

# 3. 定时任务只能调用update；未初始化时失败关闭，绝不隐式创建genesis
/home/eric/venvs/kk2/bin/python -m gcn.backtest.shadow_cli \
  --state-root /home/eric/.local/state/kk2-shadow \
  --expected-python /home/eric/venvs/kk2/bin/python update

# 4. 只从commit/generation权威链重放；只报告派生缓存问题，不写入或自动修复
/home/eric/venvs/kk2/bin/python -m gcn.backtest.shadow_cli \
  --state-root /home/eric/.local/state/kk2-shadow status
```

所有非帮助输出均为单行 canonical JSON：成功写 stdout，失败写 stderr。退出码 `0` 表示成功、
无新增或等待首个共同日，`2` 表示参数/spec配置错误，`3` 表示行情 `DATA_BLOCKED`，`4` 表示
生命周期、权限、源码、运行时或权威链异常，`1` 仅表示未预期内部错误。`initialize` 与 `update`
必须显式提供并匹配 `--expected-python`；任何 sidecar 缺失、`adjustment != adjusted`、来源不受支持或
CSV SHA-256 不一致都必须由上游行情刷新事务修复，禁止手工重算摘要绕过门禁。

---

## 机会雷达

主动发现各市场大市值标的近期的 B买 / 绝反 信号。

- **股票池** (`gcn/radar/universe.py`): 优先使用 FutuOpenD `get_stock_filter`，不可用时使用项目已有的
  yfinance Yahoo Screener；两者都设置总市值下限并分页取全量 (A股 = 沪深合并去重排序)：
  A股/港股 ≥ 100亿本币，美股 ≥ 50亿美元；同口径快照当日复用。两个动态源暂时都不可用时优先复用
  最近一次同阈值动态快照；从未成功生成动态快照时才使用各市场 100 只的静态兜底，并在 API/UI 标记
  `static-partial`（覆盖不完整）。
- **信号口径** (`gcn/radar/engine.py`): 对每只标的日K计算 EHOPT10 v4,
  提取 `B_SIGNAL` (B买) 与 `ICON_JUEFAN` (绝反); 每标的记录**最近 15 根K线**内的全部信号
  (类型/日期/相对当前交易日的天数/信号日收盘), 前端可本地切换 近3日 / 近1周 (默认) / 近2周 窗口,
  切换无需重新扫描。
- **结果缓存**: `data/radar_<market>.json`；打开雷达面板只读取快照，不隐式触发大范围扫描。
  扫描由每日 09:00 调度或“重新扫描”按钮发起。K线缓存命中时为纯本地指标计算；全市场全部失败时保留上一版；
  局部失败时保留并标记失败标的的上一版命中。
- **邮件日报**: 默认收件人 `hopetribe@gmail.com` 固定保留；雷达抽屉可添加/移除其他收件人并查看最近投递状态。
  邮件同时包含纯文本与 HTML，汇总各市场扫描/命中/失败数量及近 5 个交易日的命中标的。
- **点击穿透**: 列表行点击即关闭抽屉, 在主图以日K加载该标的。

### 雷达 API

| 方法 | 路径 | 说明 |
|---|---|---|
| GET | `/api/radar[?market=us,hk,cn]` | 各市场只读快照: 扫描进度 + 结果缓存 |
| POST | `/api/radar/scan` | 强制重扫, body `{"market": "us"\|"hk"\|"cn"\|"all"}` |
| GET | `/api/radar/email` | 邮件日程、收件人、SMTP 是否就绪及最近投递状态 |
| POST | `/api/radar/email` | 添加/移除附加收件人, body `{"action":"add"\|"remove","email":"..."}` |

### 邮件发送配置

SMTP 凭证只从服务进程环境变量读取，不写入配置文件或返回前端。以 Gmail SMTP 为例：

```bash
export GCN_SMTP_USER="sender@gmail.com"
export GCN_SMTP_PASSWORD="Gmail 应用专用密码"
python3 -m gcn.server.app
```

可选项：`GCN_SMTP_HOST`（默认 `smtp.gmail.com`）、`GCN_SMTP_PORT`（默认 `465`）、
`GCN_SMTP_FROM`（默认同用户）、`GCN_SMTP_SECURITY`（`ssl` / `starttls` / `none`）。未配置凭证时
每日扫描仍会执行，但投递状态会明确记录为失败，不会伪报已发送。

### 雷达 CLI

```bash
python3 -m gcn.radar --market all          # 立即预热日K缓存 (只刷陈旧/缺失)
python3 -m gcn.radar --market us --force   # 强制全量刷新单市场
python3 -m gcn.radar --loop                # 独立常驻守护 (缓存巡检 + 09:00 扫描发信)
```

---

## 基本面选股

八套策略均由 `docs/选股策略-条件清单.md` 结构化转写而来 (原文条件 → 字段/运算符/阈值),
全部条件需满足才算通过; 全局市值下限 50 亿元 (增长类 100 亿), 按汇率折算人民币。

| 策略 | 主题 | 市值门槛 |
|---|---|---|
| 格雷厄姆 TOP10 | 格雷厄姆数 + 存活保证 + 财务安全 | ≥50亿 |
| 增长双验 | 增长双验 + 跨周期 + 质量 + 价值护栏 | ≥100亿 |
| 沃尔特·施洛斯 TOP10 | 低PB + 账面双验 + 分红支撑 | ≥50亿 |
| 沃伦·巴菲特 TOP10 | 护城河验证 + 盈利含金量 + 增长持续 | ≥50亿 |
| 约翰·聂夫 TOP10 | 低PE + 总报酬率 TRR + 股息保底 | ≥100亿 |
| 彼得·林奇 TOP10 | PEG + 销售与存货 + 质量双验 | ≥100亿 |
| 菲利普·费雪 TOP10 | 卓越成长 | ≥100亿 |
| 戴维斯双击 TOP10 | 估值双锚 + 增长双确认 | ≥50亿 |

数据可得性: 财报源通常仅提供最近 4~5 个年度, "近7年/10年"类条件按可得年份评估并标注
`数据不足`; ETF/未盈利标的标记 `无数据`; ROIC 为近似口径 (详见 docs 文档"诚实披露"节)。

```bash
python3 -m gcn.screener --strategy graham --market us   # CLI 选股 (-v 逐条明细)
```

---

## 回测

口径: 信号 T 日收盘确认 → **T+1 开盘价成交** (无未来函数), 只做多, 双边计入成本,
对比买入持有基准。输出四部分: 策略对比表、对数资金曲线、信号预测力事件研究
(3/5/10/20 根K线胜率-均值-t值)、前60%/后40% 分段一致性。年化指标按请求的日/周/小时/分钟
周期分别换算，响应中的 `timeframe` 给出 `interval`、周期名称和每年周期数。

- 预设组合: `B买→S卖`、`B买+绝反→S卖`、`B买→S条件`、`B买+绝反→S条件`、买入持有基准;
- ★买/★卖 (九转) 已弃用, 仅在事件研究中持续观测;
- 注意: 参数为历史手工调优, 结果属**样本内**评估, 长牛标的的买入持有基线天然很强。

---

## 主图指标信号一览

| 信号 | 定义 | 展示 |
|---|---|---|
| B买 | v4 原始B；v5 将其作为Setup，5根内首次突破Setup高点且站上MA20才确认 | 蓝色 Setup / 红色 B买 徽章 |
| S卖 | DRAWICON 8 (MAJOR_TOP 顶部信号) | 绿色 S卖 徽章 |
| 绝反 | 绝地反弹 (v4: 5% 反包 + 10 日去重, 不加趋势过滤) | 橙色 绝反 徽章 |
| S条件 | S 评分达阈值 (评分离场参考) | 描边 S条件 徽章 |
| 上/下九转9 | NINE2 计数满 9 (默认折叠) | 深/紫 徽章 |

参数表 (与富途一致): `SD=20, WIDTH=2, N=4, OFFSET=15`, 均可在侧栏实时调节。

---

## 全部 API

| 方法 | 路径 | 说明 |
|---|---|---|
| POST | `/api/fetch` | 拉取K线 `{symbol, interval, count}` (缓存优先) |
| POST | `/api/compute` | 指标计算载荷 (rows/csv/sample) |
| POST | `/api/parse_csv` | 解析粘贴的 CSV (中英文列名兼容) |
| POST | `/api/backtest` | 回测 (params/cost/max_hold/years/version/**interval**), 返回 `timeframe` |
| GET | `/api/screener/meta` | 选股策略元数据 |
| POST | `/api/screener` | 运行基本面选股 `{strategy, market, symbols}` |
| GET | `/api/radar` | 雷达只读快照 |
| POST | `/api/radar/scan` | 雷达强制重扫 |
| GET/POST | `/api/radar/email` | 邮件收件人和投递状态 |
| GET | `/` · `/styles.css` · `/echarts.min.js` | 前端静态资源 |

---

## 测试

```bash
python3 tests/run_all.py   # TDX算子/黄金样本/指标/回测/CLI/服务/数据/选股/雷达/UI
```

测试全部离线运行 (不访问网络); 雷达与预热逻辑通过 monkeypatch 注入伪数据源,
覆盖 市值阈值过滤与动态快照降级 / 信号窗口提取 / 扫描排序 / 失败隔离与退避 /
每日 09:00 调度 / SMTP 邮件与收件人配置 / 缓存调度 / 并发锁 等关键路径。

---

## 已知限制

- Futu 与 Yahoo Screener 同时不可用且本机尚无动态快照时，才会退回 2025-08 静态 Top100 并明确标记覆盖不完整;
- Yahoo 免费接口偶发限流 (429), 单标的失败不影响整体扫描, 下次巡检自动重试;
- 新鲜度内置常规工作日和收盘时刻，不内置完整交易所节假日日历；特殊休市可能触发一次空刷新，
  旧缓存会保留，雷达预热进入失败退避;
- 回测与信号均为样本内评估, 不构成任何投资建议。
