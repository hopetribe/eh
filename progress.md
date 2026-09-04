# Progress Log

## Session: 2026-09-04

### Phase 12: v5 最新截面与失败模式复核
- **Status:** in_progress
- Actions taken:
  - 将上一轮 v5 固化、信号列表修复、提交与生产部署判定为有效阶段进展。
  - 重新读取当前规划、发现、进度及工作区状态，以磁盘与 Git 为准恢复上下文。
  - 新增 Phase 12～14：先复核最新截面和失败模式，再做受约束的下一版本候选验证，最后按门槛决定是否固化版本。
  - 保留现有行情缓存和规划文件，不把运行时数据差异误当作策略代码变更。
  - 核对当前提交、审计入口、报告产物和日K截止日；确认部分标的已有9月新数据但截面不齐，首轮继续使用统一截止 2026-08-28。
  - 确认现有审计模块可直接复算 v5 等价基线，并已有多维稳健性表可作为下一候选的对照证据。
  - 在 `/private/tmp/kk2-audit-v5-refresh-20260904` 完成当前数据上的576候选复算；推荐仍为 `B五日MA20确认 + 绝反即时 → S卖 + 20%跟踪止损`，主要绩效与旧报告一致。
  - 对比旧报告确认变化仅来自 NVDA 缓存校正及极小数值差，v5 基线稳定，可开始针对失败交易设计下一轮实验。
  - 将89笔 v5 等价完整交易映射回信号源与入场日特征，确认绝反即时入场的中位单笔为负且训练段极端超卖 RSI≤20 子组表现最差。
  - 明确两条待验证假设：绝反极端超卖需二次确认；MA60下方的普通B确认需额外长期趋势门槛。暂不因小样本直接改策略。
  - 核对绝反公式，确认 RSI 未参与现有信号，可作为独立风险门槛候选。
  - 核对 UI/回测默认池为11只、不含 SIVE；发现审计默认池仍为旧12只，同时 SNOW 8月28日缺K却被标为 `ok`，将先修复审计口径再固化下一版本。
  - 完成第一个 TDD 循环：新增“审计默认池必须等于生产默认池”测试，确认旧实现因多出 SIVE 红灯；改为复用 `engine.DEFAULT_SYMBOLS` 后绿灯通过。
  - 完成第二个 TDD 循环：构造请求窗口末日缺K样本，旧实现错误标 `ok`；增加 `stale-end`/`partial-window` 状态后与默认池测试共2项通过。
  - 完成第三个 TDD 循环：新增自定义截止日报告测试，确认缺少动态接口红灯；报告头现从实际 `full` 窗口生成截止日并绿灯通过。
  - 汇总并行只读审查：记录 v6 静默回退风险、旧测试年不可再称未见样本、position-aware漏点缺口及跟踪止损的大幅利润回吐问题。
  - 完成第四个 TDD 循环：新增持仓中买点不可行动、持仓中卖点可行动的集成测试；`missed_turn_table` 现可接受 incumbent 候选并重放实际持仓，测试绿灯。
  - 完成第五个 TDD 循环：新增 actionable 覆盖统计测试；报告主表、逐标的清单和Top15现仅把可交易状态的峰谷计为漏点，`tests/test_signal_audit.py` 共11项通过。
  - 用生产默认11股和共同截止日重跑完整审计，得到新的 position-aware 漏点报告；v5基线仍为全样本Sharpe 1.33/MDD14.99%。
  - 独立复算风险B选择性确认：核心10股全样本与4/5年度、成本压力、TEM外部样本方向较好，但一个验证年度退化且增益高度依赖AAOI，因此不直接晋升正式v6。
  - 完成 hard-stop 引擎前两个 TDD 循环：验证收盘触发、次日开盘穿透成交，以及 preset 参数确实传入回测；相关3项含既有max-hold测试全部通过。
  - 为 hard-stop 增加 `(0,1)` 有限数校验并完成红绿循环；拒绝0、负值、布尔值、NaN/Infinity等危险静默配置。
  - 扩展审计 `Candidate`，仅在启用时追加稳定名称后缀并透传到策略与持仓态漏点；完成名称和持仓影响两轮TDD。
  - 将12.5%与15%初始硬止损作为仅针对 incumbent v5 的受约束邻域加入候选网格，未做全笛卡尔积；唯一名称/配置测试通过。

## Session: 2026-09-02

### Phase 10: 固化正式 v5
- **Status:** in_progress
- Actions taken:
  - 接受用户将最终推荐方案正式定义为 v5 并启动本地服务的请求。
  - 决定保留 v4-exp 作为旧阶段B实验口径，新增独立 v5，避免历史结果漂移。
  - 测试先行新增 v5 配方、回测预设、HTTP 和 Web UI 契约；首次运行按预期因缺少实现而红灯。
  - 实现 v5：以完整 v4 B_SIGNAL 为 Setup，5根内首次收盘突破 Setup 高点并站上 MA20 时输出 B_SIGNAL。
  - 新增 v5 推荐回测预设：确认B+绝反即时入场、原S卖退出、20%跟踪止损。
  - 服务端将 v5 通用 Setup/确认/过期映射到兼容 payload；页面默认选择 v5，并保留 v3/v4/v4-exp。
  - 首轮4项 v5 契约测试全部通过。
  - 配方、回测、服务端和 Web UI 组合测试48项通过；完整测试套件151项通过；py_compile 与 diff whitespace 检查通过。
  - 在11只有效收藏股的冻结五年窗口逐只核对正式 v5 与审计候选：111个Setup、66个确认B，入场序列及20%止损资金曲线全部精确一致。

### Phase 11: 本地服务与交互验收
- **Status:** complete
- Actions taken:
  - 准备在 `127.0.0.1:8642` 启动正式服务并用应用内浏览器验收 v5 默认页与回测交互。
  - 首次真实服务验收通过：默认v5行情、Setup/确认、v5推荐回测与v4/v5对比均成功渲染，控制台0条日志。
  - 发现默认启动会同时运行全市场雷达预热；停止实例并按启动前快照清理本次产生的97个tracked及689个untracked缓存差异，仅保留用户原有4个日K修改和2个周K文件。
  - 新增 `--no-background-jobs`，使本次本地查看不再触发无关全市场缓存写入。
  - 以无后台任务模式重新启动服务（PID 30659），应用内浏览器复验 URL、标题、v5 默认选中、数据就绪和0条控制台日志。
  - 保留本地页面为可查看交付物；README 补充 v5 定义与轻量启动命令。

### Phase 6: 收藏池与数据审计
- **Status:** complete
- Actions taken:
  - 恢复上一轮 v4-exp 的实现与验证上下文。
  - 确认本轮需从 B_STAGE 扩展到全部 B买/绝反/S卖及漏点分析。
  - 检查工作区状态，发现既有策略实验改动和多份行情数据改动；决定全部保留。
  - 定位收藏列表为 Web UI 的 `kk2_watchlist_v2` 浏览器状态，准备读取真实收藏池。
  - 已完整加载应用内浏览器操作规范；浏览器当前无打开标签页。
  - 确认可仅启动静态页面来显示收藏下拉，避免调用行情接口和继续改写数据缓存。
  - 通过页面可见收藏下拉确认真实收藏池为12只：TQQQ、MSFT、NFLX、YINN、SNOW、TSLA、MRNA、NVDA、TEM、GOOGL、SIVE、AAOI。
  - 完成首轮数据覆盖与哈希盘点：11只有日K，SIVE缺失，TEM仅有上市后约553根。
  - 核实数据服务的复权、缓存合并和最多5000根机制；发现部分CSV与元数据哈希不一致。
  - 在线核实 SIVE 为跨市场歧义代码，当前系统会错误地按美股裸代码解析，暂不猜测替换。
  - 冻结共同分析截面为 2021-08-30～2026-08-28，排除9月2日未完成盘中K线。
  - 运行11标的数据质量检查与v4信号计数；确认OHLC有效、无重复/空值。
  - 建立 `B买+绝反→S卖` 基线：53笔、胜率67.9%、均笔28.45%，但最差-71.34%，YINN/MRNA出现严重长期回撤。
  - 新增可复现审计模块 `gcn/backtest/signal_audit.py` 和针对性测试，覆盖数据审计、逐笔信号、事后漏点、候选网格和冻结测试段。
  - 首轮针对性测试20通过、1失败；失败是10%回撤的浮点表示误差，已将断言改为近似比较。
  - 首轮216候选运行完成；识别并否决“全信号5日MA60确认”的低覆盖伪优解。
  - 增加训练/验证多数标的盈利、交易覆盖约束，并加入 S卖后3/5日跌破信号低点+MA20 的因果确认候选。
  - 第二轮360候选与22项针对性测试通过；稳健最优转为“B 3日MA20确认 + 绝反即时 + S卖 + 20%跟踪止损”。
  - 分析逐信号/趋势状态/子类型：B噪声集中在熊市恢复与阶段底，S噪声集中在牛市 major-top；S二次确认整体退化。
  - 第三轮加入MA20/MA60趋势破位退出；识别出低回撤防守方案，但冻结测试Sharpe低于基线，降级为备选。
  - 完成0.1%/0.25%/0.5%成本压力、5个逐年区间、11次逐标的留一验证；B确认+20%止损的风险调整优势对成本和单一标的均稳健。
  - 测试36组深跌后MA20恢复补漏信号及其与原信号的组合；因训练段低Sharpe、高回撤和近50%噪声全部否决。
  - 建立收益保留、回撤减半、冻结测试Sharpe/MDD四重采纳门槛；仅B五日MA20确认+S卖+20%止损（无/60日持有上限）通过，选择更简单的无上限版。
  - 生成最终报告及逐笔信号、漏点、576候选、推荐交易、消融、成本、逐年、留一、入场质量和B确认CSV。
  - 为所有输入记录SHA-256；7份CSV与旧元数据摘要不一致但OHLC结构有效，保持原样并在报告披露。
  - 审计模块6项针对性测试通过；完整测试套件147项全部通过；`py_compile` 与 `git diff --check` 通过。
  - 关闭用于读取收藏列表的临时静态服务和临时浏览器标签。

### Phase 1: 基线与代码路径确认
- **Status:** complete
- Actions taken:
  - 阅读 planning-with-files 技能并建立持久化任务记录。
  - 已完成前序 TSLA 信号归因和跨标的候选规则事件研究。
  - 定位现有配方测试、回测不变量测试、服务端序列化和 UI 信号列表。
  - 发现 v3/v4 受黄金输出列契约保护，决定新增隔离的实验版本。
  - 确认服务端版本校验和 UI 版本/对比入口需要显式扩展。
- Files created/modified:
  - `task_plan.md`（新建）
  - `findings.md`（新建）
  - `progress.md`（新建）

### Phase 2: 测试先行
- **Status:** complete
- Actions taken:
  - 新增确认窗口、MA60、到期边界和版本隔离测试。
  - 完成红灯验证：缺少 `_stage_confirmation` 时测试收集失败。
- Files created/modified:
  - `tests/test_recipe.py`
  - `tests/test_backtest.py`

### Phase 3: 实现实验版本
- **Status:** complete
- Actions taken:
  - 新增 `v4-exp`、Setup→Confirm 状态计算、实验输出列和回测预设。
  - 保持 v3/v4 原始信号路径及输出列不变。
  - 完成服务端版本校验、前端 payload、版本选择器和图表信号接入点定位。
  - 确认实验 HTTP payload 和前端静态契约的测试位置。
  - 扩展服务端实验 payload、UI版本选择、Setup图标和v4/实验版对比入口。
- Files created/modified:
  - `gcn/recipes/gcn_main.py`
  - `gcn/backtest/engine.py`
  - `gcn/server/app.py`
  - `webui/index.html`
  - `webui/styles.css`

### Phase 4: 验证与效果对比
- **Status:** complete
- Actions taken:
  - 运行配方、回测、服务端和 Web UI 针对性测试。
  - 读取前端调试与Browser技能，确定本地真实交互验证流程。
  - 在应用内Browser完成初始页面、v4-exp切换、TQQQ双版本回测和TSLA两个Setup的真实交互验证。
  - 用默认11标的重跑全样本/后40%事件研究和复合策略对比，结果记录于 findings.md。
  - 完整测试套件通过；git diff 检查无空白错误。
  - 清理浏览器验证触发的所有行情缓存改动，最终仅保留实验代码、测试和规划文件。

### Phase 5: 交付
- **Status:** complete
- Actions taken:
  - 汇总实验结果、限制和下一步优先级。
- Files created/modified:
  - `task_plan.md`
  - `findings.md`
  - `progress.md`
- Files created/modified:
  - `tests/test_server.py`
  - `tests/test_webui.py`

## Test Results
| Test | Input | Expected | Actual | Status |
|------|-------|----------|--------|--------|
| 前序事件研究 | 默认11标的，B_STAGE与5日确认 | 得到可比较基线 | 已得到，详见 findings.md | ✓ |
| 配方与回测针对测试 | `PYTHONPATH=. /opt/homebrew/bin/pytest tests/test_recipe.py tests/test_backtest.py -q` | 全部通过 | 17 passed | ✓ |
| 服务端与Web UI针对测试 | `PYTHONPATH=. /opt/homebrew/bin/pytest tests/test_server.py tests/test_webui.py -q` | 全部通过 | 27 passed | ✓ |
| 完整测试套件 | `PYTHONPATH=. /opt/homebrew/bin/pytest -q` | 全部通过 | 147 passed | ✓ |
| 工作区检查 | `git status --short`; `git diff --check` | 无data变更、无空白错误 | 符合预期 | ✓ |

## Error Log
| Timestamp | Error | Attempt | Resolution |
|-----------|-------|---------|------------|
| 2026-09-02 | Rolling对象无iloc | 1 | 使用 rolling(60).mean() Series |
| 2026-09-02 | `/usr/bin/python3` 无 pytest 模块 | 1 | 待切换到项目环境/可用pytest入口 |
| 2026-09-02 | Homebrew pytest 无法导入本地 `gcn` | 2 | 后续测试使用 `PYTHONPATH=.` |
| 2026-09-02 | `tab.playwright.screenshot` 不可用 | 1 | 改用文档API中的 `tab.screenshot` |
| 2026-09-04 | `/opt/homebrew/bin/python3` 不存在 | 1 | 检查 `pytest` shebang 和可用解释器后切换到真实环境 |
| 2026-09-04 | 受限环境禁止 `ps`/`pgrep` 读取进程列表 | 1 | 不重启任务；等待后直接检查审计输出目录，确认原进程已完成并生成全部12份产物 |
| 2026-09-04 | 临时绝反分层脚本误用 `_forward_path` 键名 `ret_pct` | 1 | 读取函数契约，改用 `return`/`mfe`/`mae` 并在展示时换算百分比 |
| 2026-09-04 | 临时候选用 `-inf` 充当“无均线门槛”，但确认函数要求门槛值有限，导致弱绝反被全部丢弃 | 1 | 改用价格域内有限常数 `0.0`，重新验证真实的“突破信号高点”确认；不采用本次错误实验结果 |
| 2026-09-04 | position-aware 测试的平坦低点使事后聚类选中第10根而非手工预期第20根 | 1 | 检查实际转折表，保持业务断言不变并把持仓起点/目标日期对齐确定性输出 |
| 2026-09-04 | 插入 hard-stop 测试时把原 max-hold 测试尾部留在新函数下方 | 1 | 按原测试作用域恢复 exposure/max-hold 三行，再继续 hard-stop 绿灯验证 |
| 2026-09-04 | complexity 测试首轮使用了非真实 baseline 名称，未到达复杂度排序 | 1 | 改用 `Candidate("v4-b+jf", "S", None, None).name` 对齐生产 lookup 后重跑 |
| 2026-09-04 | 增量评审候选表同时以 `name` 作为索引名和列名，排序时报歧义 | 1 | 取出预注册挑战者后重置索引，再按纯训练列确定唯一挑战者 |

## 2026-09-04 Phase 12/13 继续执行

- 将事件审计拆成 v4 B Setup、v5 B确认、绝反、S卖四类，并增加稳定key/角色/版本/20日完整性字段。
- 为所有事件远期路径加审计截止日，新增严格分段事件输出 `signal_events_by_split.csv`。
- 报告改为只用完整20日事件计算质量，明确Setup不是成交点，分段结果不读取下一段行情。
- 新增v5 incumbent对预注册hard-stop挑战者的增量门槛；full与已知最近一年完全不参与晋升决策。
- 报告参数说明改为从实际候选动态生成，能够正确披露初始止损的收盘确认、次开成交和跳空穿透风险。
- 定向回归：`PYTHONPATH=. /opt/homebrew/bin/pytest tests/test_recipe.py tests/test_backtest.py tests/test_signal_audit.py -q` → `40 passed`。

## 5-Question Reboot Check
| Question | Answer |
|----------|--------|
| Where am I? | 全部阶段完成 |
| Where am I going? | 交付收藏池近5年完整审计与最终推荐 |
| What's the goal? | 识别漏点和干扰B/绝反/S，并用样本外验证选出稳健方案 |
| What have I learned? | B五日MA20确认+绝反即时+原S卖+20%跟踪止损是收益与风险平衡最好的方案 |
| What have I done? | 完成11只有效收藏标的审计、576候选、补漏实验、多重稳健性验证与报告 |

## 2026-09-04 Phase 12～14 最终固化

- 将生产默认11只拆成完整5年核心10只与TEM部分历史外部样本；正式候选、交易、消融、成本、逐年和留一均只用核心10只，TEM不参与晋升。
- 修正信号审计语义：B买Setup(v4)、B确认(v5)、绝反、S卖四类分离；每个未来窗口按当前分段截止，未完成事件保留但结果为空。
- 修正position-aware漏点覆盖：覆盖信号来自实际交易的入场/退出决策，包含S卖、20%跟踪退出、hard-stop和max-hold，不再只用原始S列。
- 新增12.5%与15%两个预注册hard-stop挑战者；明确锚定实际入场开盘、收盘触发、次日开盘退出及跳空穿透，并增加有限区间、边界相等、原因优先级与异常价格测试。
- 训练段唯一选择12.5%挑战者；验证段CAGR 18.63% < 21.88%、Sharpe 1.22 < 1.31、MDD 11.74% > 11.29%，虽最差单笔改善8.19个百分点，仍按门槛否决并保持v5。
- 增加共同日历、stale-end、内部缺口、完整窗口和OHLCV有限性检查；异常或不完整标的不再进入核心选型池。
- 消除审计快照TOCTOU：在长计算前复制算法源码、行情CSV和元数据，并直接从冻结输入计算；结束时再次校验快照哈希。
- 报告明确区分11只信号诊断池、10只策略选型池与TEM外部验证，补充毛/净收益口径、hard-stop精确门槛、AAOI CAGR集中度和数据来源警告。
- 最终产物写入 `reports/signal-audit-v5-review-20260904/`：14项结果、4份源码快照、11组CSV/meta快照，run_id=`bcede8a46e7dadc596075eb524228810b26fe8258c866723c30478c7c6190a8b`。
- 最终针对性测试：`tests/test_signal_audit.py tests/test_backtest.py` → 54 passed。
- 最终完整回归（允许本地HTTP测试绑定临时端口）：187 passed；`py_compile` 与 `git diff --check` 通过。
- 独立终审确认：输出/源码/输入哈希、CSV行数、run_id、事件边界、漏点覆盖、候选门控和报告展示全部一致，P0/P1/P2均为0。
- 本轮不创建v6，也不提交或部署；等待新增时间外样本后再按冻结方案研究盈利保护。

### 本阶段新增错误记录

| Error | Attempt | Resolution |
|---|---:|---|
| 系统Python执行 `py_compile` 尝试写入沙箱外用户缓存而失败 | 1 | 使用项目Python并把 `PYTHONPYCACHEPREFIX` 指向 `/private/tmp/kk2-pycache`，静态编译通过 |
| 提交前误用系统 `python3`，其环境缺少 pytest，且 `py_compile` 默认缓存目录不可写 | 1 | 改用 Homebrew pytest，并将 `PYTHONPYCACHEPREFIX` 指向 `/private/tmp/kk2-pycache` 后重试 |
| 提交前完整回归在沙箱内有11项本地HTTP测试无法绑定临时端口 | 1 | 其余176项通过；按既定方式在允许回环端口绑定的环境重跑完整套件 |
| 沙箱禁止创建 `.git/index.lock`，首次暂存未发生任何写入 | 1 | 使用用户已授权的 Git 暂存权限重试，并继续显式排除运行时行情文件 |
| 快照测试首轮因 `_snapshot_run_materials` 尚未实现而收集失败 | 1 | 按TDD红灯确认后实现运行前快照与结束复核，相关测试及完整回归通过 |
| 完整池fallback测试因外部池改为显式partial-history后失败 | 1 | 保留旧调用方fallback，同时生产覆盖对象使用完整/部分历史显式列表 |
