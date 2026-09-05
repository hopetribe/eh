# Progress Log

## Session: 2026-09-05 r4研究结果收口

- 上轮提交bb3010b并完成生产部署，属于已验证的实质进展；本轮从当前工作区确认只有用户行情缓存变更，继续未完成的r4研究，不重复部署。
- 重读文件规划/TDD技能及已冻结r4协议，补全固定候选选型、实际成交覆盖与工件归档；默认v5及生产环境不改。
- 两个TDD循环已完成：固定候选白名单/Calmar并列顺序/覆盖门槛，以及真实快照归档、退出来源、期限计数、摘要和禁止覆写。r4专项4项通过，正式results已生成；四个新候选训练均因MDD失败，hold20/MID2另因CAGR失败，依协议不计算验证段。
- 配对同标的同入场日后，hold10改变25笔退出，15笔改善、10笔恶化，并新增29笔后续交易；NVDA 2023-12-14入场的151.34%趋势利润被截为2.74%，MRNA 2024-07-17入场的-26.10%亏损改善至-2.30%。不能只披露保护作用，也不能把配对单笔差值当作完整组合因果增益。
- r4最终工件6行训练、573行交易、1528行事件、1686行漏点；源码/协议/结果摘要复核通过，相关研究及核心回归96项通过。Phase 21按否决结论完成；继续入场趋势环境研究，未推送或再次部署。

## Session: 2026-09-05 当前代码提交与部署

- 用户最新要求“提交和部署”，本轮优先发布，不继续扩展历史研究。信号列表修复dfc1885已提交，生产现有代码e32c15e；本地r1/r2/r3提交和r4执行语义需要同步，正式策略保持v5。
- 按文件规划/TDD技能执行；Sites发布技能不适用于此现有自管Python服务器，沿用README中43.160.201.247的部署方式，不创建或迁移站点。
- 发现r4已有测试处于红灯：entry_exit_cols错误接受字符串“AB”为两列。复现业务断言失败后增加严格tuple、长度和列存在校验；待回归验证。不覆盖或提交data/缓存。
- r4本次仅固化协议、候选构建及执行语义检查点，尚未完成训练产物归档，不宣称研究通过或启用候选。
- 发布前回归：核心/研究/Web/API 124项、Shadow五模块169项，共293项通过；参数校验红灯已转绿，git diff --check通过。
- 服务器45份既有运行源码、前端、依赖与部署配置全部匹配e32c15e；新增研究模块尚不存在，适合按明确文件白名单发布。真实shadow状态目录仍不存在，不执行initialize/update。
- 代码提交bb3010b6d883a97449f41b1f4936982ada8f2006已推送GitHub main，包含r4执行语义检查点及此前r1/r2/r3研究提交。6文件发布包SHA-256为20df7c08f094ed74125c3d4f10fa30abe3e6a0b284698dd65c0e8f4778cdc75c。
- 发布使用既有.release.lock串行锁，停服后替换已核验文件；原文件、缺失清单及v5接口基线保存在/home/eric/kk2/.deploy-backups/bb3010b-release.3cirerme/，manifest.json记录DEPLOYED。已准备失败时恢复原文件、移走新增文件并重启的回滚逻辑，本次未触发。
- 服务于2026-09-05 10:16:48 CST恢复运行，active/running、NRestarts=0；全部49份运行源码/前端/配置摘要与发布提交一致，.deploy-version已更新。线上正式版本仍为v3/v4/v4-exp/v5，未注册或启用任何r4候选。
- 公网HTTPS GET首页200，index/styles与本地摘要一致；v5计算返回900根，公网计算及回测完整JSON与部署前内部基线逐项一致；HTTP入口308跳转正确。HEAD返回501是现有标准库服务未实现该方法，改用GET完成健康验收，无需修改业务代码。
- 本地11项用户行情缓存改动仍原样保留且未进入提交/发布；真实shadow目录不存在，未执行任何建账、更新或行情刷新。当前用户提交与部署请求已完成，r4研究归档工作另行继续。

## Session: 2026-09-05 历史r4持有期限与趋势失效

- 上轮为实质进展：r3本地提交b98722a，88项研究/核心及169项影子相关回归通过；本轮核对只有用户行情缓存未提交。
- 重新完整读取文件规划/TDD技能。尚无用户波段/长趋势回复，按此前声明的波段方向继续，不视为阻塞。
- 在计算候选收益前冻结4个新规则：仅新增P-stop5来源持有10/20/40根，或连续两根跌破MID退出；对照为v5和旧P-stop5，不把旧败者重新参加排序。

## Session: 2026-09-05 历史r3入场来源风控

- 上轮为有效进展：r2本地提交51ba0d4，81项相关回归通过，已确定P信号质量/覆盖改善但新增来源风险超标；本轮核对仓库，只有用户行情缓存未提交。
- 沿用文件规划和TDD，先冻结r3的P-stop5/8/12三个新候选。原v5来源（包括同日冲突）不加新止损，新增P入场才锁定各自风险比例；风险取信号日、次日实际OPEN为价格基准，CLOSE确认后下一OPEN退出。
- 完成逐入场风险锁定/重置、非法比例拒绝、研究费用路径重放、同日v5优先、实际成交覆盖等价、排除plain P补选、真实快照集成等7个TDD循环。88项研究/配方/回测及169项影子协议相关回归全部通过。
- 三个新候选均只因训练MDD失败：22.10%/23.12%/21.71%，超过16.49%门槛。P-stop5年化13.40%仍不能因此晋升；不计算候选验证收益。
- r3工件已生成：5行对照、1251行事件、372行交易、1405行漏点。全部59次初始止损属于新增P来源；原v5来源未附加风险比例。
- 初始止损收紧后仍有较高平均持仓占比42.2%，新增来源平均持有46.6根、最长365根；同时存在NVDA 147根+151.34%等大赢家，下一步检验持有时间时须披露收益被截断的代价。
- 已使用非阻塞提问询问用户偏好波段或长趋势，未回复前继续按10～40根的波段方向推进；不是等待外部数据或将目标标记阻塞。

## Session: 2026-09-05 历史r2原始触发覆盖

- 上轮为实质进展：研究r1已本地提交a1161ce，输入/源码/结果摘要及75项回归通过；本轮核验只有用户行情缓存变更，未推送或部署研究代码。
- 已完整重读文件规划/TDD技能。训练归因：无B原始触发的86个漏买中53个在MA200下、33个在MA200上；回撤中位18.2%、RSI33.7。无S原始触发漏点的60日涨幅中位28.7%。
- 在计算候选收益前新增r2协议：R超跌恢复、P趋势回调、E短周期早卖及其组合共8个主候选；沿用唯一训练选型和独立验证，不补选r1败者。
- 六个TDD循环完成：R此前高点突破、P长期趋势预热和MID穿越、E此前低点破位、组合前缀因果性、训练S质量/覆盖门槛、真实快照只训练集成。相关81项回归及diff检查通过。
- r2全候选训练门槛失败：P是唯一仅因MDD失败者；P入场20日胜率50%→52.89%、干扰48.15%→41.32%、买点覆盖11/107→22/91，但MDD14.99%→21.94%。不计算候选验证收益，不事后换门槛。
- 正式生成r2工件：8行训练比较、3620行事件、879行交易、2248行漏点；计算前冻结源码字节、计算后校验，记录协议和父快照摘要。原v5与实时行情缓存完全不变。
- P新增34笔交易中21笔亏损，11笔亏损的峰值收盘浮盈不足5%；下一阶段优先检验新增趋势回调入场失败后的退出，不把已被否决的通用盈利保护包装为解决方案。

## Session: 2026-09-05 已有数据继续优化

- 当前用户目标已转为使用既有数据持续优化并阶段版本化；旧影子协议保持冻结，未来数据等待不阻塞本轮研究。
- 上一轮只核验提交/部署，不属于指标优化进展。本轮已核对工作区仅有用户行情缓存变更；研究复用既有冻结快照，不覆盖缓存。
- 重新读取文件规划和TDD技能；使用小候选集、因果成交和回溯稳健性验证，不重复建设影子运维基础设施。
- 一次规划补丁因progress.md上下文不存在整体未应用；核对文件头后以正确上下文重新提交。一次过长读取输出被截断，后续缩小到函数级范围。
- r1九个TDD循环完成：诊断输出、已接受信号冷却、8候选因果组合、订单费用重放、20日标签边界、快照摘要、训练选型、验证拒绝和真实快照集成。针对性回归75项通过，git diff --check通过。
- 按协议先训练8候选，再验证唯一S40；验证无改善，五年CAGR21.06%→20.66%，且9/10留一均退化（去AAOI相同），否决。profit50在训练阶段因收益保留不足被否决，未偷看其验证收益。
- 已生成研究r1工件：72行分段/子集比较、2106行分组事件、924行漏点，以及交易明细、决策、环境摘要和源码快照。父数据manifest逐文件核验通过，未触碰实时data缓存或影子状态。
- 早期八股压力期v5年化−0.08%、MDD22.43%，进一步证明不能只看近五年表现。
- 下一步证据：在训练期重新计算漏点，96个可行动漏买有86个附近无B_ALL_RAW；32个可行动漏卖有31个附近无S_RAW。冷却并非主要漏点根因，下一阶段转向原始信号覆盖机制。

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
| 服务器 `/home/eric/kk2` 是发布目录而非 Git 工作区 | 1 | 放弃远端 `git pull`，沿用版本标记、部署前备份和最小文件替换的可回滚发布方式 |
| 一次部署前检查误填不存在的本地工作目录，进程未启动 | 1 | 纠正为仓库绝对路径后重跑；公网 HTTPS 入口返回200 |
| 快照测试首轮因 `_snapshot_run_materials` 尚未实现而收集失败 | 1 | 按TDD红灯确认后实现运行前快照与结束复核，相关测试及完整回归通过 |
| 完整池fallback测试因外部池改为显式partial-history后失败 | 1 | 保留旧调用方fallback，同时生产覆盖对象使用完整/部分历史显式列表 |

### 提交前兼容性复核

- 恢复无 pytest 依赖的 `python3 tests/run_all.py` 入口：新增参数边界仍逐值覆盖，核心兼容测试 `152/152` 通过。
- Homebrew pytest 完整套件在允许本地回环端口后 `179 passed`；参数化测试改为函数内逐值循环，仅测试计数变化，覆盖值未减少。

## Session: 2026-09-05

### Phase 15: v6 盈利保护候选预注册与前向影子验证
- **Status:** in_progress
- Actions taken:
  - 将上一目标轮判定为有效进展：v5审计、提交和生产部署均已完成，继续推进未完成的前向验证能力。
  - 重新读取文件规划与TDD技能，运行session catch-up，并以当前Git、规划文件和行情差异为权威状态。
  - 汇总冻结82笔交易的利润路径复算，完成Phase 15首项“失败模式量化”；确认浮盈回吐而非隔夜跳空是trail失败主因。
  - 并行发起引擎语义、统计spec和生产隔离三项只读审查；先冻结规则与测试，再按逐项红灯→绿灯实现。
  - 定位引擎尾部强平、v5 preset、VERSIONS、CLI/API/UI调用链；确认Phase 15只需兼容扩展底层状态机并新增独立shadow模块。
  - TDD循环1：先证明`terminal_policy`不存在，再实现`mark`尾部只盯市、不生成terminal交易且不扣虚假退出费；默认`liquidate`回归保持不变。
  - TDD循环2：先证明未知terminal策略会被静默当成mark，再加入严格枚举校验；`tests/test_backtest.py`当前21项通过。
  - TDD循环3：为mark模式增加稳定的open/pending-exit状态快照；先得到缺少`state`键的红灯，再实现并通过22项引擎测试。
  - TDD循环4：先证明引擎不支持盈利保护，再实现`profit_keep`的+20%激活、净保本成本线、收盘触发与次开成交；边界案例绿灯。
  - TDD循环5：分别用红灯锁定`profit_keep`必须配合trail及必须为(0,1)有限非布尔值，再加入最小校验。
  - TDD循环6：证明preset中的盈利保护未透传会导致-30%而非-10%，再接入`run_backtest`；回测文件26项通过，正式v5 preset仍无新增字段。
  - TDD循环7：用同时跌破两条线的红灯发现差异退出误归因；修正为只有收盘仍高于原trail线时才记`profit_lock`，原v5同日也会退出的场景继续归`trail`。
  - 统计功效复核推翻旧`keep20 affected>=20`门槛：5年实际仅8次差异退出。已立即撤回尚未落盘的错误门槛，候选强度将在不查看反事实收益的前提下按4年可辨识力定稿。
  - TDD循环8：为profit模式补充trail有限分数校验；先证明非法trail被静默接受，再实现并绿灯。
  - TDD循环9：为mark末端补充`pending_entry/open/pending_exit/flat`稳定状态；先得到缺少status的红灯，再实现，回测文件30项通过。
  - 明确成本压力不重算保护线和触发路径；shadow层将以0.1%规则成本冻结订单，仅对同一成交序列按0.25%重新计费。
  - 在禁止读取收益字段的前提下完成候选功效校准：keep50在36/48月具有13～18/21～26次真实差异退出，keep20只有2～7/3～8次；因此冻结唯一候选为`profit-arm20-keep50`，keep20明确排除。
  - TDD循环10～12：新增规范JSON哈希、重复字段拒绝、NaN/Infinity拒绝；逐项确认红灯后实现，shadow基础测试3项通过。
  - TDD循环13：先证明未知顶层字段会被静默接受，再冻结11个顶层字段并拒绝缺失/未知字段。
  - TDD循环14：先以缺失文件红灯锁定唯一候选、cutoff和36/48月功效门槛，再新增不可变前向spec；shadow测试5项通过。
  - TDD循环15：先证明挑战者嵌套字段可被悄然注入，再冻结全部嵌套对象字段；shadow测试6项通过。
  - 修正文档中一处候选切换后的旧描述：keep20是明确排除项，keep50是唯一候选；与功效校准及冻结spec保持一致。
  - 完成候选功效计数工件、完整统计公式、20/60日诊断标签、复权修订政策和36/48月固定端点的冻结；主spec canonical SHA-256固定为`c12f7c4932072b9fa2352bbca733481c849121afa420f3cc53ce5002e80cb57f`。
  - 新增与生产入口隔离的前向边界、确认状态重置、成熟计数、36/48月状态机、标签成熟和READY前白名单账本；候选仍未注册进`VERSIONS`、preset、API或Web UI。
  - 将`tests.test_shadow_validation`纳入裸Python兼容测试入口；同时修正雷达合成数据在周末及pandas 3.0下少一根工作日的时间相关测试假设。
  - 发布前定向回归：回测、shadow与雷达边界共`65 passed`；完整pytest在允许回环端口后`224 passed`；`python3 tests/run_all.py`为`197/197`通过。
  - 上一轮提交/部署判定为有效进展：`f68baa2`已推送，生产引擎摘要一致、服务与公网v5 API均通过；本轮继续收口Phase 15，未把部署本身误判为总体目标完成。
  - TDD循环16：构造“已接受文件与来源缓存被同步篡改”的红灯，证明单纯重叠比较可被绕过；registration升级为v2，冻结每标的首次行数、终点与精确摘要，并在每次更新前复核，相关3项通过。
  - TDD循环17：先以缺失`CURRENT`红灯锁定不可变代际协议，再让首次注册在同父临时目录一次性写入base、registration、genesis generation、commit、CURRENT和派生ledger后整体rename；首次发布及故障注入2项通过。
  - TDD循环18：先证明增量更新不会推进generation，再实现每个核心共同交易日一代、10标的一次提交、前代哈希绑定及commit→CURRENT顺序，单日前向代际测试通过。
  - TDD循环19：以“首次晚启动已含5个前向日”对“cutoff基线后批量追加”的权威文件逐字节比较得到红灯；首次注册改为只冻结`date<=cutoff`，其余共同日逐日生成相同链，比较转绿。
  - TDD循环20：先篡改CURRENT所指已提交generation的公开账本字段，确认接续流程未核对内容摘要；随后在读取CURRENT时强制验证generation文件SHA-256与引用哈希一致，篡改测试转绿。
  - TDD循环21：删除CURRENT对应的commit标记，先确认旧流程仍会继续；随后要求commit存在且内容必须逐字节等于CURRENT的canonical JSON，缺标记与内容分叉均进入DATA_BLOCKED路径。
  - TDD循环22：重算CURRENT、commit与当前generation哈希以伪造自洽尖端，但故意切断`previous_hash`；旧流程接受该分叉，新增前代唯一commit定位与链接核对后拒绝接续。
  - TDD循环23：用“上游CSV暂时回退到基线”复现派生账本从1个前向日倒退到0的红灯；新增从base逐代验证commit、generation、previous_hash、canonical JSON、K线行、accepted摘要与公开账本绑定并重放的权威读取器，接续不再依赖可变CSV缓存；shadow update专项9项通过。
  - TDD循环24：删除CURRENT得到硬失败红灯；重构头部发现为扫描全部commit、校验规范指针及generation摘要、要求序号从0连续且唯一，并由最高commit原子修复缺失/陈旧CURRENT。独立只读审查确认commit应为唯一权威选择器，孤儿generation与派生缓存不得决定状态。
  - TDD循环25：验证registration声明的`sorted_compact_utf8_lf`与实际缩进JSON不一致得到红灯；首次发布改为canonical JSON，registration→genesis链首哈希与跨调用逐字节确定性测试保持通过。
  - 新增崩溃恢复特征测试：存在“generation已落盘、commit未落盘”的同序号孤儿时，读取器保持旧权威头，随后正常追加生成唯一commit且不受孤儿内容影响。
  - TDD循环26：在篡改每代公开账本后同步重算generation、commit与CURRENT哈希，证明仅校验链完整性仍可接受语义伪造；新增公开字段白名单、spec/source/bar摘要、完整性、`elapsed_common_sessions==sequence`、非负计数及protocol/public state交叉校验，重哈希伪造转为拒绝；同时确认非头部generation摘要也逐代检查。
  - TDD循环27：把成熟计数改成另一个合法非负整数并重算整条尖端引用，得到廉价字段校验无法识别的红灯；权威加载完成后现以重放帧完整重算头部pre-READY账本并逐字段比较，杜绝自洽哈希下的指标伪造。
  - 将调用节奏确定性验证扩为三路径：首次输入已含5日、基线后一次批量追加5日、基线后逐日调用5次；registration、base、全部generation/commit与CURRENT逐字节一致。
  - TDD循环28：runner级2:1 provider调整测试先因重叠值变化被拒绝；接入冻结的1ppm统一OHLC中位缩放与独立成交量缩放后，重叠行恢复权威精确值、新行转换到首次注册价格基准，局部非统一修订仍按DATA_BLOCKED拒绝；复权专项13项及runner专项通过。
  - TDD循环29：运行环境漂移测试先因缺少环境身份接口失败；registration现冻结Python实现/精确版本、NumPy、pandas、操作系统与机器架构，接续前严格比较，模拟pandas版本变化会拒绝续写且确定性字节测试保持通过。
  - TDD循环30：用两个同步线程并发首次注册，稳定复现两次发布重叠与目录替换失败；新增实验哈希级进程内互斥和相邻`flock`，覆盖首次创建与增量全过程，两调用串行得到同一账本，runner并发/更新专项18项通过。
  - TDD循环31：registration唯一schema断言先发现四组旧顶层兼容副本；移除重复source/initial摘要，统一以`implementation`和`base`为权威，新增严格顶层、序列化协议、标的集合、base字段/哈希/行数及环境身份验证；相关19项通过。
  - TDD循环32：首次发布的文件/目录fsync测试先因无耐久写接口失败；新增统一原子字节写入，所有权威/派生文件均执行flush+文件fsync、rename、父目录fsync，首次staging各子目录和整体rename也同步目录；并发与落盘专项通过。
  - TDD循环33：把registration仅改为等价缩进JSON，旧流程直到genesis链接才报错；现读取入口先验证UTF-8/有限JSON并要求原始字节精确等于canonical编码，非规范字节在状态重放前即被拒绝。
  - TDD循环34：权威bars等价缩进编码测试先被接受；反序列化现重新生成canonical字节并逐字节比较，base编码、float hex与registration声明保持可验证一致。
  - TDD循环35：可解析但非规范的缩写十六进制浮点先被接受；base与generation的每个OHLCV现必须精确等于解析值的`float.hex()`输出，避免同值多编码破坏跨路径字节唯一性。
  - 正式评估纯模块完成独立交付与复审：纯`calculate_evaluation`不产生状态，受控`formal_evaluate`验证完整注册spec、READY/精确端点/零次、核心顺序、首列成交、订单工件与从路径推导的armed/affected计数，并只返回CAS 0→1意图；35项新测试及联合104项通过，未读取真实候选表现。
  - TDD循环36：源码清单精确断言先发现复权校验器与正式评估器未被注册摘要覆盖；两者现加入`ALGORITHM_SOURCE_PATHS`，后续任何规则/评估公式改动都会阻断既有实验接续。
  - TDD循环37：私有协议快照测试先因仅有公开账本接口失败；新增`_build_pre_ready_snapshot`，公开层继续严格只返回预注册计数，generation私有层独立保留actual start、锁窗、成熟摘要、事件/标签、评估次数与结果槽位。
  - TDD循环38：36/48月状态机测试先发现失败的36月判断只存在调用栈中；现每次到达36月固定端点都会生成链上`checkpoint_36`（端点、七项成熟计数、pass），48月锁窗必须携带同一个失败检查点，并显式记录locked_months。
  - TDD循环39：runner的READY消费测试先因无订单工件/CAS胶水失败；现从权威锁窗重算双策略OPEN成交mask与挑战者退出原因，绑定价格、日期、symbol顺序、spec/source/bar摘要生成订单工件，正式评估仅在实验锁内以READY/0次及链上36月checkpoint调用，并把CAS 0→1、工件hash与完整结果原子写入同一交易日generation。
  - 以2026-09-15至2029-09-14的合成锁窗验证真实订单构建链：10只行序固定、首列无OPEN成交、矩阵维度一致，重新计算的订单工件摘要逐字节一致；shadow validation/revision/evaluation联合回归当前106项通过。
  - TDD循环40：篡改spec引用的功效校准JSON而保持spec/sidecar不变，旧加载器仍接受；`load_spec`现限制同目录工件名、严格解析工件、核对其独立sidecar与spec内`canonical_json_sha256`，候选选择证据不再只是未验证的路径字符串。
  - TDD循环41：重启/无新增/继续追加的runner级测试先复现终态被重建为READY/0并重复正式评估；现以权威上一代`formal_evaluation_count/evaluation_result`驱动状态承接，只有首次READY/0调用评估器，后续generation保持同一结果与订单摘要。
  - TDD循环42：带锁后K线的测试证明订单工件曾绑定持续增长的全量bar hash；现逐标的只绑定`<=locked_end`的canonical bar hash，锁后数据不能改变固定评估工件。
  - TDD循环43：formal evaluator原先把日历expected日期误当成必须存在的交易日；现同时绑定`locked_end`与其后第一个共同交易日，接受expected当日或之前最后共同交易日，任意提前或晚于expected均阻断。
  - TDD循环44：删除公开`source_root`覆盖入口，算法摘要只能来自实际import仓库；同轮把spec读取收敛为加锁前唯一一次，避免两次读取落入不同实验锁。
  - TDD循环45：整链重哈希后篡改非头generation合法计数/state/checkpoint的测试先穿透浅校验；现一次因果策略重放预计算全部前缀摘要，并逐代完整重建ledger+protocol，强制评估次数、36月checkpoint、锁窗和正式结果不可逆，复杂度保持`O(symbols×sessions)`。
  - TDD循环46：1.025→1.037逐日复权与1.037批量复权先产生不同float.hex；现所有新接受OHLCV规范为12位有效数字，真正uniform换基准逐日/批量逐字节一致。1ppm仅为外层上限；带新增行而比例离散超过数值噪声时fail-closed DATA_BLOCKED。
  - TDD循环47：新增commit尾丢失检测和派生accepted_bars缓存修复；有效但超前的CURRENT不再静默回退，派生CSV损坏或提交后写失败可由权威链恢复。
  - TDD循环48：五个READY发布故障点覆盖generation写前/后、commit写前/后及CURRENT后派生缓存失败；恢复后仅一个权威CAS结果。两个独立进程竞争同一增量也只产生一个session generation。
  - TDD循环49：性能审计证明前缀成熟计数与事件列表仍可能二次增长；现改为点增量加单次`cumsum`、共享事件前缀、标签路径缓存及两遍流式generation校验。250/500/1000个合成前向会话耗时约1.285/1.583/2.202秒、峰值内存1.69/2.17/3.63MiB，增长接近线性。
  - TDD循环50：把正式结果改成合法`eligible=true`并重哈希整条链曾可穿透carry校验；现只在唯一0→1转换代重建固定订单并调用一次`formal_evaluate`，持久化结果必须与重算canonical bytes完全一致，后续代不重复bootstrap。
  - TDD循环51：共同伪造周末/休市日或同步漏掉正常交易日曾可污染样本时钟；新增离线NYSE日历工件，2026～2028使用官方闭市表、2029～2030按Rule 7.2冻结投影，并要求cutoff后会话是严格前缀。日历canonical摘要为`bbd5dad9dae12c34afd65adf61e63b44fde84b5e6d2ab7271fab00f4f296f398`，原始文件摘要为`8f11d717082a1d15732e7c04803307144e30ea98a069920dd63c80b7fa9a625e`。
  - 同轮封闭终态生命周期：到达`ELIGIBLE_FOR_V6_IMPLEMENTATION`、`REJECTED_KEEP_V5`或`INCONCLUSIVE_COVERAGE_KEEP_V5`后，权威链与accepted-bars缓存停在首个终态会话；重启只重放、验真和修复派生缓存，不再读取或追加incoming。
  - 最终影子validation/revision/evaluation联合回归`133 passed`；裸Python兼容入口在本机权限环境`297/297`全部通过。
  - 最终静态编译、`git diff --check`均通过；允许本地回环绑定的完整pytest回归为`324 passed`。
  - TDD循环52：真实构造锁窗后第59日快照并写入generation，红灯证明私有协议会提前持久化已成熟的20日及未完整60日收益/MFE/MAE；现保留事件身份但把READY前全部20/60标签值盲化为`null`，只在唯一READY→正式终态事务内重算并揭示。联合影子专项增至`134 passed`，正式评估仍只消费一次。
  - 提交/部署终审确认当前核心10股仍没有`2026-09-05`后的共同交易日，也不存在真实shadow state；本次只固化和同步影子协议模块，不初始化账本、不改变正式v5信号/API/UI。
  - 建账前运维门槛已显式保留为Phase 15未完成项：固定长期运行venv、在行情锁下校验`adjustment=adjusted`及sidecar/CSV摘要一致，并通过显式初始化门禁等待首个合法共同交易日。
  - 最终发布回归：影子三模块`134 passed`，完整pytest`325 passed`，裸Python兼容入口`298/298`；静态编译、`git diff --check`、主spec/sidecar和NYSE日历双摘要校验均通过。

### 本阶段新增错误记录

| Error | Attempt | Resolution |
|---|---:|---|
| 插入`reset_v5_confirmation_window`时函数位置误切断`derive_shadow_boundaries` | 1 | 立即恢复函数边界并以完整shadow专项回归确认 |
| 新标签边界测试首个补丁上下文已变化 | 1 | 重新定位目标测试；失败补丁未产生部分修改 |
| spec扩充后sidecar与代码注册哈希仍为旧值，8项专项校验失败 | 1 | 计算canonical JSON摘要并同时更新两处注册值；专项`33 passed` |
| 沙箱内完整回归有11项HTTP测试无法绑定回环端口 | 1 | 在获准的本机测试环境重跑，完整`224 passed` |
| 雷达合成测试假设周末`bdate_range(..., periods=30)`仍返回30根；pandas 3.0只返回29根 | 1 | 先回滚到最近工作日再构造固定30根，并按生产日龄公式断言 |
| 本轮定向测试误用系统Python，其环境无pytest | 1 | 恢复使用`PYTHONPATH=. /opt/homebrew/bin/pytest`执行逐项TDD |
| 一次复权专项测试调用误填不存在的workdir，进程未启动 | 1 | 纠正为仓库绝对路径后重跑，13项全部通过 |
| 一次registration专项组合测试再次误填不存在的workdir，进程未启动 | 1 | 立即纠正绝对路径并完成19项回归；后续命令复用固定仓库路径 |
| 首次点名源码清单测试时使用了不存在的测试函数名 | 1 | 用`rg`定位实际函数名后重跑，先得到预期红灯再完成源码覆盖修复 |
| 裸Python完整入口在沙箱内有11项本地HTTP绑定及3项Futu日志目录权限失败 | 1 | 按既定审批在本机权限环境重跑，`272/272`通过；无业务断言失败 |
| 发布前静态检查误用不存在的`/opt/homebrew/bin/python3` | 1 | 从pytest shebang定位固定解释器`/opt/homebrew/opt/python@3.11/bin/python3.11`，静态编译与摘要复核随后通过 |

## 2026-09-05 Phase 15 官方运维闭环与发布准备

- TDD循环53：先用两个线程同步读到相同旧序号，稳定复现“实际只追加一代、两个调用均返回UPDATED”；随后将状态重验、权限扫描、十股快照、提交及提交后读取纳入同一实验独占锁，结果精确收敛为一个UPDATED和一个NO_CHANGE。
- TDD循环54：普通文件状态根先被误报UNINITIALIZED；新增仓库外绝对路径、真实目录、当前所有者和私有权限契约，且在所有公开运维操作入口统一执行。
- TDD循环55：canonical但领先的伪造CURRENT先否决只读status；现只读路径完全服从连续commit/generation权威链并将CURRENT标记stale，写路径继续保留潜在commit尾丢失保护。
- TDD循环56：行情`flock`异常先裸抛PermissionError；现获取/释放失败稳定映射为DATA_BLOCKED。初始化状态锁/写入OSError也稳定映射为STATE_BLOCKED，不再成为INTERNAL_ERROR。
- 官方CLI、README运行手册、源码身份绑定和裸Python模块清单均已补齐；真实preflight聚合报告五只核心标的sidecar摘要不一致，按设计零状态写入并禁止初始化。
- 发布前Shadow专项`167 passed`；最后增加状态锁错误分类后，完整pytest为`359 passed`，裸Python兼容入口为`332/332`通过。静态编译、摘要与差异检查在提交前继续单独核验。
- Homebrew Python 3.11与系统Python双解释器`py_compile`、`git diff --check`均通过；冻结spec canonical SHA-256仍为`c12f7c4932072b9fa2352bbca733481c849121afa420f3cc53ce5002e80cb57f`，仓库内不存在真实registration/CURRENT/ledger状态文件。

### 本轮新增错误记录

| Error | Attempt | Resolution |
|---|---:|---|
| 首版并发测试在`run_shadow_snapshot`前使用Barrier，被进程级umask互斥提前串行导致超时 | 1 | 把Barrier移到旧状态读取完成后，稳定得到两个UPDATED红灯，再完成锁边界修复 |
| 新行情锁错误测试首次补丁插入到上一测试循环末句之前，引发IndentationError | 1 | 立即检查局部结构，把原断言移回循环并重跑得到预期业务红灯；未触及生产代码 |
| 完整pytest与裸Python兼容入口在30秒工具窗口内仍运行 | 1 | 按会话ID继续轮询，分别得到358项和332/332通过结果 |
| 首次白名单暂存被沙箱禁止创建`.git/index.lock` | 1 | 使用获准的Git暂存权限重试；缓存文件排除断言与cached diff检查均通过 |

## 2026-09-05 初始化/更新混合竞争复核

- 提交`1d4bfab`已推送；自动审批在核对生产主机后仍拒绝私有源码上传，要求用户对具体目标地址授权。尚未上传或部署本阶段工件。
- 本轮从现有代码确认初始化持有umask互斥后等待实验锁，而update持有实验锁后等待umask互斥；新增子进程隔离的真实线程竞争测试，覆盖延迟初始化被另一初始化超越后与update同时运行的场景。
- TDD循环57：混合竞争测试先稳定复现`initialize and update deadlocked while acquiring locks`，统一实验锁→umask互斥的顺序后转绿；锁目录及文件在创建时直接指定0700/0600，初始化的权限扫描、提交及响应序号读取也收进同一实验锁。
- 接口重构后，原状态锁错误注入测试仍指向已移除的`shadow_operations.run_shadow_snapshot`导入而失败；将注入点移至实际`shadow_runner._experiment_lock`，保留原错误分类与零状态写入断言。
- 修复后运维+CLI专项35项、全部Shadow五模块169项通过；静态编译通过，冻结spec摘要保持不变。测试只使用临时合成输入，真实行情与状态未修改。
- 系统Python无pytest依赖的运维/CLI兼容入口`35/35`通过；本次验证范围覆盖所有修改的锁入口及原有代际链、恢复、复权与评估协议。

## 2026-09-05 Shadow 运维发布完成

- 权限环境更新后恢复已授权的部署，核对GitHub main与代码提交`e32c15eaeefce26e8a2f1596b8c39d2f173c91d6`一致；仅发布README与三个Shadow运维模块。
- 发布包SHA-256为`5ef6cef1f46570dd156d3b0ca76fa633de54937806d11bbcd0aa091e7f60b9c1`。服务器发布前备份与清单位于`/home/eric/kk2/.deploy-backups/e32c15e-release.BKZyyV/manifest.txt`；自动回滚路径已随发布脚本准备，本次发布无需触发回滚。
- 候选文件及全部未修改算法依赖摘要在发布前、发布后均验证一致；服务器使用既有Python 3.12.3 / numpy 2.5.2 / pandas 3.0.5。`kk2`于2026-09-05 08:34:29 CST重新启动，验收时active、NRestarts=0。
- 内部及公网v5 sample API均返回900根K线；公网HTTPS首页200，HTTP按308跳转至`https://43.160.201.247:8443/`。本地11个行情缓存变更未进入提交或部署。
- 官方status返回UNINITIALIZED，并确认`/home/eric/.local/state/kk2-shadow`不存在。没有运行initialize/update，也没有生成真实前向绩效。
- 生产preflight发现YINN/NVDA/GOOGL缺少来源元数据。使用原数据服务的行情事务锁先备份至`/home/eric/kk2/.deploy-backups/e32c15e-inputs-25f_6tk_`，再各尝试一次正常提供方刷新。
- YINN经Yahoo成功刷新至2026-09-03，2501根，adjusted来源与CSV/sidecar摘要校验通过。NVDA和GOOGL被Yahoo HTTP 429限流，服务回退保留旧缓存，没有手工生成或重算其元数据。
- 刷新后复核preflight仅剩NVDA/GOOGL两项来源缺失，退出码3/DATA_BLOCKED；实验目录仍不存在，kk2保持active。
- 部署阻塞已解除；当前剩余条件为补齐可信行情来源，以及等待2026-09-05之后合法共同交易日及后续成熟样本。v6仍是预注册候选，生产v5未变。

### 本次发布运行记录

| 现象 | 处理与结果 |
|---|---|
| 远端bash提示zh_CN.UTF-8 locale不可用 | 非致命警告；摘要、发布与健康检查全部成功 |
| NVDA/GOOGL提供方返回HTTP 429 | 每只仅尝试一次，保留已有缓存及备份；停止连续重试并保持DATA_BLOCKED |
