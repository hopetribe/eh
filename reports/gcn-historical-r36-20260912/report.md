# r36：五类新技术指标与v5信号置信度

## 决策

五个单族模型和唯一等权组合均未达到全部预注册条件。没有上线置信度字段或修改v5信号。结论只针对本轮固定参数/模型/快照，不能推论所有技术指标都无效，也不能把回溯候选称为未来最佳方案。

本轮构建280个2017年以来信号事件；五年预测期2021-08-27至2026-08-27共有165条成熟事件：B买60、绝反44、S卖61。4条截至快照未成熟的事件和4条预热不足的早期事件被明确标记。每年开始前只用已经完成20根标签的历史训练；第二套评估还排除待预测股票。固定费用每边0.1%；S正确标签是随后跌幅超过费用，不代表卖空策略收益。

## 方法改进

五个新族为CMF21与5根变化、MFI14与5根变化、OBV20归一化流量与5根变化、ADX14与DI方向差、ATR14相对过去60根波动与当日TR冲击。每族两特征，B/JF/S分别拟合，历史Beta(2,2)收缩基率作为固定概率偏移，L2=8，不搜索参数。

除正确率外，评估Brier概率损失、对数损失、五档可靠性、high覆盖/干扰、逐年/逐股稳定性，并用股票簇配对bootstrap与18项比较校正下界审计。Brier变好同时可能来自区分能力或校准改善，不能单独证明概率准确校准。

## 主要结果

| 信号及较有价值的指标 | 时间隔离Brier改善 | 时间+股票隔离改善 | 后者high条数 | high正确率/总体 | high干扰率/总体 | 判定 |
|---|---:|---:|---:|---|---|---|
| B买—ADX/DI | +0.001934 | +0.002751 | 10/60 | 70.00% / 53.33% | 30.00% / 43.33% | 覆盖16.67%，数量及概率增益不足 |
| 绝反—CMF | +0.009511 | +0.008476 | 0/44 | 不可估计 / 68.18% | 不可估计 / 31.82% | 概率误差下降，但无合格high层 |
| S卖—ADX/DI | +0.004789 | +0.006401 | 0/61 | 不可估计 / 54.10% | 不可估计 / 39.34% | 概率误差下降，尚无可用高置信层 |

改善为同折历史基率模型Brier减去候选Brier，正数越大越好。上表选择只作结果描述，不是晋级或改变研究方案。三项的多重比较股票簇下界分别为−0.024794、−0.008124、−0.019481，均不能排除恶化。

B买ADX/DI两套评估均为10条7胜，Wilson95%区间约39.68%–89.22%，远不足以保证70%正确率。只看sidecar一致的5股，high仅4条2胜2干扰；未通过该标记的另一组为6条5胜。该分组是数据质量敏感性，不能选择较有利的一组上线。

OBV给B买的高层在时间隔离中为6条5胜，时间+股票隔离中为5条全胜；覆盖分别仅10%和8.33%，整体Brier反而略差。不能把5/5包装成100%可信。

## B买ADX/DI高层反例与成功例

以下均来自时间+股票隔离评分，收益为次OPEN至第20根CLOSE扣双边0.1%。

| 股票/信号日 | 模型概率 | 20根净收益 | 结果 |
|---|---:|---:|---|
| NFLX 2022-12-01 | 60.33% | −5.22% | 干扰 |
| TSLA 2024-04-29 | 64.85% | −5.66% | 干扰 |
| NVDA 2025-01-06 | 62.23% | −18.59% | 干扰 |
| NFLX 2023-10-24 | 64.59% | +13.94% | 正确 |
| TSLA 2024-10-25 | 62.16% | +30.32% | 正确 |
| YINN 2025-01-27 | 61.01% | +37.36% | 正确 |

其他逐笔结果保存在`results/predictions.csv`。没有通过high门槛的信号仍完整保留，不视为删除后正确率。

## 下一项独立算法问题

绝反概率存在明显低估：最后一年CMF平均预测53.52%，实际9条7胜；历史基率模型平均仅50.38%。低估贯穿多个年份，表明扩展历史基率可能响应较慢。因此另行注册r37的单一时间衰减基率实验，不调整r36的指标、模型或阈值。新实验同样是被旧结果启发的历史探索，并非独立验证。

## 来源与边界

CMF公式来自[Fidelity CMF](https://www.fidelity.com/learning-center/trading-investing/technical-analysis/technical-indicator-guide/cmf)；MFI见[Fidelity MFI](https://www.fidelity.com/learning-center/trading-investing/technical-analysis/technical-indicator-guide/mfi)，OBV见[Fidelity OBV](https://www.fidelity.com/learning-center/trading-investing/technical-analysis/technical-indicator-guide/OBV)。ADX强度与DI方向需区别，见[Fidelity ADX](https://www.fidelity.com/viewpoints/active-investor/average-directional-index-ADX)；ATR反映波动，不代表方向，见[Fidelity ATR](https://www.fidelity.com/learning-center/trading-investing/technical-analysis/technical-indicator-guide/atr)。概率损失与校准区别见[scikit-learn calibration](https://scikit-learn.org/stable/modules/calibration.html)。

快照来源/复权/摘要一致标记仅5/10股通过，收藏池也有幸存者偏差。全部历史已被以前研究查看，股票间存在共同市场风险，聚类区间仅作探索稳健检查。当前不能晋升为真实生产胜率。

复现：`PYTHONPATH=. /opt/homebrew/bin/python3.11 -m gcn.backtest.signal_research_r36 --output /tmp/kk2-r36-new-run`，要求新的空目录。`results/manifest.json`绑定协议、输入父清单、源码快照与每个输出。
