# 全量三模型评测结果（2026-10-20）

使用 FLORES-101 devtest，全量 1,012 个中文源句 × 8 个目标方向；三个模型共翻译 24,288 条。统一 BF16、batch=128、贪心解码、max-new-tokens=512，全部运行于 GPU 1。

| 模型 | 宏平均 spBLEU | 宏平均 chrF++ | 总体样本/秒 | 空输出 | 截断输出 | 峰值 allocated/reserved GiB |
|---|---:|---:|---:|---:|---:|---:|
| index | 25.90 | 43.30 | 19.89 | 0 | 0 | 8.65/10.04 |
| hy | 23.71 | 41.75 | 24.40 | 0 | 0 | 5.45/6.00 |
| lmt | 24.22 | 41.45 | 16.68 | 0 | 8 | 12.23/23.18 |

spBLEU/chrF++ 衡量与 FLORES 参考译文的相似度，不是准确率百分比。正式逐语言指标、时延分位数、字符/token吞吐和译文记录见[完整对比表](comparison.md)。

batch=256 下 LMT 在长方向显存不足；batch=128 下三模型全量运行均成功。LMT 有少数输出达到统一 512-token 上限，保留并计入分数。Index 使用当前 Transformers 的 PyTorch 线性注意力 fallback。数据公开且 NiuTrans 已披露 FLORES dev 训练暴露，限制详见[评测说明](../../../docs/ZH_TRANSLATION_BENCHMARK.md)。

批量吞吐用于离线大规模评测；整批 p50/p95 与“摊销秒/条”不能代替 batch=1 的在线单句延迟。
