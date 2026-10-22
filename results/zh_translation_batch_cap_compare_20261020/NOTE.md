# 同一 256 条中文→英文样本的吞吐试跑

batch=128 与 batch=256 使用完全相同的 256 个源句，比较表分别保存在 `batch128/comparison.md` 和 `batch256/comparison.md`。样本吞吐（条/秒）分别是：Index 22.14/22.85、Hy-MT2 34.49/36.67、LMT 36.38/38.92。对应峰值 allocated 显存 GiB：Index 8.63/13.12、Hy-MT2 4.94/6.52、LMT 5.88/8.55。

这只是英语单方向的吞吐试跑。LMT 的完整八语言 batch=256 运行在长方向遇到 CUDA OOM，因此正式全量比较统一使用 batch=128。
