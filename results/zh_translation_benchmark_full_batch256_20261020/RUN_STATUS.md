# batch=256 全量运行状态

这次运行被显存不足中断：Index-Translate-2B 和 Hy-MT2-1.8B 各有 8,096 条完整预测；LMT-60-1.7B 完成 4,560 条后，在俄语方向遇到 CUDA OOM。该目录没有三模型完整同范围报告，不可用于最终排名。

最终完整比较位于 `results/zh_translation_benchmark_full_batch128_20261020/`：三个模型各 8,096 条、统一 batch=128，并有逐方向 `comparison.md`。
