#!/usr/bin/env python3
"""Build measured six-backend results from the fresh unified scoring artifacts."""
from __future__ import annotations

from collections import defaultdict
import csv
import json
from pathlib import Path
import statistics
import numpy as np

RUN = Path(__file__).resolve().parent
LANGS = ("en", "fr", "de", "es", "ru", "ja", "ko", "ar")
NAMES = {"en":"英语", "fr":"法语", "de":"德语", "es":"西班牙语", "ru":"俄语", "ja":"日语", "ko":"韩语", "ar":"阿拉伯语"}
KEYS = ("bf16_vllm", "fp8_vllm", "nvfp4_vllm", "bf16_transformers", "fp8_transformers", "nvfp4_transformers")
DISPLAY = {"bf16_vllm":"BF16 / vLLM", "fp8_vllm":"FP8 / vLLM", "nvfp4_vllm":"NVFP4 / vLLM",
           "bf16_transformers":"BF16 / Transformers", "fp8_transformers":"FP8 / Transformers", "nvfp4_transformers":"NVFP4 / Transformers"}
BOOTSTRAP_SAMPLES = 10_000
BOOTSTRAP_SEED = 20261021
NEURAL_METRICS = ("COMET", "MetricX", "XCOMET")


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def rows(path):
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def paired_source_bootstrap(scores_a, scores_b, *, samples=BOOTSTRAP_SAMPLES, seed=BOOTSTRAP_SEED):
    """Macro A-B CI; resample Chinese source IDs with all eight languages clustered."""
    assert scores_a.keys() == scores_b.keys()
    ids = sorted({str(ident) for _, ident in scores_a}, key=int)
    assert len(ids) == 1012
    delta = np.asarray([[scores_a[(lang, ident)] - scores_b[(lang, ident)] for ident in ids]
                        for lang in LANGS], dtype=np.float64)
    observed = float(delta.mean())
    estimates = np.empty(samples, dtype=np.float64)
    rng = np.random.default_rng(seed)
    for start in range(0, samples, 200):
        count = min(200, samples - start)
        sampled_ids = rng.integers(0, len(ids), size=(count, len(ids)))
        per_language = delta[:, sampled_ids].mean(axis=2)
        estimates[start:start + count] = per_language.mean(axis=0)
    low, high = np.percentile(estimates, [2.5, 97.5])
    return {"delta_a_minus_b": observed, "ci95_low": float(low), "ci95_high": float(high),
            "ci_excludes_zero": bool(low > 0 or high < 0), "samples": samples,
            "seed": seed, "cluster_unit": "Chinese source ID shared across eight languages"}


def main():
    coverage = read(RUN / "scoring/neural_mapping_coverage.json")
    assert coverage["all_mapped"] and coverage["mapped_rows"] == 48576
    langs, macros, model_data, neural_scores = {}, {}, {}, {}
    for key in KEYS:
        pred = rows(RUN / key / "predictions.jsonl")
        assert len(pred) == 8096
        model_data[key] = read(RUN / key / "metadata.json")
        assert model_data[key].get("status") == "complete" and model_data[key].get("rows_written") == 8096
        sm = read(RUN / key / "summary.json")
        bleu = read(RUN / "scoring/bleu-13a-six" / key / "summary.json")
        score_rows = rows(RUN / "scoring" / key / "neural_scores.jsonl")
        assert len(score_rows) == 8096
        neural_scores[key] = {
            metric: {(str(r["language"]), str(r["id"])): float(r[metric.lower()]) for r in score_rows}
            for metric in NEURAL_METRICS
        }
        assert all(len(values) == 8096 for values in neural_scores[key].values())
        neural = {r["language"]: defaultdict(list) for r in score_rows}
        for r in score_rows:
            for m in ("comet", "metricx", "xcomet"):
                neural[r["language"]][m].append(float(r[m]))
        by_language = {}
        for lang in LANGS:
            v = sm["languages"][lang]
            n = neural[lang]
            by_language[lang] = {"spBLEU": v["spBLEU"], "chrFpp": v["chrFpp"],
                "BLEU13a": bleu["languages"].get(lang, {}).get("BLEU"),
                "COMET": statistics.fmean(n["comet"]), "MetricX": statistics.fmean(n["metricx"]),
                "XCOMET": statistics.fmean(n["xcomet"]),
                "sentences_per_second": model_data[key]["language_results"][lang]["sentences_per_second"],
                "generated_tokens_per_second": model_data[key]["language_results"][lang]["generation_tokens_per_second"],
                "truncated": v["truncated_outputs"]}
        macros[key] = {m: statistics.fmean(by_language[l][m] for l in LANGS if by_language[l][m] is not None)
                       for m in ("spBLEU", "chrFpp", "COMET", "MetricX", "XCOMET")}
        macros[key]["BLEU13a"] = statistics.fmean(bleu["languages"][l]["BLEU"] for l in bleu["languages"])
        macros[key]["sentences_per_second"] = float(model_data[key]["sentences_per_second"])
        macros[key]["generated_tokens_per_second"] = float(model_data[key]["generated_tokens_per_second"])
        macros[key]["truncated"] = sum(by_language[l]["truncated"] for l in LANGS)
        langs[key] = by_language

    pairs = [
        ("fp8_vllm", "fp8_transformers", "FP8 vLLM − FP8 Transformers"),
        ("nvfp4_vllm", "nvfp4_transformers", "NVFP4 vLLM − NVFP4 Transformers"),
        ("bf16_vllm", "bf16_transformers", "BF16 vLLM − BF16 Transformers"),
        ("fp8_vllm", "bf16_vllm", "FP8 − BF16 within vLLM"),
        ("nvfp4_vllm", "bf16_vllm", "NVFP4 − BF16 within vLLM"),
        ("fp8_transformers", "bf16_transformers", "FP8 − BF16 within Transformers"),
        ("nvfp4_transformers", "bf16_transformers", "NVFP4 − BF16 within Transformers"),
    ]
    paired_comparisons = {}
    for a, b, label in pairs:
        paired_comparisons[label] = {metric: paired_source_bootstrap(
            neural_scores[a][metric], neural_scores[b][metric]) for metric in NEURAL_METRICS}

    memory_peak = {key: None for key in KEYS}
    modes = {}
    telemetry = RUN / "generation_gpu1.csv"
    if telemetry.is_file():
        with telemetry.open(newline="", encoding="utf-8") as f:
            for sample in csv.DictReader(f):
                for key in KEYS:
                    if sample["phase"].startswith(key + ":"):
                        value = float(sample["memory_used_mib"]) / 1024
                        memory_peak[key] = max(memory_peak[key] or 0.0, value)

    report = ["# 六种 Transformers / vLLM 后端实测对照", "",
        "每种后端覆盖 FLORES-101 devtest 的 8,096 条中文源句到八种目标语言翻译，每语种 1,012 条。BLEU、chrF++、COMET、MetricX、XCOMET 均使用本目录中的新鲜评分结果。三个神经指标先按 `(source, hypothesis, reference, language)` 精确去重，再将分数映射回全部系统输出。",
        f"本次对照共映射 {coverage['mapped_rows']:,} 个系统输出，神经模型只需评分 {coverage['unique_inputs']:,} 个不同输入；覆盖检查通过。",
        "", "运行设置：vLLM FP8 / NVFP4 是 W8A16 / W4A16；Transformers FP8 / NVFP4 是 W8A8 / W4A4，使用 BF16 计算并解压权重后通过 compressed-tensors 软件 QDQ 执行。两种后端的量化路径不同。Transformers 日志还显示 Qwen3.5 fast linear attention 依赖未安装，因而回退至 PyTorch 实现；速度只代表本次环境。速度来自生成阶段记录，指标评分耗时不计入翻译吞吐。",
        "", "## 运行记录", "", "| 系统 | 运行栈 | 实际计算路径 | 运行版本 | 数据集 SHA-256 | 生成显存峰值¹ |", "|---|---|---|---|---|---:|"]
    for key in KEYS:
        meta = model_data[key]
        marlin = False
        if key.endswith("_transformers"):
            versions = meta.get("versions", {})
            version = ", ".join(f"{name} {value}" for name, value in versions.items()) or "版本未记录"
            peak = meta.get("peak_allocated_bytes")
            memory = f"PyTorch allocated {peak / 1024**3:.2f} GiB" if peak is not None else "—"
            runtime = "Transformers / PyTorch"
        else:
            config = read(RUN / key / "server_config.json")
            command = config.get("command", [])
            marlin = "marlin" in command
            linear_backend = command[command.index("--linear-backend") + 1] if "--linear-backend" in command else "auto"
            version = f"vLLM 0.26.0; linear_backend={linear_backend}"
            if memory_peak[key] is not None:
                memory = f"GPU1 sampled peak {memory_peak[key]:.2f} GiB"
            else:
                snapshot = meta.get("gpu_snapshot_start", "")
                gpu1 = next((line for line in snapshot.splitlines() if line.strip().startswith("1,")), None)
                if gpu1 and key == "bf16_vllm":
                    mib = int(gpu1.split(",")[3].strip().split()[0])
                    memory = f"GPU1 start snapshot {mib / 1024:.2f} GiB"
                else:
                    memory = "—"
            runtime = "vLLM"
        mode = {"bf16_vllm":"BF16", "fp8_vllm":"W8A16 weight-only Marlin" if marlin else "FP8 backend not confirmed",
                "nvfp4_vllm":"W4A16 weight-only Marlin" if marlin else "NVFP4 backend not confirmed",
                "bf16_transformers":"BF16", "fp8_transformers":"W8A8 software QDQ",
                "nvfp4_transformers":"W4A4 software QDQ"}[key]
        modes[key] = mode
        report.append(f"| {DISPLAY[key]} | {runtime} | {mode} | {version} | `{meta.get('dataset_sha256', 'unknown')}` | {memory} |")
    report += ["", "¹ Transformers 为 PyTorch 峰值 allocated；vLLM 为 GPU1 总显存。BF16 / vLLM 使用客户端记录的生成开始快照，其他 vLLM 行使用 generation_gpu1.csv 中的采样峰值；统计口径不同。", "",
        "## 指标说明", "", "- spBLEU 与 BLEU-13a：SacreBLEU corpus BLEU，分别使用 FLORES-200 SentencePiece 与 13a tokenizer。BLEU = 100 × BP × exp((1/4)Σₙ₌₁⁴ log pₙ)，其中 pₙ 为截断 n-gram 精确率，BP = min(1, exp(1 − 参考长度/译文长度))。BLEU-13a 仅报告英语、法语、德语、西班牙语、俄语、阿拉伯语六语。", "- chrF++：字符 1–6 gram 与词 1–2 gram 的匹配精确率/召回率，使用 Fβ = (1+β²)PR/(β²P+R)，β=2。", "- COMET WMT22-DA：结合源文、译文、参考译文预测人工质量分；越高越好。", "- MetricX-24 Hybrid XL：reference-based 源文、译文、参考译文错误分；越低越好。", "- XCOMET-XL：逐句整体质量分，错误跨度另存；越高越好。", "- 这些分数衡量参考译文相似度或模型预测的质量，不是正确率百分比。神经指标按每语种 1,012 个句分求均值，再对八语等权宏平均。", "",
        "## 神经指标配对差值与 95% bootstrap 区间", "",
        "差值均为左侧系统减右侧系统，按中文源句 ID 配对；每次抽取源句时同时保留该句对应的八个语种。采用 10,000 次 bootstrap（seed 20261021），表中展示八语等权宏平均差及百分位 95% 区间。COMET、XCOMET 正差倾向左侧更高，MetricX 负差倾向左侧错误分更低。", "",
        "| 对照 | COMET Δ [95% CI] | MetricX Δ [95% CI] | XCOMET Δ [95% CI] |", "|---|---|---|---|"]
    for _, _, label in pairs:
        cells = [label]
        for metric in NEURAL_METRICS:
            result = paired_comparisons[label][metric]
            cells.append(f"{result['delta_a_minus_b']:+.6f} [{result['ci95_low']:+.6f}, {result['ci95_high']:+.6f}]")
        report.append("| " + " | ".join(cells) + " |")
    report += ["", "区间覆盖 0 表示本数据集未显示宏平均差异与 0 明确分离，不构成两系统等效的证明。", "",
        "## 八语宏平均与总吞吐", "",
        "| 系统 | spBLEU ↑ | BLEU-13a ↑（6语） | chrF++ ↑ | COMET ↑ | MetricX ↓ | XCOMET ↑ | 全量句/秒 | 生成 token/秒 | 截断 |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for key in KEYS:
        v = macros[key]
        report.append(f"| {DISPLAY[key]} | {v['spBLEU']:.3f} | {v['BLEU13a']:.3f} | {v['chrFpp']:.3f} | {v['COMET']:.6f} | {v['MetricX']:.6f} | {v['XCOMET']:.6f} | {v['sentences_per_second']:.2f} | {v['generated_tokens_per_second']:.2f} | {v['truncated']} |")
    for key in KEYS:
        report += ["", f"## {DISPLAY[key]}：分语种", "",
            "| 语种 | spBLEU ↑ | BLEU-13a ↑ | chrF++ ↑ | COMET ↑ | MetricX ↓ | XCOMET ↑ | 句/秒 | token/秒 |",
            "|---|---:|---:|---:|---:|---:|---:|---:|---:|"]
        for lang in LANGS:
            v = langs[key][lang]
            bleu_value = "—" if v["BLEU13a"] is None else f"{v['BLEU13a']:.3f}"
            report.append(f"| {NAMES[lang]} | {v['spBLEU']:.3f} | {bleu_value} | {v['chrFpp']:.3f} | {v['COMET']:.6f} | {v['MetricX']:.6f} | {v['XCOMET']:.6f} | {v['sentences_per_second']:.2f} | {v['generated_tokens_per_second']:.2f} |")
    report += ["", "spBLEU 与 BLEU-13a 是两种不同 tokenizer 设定下的 BLEU，分数不应直接互换；BLEU-13a 仅对其支持的六种语言报告。chrF++ / BLEU 是参考译文相似度，不代表准确率百分比。COMET、XCOMET 越高越好，MetricX 越低越好。", "",
        "神经评分产物：[去重输入清单](scoring/unique_inputs.jsonl)、[系统到唯一输入映射](scoring/system_unique_mapping.json)、[新鲜分数与运行时间](scoring/unique/)、[XCOMET 错误跨度映射](scoring/)（各系统 `xcomet_error_spans.jsonl`）、[映射覆盖检查](scoring/neural_mapping_coverage.json)。"]
    (RUN / "BACKEND_COMPARISON.md").write_text("\n".join(report) + "\n", encoding="utf-8")
    summary = {"systems": KEYS, "unique_inputs": coverage["unique_inputs"], "mapped_rows": coverage["mapped_rows"],
               "macros": macros, "languages": langs, "modes": modes,
               "paired_source_bootstrap": {"samples": BOOTSTRAP_SAMPLES, "seed": BOOTSTRAP_SEED,
                   "cluster_unit": "Chinese source ID shared across eight languages",
                   "comparisons": paired_comparisons}}
    (RUN / "backend_comparison_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(RUN / "BACKEND_COMPARISON.md")


if __name__ == "__main__":
    main()
