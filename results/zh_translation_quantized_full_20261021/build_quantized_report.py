#!/usr/bin/env python3
"""Build the concise full-corpus report for the three Index GGUF quants."""

from __future__ import annotations

from collections import defaultdict
import csv
from datetime import datetime
import json
import math
from pathlib import Path
import statistics


ROOT = Path(__file__).resolve().parents[2]
RUN = Path(__file__).resolve().parent
BASELINE = ROOT / "results/zh_translation_benchmark_full_batch128_20261020/full_language_results.json"
MODEL_KEYS = ("q8_0", "q6_k", "q4_k_m")
DISPLAY = {
    "q8_0": "Q8_0",
    "q6_k": "Q6_K",
    "q4_k_m": "Q4_K_M",
    "bf16": "BF16 Transformers（参考）",
}
METRIC_KEY = {"q8_0": "index", "q6_k": "hy", "q4_k_m": "lmt"}
LANGUAGES = ("en", "fr", "de", "es", "ru", "ja", "ko", "ar")
LANG_NAMES = {
    "en": "英语", "fr": "法语", "de": "德语", "es": "西班牙语",
    "ru": "俄语", "ja": "日语", "ko": "韩语", "ar": "阿拉伯语",
}


def read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def read_jsonl(path: Path):
    with path.open(encoding="utf-8") as stream:
        for line in stream:
            if line.strip():
                yield json.loads(line)


def fmt(value, digits=3):
    if value is None:
        return "—"
    return f"{value:.{digits}f}"


def main() -> None:
    baseline = read_json(BASELINE)
    comet = read_json(RUN / "scoring/comet/summary.json")
    xcomet = read_json(RUN / "scoring/xcomet/summary.json")

    means = defaultdict(dict)
    per_language = defaultdict(dict)
    truncated = {}
    for key in MODEL_KEYS:
        m = METRIC_KEY[key]
        sacre = read_json(RUN / key / "summary.json")
        metadata = read_json(RUN / key / "metadata.json")
        for lang in LANGUAGES:
            item = sacre["languages"][lang]
            per_language[key].setdefault(lang, {})["spBLEU"] = item["spBLEU"]
            per_language[key][lang]["chrF++"] = item["chrFpp"]
            per_language[key][lang]["COMET"] = comet["systems"][m]["languages"][lang]["mean"]
            per_language[key][lang]["XCOMET"] = xcomet["models"][m]["languages"][lang]["mean"]
            per_language[key][lang]["MetricX"] = None
        means[key]["spBLEU"] = sum(sacre["languages"][l]["spBLEU"] for l in LANGUAGES) / len(LANGUAGES)
        means[key]["chrF++"] = sum(sacre["languages"][l]["chrFpp"] for l in LANGUAGES) / len(LANGUAGES)
        means[key]["COMET"] = comet["systems"][m]["macro_mean"]
        means[key]["XCOMET"] = xcomet["models"][m]["macro_mean_8_languages"]
        means[key]["MetricX"] = None
        means[key]["speed_rows_s"] = metadata["sentences_per_second"]
        means[key]["speed_tokens_s"] = metadata["generated_tokens_per_second"]
        truncated[key] = sum(
            metadata["language_results"][lang]["truncated_outputs"] for lang in LANGUAGES
        )

    metricx_rows = list(read_jsonl(RUN / "scoring/metricx/output_all.jsonl"))
    metricx_scores = defaultdict(lambda: defaultdict(list))
    for row in metricx_rows:
        metricx_scores[row["model_key"]][row["language"]].append(float(row["prediction"]))
    for key in MODEL_KEYS:
        langs = metricx_scores[key]
        if set(langs) != set(LANGUAGES) or any(len(langs[l]) != 1012 for l in LANGUAGES):
            raise ValueError(f"MetricX coverage mismatch for {key}: { {k: len(v) for k,v in langs.items()} }")
        per_lang_means = {lang: sum(langs[lang]) / len(langs[lang]) for lang in LANGUAGES}
        for lang in LANGUAGES:
            per_language[key][lang]["MetricX"] = per_lang_means[lang]
        means[key]["MetricX"] = sum(per_lang_means.values()) / len(LANGUAGES)

    six = ("en", "fr", "de", "es", "ru", "ar")
    bleu13a = {}
    for key in MODEL_KEYS:
        summary = read_json(RUN / "scoring/bleu-13a-six" / key / "summary.json")
        if set(summary["languages"]) != set(six):
            raise ValueError(f"BLEU-13a coverage mismatch for {key}")
        bleu13a[key] = summary
        means[key]["BLEU-13a"] = sum(summary["languages"][l]["BLEU"] for l in six) / len(six)

    bf16 = {
        "spBLEU": baseline["spBLEU_chrFpp"]["index"]["spBLEU_macro"],
        "chrF++": baseline["spBLEU_chrFpp"]["index"]["chrF++_macro"],
        "COMET": baseline["comet_wmt22_da"]["index"]["macro_mean"],
        "MetricX": baseline["metricx24"]["scores"]["index"]["macro_mean"],
        "XCOMET": baseline["xcomet"]["scores"]["index"]["macro_mean_8_languages"],
        "BLEU-13a": baseline["sacrebleu_13a"]["scores"]["index"]["macro_6_language"],
    }

    report = [
        "# Index-Translate-2B 量化版全量评测",
        "",
        "- 数据集：FLORES-101 devtest，中文→英语、法语、德语、西班牙语、俄语、日语、韩语、阿拉伯语。",
        "- 每语种 1,012 条；每个量化版本 8,096 条，三版合计 24,288 条。",
        "- 推理：官方 GGUF 权重，llama.cpp CUDA，GPU1，温度 0，最大生成 512 tokens；三种权重逐个运行，相同提示和批处理参数。",
        "- spBLEU、BLEU、chrF++ 按该语种全量语料计算；COMET、MetricX、XCOMET 取句级分数均值；质量宏平均对八个语种等权。",
        "- BLEU 同时列 spBLEU（FLORES-200 tokenizer，覆盖 8 语）和 BLEU-13a（覆盖 6 语；日、韩语不适用）。",
        "- 量化版推理使用 128 并发、batch 8192、ubatch 2048、提示缓存关闭；服务逐个运行。",
        "- Q6_K 有 1 条俄语输出（ID 327）重复至 512-token 上限，作为模型输出参与评分。",
        "- BF16 行来自此前 Transformers 全量基准，作为参考；它与 GGUF 的推理后端不同，分数差也可能含后端差异。",
        "",
        "## 指标说明",
        "",
        "- **spBLEU / BLEU-13a**：分别按 FLORES-200 SentencePiece tokenizer 和 13a tokenizer 计算 BLEU，衡量译文与参考译文的 n-gram 重合度；越高越好。BLEU-13a 只列出六个适用语种。",
        "- **chrF++**：按字符 n-gram F-score，并加入最高 2-gram 的词级匹配；越高越好。",
        "- **COMET WMT22-DA**：用源文、译文和参考译文预测人工质量评分，作为语义质量估计；越高越好。",
        "- **MetricX-24**：用源文、译文和参考译文回归翻译质量误差；越低越好。",
        "- **XCOMET-XL**：预测整体译文质量并标注局部错误片段；本表列整体分，越高越好。",
        "- **速度**：条/秒为全量句子吞吐；token/s 为生成 token 吞吐。质量分数不是准确率百分比。",
        "",
        "指标实现：[FLORES-101](https://github.com/facebookresearch/flores/blob/main/previous_releases/flores101/README.md)、[COMET / XCOMET](https://github.com/Unbabel/COMET)、[MetricX-24](https://github.com/google-research/metricx)。",
        "",
        "## 宏平均",
        "",
        "| 版本 | spBLEU ↑ | BLEU-13a ↑（6语） | chrF++ ↑ | COMET WMT22-DA ↑ | MetricX-24 ↓ | XCOMET-XL ↑ | 速度（条/秒） | 生成（token/s） | 截断条数 |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for key in (*MODEL_KEYS, "bf16"):
        if key == "bf16":
            vals = bf16
            bf16_languages = baseline["speed_full_inference"]["index"]["languages"]
            language_seconds = {
                lang: bf16_languages[lang]["count"] / bf16_languages[lang]["sentences_per_second"]
                for lang in LANGUAGES
            }
            total_seconds = sum(language_seconds.values())
            speed = sum(bf16_languages[lang]["count"] for lang in LANGUAGES) / total_seconds
            generated_tokens = sum(
                bf16_languages[lang]["generated_tokens_per_second_e2e"] * language_seconds[lang]
                for lang in LANGUAGES
            )
            token_speed = generated_tokens / total_seconds
            truncation_count = sum(
                bf16_languages[lang]["truncated_outputs"]
                for lang in LANGUAGES
            )
        else:
            vals = means[key]
            speed = vals["speed_rows_s"]
            token_speed = vals["speed_tokens_s"]
            truncation_count = truncated[key]
        report.append(
            f"| {DISPLAY[key]} | {fmt(vals['spBLEU'])} | {fmt(vals['BLEU-13a'])} | "
            f"{fmt(vals['chrF++'])} | {fmt(vals['COMET'],4)} | {fmt(vals['MetricX'],4)} | "
            f"{fmt(vals['XCOMET'],4)} | {fmt(speed,2)} | {fmt(token_speed,2)} | {truncation_count} |"
        )

    report.extend(["", "## 八语种分数", "", "| 语种 | 量化版本 | spBLEU ↑ | chrF++ ↑ | COMET ↑ | MetricX ↓ | XCOMET ↑ |", "|---|---|---:|---:|---:|---:|---:|"])
    for lang in LANGUAGES:
        for key in MODEL_KEYS:
            v = per_language[key][lang]
            report.append(
                f"| {LANG_NAMES[lang]} | {DISPLAY[key]} | {fmt(v['spBLEU'])} | {fmt(v['chrF++'])} | "
                f"{fmt(v['COMET'],4)} | {fmt(v['MetricX'],4)} | {fmt(v['XCOMET'],4)} |"
            )
        b = baseline["spBLEU_chrFpp"]["index"]["languages"][lang]
        c = baseline["comet_wmt22_da"]["index"]["languages"][lang]["mean"]
        mx = baseline["metricx24"]["scores"]["index"]["languages"][lang]["mean"]
        xc = baseline["xcomet"]["scores"]["index"]["languages"][lang]["mean"]
        report.append(
            f"| {LANG_NAMES[lang]} | {DISPLAY['bf16']} | {fmt(b['spBLEU'])} | {fmt(b['chrF++'])} | "
            f"{fmt(c,4)} | {fmt(mx,4)} | {fmt(xc,4)} |"
        )

    report.extend(["", "## BLEU-13a 分语种（6语）", "", "| 语种 | BF16 参考 | Q8_0 | Q6_K | Q4_K_M |", "|---|---:|---:|---:|---:|"])
    base_bleu = baseline["sacrebleu_13a"]["scores"]["index"]
    for lang in six:
        values = []
        for key in MODEL_KEYS:
            score = bleu13a[key]["languages"][lang]["BLEU"]
            values.append(fmt(score))
        report.append(f"| {LANG_NAMES[lang]} | {fmt(base_bleu[lang]['bleu'])} | " + " | ".join(values) + " |")

    report.extend(["", "## 速度", "", "| 语种 | Q8_0 条/秒 | Q6_K 条/秒 | Q4_K_M 条/秒 |", "|---|---:|---:|---:|"])
    for lang in LANGUAGES:
        cells = []
        for key in MODEL_KEYS:
            meta = read_json(RUN / key / "metadata.json")
            cells.append(fmt(meta["language_results"][lang]["sentences_per_second"], 2))
        report.append(f"| {LANG_NAMES[lang]} | " + " | ".join(cells) + " |")
    report.append("| 分语种速度算术均值 | " + " | ".join(
        fmt(statistics.fmean(read_json(RUN / k / "metadata.json")["language_results"][lang]["sentences_per_second"]
                             for lang in LANGUAGES), 2) for k in MODEL_KEYS) + " |")
    report.append("| 全量吞吐（8,096 / 总耗时） | " + " | ".join(fmt(means[k]["speed_rows_s"], 2) for k in MODEL_KEYS) + " |")

    telemetry_path = RUN / "gpu1_telemetry.csv"
    if telemetry_path.is_file():
        with telemetry_path.open(encoding="utf-8", newline="") as stream:
            samples = list(csv.DictReader(stream))
        report.extend([
            "",
            "## 推理期间 GPU1 使用",
            "",
            "| 量化版本 | P90 利用率 | 峰值利用率 | 峰值显存 | 采样点 |",
            "|---|---:|---:|---:|---:|",
        ])
        for key in MODEL_KEYS:
            metadata = read_json(RUN / key / "metadata.json")
            start = datetime.fromisoformat(metadata["started_utc"])
            end = datetime.fromisoformat(metadata["completed_utc"])
            rows = [
                row for row in samples
                if row["phase"] == key and row["gpu_index"] == "1"
                and start <= datetime.fromisoformat(row["timestamp_utc"]) <= end
            ]
            if rows:
                util = sorted(float(row["gpu_utilization_percent"]) for row in rows)
                p90 = util[min(len(util) - 1, math.ceil(0.9 * (len(util) - 1)))]
                peak_memory = max(float(row["memory_used_mib"]) for row in rows) / 1024
                report.append(
                    f"| {DISPLAY[key]} | {p90:.0f}% | {max(util):.0f}% | "
                    f"{peak_memory:.2f} GiB | {len(rows)} |"
                )
        report.append("采样间隔 5 秒；采集器启动较晚，Q8_0 仅覆盖推理末段，Q6_K 和 Q4_K_M 覆盖了大部分推理过程。")

    report.extend([
        "",
        "Q8_0、Q6_K、Q4_K_M 的模型文件分别约 2.08 GB、1.61 GB、1.31 GB。",
        "指标计算定义、优缺点和 FLORES 数据限制见此前的 [基准报告](../zh_translation_benchmark_full_batch128_20261020/RESULTS_SUMMARY.md)。",
        "",
    ])
    control_report = RUN / "F16_CONTROL_REPORT.md"
    if control_report.is_file():
        report.extend(["", "---", ""])
        report.extend("#" + line if line.startswith("#") else line for line in control_report.read_text(encoding="utf-8").splitlines())
    destination = RUN / "RESULTS_SUMMARY.md"
    destination.write_text("\n".join(report), encoding="utf-8")
    print(destination)


if __name__ == "__main__":
    main()
