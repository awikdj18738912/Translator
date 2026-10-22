#!/usr/bin/env python3
"""Compare official F16 GGUF, Q8_0 GGUF, and the original HF BF16 outputs."""
from __future__ import annotations

from collections import Counter, defaultdict
import csv
from datetime import datetime
import json
from pathlib import Path
import statistics

import numpy as np

RUN = Path(__file__).resolve().parent
ROOT = RUN.parents[1]
F16 = RUN / "f16_gguf"
BASE = ROOT / "results/zh_translation_benchmark_full_batch128_20261020"
LANGS = ("en", "fr", "de", "es", "ru", "ja", "ko", "ar")
NAMES = dict(zip(LANGS, ("英语", "法语", "德语", "西班牙语", "俄语", "日语", "韩语", "阿拉伯语")))
DISPLAY = {"f16": "F16 GGUF / llama.cpp", "q8": "Q8_0 GGUF / llama.cpp", "hf": "BF16 / Transformers"}
SAMPLES = 10000


def read(path):
    return json.loads(path.read_text(encoding="utf-8"))


def rows(path):
    with path.open(encoding="utf-8") as stream:
        return [json.loads(line) for line in stream if line.strip()]


def keyed(items):
    result = {(row["language"], str(row["id"])): row for row in items}
    assert len(result) == len(items) == 8096
    assert Counter(row["language"] for row in items) == Counter({l: 1012 for l in LANGS})
    return result


def get_scores(path, field, model_key=None):
    items = rows(path)
    if model_key is not None:
        items = [row for row in items if row["model_key"] == model_key]
    return {key: float(row[field]) for key, row in keyed(items).items()}


def paired_bootstrap(a, b):
    """Resample Chinese source IDs together across the eight target languages."""
    assert a.keys() == b.keys()
    ids = sorted({ident for lang, ident in a}, key=int)
    delta = np.asarray([[a[(lang, i)] - b[(lang, i)] for i in ids] for lang in LANGS])
    observed = np.concatenate([delta.mean(axis=1), [delta.mean()]])
    estimates = np.empty((len(LANGS) + 1, SAMPLES))
    rng = np.random.default_rng(20261021)
    for start in range(0, SAMPLES, 200):
        n = min(200, SAMPLES - start)
        indices = rng.integers(0, len(ids), size=(n, len(ids)))
        language_means = delta[:, indices].mean(axis=2)
        estimates[:-1, start:start + n] = language_means
        estimates[-1, start:start + n] = language_means.mean(axis=0)
    lo, hi = np.percentile(estimates, [2.5, 97.5], axis=1)
    return {scope: {"delta_a_minus_b": float(observed[i]), "ci95_low": float(lo[i]),
                    "ci95_high": float(hi[i]), "ci_excludes_zero": bool(lo[i] > 0 or hi[i] < 0)}
            for i, scope in enumerate((*LANGS, "macro"))}


def fmt(value, digits=4):
    return f"{value:.{digits}f}"


def main():
    baseline = read(BASE / "full_language_results.json")
    alignment = read(F16 / "gguf_alignment.json")
    assert not alignment["model_metadata_differences"] and not alignment["tensor_shape_differences"]
    prompt_alignment = read(F16 / "prompt_token_alignment.json")
    prompt_matches = sum(check["prompt_text_equal"] and check["token_ids_equal"]
                         for check in prompt_alignment["sample_checks"])
    predictions = {"f16": keyed(rows(F16 / "predictions.jsonl")),
                   "q8": keyed(rows(RUN / "q8_0/predictions.jsonl")),
                   "hf": keyed(rows(BASE / "index/predictions.jsonl"))}
    for system in ("f16", "q8"):
        assert predictions[system].keys() == predictions["hf"].keys()
        for key, row in predictions[system].items():
            assert all(row[field] == predictions["hf"][key][field] for field in ("source", "reference"))

    score_files = {
        "f16": {"COMET": (F16 / "scoring/comet/index.scores.jsonl", "comet", None),
                "MetricX": (F16 / "scoring/metricx/output_all.jsonl", "prediction", "f16_gguf"),
                "XCOMET": (F16 / "scoring/xcomet/index.scores.jsonl", "xcomet_score", None)},
        "q8": {"COMET": (RUN / "scoring/comet/index.scores.jsonl", "comet", None),
               "MetricX": (RUN / "scoring/metricx/output_all.jsonl", "prediction", "q8_0"),
               "XCOMET": (RUN / "scoring/xcomet/index.scores.jsonl", "xcomet_score", None)},
        "hf": {"COMET": (BASE / "comet_wmt22/index.scores.jsonl", "comet", None),
               "MetricX": (BASE / "metricx/metricx24_xl_batch128.jsonl", "prediction", "index"),
               "XCOMET": (BASE / "xcomet/index.scores.jsonl", "xcomet_score", None)},
    }
    scores = {system: {metric: get_scores(*args) for metric, args in files.items()}
              for system, files in score_files.items()}
    per_language = defaultdict(dict)
    macros = defaultdict(dict)
    for system in DISPLAY:
        for lang in LANGS:
            per_language[system][lang] = {}
        if system != "hf":
            folder = F16 if system == "f16" else RUN / "q8_0"
            sacre = read(folder / "summary.json")
            for lang in LANGS:
                item = sacre["languages"][lang]
                assert item["count"] == 1012
                per_language[system][lang].update(spBLEU=item["spBLEU"], chrFpp=item["chrFpp"])
            bleu13a = read(RUN / "scoring/bleu-13a-six" / ("f16_gguf" if system == "f16" else "q8_0") / "summary.json")
            macros[system]["BLEU13a"] = statistics.fmean(v["BLEU"] for v in bleu13a["languages"].values())
            metadata = read(folder / "metadata.json")
            macros[system].update(speed=metadata["sentences_per_second"], tokens_per_second=metadata["generated_tokens_per_second"],
                                  truncated=sum(bool(r.get("truncated")) for r in predictions[system].values()))
        else:
            for lang in LANGS:
                item = baseline["spBLEU_chrFpp"]["index"]["languages"][lang]
                per_language[system][lang].update(spBLEU=item["spBLEU"], chrFpp=item["chrF++"])
            macros[system]["BLEU13a"] = baseline["sacrebleu_13a"]["scores"]["index"]["macro_6_language"]
            speed = baseline["speed_full_inference"]["index"]["languages"]
            seconds = {l: speed[l]["count"] / speed[l]["sentences_per_second"] for l in LANGS}
            macros[system].update(speed=8096 / sum(seconds.values()), tokens_per_second=sum(
                speed[l]["generated_tokens_per_second_e2e"] * seconds[l] for l in LANGS) / sum(seconds.values()),
                truncated=sum(bool(r.get("truncated")) for r in predictions[system].values()))
        for metric in ("COMET", "MetricX", "XCOMET"):
            for lang in LANGS:
                per_language[system][lang][metric] = statistics.fmean(
                    value for (language, ident), value in scores[system][metric].items() if language == lang)
        for metric in ("spBLEU", "chrFpp", "COMET", "MetricX", "XCOMET"):
            macros[system][metric] = statistics.fmean(per_language[system][lang][metric] for lang in LANGS)

    comparisons = {}
    for a, b in (("q8", "f16"), ("f16", "hf")):
        comparisons[f"{a}_minus_{b}"] = {m: paired_bootstrap(scores[a][m], scores[b][m])
                                           for m in ("COMET", "MetricX", "XCOMET")}
    agreement = {}
    for a, b in (("q8", "f16"), ("f16", "hf"), ("q8", "hf")):
        same = sum(predictions[a][key]["hypothesis"].strip() == predictions[b][key]["hypothesis"].strip()
                   for key in predictions[a])
        agreement[f"{a}_vs_{b}"] = {"same": same, "total": 8096, "percent": same / 8096 * 100}
    report = ["# F16 GGUF 同后端对照", "",
        "官方 F16 GGUF 与 Q8_0 使用同一 llama.cpp 构建、提示、温度 0、最大 512 tokens、128 并发以及相同服务参数；各测试 8,096 条，每个语种 1,012 条。",
        "两份 GGUF 的模型配置、分词器、聊天模板及 335 个张量形状全部一致；八语种各抽一条检查实际提示文本和 token ID，" + f"{prompt_matches}/8 条与 Transformers 一致。",
        "原始 Transformers 使用 BF16，因此 F16 llama.cpp 与 BF16 Transformers 的差异仍包含精度和后端两项因素。",
        "", "## 八语宏平均与速度", "",
        "| 版本 | spBLEU ↑ | BLEU-13a ↑（6语） | chrF++ ↑ | COMET ↑ | MetricX ↓ | XCOMET ↑ | 全量句/秒 |",
        "|---|---:|---:|---:|---:|---:|---:|---:|"]
    for system in DISPLAY:
        v = macros[system]
        report.append(f"| {DISPLAY[system]} | {v['spBLEU']:.3f} | {v['BLEU13a']:.3f} | {v['chrFpp']:.3f} | "
                      f"{v['COMET']:.6f} | {v['MetricX']:.6f} | {v['XCOMET']:.6f} | {v['speed']:.2f} |")
    f16_metadata = read(F16 / "metadata.json")
    q8_metadata = read(RUN / "q8_0/metadata.json")
    with (F16 / "gpu1_telemetry.csv").open(newline="") as stream:
        telemetry = list(csv.DictReader(stream))
    begin = datetime.fromisoformat(f16_metadata["started_utc"])
    end = datetime.fromisoformat(f16_metadata["completed_utc"])
    telemetry = [r for r in telemetry if begin <= datetime.fromisoformat(r["timestamp_utc"]) <= end]
    peak_gib = max(float(r["memory_used_mib"]) for r in telemetry) / 1024
    faster = (macros["q8"]["speed"] / macros["f16"]["speed"] - 1) * 100
    report.extend(["", f"F16 / Q8_0 模型文件分别为 {f16_metadata['model_bytes']/1e9:.2f} / {q8_metadata['model_bytes']/1e9:.2f} GB；"
        f"F16 推理观测显存峰值 {peak_gib:.2f} GiB。本次单轮全量测速中，Q8_0 吞吐比 F16 高 {faster:.1f}%。"])
    report.extend(["", "## Q8_0 − F16：同后端质量差", "",
        "| 指标 | 分数差 | 95% 配对 bootstrap 区间 | 区间是否覆盖 0 |", "|---|---:|---|---|"])
    for metric, direction in (("COMET", "↑"), ("MetricX", "↓"), ("XCOMET", "↑")):
        stats = comparisons["q8_minus_f16"][metric]["macro"]
        report.append(f"| {metric} {direction} | {stats['delta_a_minus_b']:+.6f} | "
                      f"[{stats['ci95_low']:+.6f}, {stats['ci95_high']:+.6f}] | " + ("否" if stats["ci_excludes_zero"] else "是") + " |")
    if not any(comparisons["q8_minus_f16"][m]["macro"]["ci_excludes_zero"] for m in ("COMET", "MetricX", "XCOMET")):
        report.extend(["", "三个神经指标的差值区间均覆盖 0，本测试集没有给出 Q8_0 优于 F16 的明确统计支持。"])
    report.extend(["", "## F16 llama.cpp − BF16 Transformers", "",
        "| 指标 | 分数差 | 95% 配对 bootstrap 区间 | 区间是否覆盖 0 |", "|---|---:|---|---|"])
    for metric, direction in (("COMET", "↑"), ("MetricX", "↓"), ("XCOMET", "↑")):
        stats = comparisons["f16_minus_hf"][metric]["macro"]
        report.append(f"| {metric} {direction} | {stats['delta_a_minus_b']:+.6f} | "
                      f"[{stats['ci95_low']:+.6f}, {stats['ci95_high']:+.6f}] | " + ("否" if stats["ci_excludes_zero"] else "是") + " |")
    report.extend(["", "按中文源句 ID 成组配对重采样 10,000 次，同一源句的八个翻译方向一起抽取；表中为各指标均值差的 95% 区间。它反映本测试集的抽样不确定性，各指标分别展示，未计算加权综合分。",
        "连续批处理的实际调度未逐 token 固定，因此微小差异仍可能包含运行数值变化。",
        "", "## F16 全量分语种结果", "",
        "| 语种 | spBLEU ↑ | BLEU-13a ↑ | chrF++ ↑ | COMET ↑ | MetricX ↓ | XCOMET ↑ | 句/秒 |",
        "|---|---:|---:|---:|---:|---:|---:|---:|"])
    f16_bleu = read(RUN / "scoring/bleu-13a-six/f16_gguf/summary.json")
    for lang in LANGS:
        v = per_language["f16"][lang]
        bleu = f"{f16_bleu['languages'][lang]['BLEU']:.3f}" if lang in f16_bleu["languages"] else "—"
        report.append(f"| {NAMES[lang]} | {v['spBLEU']:.3f} | {bleu} | {v['chrFpp']:.3f} | {v['COMET']:.6f} | "
                      f"{v['MetricX']:.6f} | {v['XCOMET']:.6f} | {f16_metadata['language_results'][lang]['sentences_per_second']:.2f} |")
    report.extend(["", "## 输出文本完全一致比例", "", "| 对照 | 一致条数 / 全量 | 比例 |", "|---|---:|---:|"])
    for a, b in (("q8", "f16"), ("f16", "hf"), ("q8", "hf")):
        v = agreement[f"{a}_vs_{b}"]
        report.append(f"| {DISPLAY[a]} vs {DISPLAY[b]} | {v['same']} / 8096 | {v['percent']:.2f}% |")
    report.extend(["", "完全一致按去除首尾空白后的译文逐字比较；文字不同也可能语义等价，该比例不表示错误率。",
        "", f"F16 截断输出：{macros['f16']['truncated']} 条，按原输出参与评分。",
        "权重来源：[IndexTeam 官方 GGUF](https://huggingface.co/IndexTeam/Index-Translate-2B-GGUF/tree/3a0eca6ce931d88d00614d427ab3ed8ff0743902)；本次使用 `Index-Translate-2B.f16.gguf`，不是本地转换的 BF16 文件。",
        "记录：[下载与 SHA-256](f16_gguf/download_manifest.json)、[GGUF 配置对齐](f16_gguf/gguf_alignment.json)、[实际提示与 token 对齐](f16_gguf/prompt_token_alignment.json)。",
        ""])
    summary = {"dataset_rows_per_system": 8096, "rows_per_language": 1012, "bootstrap_samples": SAMPLES,
               "bootstrap_unit": "Chinese source ID shared across eight target languages", "macros": macros,
               "languages": per_language, "comparisons": comparisons, "exact_output_agreement": agreement,
               "gguf_alignment": alignment, "prompt_token_alignment": prompt_alignment}
    (F16 / "control_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n")
    path = RUN / "F16_CONTROL_REPORT.md"
    path.write_text("\n".join(report), encoding="utf-8")
    print(path)


if __name__ == "__main__":
    main()
