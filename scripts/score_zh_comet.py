#!/usr/bin/env python3
"""Score saved Chinese-to-foreign translations with COMET and compare systems.

COMET is kept in a separate environment because it requires Transformers <5,
while the translation benchmark uses Transformers 5.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import gc
import hashlib
import json
import os
from pathlib import Path
import statistics
import time

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_INPUT = ROOT / "results/zh_translation_benchmark_full_batch128_20261020"
DEFAULT_OUTPUT = DEFAULT_INPUT / "comet_wmt22"
MODELS = ("index", "hy", "lmt")
LANGUAGES = ("en", "fr", "de", "es", "ru", "ja", "ko", "ar")
MODEL_ID = "Unbabel/wmt22-comet-da"


def read_jsonl(path: Path) -> list[dict]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    temp.replace(path)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def sort_key(row: dict) -> tuple[int, int]:
    try:
        sentence_id = int(row["id"])
    except (TypeError, ValueError):
        sentence_id = str(row["id"])
    return LANGUAGES.index(row["language"]), sentence_id


def load_system_rows(input_dir: Path, models: list[str], languages: list[str], limit: int | None) -> dict[str, list[dict]]:
    systems: dict[str, list[dict]] = {}
    reference_rows: list[dict] | None = None
    reference_keys: list[tuple[str, str]] | None = None
    for name in models:
        path = input_dir / name / "predictions.jsonl"
        if not path.is_file():
            raise FileNotFoundError(path)
        rows = read_jsonl(path)
        rows = [row for row in rows if row.get("language") in languages]
        rows.sort(key=sort_key)
        if limit is not None:
            chosen = set()
            for language in languages:
                candidates = [r for r in rows if r["language"] == language]
                chosen.update((str(r["id"]), language) for r in candidates[:limit])
            rows = [r for r in rows if (str(r["id"]), r["language"]) in chosen]
        for row in rows:
            if row.get("error"):
                raise ValueError(f"{name} has a failed translation: {row['id']} {row['language']}: {row['error']}")
            if not row.get("hypothesis", "").strip():
                raise ValueError(f"{name} has an empty translation: {row['id']} {row['language']}")
        keys = [(str(r["id"]), r["language"]) for r in rows]
        if len(keys) != len(set(keys)):
            raise ValueError(f"Duplicate case IDs in {path}")
        if reference_keys is None:
            reference_keys = keys
            reference_rows = rows
        elif keys != reference_keys:
            raise ValueError(f"Systems do not contain the same ID/language rows: {name}")
        else:
            assert reference_rows is not None
            for expected, current in zip(reference_rows, rows):
                for field in ("source", "reference"):
                    if expected[field] != current[field]:
                        raise ValueError(f"Input {field} differs at {current['id']} {current['language']}")
        systems[name] = rows
    return systems


def extract_scores(prediction: object) -> list[float]:
    values = getattr(prediction, "scores", None)
    if values is None and isinstance(prediction, dict):
        values = prediction.get("scores")
    if values is None:
        raise RuntimeError(f"COMET returned no sentence scores: {type(prediction)}")
    return [float(value) for value in values]


def score_system(model: object, rows: list[dict], batch_size: int, device_count: int, torch: object) -> tuple[list[float], int]:
    data = [{"src": r["source"], "mt": r["hypothesis"], "ref": r["reference"]} for r in rows]
    current_batch_size = batch_size
    while True:
        try:
            prediction = model.predict(data, batch_size=current_batch_size, gpus=device_count, num_workers=4)
            scores = extract_scores(prediction)
            if len(scores) != len(rows):
                raise RuntimeError(f"COMET returned {len(scores)} scores for {len(rows)} rows")
            return scores, current_batch_size
        except RuntimeError as error:
            if "out of memory" not in str(error).lower() or current_batch_size <= 16:
                raise
            next_batch_size = max(16, current_batch_size // 2)
            print(f"CUDA OOM at batch={current_batch_size}; retrying this model at batch={next_batch_size}", flush=True)
            gc.collect()
            try:
                model.cpu()
            finally:
                torch.cuda.empty_cache()
            current_batch_size = next_batch_size


def grouped_scores(models: list[str], systems: dict[str, list[dict]], score_map: dict[str, dict[tuple[str, str], float]]) -> dict:
    result = {}
    for model_name in models:
        values = score_map[model_name]
        by_language = {}
        for language in LANGUAGES:
            scores = [values[(str(row["id"]), language)] for row in systems[model_name] if row["language"] == language]
            if scores:
                by_language[language] = {"count": len(scores), "mean": statistics.fmean(scores), "stddev": statistics.pstdev(scores)}
        macro = statistics.fmean(x["mean"] for x in by_language.values())
        result[model_name] = {"macro_mean": macro, "languages": by_language}
    return result


def bootstrap_delta(strata: list[list[float]], samples: int, seed: int) -> dict:
    """Stratified paired bootstrap for A-B; language weights stay equal."""
    import numpy as np

    observed = statistics.fmean(statistics.fmean(group) for group in strata)
    rng = np.random.default_rng(seed)
    estimates = np.empty(samples, dtype=np.float64)
    chunk_size = 256
    for start in range(0, samples, chunk_size):
        count = min(chunk_size, samples - start)
        language_means = []
        for group in strata:
            arr = np.asarray(group, dtype=np.float64)
            indices = rng.integers(0, len(arr), size=(count, len(arr)))
            language_means.append(arr[indices].mean(axis=1))
        estimates[start:start + count] = np.stack(language_means, axis=0).mean(axis=0)
    lo, hi = np.percentile(estimates, [2.5, 97.5])
    p_lo = (int(np.count_nonzero(estimates <= 0)) + 1) / (samples + 1)
    p_hi = (int(np.count_nonzero(estimates >= 0)) + 1) / (samples + 1)
    return {
        "delta_a_minus_b": float(observed),
        "ci95_low": float(lo),
        "ci95_high": float(hi),
        "p_two_sided": float(min(1.0, 2 * min(p_lo, p_hi))),
        "bootstrap_samples": samples,
    }


def holm_adjust(items: list[dict], family_key: str) -> None:
    families: dict[str, list[dict]] = {}
    for item in items:
        families.setdefault(str(item[family_key]), []).append(item)
    for group in families.values():
        ordered = sorted(group, key=lambda item: item["p_two_sided"])
        n = len(ordered)
        running = 0.0
        for rank, item in enumerate(ordered):
            adjusted = min(1.0, (n - rank) * item["p_two_sided"])
            running = max(running, adjusted)
            item["p_holm"] = running


def make_comparisons(models: list[str], systems: dict[str, list[dict]], score_map: dict[str, dict[tuple[str, str], float]], samples: int, seed: int) -> list[dict]:
    comparisons = []
    pairs = [(models[i], models[j]) for i in range(len(models)) for j in range(i + 1, len(models))]
    for language in LANGUAGES:
        keys = [(str(r["id"]), language) for r in systems[models[0]] if r["language"] == language]
        if not keys:
            continue
        for a, b in pairs:
            deltas = [score_map[a][key] - score_map[b][key] for key in keys]
            stats = bootstrap_delta([deltas], samples, seed + len(comparisons) * 17)
            comparisons.append({"scope": language, "model_a": a, "model_b": b, **stats})
    if len(models) > 1:
        language_strata = []
        for language in LANGUAGES:
            language_strata.append([
                score_map[models[0]][(str(r["id"]), language)] - score_map[models[1]][(str(r["id"]), language)]
                for r in systems[models[0]] if r["language"] == language
            ])
        # Macro comparisons are computed independently for every pair.
        for pair_index, (a, b) in enumerate(pairs):
            strata = []
            for language in LANGUAGES:
                strata.append([
                    score_map[a][(str(r["id"]), language)] - score_map[b][(str(r["id"]), language)]
                    for r in systems[a] if r["language"] == language
                ])
            if all(strata):
                stats = bootstrap_delta(strata, samples, seed + 100000 + pair_index * 17)
                comparisons.append({"scope": "macro_equal_language_weight", "model_a": a, "model_b": b, **stats})
    # Correct the three pairwise comparisons within each target language and
    # separately across the three macro comparisons.
    for item in comparisons:
        item["holm_family"] = item["scope"]
    holm_adjust(comparisons, "holm_family")
    return comparisons


def write_markdown(path: Path, summary: dict, comparisons: list[dict], models: list[str]) -> None:
    lines = [
        "# COMET 补充评测",
        "",
        f"- 指标：`{summary['model_id']}` (unbabel-comet {summary['comet_version']})",
        f"- 样本：{summary['total_cases']:,} 条；目标语言宏平均按语言等权",
        f"- GPU：物理 GPU {summary['physical_gpu']}；请求 batch={summary['batch_size']}；实际 batch=" + ", ".join(f"{name}:{summary['runtime'][name]['batch_size_used']}" for name in models),
        "- COMET 分数用于模型相对排序，不是正确率百分比；本评分器训练覆盖范围虽含这些语言，但主要训练方向与中文→多语方向不完全相同。",
        "",
        "## 按目标语言的分数",
        "",
        "| 模型 | 宏平均 | " + " | ".join(LANGUAGES) + " |",
        "|---|---:|" + "---:|" * len(LANGUAGES),
    ]
    for model in models:
        stats = summary["systems"][model]
        cells = [model, f"{stats['macro_mean']:.4f}"]
        cells.extend(f"{stats['languages'][lang]['mean']:.4f}" if lang in stats["languages"] else "—" for lang in LANGUAGES)
        lines.append("| " + " | ".join(cells) + " |")
    lines += [
        "",
        "## 配对 bootstrap",
        "",
        "Δ=模型 A 分数−模型 B 分数；95% 区间按句对分层重采样。Holm 校正按目标语言分别进行，宏平均另成一组。正值表示 A 分数较高。",
        "",
        "| 范围 | A | B | Δ | 95% CI | p | Holm p |",
        "|---|---|---|---:|---:|---:|---:|",
    ]
    for item in comparisons:
        ci = f"[{item['ci95_low']:.4f}, {item['ci95_high']:.4f}]"
        lines.append(f"| {item['scope']} | {item['model_a']} | {item['model_b']} | {item['delta_a_minus_b']:.4f} | {ci} | {item['p_two_sided']:.4g} | {item['p_holm']:.4g} |")
    lines.append("")
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--models", default=",".join(MODELS), help="Comma-separated subset: index,hy,lmt")
    parser.add_argument("--languages", default=",".join(LANGUAGES))
    parser.add_argument("--limit-per-language", type=int, default=0, help="Debug only; 0 scores all saved rows")
    parser.add_argument("--model-id", default=MODEL_ID)
    parser.add_argument("--gpu", type=int, default=1, help="Physical GPU index")
    parser.add_argument("--hf-endpoint", default="https://huggingface.co", help="Model hub URL; defaults to the official hub")
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--matmul-precision", choices=("highest", "high", "medium"), default="high")
    parser.add_argument("--bootstrap-samples", type=int, default=10000)
    parser.add_argument("--seed", type=int, default=20261020)
    args = parser.parse_args()
    models = args.models.split(",")
    languages = args.languages.split(",")
    if not models or any(model not in MODELS for model in models):
        parser.error(f"--models must be a comma-separated subset of {MODELS}")
    if not languages or any(language not in LANGUAGES for language in languages):
        parser.error(f"--languages must be a comma-separated subset of {LANGUAGES}")
    if args.batch_size < 1 or args.bootstrap_samples < 100:
        parser.error("batch size must be positive and bootstrap samples must be at least 100")
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
    os.environ["HF_ENDPOINT"] = args.hf_endpoint
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

    systems = load_system_rows(args.input_dir, models, languages, args.limit_per_language or None)
    args.output.mkdir(parents=True, exist_ok=True)
    print(f"Loading {args.model_id}; physical GPU {args.gpu}; {len(systems[models[0]]):,} cases/system; batch={args.batch_size}", flush=True)

    from comet import __version__ as comet_version
    from comet import download_model, load_from_checkpoint
    import torch

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable; refusing to silently run the planned GPU score on CPU")
    torch.cuda.init()
    torch.cuda.set_device(0)
    torch.set_float32_matmul_precision(args.matmul_precision)
    if torch.cuda.device_count() != 1:
        print(f"Visible CUDA devices after masking: {torch.cuda.device_count()}", flush=True)
    checkpoint = download_model(args.model_id)
    metric = load_from_checkpoint(checkpoint)
    checkpoint_path = Path(checkpoint)
    score_map: dict[str, dict[tuple[str, str], float]] = {}
    run_stats = {}
    for model_name in models:
        print(f"Scoring {model_name}...", flush=True)
        torch.cuda.reset_peak_memory_stats(0)
        started = time.perf_counter()
        scores, batch_size_used = score_system(metric, systems[model_name], args.batch_size, 1, torch)
        elapsed = time.perf_counter() - started
        score_map[model_name] = {}
        score_path = args.output / f"{model_name}.scores.jsonl"
        with score_path.open("w", encoding="utf-8") as handle:
            for row, value in zip(systems[model_name], scores):
                key = (str(row["id"]), row["language"])
                score_map[model_name][key] = value
                handle.write(json.dumps({"id": row["id"], "language": row["language"], "comet": value}, ensure_ascii=False) + "\n")
        run_stats[model_name] = {
            "batch_size_used": batch_size_used,
            "scoring_seconds": elapsed,
            "scores_sha256": sha256(score_path),
            "gpu_peak_allocated_bytes": int(torch.cuda.max_memory_allocated(0)),
            "gpu_peak_reserved_bytes": int(torch.cuda.max_memory_reserved(0)),
        }
        print(f"Finished {model_name}: mean={statistics.fmean(scores):.4f}, {elapsed:.1f}s", flush=True)

    system_summary = grouped_scores(models, systems, score_map)
    comparisons = make_comparisons(models, systems, score_map, args.bootstrap_samples, args.seed) if len(models) > 1 else []
    summary = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "input_dir": str(args.input_dir.resolve()),
        "model_id": args.model_id,
        "comet_version": comet_version,
        "checkpoint_path": str(checkpoint_path.resolve()),
        "checkpoint_size_bytes": checkpoint_path.stat().st_size,
        "checkpoint_sha256": sha256(checkpoint_path),
        "physical_gpu": args.gpu,
        "gpu_name": torch.cuda.get_device_name(0),
        "torch_version": torch.__version__,
        "batch_size": args.batch_size,
        "matmul_precision": args.matmul_precision,
        "total_cases": len(systems[models[0]]),
        "languages": languages,
        "models": models,
        "bootstrap_samples": args.bootstrap_samples,
        "seed": args.seed,
        "systems": system_summary,
        "runtime": run_stats,
        "limitations": [
            "COMET score is a learned quality estimate, not a percentage-correct accuracy.",
            "XLM-R covers the evaluated languages, while the metric training directions do not fully match Chinese-to-eight-language translation.",
            "FLORES-101 devtest is public written text, not domain-specific speech-recognition output.",
        ],
    }
    write_json(args.output / "summary.json", summary)
    write_json(args.output / "paired_bootstrap.json", {"comparisons": comparisons})
    write_markdown(args.output / "report.md", summary, comparisons, models)
    print(f"Wrote {args.output / 'report.md'}", flush=True)


if __name__ == "__main__":
    main()
