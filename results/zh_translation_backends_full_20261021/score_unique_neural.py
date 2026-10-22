#!/usr/bin/env python3
"""Fresh, resumable neural scoring for exact-input-deduplicated outputs.

Input records are prepared by run_metric_pipeline.py. The score file is indexed
by unique_id and deliberately independent of backend/model keys.
"""
from __future__ import annotations

import argparse
import gc
import hashlib
import json
import os
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
RUN = Path(__file__).resolve().parent
COMET_PY = "/home/aim0/data/venvs/translation-comet/bin/python"
METRICX_PY = "/home/aim0/data/venvs/translation-metricx/bin/python"
METRICX_SCRIPT = ROOT / "results/zh_translation_benchmark_full_batch128_20261020/metricx/score_metricx24.py"
XCOMET_SCRIPT = ROOT / "results/zh_translation_benchmark_full_batch128_20261020/metricx/score_xcomet_full.py"
CHECKPOINT = "/home/aim0/data/hf-cache/hub/models--Unbabel--XCOMET-XL/snapshots/6a123c5e8e6dccab25e5fcffa3c8b417abadb462/checkpoints/model.ckpt"


def read(path):
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def sha256(path):
    digest = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def save(path, obj):
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    tmp = p.with_suffix(p.suffix + ".tmp")
    tmp.write_text(json.dumps(obj, ensure_ascii=False, indent=2) + "\n")
    tmp.replace(p)


def restore_progress(out, rows):
    progress = Path(out).with_suffix(".progress.jsonl")
    if not progress.is_file():
        return [], progress
    saved = read(progress)
    if len(saved) > len(rows) or any(item.get("unique_id") != i for i, item in enumerate(saved)):
        progress.unlink()
        return [], progress
    return [float(item["score"]) for item in saved], progress


def checkpoint(progress, rows, values):
    tmp = progress.with_suffix(progress.suffix + ".tmp")
    with tmp.open("w", encoding="utf-8") as f:
        for row, score in zip(rows, values):
            f.write(json.dumps({"unique_id": row["unique_id"], "score": float(score)}, ensure_ascii=False) + "\n")
        f.flush()
        os.fsync(f.fileno())
    tmp.replace(progress)


def env_for_gpu():
    env = os.environ.copy()
    env.update(CUDA_VISIBLE_DEVICES="1", HF_HOME="/home/aim0/data/hf-cache",
               HF_HUB_CACHE="/home/aim0/data/hf-cache/hub", HF_ENDPOINT="https://huggingface.co",
               HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1", PYTHONUNBUFFERED="1")
    return env


def score_comet(rows, out, batch_size):
    import torch
    from comet import load_from_checkpoint
    torch.set_float32_matmul_precision("high")
    model_path = "/home/aim0/data/hf-cache/hub/models--Unbabel--wmt22-comet-da/snapshots"
    candidates = list(Path(model_path).glob("*/checkpoints/model.ckpt"))
    if not candidates:
        raise FileNotFoundError("Cached Unbabel/wmt22-comet-da checkpoint missing")
    model = load_from_checkpoint(str(candidates[0]))
    results, progress = restore_progress(out, rows)
    started = time.perf_counter()
    if not results:
        current = batch_size
        while True:
            try:
                pred = model.predict([{"src": r["source"], "mt": r["hypothesis"], "ref": r["reference"]} for r in rows],
                                     batch_size=current, gpus=1, num_workers=4)
                results = [float(x) for x in pred.scores]
                checkpoint(progress, rows, results)
                batch_size = current
                break
            except RuntimeError as error:
                if "out of memory" not in str(error).lower() or current <= 16:
                    raise
                current = max(16, current // 2)
                gc.collect()
                torch.cuda.empty_cache()
                print(f"COMET CUDA OOM; retrying batch_size={current}", flush=True)
        print(f"COMET {len(results)}/{len(rows)}", flush=True)
    del model
    gc.collect()
    torch.cuda.empty_cache()
    return results, time.perf_counter() - started, batch_size


def score_metricx(rows, out, batch_size):
    import torch
    import transformers
    from metricx24.models import MT5ForRegression
    torch.set_num_threads(min(torch.get_num_threads(), 4))
    tokenizer = transformers.AutoTokenizer.from_pretrained("google/mt5-xl", use_fast=False)
    model = MT5ForRegression.from_pretrained("google/metricx-24-hybrid-xl-v2p6-bfloat16", torch_dtype="auto")
    model.to("cuda").eval()
    results, progress = restore_progress(out, rows)
    started = time.perf_counter()
    pos = len(results)
    while pos < len(rows):
        batch = rows[pos:pos + batch_size]
        texts = ["source: " + r["source"] + " candidate: " + r["hypothesis"] + " reference: " + r["reference"] for r in batch]
        enc = tokenizer(texts, max_length=1536, truncation=True, padding=False)
        ids = [x[:-1] for x in enc["input_ids"]]
        masks = [x[:-1] for x in enc["attention_mask"]]
        padded = None
        try:
            padded = tokenizer.pad({"input_ids": ids, "attention_mask": masks}, padding=True, return_tensors="pt")
            padded = {k: v.to("cuda") for k, v in padded.items()}
            with torch.inference_mode():
                values = model(**padded, use_cache=False).predictions.float().cpu().tolist()
        except RuntimeError as error:
            if "out of memory" not in str(error).lower() or batch_size <= 1:
                raise
            batch_size = max(1, batch_size // 2)
            del padded
            gc.collect()
            torch.cuda.empty_cache()
            print(f"MetricX CUDA OOM; retrying batch_size={batch_size}", flush=True)
            continue
        results.extend(float(x) for x in values)
        checkpoint(progress, rows, results)
        pos += len(batch)
        if len(results) % 1024 < len(batch) or len(results) == len(rows):
            print(f"MetricX {len(results)}/{len(rows)}", flush=True)
    del model, tokenizer
    gc.collect()
    torch.cuda.empty_cache()
    return results, time.perf_counter() - started, batch_size


def score_xcomet(rows, out, batch_size, include_error_spans=True):
    import torch
    from comet import load_from_checkpoint
    torch.set_num_threads(min(torch.get_num_threads(), 4))
    if not Path(CHECKPOINT).is_file():
        raise FileNotFoundError(CHECKPOINT)
    model = load_from_checkpoint(CHECKPOINT)
    results, progress = restore_progress(out, rows)
    spans_path = Path(out).with_suffix(".error_spans.jsonl")
    if include_error_spans and results and not spans_path.is_file():
        # The primary XCOMET output includes a parallel localization artifact.
        # Recompute a completed-but-uncommitted score pass if spans were lost.
        results = []
        progress.unlink(missing_ok=True)
    error_spans = None
    started = time.perf_counter()
    if not results:
        current = batch_size
        while True:
            try:
                pred = model.predict([{"src": r["source"], "mt": r["hypothesis"], "ref": r["reference"]} for r in rows],
                                     batch_size=current, gpus=1, num_workers=0,
                                     progress_bar=True, length_batching=True)
                results = [float(x) for x in pred.scores]
                metadata = getattr(pred, "metadata", None)
                error_spans = getattr(metadata, "error_spans", None) if metadata is not None else None
                if include_error_spans:
                    if error_spans is None:
                        error_spans = [[] for _ in rows]
                    if len(error_spans) != len(rows):
                        raise RuntimeError(f"XCOMET returned {len(error_spans)} error-span rows for {len(rows)} inputs")
                    with spans_path.open("w", encoding="utf-8") as f:
                        for row, spans in zip(rows, error_spans):
                            f.write(json.dumps({"unique_id": row["unique_id"], "error_spans": spans or []},
                                               ensure_ascii=False, default=str) + "\n")
                checkpoint(progress, rows, results)
                batch_size = current
                break
            except RuntimeError as error:
                if "out of memory" not in str(error).lower() or current <= 1:
                    raise
                current = max(1, current // 2)
                gc.collect()
                torch.cuda.empty_cache()
                print(f"XCOMET CUDA OOM; retrying batch_size={current}", flush=True)
        print(f"XCOMET {len(results)}/{len(rows)}", flush=True)
    del model
    gc.collect()
    torch.cuda.empty_cache()
    return results, time.perf_counter() - started, batch_size


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("metric", choices=("comet", "metricx", "xcomet"))
    ap.add_argument("--input", type=Path, default=RUN / "scoring/unique_inputs.jsonl")
    ap.add_argument("--output", type=Path, default=None)
    ap.add_argument("--batch-size", type=int, default=None)
    ap.add_argument("--scores-only", action="store_true",
                    help="For XCOMET, discard batch error-span metadata and keep sentence scores only.")
    a = ap.parse_args()
    rows = read(a.input)
    input_sha = sha256(a.input)
    out = a.output or RUN / "scoring/unique" / f"{a.metric}.scores.jsonl"
    out.parent.mkdir(parents=True, exist_ok=True)
    summary_path = out.with_suffix(".summary.json")
    if out.is_file() and summary_path.is_file():
        saved = read(out)
        old_summary = json.loads(summary_path.read_text(encoding="utf-8"))
        spans_ready = a.metric != "xcomet" or out.with_suffix(".error_spans.jsonl").is_file()
        if (len(saved) == len(rows) and all(r.get("unique_id") == i for i, r in enumerate(saved))
                and old_summary.get("input_sha256") == input_sha and spans_ready):
            print(f"{a.metric} already complete ({len(saved)} rows)", flush=True)
            return
        out.unlink(missing_ok=True)
        summary_path.unlink(missing_ok=True)
        out.with_suffix(".progress.jsonl").unlink(missing_ok=True)
        out.with_suffix(".error_spans.jsonl").unlink(missing_ok=True)
    batch = a.batch_size or {"comet": 256, "metricx": 128, "xcomet": 16}[a.metric]
    fn = {"comet": score_comet, "metricx": score_metricx, "xcomet": score_xcomet}[a.metric]
    if a.metric == "xcomet":
        values, seconds, actual_batch = fn(rows, out, batch, include_error_spans=not a.scores_only)
    else:
        values, seconds, actual_batch = fn(rows, out, batch)
    if len(values) != len(rows):
        raise RuntimeError(f"{a.metric}: got {len(values)} scores for {len(rows)} unique inputs")
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", encoding="utf-8") as f:
        for row, score in zip(rows, values):
            f.write(json.dumps({"unique_id": row["unique_id"], "language": row["language"], "score": score}, ensure_ascii=False) + "\n")
    out.with_suffix(".progress.jsonl").unlink(missing_ok=True)
    save(summary_path, {"metric": a.metric, "examples": len(rows), "batch_size": actual_batch,
                        "score_seconds": seconds, "items_per_second": len(rows) / max(seconds, 1e-9),
                        "input": str(a.input), "input_sha256": input_sha, "fresh_unique_inputs": True,
                        "error_spans_saved": a.metric == "xcomet" and not a.scores_only})
    print(f"{a.metric} complete: {len(rows)} fresh unique inputs in {seconds:.1f}s", flush=True)


if __name__ == "__main__":
    main()
