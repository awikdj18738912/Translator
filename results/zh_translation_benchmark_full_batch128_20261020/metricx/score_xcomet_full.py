#!/usr/bin/env python3
"""Run full-corpus XCOMET-XL scoring over the saved translation outputs."""

import argparse
import gc
import json
import pathlib
import statistics
import time

import torch
from comet import download_model, load_from_checkpoint


MODELS = ("index", "hy", "lmt")
LANGUAGES = ("en", "fr", "de", "es", "ru", "ja", "ko", "ar")


def read_jsonl(path):
    with open(path, encoding="utf-8") as stream:
        return [json.loads(line) for line in stream if line.strip()]


def write_model_scores(model_key, rows, scores, error_spans, output_dir, batch_size):
    if len(scores) != len(rows):
        raise RuntimeError(f"{model_key}: XCOMET returned {len(scores)} scores for {len(rows)} rows")
    if error_spans is None:
        error_spans = [[] for _ in rows]
    if len(error_spans) != len(rows):
        raise RuntimeError(
            f"{model_key}: XCOMET returned {len(error_spans)} span lists for {len(rows)} rows"
        )

    output_path = output_dir / f"{model_key}.scores.jsonl"
    grouped = {language: [] for language in LANGUAGES}
    with output_path.open("w", encoding="utf-8") as stream:
        for row, score, spans in zip(rows, scores, error_spans):
            score = float(score)
            grouped[row["language"]].append(score)
            record = {
                "id": row["id"],
                "language": row["language"],
                "xcomet_score": score,
                "error_spans": spans or [],
            }
            stream.write(json.dumps(record, ensure_ascii=False, default=str) + "\n")

    per_language = {
        language: {"count": len(grouped[language]), "mean": statistics.mean(grouped[language])}
        for language in LANGUAGES
    }
    if any(per_language[language]["count"] != 1012 for language in LANGUAGES):
        raise RuntimeError(f"{model_key}: unexpected per-language score counts: {per_language}")
    summary = {
        "model_key": model_key,
        "model": "Unbabel/XCOMET-XL",
        "count": len(rows),
        "batch_size_used": batch_size,
        "score_direction": "higher is better",
        "macro_mean_8_languages": statistics.mean(
            per_language[language]["mean"] for language in LANGUAGES
        ),
        "languages": per_language,
        "scores_file": output_path.name,
    }
    (output_dir / f"{model_key}.summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    return summary


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--predictions_dir", type=pathlib.Path, required=True)
    parser.add_argument("--output_dir", type=pathlib.Path, required=True)
    parser.add_argument("--cache_dir", type=pathlib.Path, required=True)
    parser.add_argument("--model", default="Unbabel/XCOMET-XL")
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--models", default=",".join(MODELS), help="Comma-separated systems to score")
    args = parser.parse_args()
    models = args.models.split(",")
    if not models or len(models) != len(set(models)) or any(key not in MODELS for key in models):
        parser.error("--models must contain distinct keys from index,hy,lmt")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    args.cache_dir.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(min(torch.get_num_threads(), 4))
    if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
        raise RuntimeError("Run with CUDA_VISIBLE_DEVICES=1 so only GPU1 is visible")
    torch.cuda.set_device(0)

    model_started = time.perf_counter()
    local_checkpoint = pathlib.Path(args.model).expanduser()
    if local_checkpoint.is_file():
        # Keep the snapshot symlink path so load_from_checkpoint can locate
        # hparams.yaml beside the checkpoint's parent directory.
        checkpoint = str(local_checkpoint)
    else:
        checkpoint = download_model(args.model, saving_directory=args.cache_dir)
    model = load_from_checkpoint(checkpoint)
    model_load_seconds = time.perf_counter() - model_started
    torch.cuda.reset_peak_memory_stats(0)
    print(f"Loaded {args.model}; scoring with initial batch_size={args.batch_size}", flush=True)

    all_summaries = []
    for model_key in models:
        rows = read_jsonl(args.predictions_dir / model_key / "predictions.jsonl")
        if len(rows) != 8096:
            raise RuntimeError(f"{model_key}: expected 8,096 rows, found {len(rows)}")
        data = [{"src": row["source"], "mt": row["hypothesis"], "ref": row["reference"]}
                for row in rows]

        batch_size = args.batch_size
        while True:
            started = time.perf_counter()
            try:
                prediction = model.predict(
                    data,
                    batch_size=batch_size,
                    gpus=1,
                    num_workers=0,
                    progress_bar=True,
                    length_batching=True,
                )
                score_seconds = time.perf_counter() - started
                break
            except RuntimeError as error:
                if "out of memory" not in str(error).lower() or batch_size <= 1:
                    raise
                batch_size = max(1, batch_size // 2)
                torch.cuda.empty_cache()
                gc.collect()
                print(f"{model_key}: CUDA OOM; retrying at batch_size={batch_size}", flush=True)

        metadata = getattr(prediction, "metadata", None)
        spans = getattr(metadata, "error_spans", None) if metadata is not None else None
        summary = write_model_scores(
            model_key, rows, prediction.scores, spans, args.output_dir, batch_size
        )
        summary["score_seconds"] = score_seconds
        summary["score_items_per_second"] = len(rows) / max(score_seconds, 1e-9)
        summary["torch_peak_allocated_gib_after_model"] = torch.cuda.max_memory_allocated(0) / 1024**3
        summary["torch_peak_reserved_gib_after_model"] = torch.cuda.max_memory_reserved(0) / 1024**3
        (args.output_dir / f"{model_key}.summary.json").write_text(
            json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
        )
        all_summaries.append(summary)
        print(
            f"{model_key}: {len(rows)}/{len(rows)} rows, macro={summary['macro_mean_8_languages']:.6f}, "
            f"batch_size={batch_size}, {summary['score_items_per_second']:.1f} rows/s",
            flush=True,
        )
        del prediction, data, rows
        gc.collect()
        torch.cuda.empty_cache()

    combined = {
        "model": args.model,
        "tokenizer": "model-defined COMET tokenizer",
        "total_rows": sum(item["count"] for item in all_summaries),
        "model_load_seconds": model_load_seconds,
        "models": {item["model_key"]: item for item in all_summaries},
        "macro_mean_8_languages": {
            item["model_key"]: item["macro_mean_8_languages"] for item in all_summaries
        },
    }
    (args.output_dir / "summary.json").write_text(
        json.dumps(combined, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    del model
    gc.collect()
    torch.cuda.empty_cache()
    print(json.dumps(combined, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
