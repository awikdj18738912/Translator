#!/usr/bin/env python3
"""Score reference-based MetricX-24 predictions with dynamic batch padding."""

import argparse
import gc
import json
import time
from pathlib import Path

import torch
import transformers
from metricx24.models import MT5ForRegression


def read_jsonl(path):
    with open(path, encoding="utf-8") as stream:
        return [json.loads(line) for line in stream if line.strip()]


def score_batch(model, tokenizer, examples, device, max_input_length):
    texts = [
        "source: " + row["source"]
        + " candidate: " + row["hypothesis"]
        + " reference: " + row["reference"]
        for row in examples
    ]
    encoded = tokenizer(
        texts,
        max_length=max_input_length,
        truncation=True,
        padding=False,
    )
    # Match the official MetricX-24 inference example, which removes the final EOS.
    input_ids = [ids[:-1] for ids in encoded["input_ids"]]
    attention_masks = [mask[:-1] for mask in encoded["attention_mask"]]
    batch = tokenizer.pad(
        {"input_ids": input_ids, "attention_mask": attention_masks},
        padding=True,
        return_tensors="pt",
    )
    batch = {key: value.to(device) for key, value in batch.items()}
    with torch.inference_mode():
        # This regressor emits a score from a single decoder token; caching is
        # unnecessary and avoids a decoder-mask incompatibility in newer HF.
        return model(**batch, use_cache=False).predictions.float().cpu().tolist()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_file", required=True)
    parser.add_argument("--output_file", required=True)
    parser.add_argument(
        "--model_name_or_path",
        default="google/metricx-24-hybrid-xl-v2p6-bfloat16",
    )
    parser.add_argument("--tokenizer", default="google/mt5-xl")
    parser.add_argument("--max_input_length", type=int, default=1536)
    parser.add_argument("--batch_size", type=int, default=64)
    args = parser.parse_args()

    rows = read_jsonl(args.input_file)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type == "cuda":
        torch.cuda.set_device(0)
    tokenizer = transformers.AutoTokenizer.from_pretrained(
        args.tokenizer, use_fast=False
    )
    load_started = time.perf_counter()
    model = MT5ForRegression.from_pretrained(
        args.model_name_or_path, torch_dtype="auto"
    )
    model.to(device)
    model.eval()
    load_seconds = time.perf_counter() - load_started
    if device.type == "cuda":
        torch.set_num_threads(min(torch.get_num_threads(), 4))
        torch.cuda.reset_peak_memory_stats(device)

    output_path = Path(args.output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    score_started = time.perf_counter()
    index = 0
    batch_size = min(args.batch_size, max(1, len(rows)))
    with output_path.open("w", encoding="utf-8") as stream:
        while index < len(rows):
            end = min(index + batch_size, len(rows))
            try:
                scores = score_batch(
                    model,
                    tokenizer,
                    rows[index:end],
                    device,
                    args.max_input_length,
                )
            except RuntimeError as error:
                if "out of memory" not in str(error).lower() or batch_size == 1:
                    raise
                batch_size = max(1, batch_size // 2)
                if device.type == "cuda":
                    torch.cuda.empty_cache()
                print(f"CUDA OOM; retrying with batch_size={batch_size}", flush=True)
                continue

            for row, score in zip(rows[index:end], scores):
                result = dict(row)
                result["prediction"] = float(score)
                stream.write(json.dumps(result, ensure_ascii=False) + "\n")
            index = end
            if index % 1024 == 0 or index == len(rows):
                elapsed = time.perf_counter() - score_started
                print(
                    f"scored {index}/{len(rows)}; "
                    f"{index / max(elapsed, 1e-9):.1f} items/s; "
                    f"batch_size={batch_size}",
                    flush=True,
                )

    score_seconds = time.perf_counter() - score_started
    peak_allocated = (
        torch.cuda.max_memory_allocated(device) / 1024**3
        if device.type == "cuda" else None
    )
    peak_reserved = (
        torch.cuda.max_memory_reserved(device) / 1024**3
        if device.type == "cuda" else None
    )
    del model
    gc.collect()
    if device.type == "cuda":
        torch.cuda.synchronize(device)
        torch.cuda.empty_cache()

    summary = {
        "model": args.model_name_or_path,
        "tokenizer": args.tokenizer,
        "max_input_length": args.max_input_length,
        "requested_batch_size": args.batch_size,
        "examples": len(rows),
        "load_seconds": load_seconds,
        "score_seconds": score_seconds,
        "score_items_per_second": len(rows) / max(score_seconds, 1e-9),
        "device": str(device),
        "torch_peak_allocated_gib": peak_allocated,
        "torch_peak_reserved_gib": peak_reserved,
    }
    summary_path = output_path.with_suffix(".summary.json")
    summary_path.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
