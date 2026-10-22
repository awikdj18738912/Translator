#!/usr/bin/env python3
"""Run a full Chinese-to-eight-language Index GGUF translation benchmark."""

from __future__ import annotations

import argparse
from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import statistics
import subprocess
import sys
import threading
import time
import uuid

import requests


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts"))
from benchmark_zh_translation import LANGS, prompt_for, read_jsonl  # noqa: E402


DEFAULT_DATA = ROOT / "data/eval/zh_translation/flores101_devtest.jsonl"
MODEL_ROOT = Path("/home/aim0/data/models/ASR/Index-Translate-2B-GGUF")
MODEL_FILES = {
    "f16_gguf": "Index-Translate-2B.f16.gguf",
    "bf16_gguf": "Index-Translate-2B.bf16.gguf",
    "q8_0": "Index-Translate-2B.Q8_0.gguf",
    "q6_k": "Index-Translate-2B.Q6_K.gguf",
    "q4_k_m": "Index-Translate-2B.Q4_K_M.gguf",
}
THREAD_LOCAL = threading.local()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def save_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n")
    temporary.replace(path)


def gpu_snapshot() -> str:
    try:
        return subprocess.check_output(
            [
                "nvidia-smi",
                "--query-gpu=index,uuid,name,memory.used,memory.total,utilization.gpu",
                "--format=csv,noheader",
            ],
            text=True,
            timeout=10,
        ).strip()
    except (OSError, subprocess.SubprocessError) as error:
        return str(error)


def percentile(values: list[float], q: float) -> float | None:
    values = sorted(values)
    if not values:
        return None
    position = (len(values) - 1) * q
    low = int(position)
    high = min(low + 1, len(values) - 1)
    return values[low] + (values[high] - values[low]) * (position - low)


def active_wall_seconds(records: list[dict]) -> float:
    intervals = sorted(
        (float(record["request_started_at"]), float(record["request_ended_at"]))
        for record in records
        if record.get("request_started_at") is not None
        and record.get("request_ended_at") is not None
    )
    total = 0.0
    current_start = current_end = None
    for start, end in intervals:
        if current_start is None:
            current_start, current_end = start, end
        elif start <= current_end:
            current_end = max(current_end, end)
        else:
            total += current_end - current_start
            current_start, current_end = start, end
    if current_start is not None:
        total += current_end - current_start
    return total


def worker_session() -> requests.Session:
    session = getattr(THREAD_LOCAL, "session", None)
    if session is None:
        session = requests.Session()
        adapter = requests.adapters.HTTPAdapter(
            pool_connections=1,
            pool_maxsize=1,
            max_retries=0,
        )
        session.mount("http://", adapter)
        THREAD_LOCAL.session = session
    return session


def translate_one(index: int, row: dict, args: argparse.Namespace) -> dict:
    prompt = prompt_for("index", row["source"], row["language"])
    payload = {
        "model": "Index-Translate-2B",
        "messages": [{"role": "user", "content": prompt}],
        "temperature": 0.0,
        "top_p": 1.0,
        "max_tokens": args.max_tokens,
        "stream": False,
    }
    session = worker_session()
    endpoint = args.base_url.rstrip("/") + "/v1/chat/completions"
    last_error: Exception | None = None
    for attempt in range(args.retries + 1):
        request_started = time.time()
        monotonic_started = time.perf_counter()
        try:
            response = session.post(endpoint, json=payload, timeout=(30, args.timeout))
            if response.status_code >= 500 or response.status_code == 429:
                response.raise_for_status()
            response.raise_for_status()
            result = response.json()
            choice = result["choices"][0]
            message = choice.get("message") or {}
            hypothesis = (message.get("content") or "").strip()
            if not hypothesis:
                raise RuntimeError(f"Empty response for {row['language']}:{row['id']}")
            request_ended = time.time()
            latency = time.perf_counter() - monotonic_started
            usage = result.get("usage") or {}
            timings = result.get("timings") or {}
            completion_tokens = int(usage.get("completion_tokens") or timings.get("predicted_n") or 0)
            return {
                **row,
                "hypothesis": hypothesis,
                "raw_hypothesis": hypothesis,
                "input_tokens": int(usage.get("prompt_tokens") or timings.get("prompt_n") or 0),
                "generated_tokens_including_eos": completion_tokens,
                "content_tokens": completion_tokens,
                "generation_seconds": latency,
                "latency_seconds": latency,
                "request_started_at": request_started,
                "request_ended_at": request_ended,
                "finish_reason": choice.get("finish_reason"),
                "eos": choice.get("finish_reason") == "stop",
                "truncated": choice.get("finish_reason") == "length",
                "empty_output": False,
                "server_timings": timings,
                "error": None,
                "request_index": index,
            }
        except Exception as error:
            last_error = error
            if attempt >= args.retries:
                raise RuntimeError(
                    f"Translation failed for {row['language']}:{row['id']} after "
                    f"{args.retries + 1} attempts"
                ) from error
            time.sleep(min(2**attempt, 16))
    raise RuntimeError("Unreachable retry state") from last_error


def read_progress(path: Path) -> dict[tuple[str, str], dict]:
    completed: dict[tuple[str, str], dict] = {}
    if not path.exists():
        return completed
    with path.open(encoding="utf-8") as stream:
        for line in stream:
            if not line.strip():
                continue
            row = json.loads(line)
            if row.get("error") is None and row.get("hypothesis", "").strip():
                completed[(row["language"], str(row["id"]))] = row
    return completed


def write_record(stream, record: dict) -> None:
    stream.write(json.dumps(record, ensure_ascii=False) + "\n")
    stream.flush()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-key", choices=tuple(MODEL_FILES), required=True)
    parser.add_argument("--data", type=Path, default=DEFAULT_DATA)
    parser.add_argument("--model-path", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--base-url", default="http://127.0.0.1:8088")
    parser.add_argument("--concurrency", type=int, default=128)
    parser.add_argument("--max-tokens", type=int, default=512)
    parser.add_argument("--timeout", type=int, default=600)
    parser.add_argument("--retries", type=int, default=4)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()

    if args.concurrency < 1 or args.max_tokens < 1:
        parser.error("concurrency and max-tokens must be positive")
    model_path = args.model_path or MODEL_ROOT / MODEL_FILES[args.model_key]
    if not model_path.is_file():
        raise FileNotFoundError(model_path)
    rows = read_jsonl(args.data)
    expected = {language: 1012 for language in LANGS}
    counts = {language: 0 for language in LANGS}
    for row in rows:
        counts[row["language"]] += 1
    if len(rows) != 8096 or counts != expected:
        raise ValueError(f"Expected 8,096 rows (1,012/language), got {len(rows)} / {counts}")

    output = args.output
    output.parent.mkdir(parents=True, exist_ok=True)
    progress_path = output.with_suffix(".progress.jsonl")
    if output.exists() and not args.resume:
        raise FileExistsError(f"Refusing to overwrite existing output: {output}")
    completed = read_progress(progress_path) if args.resume else {}
    selected_keys = {(row["language"], str(row["id"])) for row in rows}
    if not completed.keys() <= selected_keys:
        raise ValueError("Progress file includes rows outside the requested full dataset")

    run_id = uuid.uuid4().hex
    metadata_path = output.parent / "metadata.json"
    started_utc = datetime.now(timezone.utc).isoformat()
    metadata = {
        "status": "running",
        "model_key": args.model_key,
        "model_file": model_path.name,
        "model_path": str(model_path.resolve()),
        "model_bytes": model_path.stat().st_size,
        "model_sha256": sha256(model_path),
        "dataset": "FLORES-101 devtest",
        "dataset_path": str(args.data.resolve()),
        "dataset_sha256": sha256(args.data),
        "source_language": "zh",
        "languages": list(LANGS),
        "sentences_per_language": 1012,
        "total_cases": len(rows),
        "prompt_protocol": "scripts/benchmark_zh_translation.py prompt_for('index', ...)",
        "decode": {
            "temperature": 0.0,
            "top_p": 1.0,
            "max_tokens": args.max_tokens,
            "reasoning": "off",
        },
        "runtime": "llama.cpp CUDA llama-server / OpenAI chat completions",
        "physical_gpu": 1,
        "server_url": args.base_url,
        "client_concurrency": args.concurrency,
        "retry_count": args.retries,
        "run_id": run_id,
        "started_utc": started_utc,
        "gpu_snapshot_start": gpu_snapshot(),
        "resumed_rows": len(completed),
    }
    save_json(metadata_path, metadata)

    pending_indices = [
        index
        for index, row in enumerate(rows)
        if (row["language"], str(row["id"])) not in completed
    ]
    if not pending_indices:
        print("All rows already present in progress file; finalizing output.", flush=True)
    else:
        total_started = time.perf_counter()
        done_count = len(completed)
        new_done_count = 0
        with progress_path.open("a", encoding="utf-8") as progress:
            with ThreadPoolExecutor(max_workers=args.concurrency) as pool:
                pending = {}
                indices = iter(pending_indices)

                def fill_queue() -> None:
                    while len(pending) < args.concurrency:
                        try:
                            index = next(indices)
                        except StopIteration:
                            return
                        pending[pool.submit(translate_one, index, rows[index], args)] = index

                fill_queue()
                while pending:
                    finished, _ = wait(pending, return_when=FIRST_COMPLETED)
                    for future in finished:
                        index = pending.pop(future)
                        record = future.result()
                        record["session_id"] = run_id
                        record["batch_size"] = args.concurrency
                        completed[(record["language"], str(record["id"]))] = record
                        write_record(progress, record)
                        done_count += 1
                        new_done_count += 1
                        if done_count % 256 == 0 or done_count == len(rows):
                            elapsed = time.perf_counter() - total_started
                            print(
                                f"{args.model_key}: {done_count}/{len(rows)} rows, "
                                f"{new_done_count / max(elapsed, 1e-9):.1f} new rows/s, "
                                f"active={len(pending)}",
                                flush=True,
                            )
                    fill_queue()
        elapsed = time.perf_counter() - total_started

    ordered = [completed[(row["language"], str(row["id"]))] for row in rows]
    temporary = output.with_suffix(output.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as stream:
        for record in ordered:
            stream.write(json.dumps(record, ensure_ascii=False) + "\n")
    temporary.replace(output)

    session_groups: dict[str, list[dict]] = {}
    for record in ordered:
        session_groups.setdefault(record.get("session_id", "unknown"), []).append(record)
    wall_seconds = sum(active_wall_seconds(group) for group in session_groups.values())
    language_summary = {}
    for language in LANGS:
        group = [record for record in ordered if record["language"] == language]
        latencies = [float(record["generation_seconds"]) for record in group]
        language_sessions: dict[str, list[dict]] = {}
        for record in group:
            language_sessions.setdefault(record.get("session_id", "unknown"), []).append(record)
        language_wall = sum(active_wall_seconds(values) for values in language_sessions.values())
        language_summary[language] = {
            "count": len(group),
            "mean_request_latency_seconds": statistics.fmean(latencies),
            "p50_request_latency_seconds": percentile(latencies, 0.50),
            "p95_request_latency_seconds": percentile(latencies, 0.95),
            "wall_seconds": language_wall,
            "sentences_per_second": len(group) / max(language_wall, 1e-9),
            "generated_tokens": sum(int(record["generated_tokens_including_eos"]) for record in group),
            "generation_tokens_per_second": sum(
                int(record["generated_tokens_including_eos"]) for record in group
            ) / max(language_wall, 1e-9),
            "empty_outputs": sum(not record["hypothesis"].strip() for record in group),
            "truncated_outputs": sum(bool(record["truncated"]) for record in group),
        }

    metadata.update(
        status="complete",
        completed_utc=datetime.now(timezone.utc).isoformat(),
        rows_written=len(ordered),
        predictions_sha256=sha256(output),
        wall_seconds=wall_seconds,
        sentences_per_second=len(ordered) / max(wall_seconds, 1e-9),
        generated_tokens=sum(int(record["generated_tokens_including_eos"]) for record in ordered),
        generated_tokens_per_second=sum(
            int(record["generated_tokens_including_eos"]) for record in ordered
        ) / max(wall_seconds, 1e-9),
        language_results=language_summary,
        gpu_snapshot_end=gpu_snapshot(),
    )
    save_json(metadata_path, metadata)
    if progress_path.exists():
        progress_path.unlink()
    print(
        f"COMPLETE {args.model_key}: {len(ordered)}/{len(rows)} rows, "
        f"{metadata['sentences_per_second']:.2f} rows/s, {wall_seconds:.1f}s",
        flush=True,
    )


if __name__ == "__main__":
    main()
