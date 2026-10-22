#!/usr/bin/env python3
"""Resume the official F16 GGUF control, with one GPU1 workload at a time."""
from __future__ import annotations

from collections import Counter
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time
import urllib.request

RUN = Path(__file__).resolve().parent
ROOT = RUN.parents[1]
OUT = RUN / "f16_gguf"
BENCH_PY = "/home/aim0/data/venvs/translation-bench/bin/python"
COMET_PY = "/home/aim0/data/venvs/translation-comet/bin/python"
METRICX_PY = "/home/aim0/data/venvs/translation-metricx/bin/python"
OLD = ROOT / "results/zh_translation_benchmark_full_batch128_20261020/metricx"
MODEL = Path("/home/aim0/data/models/ASR/Index-Translate-2B-GGUF/Index-Translate-2B.f16.gguf")
CHECKPOINT = "/home/aim0/data/hf-cache/hub/models--Unbabel--XCOMET-XL/snapshots/6a123c5e8e6dccab25e5fcffa3c8b417abadb462/checkpoints/model.ckpt"
LANGS = ("en", "fr", "de", "es", "ru", "ja", "ko", "ar")
STATE = OUT / "pipeline_status.json"


def save(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n")
    temporary.replace(path)


def event(stage, **kwargs):
    record = {"stage": stage, "utc": datetime.now(timezone.utc).isoformat(), **kwargs}
    save(STATE, record)
    print(json.dumps(record, ensure_ascii=False), flush=True)


def read_rows(path):
    with path.open(encoding="utf-8") as stream:
        return [json.loads(line) for line in stream if line.strip()]


def predictions_complete():
    path = OUT / "predictions.jsonl"
    if not path.is_file():
        return False
    rows = read_rows(path)
    assert len(rows) == 8096
    assert Counter(r["language"] for r in rows) == Counter({l: 1012 for l in LANGS})
    assert len({(r["language"], str(r["id"])) for r in rows}) == 8096
    assert not any(r.get("error") or not r.get("hypothesis", "").strip() for r in rows)
    metadata = json.loads((OUT / "metadata.json").read_text())
    assert metadata["status"] == "complete" and metadata["rows_written"] == 8096
    return True


def stage(stage_name, command, env=None, finished=None):
    if finished is not None and finished():
        event(stage_name, status="already_complete")
        return
    log_path = OUT / f"{stage_name}.log"
    with log_path.open("a") as log:
        process = subprocess.Popen(command, cwd=ROOT, env=env, stdout=log, stderr=subprocess.STDOUT,
                                   start_new_session=True)
        event(stage_name, status="running", pid=process.pid, command=command, log=str(log_path))
        complete_since = None
        while process.poll() is None:
            # Some HF download threads can keep MetricX alive after its final
            # output and summary are safely closed. Release only that completed child.
            if stage_name == "metricx" and finished is not None and finished():
                complete_since = complete_since or time.monotonic()
                if time.monotonic() - complete_since > 30:
                    process.terminate()
                    process.wait(timeout=20)
                    event(stage_name, status="complete", completed_child_terminated=True)
                    return
            time.sleep(3)
        if process.returncode:
            raise RuntimeError(f"{stage_name} failed with exit {process.returncode}; see {log_path}")
    if finished is not None and not finished():
        raise RuntimeError(f"{stage_name} exited without complete output")
    event(stage_name, status="complete")


def score_complete(name):
    folder = OUT / "scoring" / name
    if name == "metricx":
        summary = folder / "output_all.summary.json"
        scores = folder / "output_all.jsonl"
        if not summary.is_file() or not scores.is_file():
            return False
        if json.loads(summary.read_text()).get("examples") != 8096:
            return False
    else:
        summary = folder / "summary.json"
        scores = folder / "index.scores.jsonl"
        if not summary.is_file() or not scores.is_file():
            return False
    rows = read_rows(scores)
    return len(rows) == 8096 and Counter(r["language"] for r in rows) == Counter({l: 1012 for l in LANGS})


def wait_for_gpu():
    """Let an unrelated running experiment finish before measuring throughput."""
    uuid = subprocess.check_output(["nvidia-smi", "-i", "1", "--query-gpu=uuid", "--format=csv,noheader"], text=True).strip()
    idle = 0
    previous = None
    while idle < 3:
        output = subprocess.check_output(["nvidia-smi", "--query-compute-apps=gpu_uuid,pid,process_name",
                                          "--format=csv,noheader"], text=True)
        busy = [line.strip() for line in output.splitlines() if line.startswith(uuid + ",")]
        if busy != previous:
            event("gpu_wait", status="waiting" if busy else "checking_idle", processes=busy)
            previous = busy
        idle = 0 if busy else idle + 1
        time.sleep(5)


def prepare_inputs():
    rows = read_rows(OUT / "predictions.jsonl")
    original = {(r["language"], str(r["id"])): r for r in read_rows(RUN / "q8_0/predictions.jsonl")}
    for row in rows:
        other = original[(row["language"], str(row["id"]))]
        assert all(row[k] == other[k] for k in ("source", "reference"))
    path = OUT / "scoring/neural-input/index/predictions.jsonl"
    path.parent.mkdir(parents=True, exist_ok=True)
    if not path.exists():
        path.symlink_to(OUT / "predictions.jsonl")
    metricx = OUT / "scoring/metricx-input-all.jsonl"
    with metricx.open("w", encoding="utf-8") as stream:
        for i, row in enumerate(rows):
            record = {k: row[k] for k in ("id", "language", "source", "hypothesis", "reference")}
            record.update(bench_row_id=i, model_key="f16_gguf", model_name="Index-Translate-2B F16 GGUF")
            stream.write(json.dumps(record, ensure_ascii=False) + "\n")
    bleu = RUN / "scoring/bleu-13a-six/f16_gguf/predictions.jsonl"
    bleu.parent.mkdir(parents=True, exist_ok=True)
    with bleu.open("w", encoding="utf-8") as stream:
        for row in rows:
            if row["language"] in ("en", "fr", "de", "es", "ru", "ar"):
                stream.write(json.dumps(row, ensure_ascii=False) + "\n")


def translate():
    if predictions_complete():
        event("translation", status="already_complete")
        return
    wait_for_gpu()
    server_args = ["/home/aim0/data/tools/llama.cpp/build-cuda/bin/llama-server", "-m", str(MODEL),
        "--host", "127.0.0.1", "--port", "8088", "--n-gpu-layers", "all",
        "--ctx-size", "131072", "--parallel", "128", "--batch-size", "8192", "--ubatch-size", "2048",
        "--flash-attn", "on", "--reasoning", "off", "--threads", "12", "--threads-batch", "16",
        "--cache-ram", "0", "--no-cache-prompt", "--no-cache-idle-slots", "--ctx-checkpoints", "0",
        "--fit", "off", "--verbosity", "1", "--no-webui"]
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = "1"
    env["LD_LIBRARY_PATH"] = "/home/aim0/data/venvs/llama-runtime/lib/python3.12/site-packages/nvidia/cu13/lib"
    save(OUT / "server_config.json", {"args": server_args, "CUDA_VISIBLE_DEVICES": "1"})
    (OUT / "STOP_TELEMETRY").unlink(missing_ok=True)
    (OUT / "ACTIVE_MODEL.txt").write_text("f16_gguf\n")
    collector = subprocess.Popen([sys.executable, str(RUN / "collect_gpu1_telemetry.py"),
        str(OUT / "gpu1_telemetry.csv"), str(OUT / "ACTIVE_MODEL.txt"), str(OUT / "STOP_TELEMETRY"), "5"], cwd=ROOT)
    with (OUT / "server.log").open("a") as log:
        server = subprocess.Popen(server_args, env=env, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT,
                                  start_new_session=True)
        try:
            event("server_start", status="running", pid=server.pid)
            for _ in range(120):
                if server.poll() is not None:
                    raise RuntimeError("llama-server exited before health check")
                try:
                    with urllib.request.urlopen("http://127.0.0.1:8088/health", timeout=2) as response:
                        if response.status == 200:
                            break
                except Exception:
                    time.sleep(1)
            else:
                raise RuntimeError("llama-server health check timed out")
            with urllib.request.urlopen("http://127.0.0.1:8088/props", timeout=10) as response:
                save(OUT / "server_properties.json", json.load(response))
            command = [BENCH_PY, str(RUN / "run_gguf_translation.py"), "--model-key", "f16_gguf",
                "--output", str(OUT / "predictions.jsonl"), "--base-url", "http://127.0.0.1:8088",
                "--concurrency", "128", "--max-tokens", "512", "--timeout", "600", "--retries", "4"]
            if (OUT / "predictions.progress.jsonl").exists():
                command.append("--resume")
            stage("translation", command, finished=predictions_complete)
        finally:
            if server.poll() is None:
                server.terminate()
                server.wait(timeout=30)
            (OUT / "STOP_TELEMETRY").touch()
            try:
                collector.wait(timeout=10)
            except subprocess.TimeoutExpired:
                collector.terminate()
                collector.wait(timeout=10)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    assert MODEL.is_file(), f"Official F16 file missing: {MODEL}"
    translate()
    prepare_inputs()
    stage("spbleu_chrf", [BENCH_PY, str(ROOT / "scripts/benchmark_zh_translation.py"), "score",
        "--predictions", str(OUT / "predictions.jsonl"), "--bleu-tokenizer", "flores200"])
    stage("bleu13a", [BENCH_PY, str(ROOT / "scripts/benchmark_zh_translation.py"), "score",
        "--predictions", str(RUN / "scoring/bleu-13a-six/f16_gguf/predictions.jsonl"), "--bleu-tokenizer", "13a"])
    env = os.environ.copy()
    env.update(CUDA_VISIBLE_DEVICES="1", HF_HOME="/home/aim0/data/hf-cache",
        HF_HUB_CACHE="/home/aim0/data/hf-cache/hub", HF_ENDPOINT="https://huggingface.co",
        HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1", PYTHONUNBUFFERED="1")
    if not score_complete("comet"):
        wait_for_gpu()
    stage("comet", [COMET_PY, str(ROOT / "scripts/score_zh_comet.py"), "--input-dir", str(OUT / "scoring/neural-input"),
        "--output", str(OUT / "scoring/comet"), "--models", "index", "--batch-size", "256", "--gpu", "1"],
        env=env, finished=lambda: score_complete("comet"))
    metricx_env = env.copy()
    metricx_env["PYTHONPATH"] = "/home/aim0/data/metricx-source"
    if not score_complete("metricx"):
        wait_for_gpu()
    stage("metricx", [METRICX_PY, str(OLD / "score_metricx24.py"), "--input_file", str(OUT / "scoring/metricx-input-all.jsonl"),
        "--output_file", str(OUT / "scoring/metricx/output_all.jsonl"), "--batch_size", "128"],
        env=metricx_env, finished=lambda: score_complete("metricx"))
    if not score_complete("xcomet"):
        wait_for_gpu()
    stage("xcomet", [COMET_PY, str(OLD / "score_xcomet_full.py"), "--predictions_dir", str(OUT / "scoring/neural-input"),
        "--output_dir", str(OUT / "scoring/xcomet"), "--cache_dir", "/home/aim0/data/hf-cache/hub",
        "--model", CHECKPOINT, "--batch_size", "16", "--models", "index"],
        env=env, finished=lambda: score_complete("xcomet"))
    stage("control_report", [BENCH_PY, str(RUN / "build_f16_control_report.py")])
    stage("main_report", [BENCH_PY, str(RUN / "build_quantized_report.py")])
    event("all", status="complete", report=str(RUN / "RESULTS_SUMMARY.md"))


if __name__ == "__main__":
    try:
        main()
    except Exception as error:
        event("failed", status="failed", error=str(error))
        raise
