#!/usr/bin/env python3
"""Prepare, freshly score, collect, and report the six backend outputs.

Run with `--stage prepare`, then `--stage score`, then `--stage report`.
Neural scoring uses one exact-input-deduplicated corpus shared by all systems.
"""
from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
import os
from pathlib import Path
import statistics
import subprocess
import sys

RUN = Path(__file__).resolve().parent
ROOT = RUN.parents[1]
LANGS = ("en", "fr", "de", "es", "ru", "ja", "ko", "ar")
KEYS = ("bf16_vllm", "fp8_vllm", "nvfp4_vllm", "bf16_transformers", "fp8_transformers", "nvfp4_transformers")
BENCH_PY = "/home/aim0/data/venvs/translation-bench/bin/python"
COMET_PY = "/home/aim0/data/venvs/translation-comet/bin/python"
METRICX_PY = "/home/aim0/data/venvs/translation-metricx/bin/python"
NEURAL = RUN / "score_unique_neural.py"


def read(path):
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def save(path, data):
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    tmp = p.with_suffix(p.suffix + ".tmp")
    tmp.write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    tmp.replace(p)


def sha256(path):
    digest = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def event(stage, **kw):
    rec = {"stage": stage, **kw}
    save(RUN / "scoring/pipeline_status.json", rec)
    print(json.dumps(rec, ensure_ascii=False), flush=True)


def run_logged(label, command, *, cwd=ROOT, env=None):
    log_path = RUN / "scoring/logs" / f"{label}.log"
    log_path.parent.mkdir(parents=True, exist_ok=True)
    event(label, status="running", command=command, log=str(log_path))
    with log_path.open("a", encoding="utf-8") as log:
        subprocess.run(command, cwd=cwd, env=env, stdout=log, stderr=subprocess.STDOUT, check=True)
    event(label, status="complete", log=str(log_path))


def prepare():
    all_rows, input_to_uid = [], {}
    systems = {}
    base = None
    for key in KEYS:
        path = RUN / key / "predictions.jsonl"
        if not path.is_file():
            raise FileNotFoundError(path)
        rows = read(path)
        if len(rows) != 8096 or Counter(r["language"] for r in rows) != Counter({l: 1012 for l in LANGS}):
            raise ValueError(f"{key}: expected 8,096 rows and 1,012 per language; found {len(rows)}")
        keyed = {(r["language"], str(r["id"])): r for r in rows}
        if len(keyed) != 8096 or any(r.get("error") or not r.get("hypothesis", "").strip() for r in rows):
            raise ValueError(f"{key}: duplicate keys, failed, or empty predictions")
        if base is None:
            base = keyed
        elif keyed.keys() != base.keys():
            raise ValueError(f"{key}: prediction IDs differ from {KEYS[0]}")
        else:
            for k, row in keyed.items():
                if any(row[field] != base[k][field] for field in ("source", "reference")):
                    raise ValueError(f"{key}: source/reference mismatch for {k}")
        for row in rows:
            identity = (row["source"], row["hypothesis"], row["reference"], row["language"])
            uid = input_to_uid.get(identity)
            if uid is None:
                uid = len(all_rows)
                input_to_uid[identity] = uid
                all_rows.append({"unique_id": uid, "language": row["language"], "source": row["source"],
                                 "hypothesis": row["hypothesis"], "reference": row["reference"]})
        systems[key] = rows
    scoring = RUN / "scoring"
    scoring.mkdir(exist_ok=True)
    with (scoring / "unique_inputs.jsonl").open("w", encoding="utf-8") as f:
        for row in all_rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    mapping = {}
    for key, rows in systems.items():
        maprows = []
        for row in rows:
            uid = input_to_uid[(row["source"], row["hypothesis"], row["reference"], row["language"])]
            maprows.append({"id": row["id"], "language": row["language"], "unique_id": uid})
        mapping[key] = maprows
    save(scoring / "system_unique_mapping.json", mapping)
    save(scoring / "unique_input_manifest.json", {"systems": list(KEYS), "rows_per_system": 8096,
        "mapped_rows": len(KEYS) * 8096, "unique_exact_inputs": len(all_rows),
        "dedup_key": ["source", "hypothesis", "reference", "language"], "fresh_scores": True})
    # Reference-based metrics are calculated independently from each system's predictions.
    for key in KEYS:
        (scoring / "bleu-13a-six" / key).mkdir(parents=True, exist_ok=True)
        six = [r for r in systems[key] if r["language"] in ("en", "fr", "de", "es", "ru", "ar")]
        with (scoring / "bleu-13a-six" / key / "predictions.jsonl").open("w", encoding="utf-8") as f:
            for row in six:
                f.write(json.dumps(row, ensure_ascii=False) + "\n")
    event("prepare", status="complete", unique_inputs=len(all_rows), mapped_rows=len(KEYS) * 8096)


def score():
    if not (RUN / "scoring/unique_inputs.jsonl").is_file():
        prepare()
    # Run corpus metrics per system, with the same maintained benchmark scorer used in prior work.
    for key in KEYS:
        for metric, pred, tokenizer in (
            ("spbleu_chrf", RUN / key / "predictions.jsonl", "flores200"),
            ("bleu13a", RUN / "scoring/bleu-13a-six" / key / "predictions.jsonl", "13a"),
        ):
            summary = pred.parent / "summary.json"
            if summary.is_file():
                continue
            cmd = [BENCH_PY, str(ROOT / "scripts/benchmark_zh_translation.py"), "score", "--predictions", str(pred), "--bleu-tokenizer", tokenizer]
            run_logged(f"{key}_{metric}", cmd)
        event("reference_metrics", system=key, status="complete")
    env = os.environ.copy()
    env.update(CUDA_VISIBLE_DEVICES="1", HF_HOME="/home/aim0/data/hf-cache", HF_HUB_CACHE="/home/aim0/data/hf-cache/hub",
               HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1", PYTHONUNBUFFERED="1")
    missing = []
    unique_sha = sha256(RUN / "scoring/unique_inputs.jsonl")
    for metric in ("comet", "metricx", "xcomet"):
        out = RUN / "scoring/unique" / f"{metric}.scores.jsonl"
        summary = out.with_suffix(".summary.json")
        current = False
        if summary.is_file() and out.is_file():
            summary_obj = json.loads(summary.read_text(encoding="utf-8"))
            current = summary_obj.get("input_sha256") == unique_sha
            if metric == "xcomet":
                current = current and out.with_suffix(".error_spans.jsonl").is_file()
        if not current:
            missing.append(metric)
    if missing:
        scoring_dir = RUN / "scoring"
        phase_file = scoring_dir / "neural_active_phase.txt"
        stop_file = scoring_dir / "neural_telemetry.STOP"
        telemetry_file = scoring_dir / "neural_gpu1.csv"
        stop_file.unlink(missing_ok=True)
        phase_file.write_text("neural_startup", encoding="utf-8")
        collector = subprocess.Popen([BENCH_PY, str(ROOT / "results/zh_translation_quantized_full_20261021/collect_gpu1_telemetry.py"),
                                      str(telemetry_file), str(phase_file), str(stop_file), "1"], cwd=ROOT, env=env)
        try:
            for metric in ("comet", "metricx", "xcomet"):
                if metric not in missing:
                    continue
                phase_file.write_text(f"{metric}:loading_scoring", encoding="utf-8")
                batch = {"comet": "256", "metricx": "128", "xcomet": "16"}[metric]
                metric_env = env.copy()
                if metric == "metricx":
                    metric_env["PYTHONPATH"] = "/home/aim0/data/metricx-source"
                python = METRICX_PY if metric == "metricx" else COMET_PY
                run_logged(f"neural_{metric}", [python, str(NEURAL), metric, "--batch-size", batch], env=metric_env)
                event("neural_metric", metric=metric, status="complete")
        finally:
            phase_file.write_text("neural_stopping", encoding="utf-8")
            stop_file.touch()
            try:
                collector.wait(timeout=10)
            except subprocess.TimeoutExpired:
                collector.terminate()
                collector.wait(timeout=10)
    collect()


def collect():
    inputs = read(RUN / "scoring/unique_inputs.jsonl")
    scores = {}
    for metric in ("comet", "metricx", "xcomet"):
        values = read(RUN / "scoring/unique" / f"{metric}.scores.jsonl")
        if len(values) != len(inputs) or any(row["unique_id"] != i for i, row in enumerate(values)):
            raise ValueError(f"{metric}: score coverage mismatch")
        scores[metric] = {row["unique_id"]: row["score"] for row in values}
    mapping = json.loads((RUN / "scoring/system_unique_mapping.json").read_text(encoding="utf-8"))
    for key in KEYS:
        path = RUN / "scoring" / key / "neural_scores.jsonl"
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", encoding="utf-8") as f:
            for row in mapping[key]:
                f.write(json.dumps({**row, **{m: scores[m][row["unique_id"]] for m in scores}}, ensure_ascii=False) + "\n")
    unique_spans = read(RUN / "scoring/unique/xcomet.scores.error_spans.jsonl")
    if len(unique_spans) != len(inputs) or any(r["unique_id"] != i for i, r in enumerate(unique_spans)):
        raise ValueError("XCOMET error-span coverage mismatch")
    span_by_uid = {row["unique_id"]: row["error_spans"] for row in unique_spans}
    for key in KEYS:
        with (RUN / "scoring" / key / "xcomet_error_spans.jsonl").open("w", encoding="utf-8") as f:
            for row in mapping[key]:
                f.write(json.dumps({"id": row["id"], "language": row["language"],
                                    "unique_id": row["unique_id"],
                                    "error_spans": span_by_uid[row["unique_id"]]},
                                   ensure_ascii=False, default=str) + "\n")
    save(RUN / "scoring/neural_mapping_coverage.json", {"unique_inputs": len(inputs), "mapped_rows": sum(len(v) for v in mapping.values()),
        "expected_mapped_rows": len(KEYS) * 8096, "all_mapped": sum(len(v) for v in mapping.values()) == len(KEYS) * 8096})
    event("collect", status="complete", unique_inputs=len(inputs), mapped_rows=len(KEYS) * 8096)


def report():
    run_logged("report", [BENCH_PY, str(RUN / "build_backend_comparison_report.py")])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", choices=("prepare", "score", "collect", "report", "all"), required=True)
    a = ap.parse_args()
    if a.stage == "prepare": prepare()
    elif a.stage == "score": score()
    elif a.stage == "collect": collect()
    elif a.stage == "report": report()
    else:
        prepare(); score(); report()


if __name__ == "__main__":
    main()
