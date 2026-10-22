#!/usr/bin/env python3
"""Validate the three full prediction sets and prepare common metric inputs."""

from __future__ import annotations

from collections import Counter
import hashlib
import json
import os
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
RUN = Path(__file__).resolve().parent
MODELS = {
    "q8_0": ("index", "Index-Translate-2B GGUF Q8_0"),
    "q6_k": ("hy", "Index-Translate-2B GGUF Q6_K"),
    "q4_k_m": ("lmt", "Index-Translate-2B GGUF Q4_K_M"),
}
LANGUAGES = ("en", "fr", "de", "es", "ru", "ja", "ko", "ar")
BLEU_13A_LANGUAGES = ("en", "fr", "de", "es", "ru", "ar")


def read_jsonl(path: Path) -> list[dict]:
    with path.open(encoding="utf-8") as stream:
        return [json.loads(line) for line in stream if line.strip()]


def digest(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as stream:
        for row in rows:
            stream.write(json.dumps(row, ensure_ascii=False) + "\n")
    temporary.replace(path)


def make_link(destination: Path, target: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.is_symlink():
        destination.unlink()
    elif destination.exists():
        raise FileExistsError(f"Refusing to overwrite {destination}")
    destination.symlink_to(target.resolve())


def main() -> None:
    loaded: dict[str, list[dict]] = {}
    expected_keys = None
    reference_hash = None
    manifest = {"models": {}, "languages": list(LANGUAGES)}
    for model_key, (metric_key, model_name) in MODELS.items():
        path = RUN / model_key / "predictions.jsonl"
        rows = read_jsonl(path)
        if len(rows) != 8096:
            raise ValueError(f"{model_key}: expected 8,096 rows, found {len(rows)}")
        counts = Counter(row["language"] for row in rows)
        if counts != Counter({language: 1012 for language in LANGUAGES}):
            raise ValueError(f"{model_key}: unexpected language counts: {counts}")
        keys = [(str(row["id"]), row["language"]) for row in rows]
        if len(keys) != len(set(keys)):
            raise ValueError(f"{model_key}: duplicate (id, language) rows")
        if any(row.get("error") for row in rows):
            raise ValueError(f"{model_key}: contains failed translations")
        if any(not row.get("hypothesis", "").strip() for row in rows):
            raise ValueError(f"{model_key}: contains an empty hypothesis")
        if expected_keys is None:
            expected_keys = keys
            reference_hash = digest("\n".join(
                f"{row['id']}\t{row['language']}\t{row['source']}\t{row['reference']}"
                for row in rows
            ))
        else:
            if keys != expected_keys:
                raise ValueError(f"{model_key}: row order/IDs differ from the first model")
            current_hash = digest("\n".join(
                f"{row['id']}\t{row['language']}\t{row['source']}\t{row['reference']}"
                for row in rows
            ))
            if current_hash != reference_hash:
                raise ValueError(f"{model_key}: sources or references differ")
        loaded[model_key] = rows
        manifest["models"][model_key] = {
            "metric_key": metric_key,
            "model_name": model_name,
            "rows": len(rows),
            "predictions_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        }

    staging = RUN / "scoring"
    metricx_rows = []
    bleu13a_counts = {}
    for model_key, (metric_key, model_name) in MODELS.items():
        predictions = RUN / model_key / "predictions.jsonl"
        for scorer in ("comet-input", "xcomet-input"):
            make_link(staging / scorer / metric_key / "predictions.jsonl", predictions)
        for scorer in ("spbleu-chrf",):
            make_link(staging / scorer / model_key / "predictions.jsonl", predictions)

        six_languages = [row for row in loaded[model_key] if row["language"] in BLEU_13A_LANGUAGES]
        write_jsonl(staging / "bleu-13a-six" / model_key / "predictions.jsonl", six_languages)
        bleu13a_counts[model_key] = len(six_languages)
        for row in loaded[model_key]:
            metricx_rows.append({
                "bench_row_id": len(metricx_rows),
                "model_key": model_key,
                "model_name": model_name,
                "id": row["id"],
                "language": row["language"],
                "source": row["source"],
                "hypothesis": row["hypothesis"],
                "reference": row["reference"],
            })

    write_jsonl(staging / "metricx-input-all.jsonl", metricx_rows)
    manifest["total_model_rows"] = len(metricx_rows)
    manifest["bleu_13a_languages"] = list(BLEU_13A_LANGUAGES)
    manifest["bleu_13a_rows_per_model"] = bleu13a_counts
    manifest["source_reference_alignment_sha256"] = reference_hash
    (staging / "input_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
