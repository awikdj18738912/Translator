"""Run a CPU FP32 full-input LMT quality control on streaming MT examples.

This is a translation-quality control only. Its timings are CPU timings and
must not be compared with GPU latency results.
"""
import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
from time import perf_counter


DEFAULT_MODEL = Path("/home/aim0/data/models/ASR/LMT-60-0.6B")
DEFAULT_CASES = Path("examples/streaming_mt_cases.json")
DEFAULT_OUTPUT = Path("results/lmt600m_comparison_20261017/native_fp32_cpu")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-path", type=Path, default=DEFAULT_MODEL)
    parser.add_argument("--cases", type=Path, default=DEFAULT_CASES)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()

    case_path = args.output_dir / "LMT-60-0.6B.cases.jsonl"
    summary_path = args.output_dir / "LMT-60-0.6B.summary.json"
    args.output_dir.mkdir(parents=True, exist_ok=True)
    if case_path.exists() or summary_path.exists():
        raise FileExistsError(f"Refusing to overwrite existing native control output in {args.output_dir}")

    # Hide CUDA before importing torch so this control cannot use a GPU.
    os.environ["CUDA_VISIBLE_DEVICES"] = ""
    import torch
    import transformers

    torch.set_num_threads(4)
    started = perf_counter()
    tokenizer = transformers.AutoTokenizer.from_pretrained(
        args.model_path,
        local_files_only=True,
        padding_side="left",
    )
    model = transformers.AutoModelForCausalLM.from_pretrained(
        args.model_path,
        local_files_only=True,
        dtype=torch.float32,
        low_cpu_mem_usage=True,
    ).eval().to(torch.device("cpu"))
    parameter_count = sum(parameter.numel() for parameter in model.parameters())
    generation = json.loads((args.model_path / "generation_config.json").read_text())
    eos_ids = generation["eos_token_id"]
    if isinstance(eos_ids, int):
        eos_ids = [eos_ids]
    eos_ids = [int(token_id) for token_id in eos_ids]

    cases = json.loads(args.cases.read_text())["cases"]
    manifest = json.loads((args.model_path / "DOWNLOAD_MANIFEST.json").read_text())
    special_ids = set(tokenizer.all_special_ids)
    results = []
    with case_path.open("x", encoding="utf-8") as output_file:
        for case in cases:
            source = case["prefixes"][-1] if case.get("prefixes") else "".join(case["chunks"])
            language = case["language"]
            language_name = {
                "en": "English", "fr": "French", "de": "German", "es": "Spanish",
                "ru": "Russian", "ja": "Japanese", "ko": "Korean", "ar": "Arabic",
            }[language]
            prompt = (
                f"Translate the following text from {language_name} into Chinese:\n"
                f"{language_name}: {source}\nChinese:"
            )
            prompt_ids = tokenizer.apply_chat_template(
                [{"role": "user", "content": prompt}],
                tokenize=True,
                add_generation_prompt=True,
            )
            input_ids = torch.tensor([prompt_ids], dtype=torch.long, device=torch.device("cpu"))
            attention_mask = torch.ones_like(input_ids, device=torch.device("cpu"))

            call_started = perf_counter()
            with torch.inference_mode():
                generated = model.generate(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    num_beams=3,
                    do_sample=False,
                    max_new_tokens=96,
                    use_cache=True,
                    eos_token_id=eos_ids,
                    pad_token_id=tokenizer.pad_token_id,
                )
            elapsed = perf_counter() - call_started
            generated_ids = generated[0, len(prompt_ids):].tolist()
            final_translation = tokenizer.decode(generated_ids, skip_special_tokens=True)
            item = {
                "case_id": case["id"],
                "language": language,
                "mode": "retranslate",
                "source": source,
                "reference": case["reference"],
                "final_translation": final_translation,
                "offline_beam3_translation": final_translation,
                "elapsed_seconds": elapsed,
                "generated_token_ids": generated_ids,
                "generated_token_count": len(generated_ids),
                "eos": bool(generated_ids and generated_ids[-1] in eos_ids),
                "unknown_tokens": sum(token_id == tokenizer.unk_token_id for token_id in generated_ids),
                "replacement_characters": final_translation.count("\ufffd"),
            }
            # Keep the special-token variable explicit for parity with the MT
            # runtime's output accounting; full generated IDs are preserved above.
            item["non_special_generated_token_count"] = sum(
                token_id not in special_ids for token_id in generated_ids
            )
            output_file.write(json.dumps(item, ensure_ascii=False) + "\n")
            output_file.flush()
            results.append(item)

    summary = {
        "status": "completed",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "runtime": "Transformers AutoModelForCausalLM",
        "transformers": transformers.__version__,
        "model": "LMT-60-0.6B",
        "repo_id": manifest["repo_id"],
        "revision": manifest["revision"],
        "device": "cpu",
        "dtype": "float32",
        "parameter_count": parameter_count,
        "torch_threads": torch.get_num_threads(),
        "beam_size": 3,
        "do_sample": False,
        "max_new_tokens": 96,
        "use_cache": True,
        "eos_token_ids": eos_ids,
        "case_count": len(results),
        "cases_sha256": hashlib.sha256(args.cases.read_bytes()).hexdigest(),
        "total_elapsed_seconds": perf_counter() - started,
        "timing_scope": "CPU full-input generation; not comparable to GPU latency",
    }
    with summary_path.open("x", encoding="utf-8") as output_file:
        json.dump(summary, output_file, ensure_ascii=False, indent=2)
        output_file.write("\n")
    print(json.dumps({"event": "native_control_complete", **summary}, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
