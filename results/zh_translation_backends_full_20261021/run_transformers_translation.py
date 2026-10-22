#!/usr/bin/env python3
"""Full, batched HF inference for BF16 and official compressed-tensors models."""
from __future__ import annotations
import argparse
from collections import Counter
from datetime import datetime, timezone
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import statistics
import subprocess
import sys
import time
import uuid

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts"))
from benchmark_zh_translation import LANGS, prompt_for, read_jsonl

MODEL_NAMES = {"bf16_transformers": "Index-Translate-2B", "fp8_transformers": "Index-Translate-2B-FP8",
               "nvfp4_transformers": "Index-Translate-2B-FP4"}


def save(path, value):
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n")
    temporary.replace(path)


def snapshot():
    return subprocess.check_output(["nvidia-smi", "--query-gpu=index,uuid,name,memory.used,memory.total,utilization.gpu",
                                    "--format=csv,noheader"], text=True).strip()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-key", choices=tuple(MODEL_NAMES), required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--max-tokens", type=int, default=512)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    assert args.batch_size > 0 and args.max_tokens > 0
    assert os.environ.get("CUDA_VISIBLE_DEVICES") == "1", "Use physical GPU1 only"
    import torch
    from transformers import AutoTokenizer, AutoModelForImageTextToText, GenerationConfig
    from transformers.utils.quantization_config import CompressedTensorsConfig
    assert torch.cuda.is_available() and torch.cuda.device_count() == 1
    torch.cuda.set_device(0)
    torch.set_num_threads(8)
    torch.manual_seed(20261021)
    model_path = Path("/home/aim0/data/models/ASR") / MODEL_NAMES[args.model_key]
    data = ROOT / "data/eval/zh_translation/flores101_devtest.jsonl"
    rows = read_jsonl(data)
    assert len(rows) == 8096 and Counter(r["language"] for r in rows) == Counter({l:1012 for l in LANGS})
    output = args.output
    output.parent.mkdir(parents=True, exist_ok=True)
    assert args.resume or not output.exists(), f"Refusing to overwrite {output}"
    progress = output.with_suffix(".progress.jsonl")
    completed = {}
    if args.resume and progress.exists():
        for line in progress.open(encoding="utf-8"):
            record = json.loads(line)
            completed[(record["language"], str(record["id"]))] = record
    session = uuid.uuid4().hex
    metadata = {"status":"loading", "model_key":args.model_key, "model_path":str(model_path),
        "model_bytes":sum(p.stat().st_size for p in model_path.glob("*.safetensors")),
        "runtime":"Transformers", "physical_gpu":1, "batch_size":args.batch_size,
        "total_cases":8096, "sentences_per_language":1012, "dataset_sha256":hashlib.sha256(data.read_bytes()).hexdigest(),
        "decode":{"do_sample":False,"num_beams":1,"repetition_penalty":1.0,"max_tokens":args.max_tokens,
                  "eos_token_ids":[248044,248046],"enable_thinking":False},
        "started_utc":datetime.now(timezone.utc).isoformat(), "gpu_snapshot_start":snapshot(),
        "versions":{name:importlib.metadata.version(name) for name in ("torch","transformers","compressed-tensors")}}
    metadata["configured_batch_size"] = args.batch_size
    metadata["actual_batch_size_max"] = 0
    metadata["oom_retries"] = 0
    meta_path = output.parent / "metadata.json"
    save(meta_path, metadata)
    load_started = time.perf_counter()
    tokenizer = AutoTokenizer.from_pretrained(model_path, local_files_only=True)
    tokenizer.padding_side = "left"
    kwargs = {"local_files_only":True,"dtype":torch.bfloat16,"device_map":{"":"cuda:0"},"attn_implementation":"sdpa"}
    if args.model_key != "bf16_transformers":
        # In installed Transformers 5.14.1, run_compressed=False keeps ordinary
        # Linear modules while compressed-tensors installs quantization QDQ hooks,
        # compresses checkpoint weights, then decompresses them on first forward.
        kwargs["quantization_config"] = CompressedTensorsConfig(run_compressed=False)
    model = AutoModelForImageTextToText.from_pretrained(model_path, **kwargs).eval()
    # Transformers derives model.device from its first parameter. In this model
    # the visual tower is registered first, so moving that unused tower to CPU
    # makes generate() move input_ids to CPU while the text embeddings stay on
    # GPU. Keep the tower resident on GPU; text-only prompts never execute it.
    visual = getattr(getattr(model, "model", None), "visual", None)
    metadata["load_seconds"] = time.perf_counter() - load_started
    metadata["unused_vision_tower_resident_on_gpu"] = visual is not None
    config = GenerationConfig(do_sample=False,num_beams=1,use_cache=True,repetition_penalty=1.0,
        max_new_tokens=args.max_tokens,eos_token_id=[248044,248046],pad_token_id=tokenizer.pad_token_id,
        bos_token_id=tokenizer.bos_token_id)

    def generate(group, max_tokens):
        texts = [tokenizer.apply_chat_template([{"role":"user","content":prompt_for("index",r["source"],r["language"])}],
                  tokenize=False,add_generation_prompt=True,enable_thinking=False) for r in group]
        inputs = tokenizer(texts,return_tensors="pt",padding=True,add_special_tokens=False).to("cuda:0")
        with torch.inference_mode():
            result = model.generate(**inputs,generation_config=config,max_new_tokens=max_tokens)
        torch.cuda.synchronize()
        return result[:,inputs.input_ids.shape[1]:].cpu().tolist(), inputs.attention_mask.sum(dim=1).cpu().tolist()

    adaptive_batch_size = args.batch_size

    def generate_with_oom_backoff(group, max_tokens):
        """Retry a batch after CUDA OOM, halving it until it fits."""
        nonlocal adaptive_batch_size
        while True:
            try:
                bodies, input_tokens = generate(group, max_tokens)
                metadata["actual_batch_size_max"] = max(metadata["actual_batch_size_max"], len(group))
                return bodies, input_tokens, group
            except RuntimeError as error:
                if not isinstance(error, torch.cuda.OutOfMemoryError) and "out of memory" not in str(error).lower():
                    raise
                if len(group) == 1:
                    raise
                metadata["oom_retries"] += 1
                adaptive_batch_size = max(1, min(adaptive_batch_size, len(group) // 2))
                group = group[:adaptive_batch_size]
                torch.cuda.empty_cache()

    warmup_start = time.perf_counter()
    warmup_requests = 0
    while warmup_requests < args.batch_size:
        remaining = args.batch_size - warmup_requests
        _, _, warmup_group = generate_with_oom_backoff([rows[0]] * min(remaining, adaptive_batch_size),32)
        warmup_requests += len(warmup_group)
    metadata["warmup_seconds"] = time.perf_counter() - warmup_start
    metadata["warmup_requests"] = warmup_requests
    quantized = []
    for name, layer in model.named_modules():
        scheme = getattr(layer,"quantization_scheme",None)
        if scheme is not None:
            quantized.append({"name":name,"status":str(getattr(layer,"quantization_status",None)),
                "weight_bits":getattr(getattr(scheme,"weights",None),"num_bits",None),
                "activation_bits":getattr(getattr(scheme,"input_activations",None),"num_bits",None)})
    metadata["quantized_layers"] = quantized
    metadata["execution_mode"] = "BF16 PyTorch" if not quantized else "decompressed weights and compressed-tensors software QDQ"
    from transformers.models.qwen3_5 import modeling_qwen3_5
    metadata["fast_linear_attention_available"] = getattr(modeling_qwen3_5,"is_fast_path_available",None)
    metadata["status"] = "running"
    save(meta_path,metadata)
    torch.cuda.reset_peak_memory_stats(0)
    with progress.open("a",encoding="utf-8") as stream:
        for language in LANGS:
            pending = [r for r in rows if r["language"] == language and (language,str(r["id"])) not in completed]
            start = 0
            while start < len(pending):
                group = pending[start:start+adaptive_batch_size]
                started = time.time(); clock = time.perf_counter()
                bodies, input_tokens, group = generate_with_oom_backoff(group,args.max_tokens)
                decoded = []
                for row, body, n_input in zip(group,bodies,input_tokens):
                    first_eos = next((i for i,t in enumerate(body) if t in (248044,248046)),None)
                    stopped = first_eos is not None
                    kept = body[:first_eos+1] if stopped else body
                    hypothesis = tokenizer.decode(kept,skip_special_tokens=True,clean_up_tokenization_spaces=False).strip()
                    if not hypothesis:
                        raise RuntimeError(f"Empty translation: {language}:{row['id']}")
                    decoded.append((row,hypothesis,len(kept),stopped,n_input))
                elapsed = time.perf_counter() - clock; ended = time.time()
                batch_id = f"{session}:{language}:{start}"
                for row,hypothesis,n_tokens,stopped,n_input in decoded:
                    record={**row,"hypothesis":hypothesis,"raw_hypothesis":hypothesis,
                        "input_tokens":n_input,"generated_tokens_including_eos":n_tokens,"content_tokens":n_tokens,
                        "generation_seconds":elapsed,"latency_seconds":elapsed,"batch_latency_seconds":elapsed,
                        "request_started_at":started,"request_ended_at":ended,"batch_id":batch_id,"batch_size":len(group),
                        "actual_batch_size":len(group),
                        "session_id":session,"finish_reason":"stop" if stopped else "length", "eos":stopped,
                        "truncated":not stopped,"empty_output":False,"error":None}
                    completed[(language,str(row["id"]))]=record
                    stream.write(json.dumps(record,ensure_ascii=False)+"\n")
                stream.flush()
                print(f"{args.model_key}: {len(completed)}/8096, {len(group)/elapsed:.2f} rows/s, {language}",flush=True)
                start += len(group)
    ordered=[completed[(r["language"],str(r["id"]))] for r in rows]
    temporary=output.with_suffix(".jsonl.tmp")
    with temporary.open("w",encoding="utf-8") as stream:
        for row in ordered:stream.write(json.dumps(row,ensure_ascii=False)+"\n")
    temporary.replace(output)
    batch_map={r["batch_id"]:r for r in ordered}
    wall=sum(r["batch_latency_seconds"] for r in batch_map.values())
    languages={}
    for lang in LANGS:
        group=[r for r in ordered if r["language"]==lang]
        seconds=sum(r["batch_latency_seconds"] for r in batch_map.values() if r["language"]==lang)
        n_tokens=sum(r["generated_tokens_including_eos"] for r in group)
        languages[lang]={"count":len(group),"wall_seconds":seconds,"sentences_per_second":len(group)/seconds,
            "generated_tokens":n_tokens,"generation_tokens_per_second":n_tokens/seconds,
            "empty_outputs":0,"truncated_outputs":sum(r["truncated"] for r in group)}
    total_tokens=sum(r["generated_tokens_including_eos"] for r in ordered)
    metadata.update(status="complete",rows_written=len(ordered),completed_utc=datetime.now(timezone.utc).isoformat(),
        wall_seconds=wall,sentences_per_second=len(ordered)/wall,generated_tokens=total_tokens,
        generated_tokens_per_second=total_tokens/wall,language_results=languages,
        peak_allocated_bytes=torch.cuda.max_memory_allocated(0),peak_reserved_bytes=torch.cuda.max_memory_reserved(0),
        gpu_snapshot_end=snapshot(),predictions_sha256=hashlib.sha256(output.read_bytes()).hexdigest())
    metadata["final_adaptive_batch_size"] = adaptive_batch_size
    save(meta_path,metadata)
    progress.unlink()
    print(f"COMPLETE {args.model_key}: 8096 rows, {len(ordered)/wall:.2f} rows/s",flush=True)


if __name__=="__main__":
    main()
