"""GPU smoke/latency evaluation of multilingual incremental retranslation.

This exercises text prefixes, NOT streaming audio, ASR, refinement or NLLW.
No new training or persistent source-encoder cache is assumed.
"""
import argparse
from datetime import datetime, timezone
import gc
import hashlib
import importlib.util
import json
import math
import os
from pathlib import Path
import subprocess
import sys
import time
from time import perf_counter


def common_prefix(left, right):
    length = 0
    for a, b in zip(left, right):
        if a != b:
            break
        length += 1
    return left[:length]


def percentiles(values):
    values = sorted(values)
    if not values:
        return {}
    return {"count":len(values), "mean":sum(values)/len(values),
            "p50":values[math.ceil(.50*len(values))-1],
            "p95":values[math.ceil(.95*len(values))-1], "max":values[-1]}


def gpu_snapshot():
    return subprocess.check_output(["nvidia-smi", "--query-gpu=index,name,memory.total,memory.used,utilization.gpu",
                                    "--format=csv,noheader"], text=True).strip()


def process_gpu_memory():
    output = subprocess.check_output(["nvidia-smi", "--query-compute-apps=pid,used_gpu_memory", "--format=csv,noheader,nounits"],text=True)
    return sum(float(line.split(",")[1].strip()) for line in output.splitlines() if line.split(",")[0].strip()==str(os.getpid()))


def simulate_queue(durations, interval):
    """Virtual FIFO schedule on measured service times; not a live speech metric."""
    finished, waits, delays = 0.0, [], []
    for index, duration in enumerate(durations):
        arrival = index * interval
        start = max(arrival, finished)
        waits.append(start-arrival)
        finished = start+duration
        delays.append(finished-arrival)
    return {"arrival_interval_seconds":interval, "queue_wait":percentiles(waits), "response_delay":percentiles(delays)}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", choices=["m2m100_418M", "small100", "nllb200_600M", "LMT-60-0.6B"], required=True)
    parser.add_argument("--model-root", type=Path, default=Path("/home/aim0/data/models/ASR"))
    parser.add_argument("--cases", type=Path, default=Path("examples/streaming_mt_cases.json"))
    parser.add_argument("--output", type=Path, default=Path("results/translation_streaming_20261017"))
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--backend", choices=["transformers", "ctranslate2"], default="ctranslate2")
    parser.add_argument("--memory-fraction", type=float, default=.055)
    parser.add_argument("--case-limit", type=int)
    parser.add_argument("--stress-sentences", type=int, default=12)
    parser.add_argument("--realtime-interval", type=float, default=0,
                        help="Actually feed synthetic text updates this many seconds apart; zero runs offline replay")
    parser.add_argument("--stream-beams", type=int, default=1)
    args = parser.parse_args()
    import torch
    import transformers
    from transformers import M2M100ForConditionalGeneration, M2M100Tokenizer
    from transformers.generation.streamers import BaseStreamer

    torch.set_num_threads(2)
    torch.manual_seed(0)
    ct2_mode = args.backend == "ctranslate2"
    device = f"cuda:{args.gpu}"
    if not ct2_mode:
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA unavailable; refusing silent CPU fallback")
        torch.cuda.set_device(args.gpu)
        torch.cuda.set_per_process_memory_fraction(args.memory_fraction, args.gpu)
        free, total = torch.cuda.mem_get_info(args.gpu)
    else:
        import ctranslate2
        if ctranslate2.get_cuda_device_count() <= args.gpu:
            raise RuntimeError("CTranslate2 CUDA unavailable; refusing CPU fallback")
        memory_lines = subprocess.check_output(["nvidia-smi","--query-gpu=memory.free,memory.total","--format=csv,noheader,nounits"],text=True).splitlines()
        free, total = [int(v.strip())*1024**2 for v in memory_lines[args.gpu].split(",")]
    if free < (850 if ct2_mode else 1500) * 1024**2:
        raise RuntimeError(f"Only {free/1024**2:.0f} MiB free on GPU {args.gpu}; existing tasks left untouched")
    args.output.mkdir(parents=True, exist_ok=True)
    model_path = args.model_root / args.model
    manifest = json.loads((model_path / "DOWNLOAD_MANIFEST.json").read_text())
    from mt_tokenizers import load_translation_tokenizer, set_source_language, chinese_decoder_prefix, chinese_language_id
    is_lmt = args.model == 'LMT-60-0.6B'
    tokenizer = None if is_lmt else load_translation_tokenizer(model_path,args.model)
    started = perf_counter()
    if is_lmt:
        from lmt_runtime import LmtRuntime,complete_unicode_prefix
        lmt = LmtRuntime(model_path,args.gpu,args.backend,memory_fraction=args.memory_fraction)
        tokenizer,model,config = lmt.tokenizer,lmt.model,lmt.config
    elif ct2_mode:
        model = ctranslate2.Translator(str(model_path/"ct2-int8"),device="cuda",device_index=args.gpu,
                                      compute_type="int8_float16",inter_threads=1,intra_threads=2)
        config = json.loads((model_path/"config.json").read_text())
    else:
        model = M2M100ForConditionalGeneration.from_pretrained(model_path, local_files_only=True,
            torch_dtype=torch.float16, low_cpu_mem_usage=True, attn_implementation="eager").eval().to(device)
        torch.cuda.synchronize()
        config = model.config.to_dict()
    metadata = {"model":args.model,"repo_id":manifest["repo_id"],"revision":manifest["revision"],
        "created_at":datetime.now(timezone.utc).isoformat(),"torch":torch.__version__,
        "transformers":transformers.__version__,"cuda":torch.version.cuda,
        "gpu":subprocess.check_output(["nvidia-smi","--query-gpu=name","--format=csv,noheader"],text=True).splitlines()[args.gpu],
        "gpu_index":args.gpu,"dtype":"int8_float16" if ct2_mode else "float16", "backend":args.backend,
        "ctranslate2":ctranslate2.__version__ if ct2_mode else None,
        "gpu_snapshot_start":gpu_snapshot(),"initial_free_mib":free/1024**2,
        "memory_fraction":args.memory_fraction,"batch_size":1,"stream_beam_size":args.stream_beams,
        "max_new_tokens":96,"parameter_count":None if ct2_mode else sum(p.numel() for p in model.parameters()),
        "encoder_layers":config.get("encoder_layers",0),"decoder_layers":config.get("decoder_layers",config.get('num_hidden_layers')),
        "load_seconds":perf_counter()-started,
        "scope":"Synthetic incremental text on a shared GPU, no ASR/audio/Refiner/NLLW, no real audio latency or production accuracy claims",
        "realtime_text_interval_seconds":args.realtime_interval,
        "cases_sha256":hashlib.sha256(args.cases.read_bytes()).hexdigest()}
    print(json.dumps({"event":"model_loaded",**metadata},ensure_ascii=False),flush=True)
    eos = config["eos_token_id"]
    special = set(tokenizer.all_special_ids)
    start_id = config.get("decoder_start_token_id")

    def allocated_mib():
        return None if ct2_mode else torch.cuda.memory_allocated()/1024**2

    def peak_mib():
        return None if ct2_mode else torch.cuda.max_memory_allocated()/1024**2

    def infer(text, language, committed=None, beams=1, streamer=None):
        if is_lmt:
            callback=None
            if streamer is not None and ct2_mode:
                def callback(step):
                    streamer.events.append({'elapsed_seconds':perf_counter()-streamer.started,
                        'ids':[tokenizer.convert_tokens_to_ids(step.token)]})
                    return False
            result,elapsed=lmt.translate([text],language,committed,beams=beams,max_new_tokens=96,
                callback=callback,streamer=streamer if not ct2_mode else None)
            item=result[0]
            return {'text':item['hypothesis'],'ids':item['body_ids'],
                'unknown_tokens':item['unknown_tokens'],'input_tokens':item['source_tokens'],
                'output_tokens':item['target_tokens'],'generation_seconds':elapsed,'seconds':elapsed,
                'ended_with_eos':item['eos']}
        t0 = perf_counter()
        set_source_language(tokenizer,args.model,language)
        if ct2_mode:
            input_ids = tokenizer.encode(text)
            if len(input_ids)>256:
                raise ValueError("Test text exceeds bounded source window")
            prefix = chinese_decoder_prefix(tokenizer,args.model)
            prefix += tokenizer.convert_ids_to_tokens(committed or [])
            options = {"beam_size":beams,"max_decoding_length":96,"return_end_token":True}
            if streamer is not None:
                def on_step(step):
                    streamer.events.append({"elapsed_seconds":perf_counter()-streamer.started,
                                            "ids":[tokenizer.convert_tokens_to_ids(step.token)]})
                    return False
                options["callback"]=on_step
            gen_start=perf_counter()
            output=model.translate_batch([tokenizer.convert_ids_to_tokens(input_ids)],
                                         target_prefix=[prefix] if prefix else None, **options)[0]
            gen_seconds=perf_counter()-gen_start
            ids=tokenizer.convert_tokens_to_ids(output.hypotheses[0])
            content=[token for token in ids if token not in special]
            if committed and content[:len(committed)]!=committed:
                raise RuntimeError("CT2 generation violated committed target prefix")
            return {"text":tokenizer.decode(content,skip_special_tokens=True),"ids":content,
                    "unknown_tokens":ids.count(tokenizer.unk_token_id),
                    "input_tokens":len(input_ids),"output_tokens":len(content),"generation_seconds":gen_seconds,
                    "seconds":perf_counter()-t0,"ended_with_eos":bool(ids and ids[-1]==eos)}
        encoded = tokenizer(text, return_tensors="pt").to(device)
        if encoded["input_ids"].shape[-1] > 256:
            raise ValueError("Test text exceeds bounded source window")
        options = dict(max_new_tokens=96, num_beams=beams, do_sample=False, use_cache=True)
        prefix_ids = [start_id]
        if args.model != "small100":
            target = chinese_language_id(tokenizer,args.model)
            options["forced_bos_token_id"] = target
            prefix_ids.append(target)
        if committed:
            prefix_ids.extend(committed)
            options["decoder_input_ids"] = torch.tensor([prefix_ids], dtype=torch.long, device=device)
        if streamer is not None:
            options["streamer"] = streamer
        torch.cuda.synchronize()
        gen_start = perf_counter()
        with torch.inference_mode():
            generated = model.generate(**encoded, **options)
        torch.cuda.synchronize()
        gen_seconds = perf_counter()-gen_start
        ids = generated[0].tolist()
        content = [token for token in ids if token not in special]
        if committed and content[:len(committed)] != committed:
            raise RuntimeError("Generation violated committed target prefix")
        text_out = tokenizer.decode(content, skip_special_tokens=True)
        return {"text":text_out,"ids":content,"input_tokens":encoded["input_ids"].shape[-1],
                "unknown_tokens":ids.count(tokenizer.unk_token_id),
                "output_tokens":len(content),"generation_seconds":gen_seconds,
                "seconds":perf_counter()-t0,"ended_with_eos":ids[-1] == eos}

    # Warmup separately; do not count initialization in steady-state latency.
    infer("Hello, please confirm the delivery date.", "en")
    if not ct2_mode:
        torch.cuda.reset_peak_memory_stats()
    metadata["process_gpu_used_mib_after_warmup"] = process_gpu_memory()

    class Probe(BaseStreamer):
        def __init__(self):
            self.events, self.skip = [], True
        def put(self, value):
            if self.skip:
                self.skip = False
                return
            self.events.append({"elapsed_seconds":perf_counter()-self.started,"ids":value.reshape(-1).tolist()})
        def end(self):
            pass
    probe = Probe()
    probe.started = perf_counter()
    probe_result = infer("Please send 300 units only after receiving the deposit.", "en", streamer=probe)
    content_events = [e for e in probe.events if any(i not in special for i in e["ids"])]
    metadata["fixed_input_token_stream_probe"] = {
        "text":probe_result["text"], "first_content_seconds":content_events[0]["elapsed_seconds"] if content_events else None,
        "total_seconds":probe_result["seconds"], "events":len(probe.events),
        "note":"One complete source was supplied first; this tests output token callbacks only"}

    cases = json.loads(args.cases.read_text())["cases"]
    if args.case_limit:
        cases = cases[:args.case_limit]
    rows = []
    raw_file = (args.output / f"{args.model}.steps.jsonl").open("w")
    cases_file = (args.output / f"{args.model}.cases.jsonl").open("w")
    summaries = []
    for case in cases:
        prefixes = case.get("prefixes")
        final_source = prefixes[-1] if prefixes else "".join(case["chunks"])
        offline = infer(final_source, case["language"], beams=3)
        for mode in ("retranslate", "local_agreement_forced_prefix"):
            source, previous_ids, committed, previous_text = "", [], [], ""
            deleted, times, records = 0, [], []
            updates = prefixes if prefixes else case["chunks"]
            replay_started = perf_counter()
            for index, chunk in enumerate(updates):
                if args.realtime_interval:
                    time.sleep(max(0, replay_started+index*args.realtime_interval-perf_counter()))
                submitted_at = perf_counter()-replay_started
                source = chunk if prefixes else source+chunk
                final = index == len(updates)-1
                prediction = infer(source, case["language"], committed if mode != "retranslate" else None,
                                   beams=args.stream_beams)
                candidate = prediction.pop("ids")
                if mode != "retranslate":
                    if final:
                        committed = candidate
                    else:
                        agreed = common_prefix(previous_ids, candidate)
                        # Hold back one decoder token even if successive prefixes agree.
                        new_length = max(len(committed), len(agreed)-1)
                        if is_lmt:
                            new_length=complete_unicode_prefix(tokenizer,candidate,new_length,len(committed))
                        committed = candidate[:new_length]
                common_chars = len(common_prefix(previous_text, prediction["text"]))
                erased = len(previous_text)-common_chars
                deleted += erased
                record = {"model":args.model,"case_id":case["id"],"language":case["language"],"mode":mode,
                    "step":index,"source":source,"final":final,**prediction,
                    "submitted_seconds_since_first_update":submitted_at,
                    "completed_seconds_since_first_update":perf_counter()-replay_started,
                    "committed_text":tokenizer.decode(committed, skip_special_tokens=True),
                    "draft_text":tokenizer.decode(candidate[len(committed):], skip_special_tokens=True),
                    "erased_characters_from_previous_display":erased,
                    "cuda_allocated_mib":allocated_mib(),
                    "cuda_peak_allocated_mib":peak_mib()}
                records.append(record)
                rows.append(record)
                raw_file.write(json.dumps(record, ensure_ascii=False)+"\n")
                raw_file.flush()
                times.append(record["seconds"])
                previous_ids, previous_text = candidate, prediction["text"]
            summary = {"case_id":case["id"],"language":case["language"],"mode":mode,
                "source":source,"reference":case["reference"],"final_translation":previous_text,
                "offline_beam3_translation":offline["text"],"offline_beam3_seconds":offline["seconds"],
                "normalized_erasure_characters":deleted/max(1,len(previous_text)),
                "missing_literal_flags":[x for x in case["critical_literals"] if x not in previous_text],
                "literal_check_note":"Surface hints only; numeric localization may be legitimate; no semantic pass/fail",
                "first_commit_step":next((r["step"] for r in records if r["committed_text"]),None),
                "latency":percentiles(times)}
            summaries.append(summary)
            cases_file.write(json.dumps(summary, ensure_ascii=False)+"\n")
            cases_file.flush()
        print(json.dumps({"event":"case_complete","model":args.model,"case":case["id"],"rows":len(rows)},ensure_ascii=False),flush=True)

    # Bounded-window long-session replay. Reset each completed sentence, never
    # append the entire session to a single encoder context.
    stress = []
    stress_file = (args.output / f"{args.model}.stress.jsonl").open("w")
    for index in range(args.stress_sentences):
        case = cases[index % len(cases)]
        source = case["prefixes"][-1] if case.get("prefixes") else "".join(case["chunks"])
        result = infer(source, case["language"], beams=args.stream_beams)
        result.pop("ids")
        item = {"sentence":index,"case_id":case["id"],"language":case["language"],**result,
                "cuda_allocated_mib":allocated_mib()}
        stress.append(item)
        stress_file.write(json.dumps(item,ensure_ascii=False)+"\n")
        stress_file.flush()
    raw_file.close()
    cases_file.close()
    stress_file.close()
    durations = [r["seconds"] for r in rows]
    metadata.update({"status":"completed","languages":sorted({c["language"] for c in cases}),
        "case_count":len(cases),"step_count":len(rows),"step_latency_seconds":percentiles(durations),
        "per_language_latency_seconds":{lang:percentiles([r["seconds"] for r in rows if r["language"]==lang]) for lang in sorted({c["language"] for c in cases})},
        "output_truncations":sum(not r["ended_with_eos"] for r in rows),
        "empty_translations":sum(not r["text"].strip() for r in rows),
        "peak_cuda_allocated_mib":peak_mib(),
        "peak_cuda_reserved_mib":None if ct2_mode else torch.cuda.max_memory_reserved()/1024**2,
        "process_gpu_used_mib_end":process_gpu_memory(),
        "normalized_erasure_by_mode":{mode:percentiles([c["normalized_erasure_characters"] for c in summaries if c["mode"]==mode]) for mode in ("retranslate","local_agreement_forced_prefix")},
        "virtual_queue_simulations":[simulate_queue(durations, dt) for dt in (.5,1.0,2.0)],
        "stress":{"sentences":len(stress),"latency":percentiles([r["seconds"] for r in stress]),
                  "allocated_mib_start":stress[0]["cuda_allocated_mib"] if stress else None,
                  "allocated_mib_end":stress[-1]["cuda_allocated_mib"] if stress else None},
        "gpu_snapshot_end":gpu_snapshot()})
    (args.output/f"{args.model}.summary.json").write_text(json.dumps(metadata,ensure_ascii=False,indent=2)+"\n")
    print(json.dumps({"event":"benchmark_complete",**metadata},ensure_ascii=False),flush=True)
    del model
    gc.collect()
    if not ct2_mode:
        torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
