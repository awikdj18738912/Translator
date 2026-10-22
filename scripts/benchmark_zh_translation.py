#!/usr/bin/env python3
"""Reproducible Chinese -> eight-language quality and GPU throughput benchmark.

prepare needs only Python's standard library. run needs PyTorch/Transformers;
score needs SacreBLEU and SentencePiece. No application services are changed.
"""
import argparse
from collections import defaultdict
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import random
import subprocess
import sys
import tarfile
import time
import urllib.request
import uuid

ROOT = Path(__file__).resolve().parents[1]
MODEL_ROOT = Path('/home/aim0/data/models/ASR')
DEFAULT_DATA = ROOT / 'data/eval/zh_translation/flores101_devtest.jsonl'
MODELS = {'index': 'Index-Translate-2B', 'hy': 'Hy-MT2-1.8B', 'lmt': 'LMT-60-1.7B'}
LANGS = {
    'en': ('eng', 'English', '英语'), 'fr': ('fra', 'French', '法语'),
    'de': ('deu', 'German', '德语'), 'es': ('spa', 'Spanish', '西班牙语'),
    'ru': ('rus', 'Russian', '俄语'), 'ja': ('jpn', 'Japanese', '日语'),
    'ko': ('kor', 'Korean', '韩语'), 'ar': ('ara', 'Arabic', '阿拉伯语'),
}
ARCHIVE_URL = 'https://dl.fbaipublicfiles.com/flores101/dataset/flores101_dataset.tar.gz'
PROTOCOL = 'zh-translation-bf16-greedy-batched-v1'


def emit(event, **fields):
    print(json.dumps({'event': event, **fields}, ensure_ascii=False), flush=True)


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as source:
        for chunk in iter(lambda: source.read(8 * 1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def save_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + '.tmp')
    temp.write_text(json.dumps(value, ensure_ascii=False, indent=2) + '\n')
    temp.replace(path)


def read_jsonl(path):
    rows = []
    keys = set()
    for line in Path(path).read_text().splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        for field in ('id', 'language', 'source', 'reference'):
            if field not in row:
                raise ValueError(f'Missing field {field}: {row}')
        if row['language'] not in LANGS:
            raise ValueError(f'Unsupported target language: {row["language"]}')
        if not isinstance(row['source'], str) or not row['source'].strip():
            raise ValueError('Empty/non-text source')
        if not isinstance(row['reference'], str) or not row['reference'].strip():
            raise ValueError('Empty/non-text reference')
        key = (str(row['id']), row['language'])
        if key in keys:
            raise ValueError(f'Duplicate case: {key}')
        keys.add(key)
        rows.append(row)
    if not rows:
        raise ValueError('Empty dataset')
    return rows


def prepare(args):
    archive = args.archive
    if not archive.exists():
        archive.parent.mkdir(parents=True, exist_ok=True)
        partial = archive.with_suffix('.partial')
        with urllib.request.urlopen(ARCHIVE_URL, timeout=120) as source, partial.open('wb') as out:
            while chunk := source.read(1024 * 1024):
                out.write(chunk)
        partial.replace(archive)
    contents = {}
    # Read members directly; never extract untrusted archive paths to disk.
    with tarfile.open(archive, 'r:gz') as tar:
        for code in ['zho_simpl', *(v[0] for v in LANGS.values())]:
            matches = [m for m in tar.getmembers() if m.name.endswith(f'/devtest/{code}.devtest')]
            if len(matches) != 1:
                raise ValueError(f'Expected exactly one devtest member for {code}')
            contents[code] = tar.extractfile(matches[0]).read().decode('utf-8').splitlines()
    if {len(lines) for lines in contents.values()} != {1012}:
        raise ValueError('Expected 1012 aligned sentences in every language')
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open('w') as out:
        for language, (code, _, _) in LANGS.items():
            for i, (source, reference) in enumerate(zip(contents['zho_simpl'], contents[code]), 1):
                out.write(json.dumps({'id': i, 'language': language, 'source_language': 'zh',
                                      'source': source, 'reference': reference}, ensure_ascii=False) + '\n')
    manifest = {
        'dataset': 'FLORES-101', 'split': 'devtest', 'license': 'CC-BY-SA-4.0',
        'official_url': ARCHIVE_URL,
        'documentation': 'https://github.com/facebookresearch/flores',
        'archive_sha256': sha256(archive), 'data_sha256': sha256(args.output),
        'source': 'zho_simpl', 'targets': {k: v[0] for k, v in LANGS.items()},
        'sentences_per_language': 1012, 'total_pairs': 8096,
        'limitations': [
            'Public written-text benchmark; not Chinese speech or business audio.',
            'Chinese is translated source text, not a corpus of Chinese-original utterances.',
            'NiuTrans publicly lists FLORES-200 dev in SFT; dev is excluded here.',
            'No guarantee against undisclosed training or model-selection exposure.',
            'BLEU/chrF are reference similarity metrics, not percent-correct accuracy.',
        ],
    }
    save_json(args.output.with_suffix('.manifest.json'), manifest)
    emit('prepared', data=str(args.output), **manifest)


def prompt_for(model, text, language):
    _, english, chinese = LANGS[language]
    if model == 'index':
        return f'请将以下文本翻译为{chinese}，直接输出翻译结果，不要进行任何解释。\n\n{text}'
    if model == 'hy':
        return f'将以下文本翻译为{chinese}，注意只需要输出翻译后的结果，不要额外解释：\n{text}'
    return f'Translate the following text from Chinese into {english}:\nChinese: {text}\n{english}:'


def select_rows(rows, languages, limit, seed):
    by_lang = {lang: {str(r['id']): r for r in rows if r['language'] == lang} for lang in languages}
    for lang, group in by_lang.items():
        if not group:
            raise ValueError(f'No data for {lang}')
    # Same Chinese source IDs in each language, and exactly identical sources.
    ids = set(next(iter(by_lang.values())))
    if any(set(group) != ids for group in by_lang.values()):
        raise ValueError('All selected language directions must contain identical source IDs')
    ordered = sorted(ids)
    for ident in ordered:
        if len({group[ident]['source'] for group in by_lang.values()}) != 1:
            raise ValueError(f'Source mismatch at {ident}')
    chosen = sorted(random.Random(seed).sample(ordered, min(limit, len(ordered)))) if limit else ordered
    return [by_lang[lang][ident] for lang in languages for ident in chosen]


def gpu_snapshot():
    try:
        return subprocess.check_output([
            'nvidia-smi', '--query-gpu=index,uuid,name,memory.used,memory.total,utilization.gpu',
            '--format=csv,noheader'], text=True, timeout=10).strip()
    except (OSError, subprocess.SubprocessError) as exc:
        return str(exc)


def run(args):
    if args.model == 'all':
        # Each model gets a new process, releasing every CUDA allocation on exit.
        for key in MODELS:
            validate_checkpoint(args.model_root / MODELS[key])
        for key in MODELS:
            forwarded = ['run', '--model', key, '--model-root', str(args.model_root),
                         '--data', str(args.data), '--output', str(args.output),
                         '--gpu', str(args.gpu), '--languages', args.languages,
                         '--limit', str(args.limit), '--seed', str(args.seed),
                         '--max-new-tokens', str(args.max_new_tokens), '--warmup', str(args.warmup),
                         '--cpu-threads', str(args.cpu_threads), '--batch-size', str(args.batch_size)]
            if args.resume:
                forwarded.append('--resume')
            subprocess.run([sys.executable, str(Path(__file__).resolve()), *forwarded], check=True)
        return
    import torch
    import transformers
    from transformers import AutoTokenizer, AutoModelForCausalLM, GenerationConfig
    from transformers.generation.streamers import BaseStreamer
    if not torch.cuda.is_available():
        raise RuntimeError('CUDA required for this benchmark; no silent CPU fallback')
    torch.cuda.set_device(args.gpu)
    if not torch.cuda.is_bf16_supported():
        raise RuntimeError('Selected GPU does not support BF16')
    torch.set_num_threads(args.cpu_threads)
    torch.manual_seed(args.seed)
    device = f'cuda:{args.gpu}'
    model_path = args.model_root / MODELS[args.model]
    validate_checkpoint(model_path)
    rows = select_rows(read_jsonl(args.data), args.languages.split(','), args.limit, args.seed)
    out = args.output / args.model
    out.mkdir(parents=True, exist_ok=True)
    protocol = {
        'protocol': PROTOCOL, 'model': args.model, 'path': str(model_path.resolve()),
        'script_sha256': sha256(Path(__file__)),
        'checkpoint_manifest_sha256': sha256(model_path / 'DOWNLOAD_MANIFEST.json'),
        'data_sha256': sha256(args.data), 'languages': args.languages.split(','),
        'case_ids': [(r['language'], r['id']) for r in rows],
        'precision': 'bfloat16', 'device': device, 'batch_size': args.batch_size,
        'max_new_tokens': args.max_new_tokens, 'seed': args.seed,
        'decode': 'greedy: do_sample=False, num_beams=1, use_cache=True, repetition_penalty=1',
        'attention': 'sdpa', 'prompt_example': prompt_for(args.model, rows[0]['source'], rows[0]['language']),
        'thinking': 'disabled explicitly for Index; checkpoint default for HY/LMT',
        'cpu_threads': args.cpu_threads, 'warmup_per_language': args.warmup,
        'torch': torch.__version__, 'transformers': transformers.__version__,
        'gpu': torch.cuda.get_device_properties(args.gpu).name,
    }
    fingerprint = hashlib.sha256(json.dumps(protocol, sort_keys=True).encode()).hexdigest()
    meta_path, predictions_path = out / 'metadata.json', out / 'predictions.jsonl'
    previous = []
    if predictions_path.exists():
        if not args.resume:
            raise FileExistsError(f'{predictions_path} exists; use a new output directory or --resume')
        old = json.loads(meta_path.read_text())
        if old['fingerprint'] != fingerprint:
            raise ValueError('Resume protocol mismatch; use a new output directory')
        previous = read_jsonl(predictions_path)
    session = uuid.uuid4().hex
    metadata = {'fingerprint': fingerprint, 'protocol': protocol,
                'created_at': datetime.now(timezone.utc).isoformat(),
                'session_id': session, 'gpu_snapshot_start': gpu_snapshot(),
                'resume_previous_rows': len(previous), 'status': 'loading'}
    save_json(meta_path, metadata)
    tokenizer = AutoTokenizer.from_pretrained(model_path, local_files_only=True)
    tokenizer.padding_side = 'left'
    loader = AutoModelForCausalLM
    if args.model == 'index':
        from transformers import AutoModelForImageTextToText
        loader = AutoModelForImageTextToText
    start = time.perf_counter()
    model = loader.from_pretrained(model_path, local_files_only=True, dtype=torch.bfloat16,
                                   attn_implementation='sdpa', device_map={'': device}).eval()
    torch.cuda.synchronize()
    metadata['load_seconds'] = time.perf_counter() - start
    metadata['parameter_count'] = sum(p.numel() for p in model.parameters())
    metadata['parameter_bytes'] = sum(p.numel() * p.element_size() for p in model.parameters())
    eos = model.generation_config.eos_token_id
    if eos is None:
        eos = getattr(model.config, 'text_config', model.config).eos_token_id
    if eos is None:
        eos = tokenizer.eos_token_id
    eos_ids = {eos} if isinstance(eos, int) else set(eos)
    pad = tokenizer.pad_token_id
    if pad is None:
        pad = min(eos_ids)
    decode_config = GenerationConfig(do_sample=False, num_beams=1, use_cache=True,
        temperature=1.0, top_p=1.0, top_k=50,
        repetition_penalty=1.0, max_new_tokens=args.max_new_tokens,
        eos_token_id=sorted(eos_ids), pad_token_id=pad, bos_token_id=tokenizer.bos_token_id)
    metadata['generation_config'] = decode_config.to_dict()
    if args.model == 'index':
        from transformers.models.qwen3_5 import modeling_qwen3_5
        metadata['index_fast_linear_attention_available'] = getattr(modeling_qwen3_5, 'is_fast_path_available', None)

    class TokenClock(BaseStreamer):
        def __init__(self):
            self.skip_prompt = True
            self.times = []

        def put(self, value):
            if self.skip_prompt:
                self.skip_prompt = False
                return
            # One callback contains one decode step for every live row.
            self.times.append(time.perf_counter())

        def end(self):
            pass

    def translate_batch(texts, language, batch_id):
        torch.cuda.synchronize()
        started = time.perf_counter()
        kwargs = {'enable_thinking': False} if args.model == 'index' else {}
        rendered = [tokenizer.apply_chat_template(
            [{'role': 'user', 'content': prompt_for(args.model, text, language)}],
            tokenize=False, add_generation_prompt=True, **kwargs) for text in texts]
        inputs = tokenizer(rendered, padding=True, return_tensors='pt', add_special_tokens=False).to(device)
        input_tokens = inputs.input_ids.shape[1]
        text_config = getattr(model.config, 'text_config', model.config)
        real_lengths = inputs.attention_mask.sum(dim=1).tolist()
        if any(length + args.max_new_tokens > text_config.max_position_embeddings for length in real_lengths):
            raise ValueError('Context overflow; no silent source truncation')
        clock = TokenClock()
        torch.cuda.synchronize()
        generate_started = time.perf_counter()
        with torch.inference_mode():
            generated = model.generate(**inputs, generation_config=decode_config, streamer=clock)
        torch.cuda.synchronize()
        generate_seconds = time.perf_counter() - generate_started
        decode_seconds = (clock.times[-1] - clock.times[0]) if len(clock.times) > 1 else None
        results = []
        for text, sequence, input_length in zip(texts, generated, real_lengths):
            body = sequence[input_tokens:].tolist()
            # Batched decoding pads rows that hit EOS early. Keep through the
            # first end token so output text and token counts do not include pads.
            stop = next((i for i, token in enumerate(body) if token in eos_ids), None)
            if stop is not None:
                body = body[:stop + 1]
            ended = bool(body and body[-1] in eos_ids)
            raw = tokenizer.decode(body, skip_special_tokens=True, clean_up_tokenization_spaces=False)
            hypothesis = raw.strip()
            if args.model == 'index' and '</think>' in hypothesis:
                hypothesis = hypothesis.split('</think>', 1)[1].strip()
            results.append({
                'hypothesis': hypothesis, 'raw_hypothesis': raw,
                'input_tokens': input_length, 'generated_tokens_including_eos': len(body),
                'content_tokens': sum(t not in tokenizer.all_special_ids for t in body),
                'batch_id': batch_id, 'batch_size': len(texts),
                'generation_seconds': generate_seconds,
                'batch_ttft_seconds': clock.times[0] - started if clock.times else None,
                'decode_seconds_after_first_token': decode_seconds,
                'decode_steps': len(clock.times),
                'eos': ended, 'truncated': len(body) >= args.max_new_tokens and not ended,
                'session_id': session, 'error': None,
            })
        batch_latency = time.perf_counter() - started
        for result in results:
            result['batch_latency_seconds'] = batch_latency
            result['per_item_amortized_latency_seconds'] = batch_latency / len(texts)
        return results

    done = {(r['language'], str(r['id'])) for r in previous}
    selected_keys = {(r['language'], str(r['id'])) for r in rows}
    if not done.issubset(selected_keys):
        raise ValueError('Resume output contains unexpected case IDs')
    metadata['status'] = 'running'
    save_json(meta_path, metadata)
    for lang in args.languages.split(','):
        emit('warmup', model=args.model, language=lang, iterations=args.warmup)
        for _ in range(args.warmup):
            translate_batch(['请确认这批货物的数量和交货日期。'] * args.batch_size, lang, 'warmup')
    torch.cuda.reset_peak_memory_stats()
    emit('model_ready', model=args.model, cases=len(rows), **{'load_seconds': metadata['load_seconds']})
    completed = len(done)
    with predictions_path.open('a' if args.resume else 'w') as output:
        for language in args.languages.split(','):
            language_rows = [r for r in rows if r['language'] == language]
            for start in range(0, len(language_rows), args.batch_size):
                batch = language_rows[start:start + args.batch_size]
                pending = [r for r in batch if (r['language'], str(r['id'])) not in done]
                if not pending:
                    continue
                batch_id = f'{session}:{language}:{start}'
                results = translate_batch([r['source'] for r in pending], language, batch_id)
                for row, result in zip(pending, results):
                    output.write(json.dumps({**row, **result}, ensure_ascii=False) + '\n')
                output.flush()
                completed += len(pending)
                if start == 0 or start % (args.batch_size * 10) == 0 or start + len(batch) == len(language_rows):
                    emit('progress', model=args.model, language=language,
                         language_done=start + len(batch), language_total=len(language_rows),
                         done=completed, total=len(rows))
    metadata.update(status='complete', gpu_snapshot_end=gpu_snapshot(),
                    peak_allocated_bytes=torch.cuda.max_memory_allocated(),
                    peak_reserved_bytes=torch.cuda.max_memory_reserved())
    save_json(meta_path, metadata)
    emit('model_complete', model=args.model, output=str(out))


def validate_checkpoint(path):
    for name in ('DOWNLOAD_MANIFEST.json', 'config.json', 'tokenizer_config.json', 'model.safetensors'):
        if not (path / name).is_file():
            raise FileNotFoundError(f'Checkpoint incomplete: {path / name}; wait for the download to finish')
    manifest = json.loads((path / 'DOWNLOAD_MANIFEST.json').read_text())
    weights = next(f for f in manifest['files'] if f['file'] == 'model.safetensors')
    if (path / 'model.safetensors').stat().st_size != weights.get('bytes', weights.get('size_bytes')):
        raise ValueError(f'Weight size differs from download manifest: {path}')


def distribution(values):
    values = sorted(v for v in values if v is not None)
    if not values:
        return None
    def percentile(p):
        rank = (len(values) - 1) * p
        low = int(rank)
        return values[low] + (values[min(low + 1, len(values)-1)] - values[low]) * (rank-low)
    return {'mean': sum(values)/len(values), 'p50': percentile(.5), 'p95': percentile(.95)}


def score(args):
    import sacrebleu
    from sacrebleu.metrics import BLEU, CHRF
    rows = read_jsonl(args.predictions)
    by_language = defaultdict(list)
    for row in rows:
        by_language[row['language']].append(row)
    languages = {}
    for language, group in by_language.items():
        hypotheses = [r['hypothesis'] for r in group]
        references = [r['reference'] for r in group]
        bleu, chrf = BLEU(tokenize=args.bleu_tokenizer), CHRF(word_order=2)
        batches = {}
        for row in group:
            batches.setdefault(row.get('batch_id', f"single:{row['id']}"), row)
        batch_rows = list(batches.values())
        total_latency = sum(r.get('batch_latency_seconds', r.get('latency_seconds', 0)) for r in batch_rows)
        decode_seconds = sum(r.get('decode_seconds_after_first_token') or 0 for r in batch_rows)
        decode_tokens = sum(max(r['generated_tokens_including_eos']-1, 0) for r in group
                            if r.get('decode_seconds_after_first_token') is not None)
        languages[language] = {
            'count': len(group), 'spBLEU' if args.bleu_tokenizer == 'flores200' else 'BLEU':
                bleu.corpus_score(hypotheses, [references]).score,
            'chrFpp': chrf.corpus_score(hypotheses, [references]).score,
            'bleu_signature': str(bleu.get_signature()), 'chrf_signature': str(chrf.get_signature()),
            'batch_count': len(batch_rows),
            'max_batch_size': max(r.get('batch_size', 1) for r in group),
            'batch_latency_seconds': distribution([r.get('batch_latency_seconds', r.get('latency_seconds')) for r in batch_rows]),
            'per_item_amortized_latency_seconds': distribution([
                r.get('batch_latency_seconds', r.get('latency_seconds')) / r.get('batch_size', 1)
                for r in batch_rows]),
            'batch_ttft_seconds': distribution([r.get('batch_ttft_seconds', r.get('ttft_seconds')) for r in batch_rows]),
            'sentences_per_second': len(group) / total_latency,
            'output_characters_per_second': sum(len(r['hypothesis']) for r in group)/total_latency,
            'generated_tokens_per_second_e2e': sum(r['generated_tokens_including_eos'] for r in group)/total_latency,
            'decode_tokens_per_second': decode_tokens / decode_seconds if decode_seconds else None,
            'empty_outputs': sum(not r['hypothesis'].strip() for r in group),
            'truncated_outputs': sum(r['truncated'] for r in group),
            'session_count': len({r['session_id'] for r in group}),
        }
    run_meta_path = args.predictions.parent / 'metadata.json'
    run_meta = json.loads(run_meta_path.read_text()) if run_meta_path.exists() else {}
    summary = {'sacrebleu': sacrebleu.__version__, 'predictions_sha256': sha256(args.predictions),
               'languages': languages, 'total_cases': len(rows),
               'peak_allocated_bytes': run_meta.get('peak_allocated_bytes'),
               'peak_reserved_bytes': run_meta.get('peak_reserved_bytes'),
               'macro_chrFpp': sum(v['chrFpp'] for v in languages.values())/len(languages),
               'note': 'Quality is reference similarity, not percent correct. Compare languages separately; tokenizers differ across models.'}
    save_json(args.predictions.parent / 'summary.json', summary)
    emit('scored', output=str(args.predictions.parent / 'summary.json'), **summary)


def report(args):
    results = []
    control = None
    metric_control = None
    for name in args.models.split(','):
        folder = args.output / name
        metadata = json.loads((folder / 'metadata.json').read_text())
        if metadata['status'] != 'complete':
            raise ValueError(f'{name} is not complete')
        summary = json.loads((folder / 'summary.json').read_text())
        if summary['predictions_sha256'] != sha256(folder / 'predictions.jsonl'):
            raise ValueError(f'Stale summary for {name}; run score again')
        p = metadata['protocol']
        predictions = read_jsonl(folder / 'predictions.jsonl')
        actual_ids = {(r['language'], str(r['id'])) for r in predictions}
        expected_ids = {(language, str(ident)) for language, ident in p['case_ids']}
        if actual_ids != expected_ids:
            raise ValueError(f'Incomplete or unexpected predictions for {name}')
        settings = {k: p[k] for k in ('protocol', 'data_sha256', 'case_ids', 'precision',
                    'batch_size', 'max_new_tokens', 'decode', 'attention', 'torch',
                    'transformers', 'gpu', 'device', 'seed', 'cpu_threads', 'warmup_per_language',
                    'script_sha256')}
        if control is not None and settings != control:
            raise ValueError('Comparison protocol/case IDs differ; cannot build a fair comparison')
        control = settings
        signatures = {lang: (v['bleu_signature'], v['chrf_signature'])
                      for lang, v in summary['languages'].items()}
        if metric_control is not None and metric_control != signatures:
            raise ValueError('Metric signatures differ; recompute scores with identical settings')
        metric_control = signatures
        results.append({'model': name, 'metadata': metadata, 'summary': summary})
    lines = ['# 中文→外文翻译对比', '',
             'spBLEU、chrF++ 是参考译文相似度，不是准确率百分比。速度仅代表本机当前推理栈。', '',
             '| 模型 | 目标语言 | 样本 | 最大批量 | chrF++ ↑ | BLEU ↑ | 批首 token p50 秒 ↓ | 整批 p50 秒 ↓ | 整批 p95 秒 ↓ | 摊销秒/条 ↓ | 样本/秒 ↑ | 字符/秒 ↑ | token/秒 ↑ | 峰值显存 allocated/reserved GiB | 截断 |',
             '|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|']
    for result in results:
        for lang, value in result['summary']['languages'].items():
            ttft = value['batch_ttft_seconds']
            mem = result['summary'].get('peak_allocated_bytes')
            reserved = result['summary'].get('peak_reserved_bytes')
            lines.append(f"| {result['model']} | zh→{lang} | {value['count']} | {value['max_batch_size']} | "
                f"{value['chrFpp']:.2f} | {value.get('spBLEU', value.get('BLEU')):.2f} | "
                f"{ttft['p50'] if ttft else float('nan'):.3f} | "
                f"{value['batch_latency_seconds']['p50']:.3f} | {value['batch_latency_seconds']['p95']:.3f} | "
                f"{value['per_item_amortized_latency_seconds']['p50']:.3f} | "
                f"{value['sentences_per_second']:.2f} | {value['output_characters_per_second']:.2f} | "
                f"{value['generated_tokens_per_second_e2e']:.2f} | "
                f"{mem/1024**3 if mem is not None else float('nan'):.2f}/"
                f"{reserved/1024**3 if reserved is not None else float('nan'):.2f} | {value['truncated_outputs']} |")
    lines.extend(['', '显存列依次为 PyTorch 峰值 allocated/reserved。reserved 包含 PyTorch 缓存池，表示进程可能占用的显存空间，不等于活跃张量本身。',
                  '批首 token 与整批延迟包含提示词构造、分词及 GPU 传输，不包含模型加载、预热和质量指标计算。摊销秒/条等于整批时延除以批量，不代表单句响应延迟。',
                  '各模型 tokenizer 不同，token/秒不宜直接排名；结合样本/秒和字符吞吐比较。',
                  '断点续跑可能跨会话；共享 GPU 的其他进程、温度及 kernel 优化状态会影响速度。',
                  '公开语料可能被模型训练或模型选择接触；这不是未公开的独立测试集。'])
    for result in results:
        if result['metadata'].get('index_fast_linear_attention_available') is False:
            lines.append('Index 的线性注意力正在使用 PyTorch fallback；速度不能代表优化 kernel 的部署性能。')
    save_json(args.output / 'comparison.json', results)
    (args.output / 'comparison.md').write_text('\n'.join(lines) + '\n')
    emit('report_complete', report=str(args.output / 'comparison.md'))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest='command', required=True)
    prep = sub.add_parser('prepare', help='Prepare aligned Chinese-source FLORES-101 devtest')
    prep.add_argument('--archive', type=Path, default=ROOT/'data/eval/flores101/flores101_dataset.tar.gz')
    prep.add_argument('--output', type=Path, default=DEFAULT_DATA)
    bench = sub.add_parser('run', help='Benchmark local BF16 checkpoints with batched inference')
    bench.add_argument('--model', choices=['all', *MODELS], required=True)
    bench.add_argument('--model-root', type=Path, default=MODEL_ROOT)
    bench.add_argument('--data', type=Path, default=DEFAULT_DATA)
    bench.add_argument('--output', type=Path, required=True)
    bench.add_argument('--gpu', type=int, default=1)
    bench.add_argument('--languages', default=','.join(LANGS))
    bench.add_argument('--limit', type=int, default=100, help='Aligned source IDs per language; 0=all')
    bench.add_argument('--seed', type=int, default=20261020)
    bench.add_argument('--max-new-tokens', type=int, default=256)
    bench.add_argument('--warmup', type=int, default=2, help='Unscored synthetic warmups per language')
    bench.add_argument('--cpu-threads', type=int, default=2)
    bench.add_argument('--batch-size', type=int, default=16)
    bench.add_argument('--resume', action='store_true')
    metric = sub.add_parser('score', help='Compute quality and speed from saved translations')
    metric.add_argument('--predictions', type=Path, required=True)
    metric.add_argument('--bleu-tokenizer', choices=['flores200', '13a', 'none'], default='flores200')
    comparison = sub.add_parser('report', help='Compare completed, identically configured runs')
    comparison.add_argument('--output', type=Path, required=True)
    comparison.add_argument('--models', default='index,hy,lmt')
    args = parser.parse_args()
    if args.command == 'run':
        languages = args.languages.split(',')
        if not languages or len(set(languages)) != len(languages) or any(l not in LANGS for l in languages):
            parser.error('Choose unique target codes from: ' + ','.join(LANGS))
        if args.limit < 0 or args.max_new_tokens < 1 or args.warmup < 0 or args.cpu_threads < 1 or args.batch_size < 1:
            parser.error('Invalid numeric option')
    {'prepare': prepare, 'run': run, 'score': score, 'report': report}[args.command](args)


if __name__ == '__main__':
    main()
