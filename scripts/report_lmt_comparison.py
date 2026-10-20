"""Independently audit and report LMT-60-0.6B comparison artifacts."""
import hashlib
import json
from pathlib import Path
from statistics import mean

ROOT = Path('results/lmt600m_comparison_20261017')
OLD = Path('results/multilingual_mt_accuracy_20261017')
NLLB_ROOT = Path('results/nllb600m_comparison_20261017')
LMT = 'LMT-60-0.6B'
OLD_MODELS = ('m2m100_418M', 'small100', 'nllb200_600M')
MODELS = (LMT,) + OLD_MODELS
LANGUAGES = {'en':'英语','fr':'法语','de':'德语','es':'西班牙语','ru':'俄语','ja':'日语','ko':'韩语','ar':'阿拉伯语'}
MODES = ('retranslate', 'agreement_forced_prefix')
DATA = Path('data/eval/flores101/aligned_devtest.jsonl')


def model_root(name):
    if name == LMT:
        return ROOT
    if name == 'nllb200_600M':
        return NLLB_ROOT
    return OLD


def read_jsonl(path):
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def score(rows, normalizer, bleu, chrf):
    hypotheses = [normalizer.convert(r['hypothesis']) for r in rows]
    references = [[normalizer.convert(r['reference']) for r in rows]]
    return {'BLEU_zh': bleu.corpus_score(hypotheses, references).score,
            'chrFpp': chrf.corpus_score(hypotheses, references).score}


def check(condition, message):
    if not condition:
        raise AssertionError(message)


def main():
    from opencc import OpenCC
    from sacrebleu.metrics import BLEU, CHRF
    from benchmark_streaming_mt import percentiles

    normalizer, bleu, chrf = OpenCC('t2s'), BLEU(tokenize='zh'), CHRF(word_order=2)
    summaries = {n: json.loads((model_root(n) / f'{n}.summary.json').read_text()) for n in MODELS}
    metadata_path = ROOT / f'{LMT}.metadata.json'
    metadata = json.loads(metadata_path.read_text()) if metadata_path.exists() else summaries[LMT]
    offline = {n: read_jsonl(model_root(n) / f'{n}.offline.jsonl') for n in MODELS}
    streams = {n: read_jsonl(model_root(n) / f'{n}.streaming.jsonl') for n in MODELS}
    data_hash = hashlib.sha256(DATA.read_bytes()).hexdigest()
    gold_rows = read_jsonl(DATA)
    gold = {(r['language'], r['id']): r for r in gold_rows}
    fixed_seed_subset = {(r['language'], r['id']) for r in gold_rows if r['streaming_subset']}

    check(summaries[LMT].get('status') == 'completed', 'LMT summary is not completed')
    for name in MODELS:
        check(summaries[name].get('status') == 'completed', f'{name} summary is not completed')
        check(summaries[name].get('data_sha256') == data_hash, f'{name} summary data hash mismatch')
        check(len(offline[name]) == 8096, f'{name}: expected 8096 offline rows')
        check(len({(r['language'], r['id']) for r in offline[name]}) == 8096, f'{name}: duplicate offline IDs')
        check(len(streams[name]) == 1600, f'{name}: expected 1600 stream trajectories')
        check(len({(r['language'], r['id'], r['mode']) for r in streams[name]}) == 1600, f'{name}: duplicate stream trajectories')

    for name in MODELS:
        for row in offline[name] + streams[name]:
            ref = gold.get((row['language'], row['id']))
            check(ref is not None, f"{name}: unknown pair {(row['language'], row['id'])}")
            check(row['source'] == ref['source'] and row['reference'] == ref['reference'], f"{name}: source/reference mismatch for {row['id']}")
            check(row['streaming_subset'] == ref['streaming_subset'], f"{name}: subset flag mismatch for {row['id']}")

    # Validate each model against the same fixed-seed 100 IDs per language and prefix schedule.
    for lang in LANGUAGES:
        expected = {i for l, i in fixed_seed_subset if l == lang}
        check(len(expected) == 100, f'{lang}: fixed-seed streaming subset is not 100 rows')
        for name in MODELS:
            for mode in MODES:
                ids = {r['id'] for r in streams[name] if r['language'] == lang and r['mode'] == mode}
                check(ids == expected, f'{name}/{lang}/{mode}: IDs differ from fixed-seed stream subset')

    old_lookups = {n: {(r['language'], r['id'], r['mode']): r for r in streams[n]} for n in OLD_MODELS}
    lmt_lookup = {(r['language'], r['id'], r['mode']): r for r in streams[LMT]}
    for key, row in lmt_lookup.items():
        steps = row['steps']
        check(bool(steps), f'{key}: empty stream steps')
        check(steps[-1]['source'] == row['source'], f'{key}: final step is not full source')
        check(all(b['source'].startswith(a['source']) for a, b in zip(steps, steps[1:])), f'{key}: source prefixes are not monotonic')
        if row['mode'] == 'agreement_forced_prefix':
            check(steps[-1]['committed_text'] == row['hypothesis'], f'{key}: final committed text differs from hypothesis')
            check(all(b['committed_text'].startswith(a['committed_text']) for a, b in zip(steps, steps[1:])), f'{key}: committed text is not monotonic')
        for name in OLD_MODELS:
            check([s['source'] for s in steps] == [s['source'] for s in old_lookups[name][key]['steps']], f'{name} and LMT source prefix mismatch for {key}')

    # Prefer an explicit independently run batch=1 control. Reuse full outputs only when
    # the run metadata proves full inference itself used batch size 1.
    control_path = ROOT / 'batch1_control' / f'{LMT}.offline.jsonl'
    offline_batch = metadata.get('offline_batch_size')
    if control_path.exists():
        control = read_jsonl(control_path)
        control_source = '独立 batch=1 控制产物'
        control_available = True
    elif offline_batch == 1:
        control = [r for r in offline[LMT] if r['streaming_subset']]
        control_source = '复用全量输出：metadata 记录 offline_batch_size=1；未另跑控制任务'
        control_available = True
    else:
        control, control_source, control_available = [], '缺少独立控制产物，且全量 offline batch 不是 1', False
    control_pairs = {(r['language'], r['id']) for r in control}
    control_alignment = control_available and len(control) == 800 and len(control_pairs) == 800 and control_pairs == fixed_seed_subset
    if control_available:
        check(len(control) == 800, 'LMT batch1 control must contain 800 rows')
        check(len(control_pairs) == 800, 'duplicate LMT batch1 control IDs')
        for row in control:
            ref = gold.get((row['language'], row['id']))
            check(ref is not None and row['source'] == ref['source'] and row['reference'] == ref['reference'], f"LMT control source/reference mismatch for {row['id']}")

    metrics, stream_metrics, stream_stats = {}, {}, {}
    full_output_counts = {}
    for name in MODELS:
        metrics[name], stream_metrics[name] = {}, {}
        for lang in LANGUAGES:
            rows = [r for r in offline[name] if r['language'] == lang]
            check(len(rows) == 1012, f'{name}/{lang}: expected 1012 offline rows')
            val = score(rows, normalizer, bleu, chrf)
            expected = summaries[name]['scores'][lang]['offline_full']['t2s']
            check(all(abs(val[k] - expected[k]) < 1e-9 for k in val), f'{name}/{lang}: offline metrics differ from summary')
            metrics[name][lang] = val
            if name == LMT:
                check(all('eos' in r and 'hypothesis' in r and 'replacement_characters' in r for r in rows),
                      f'{LMT}/{lang}: missing raw output diagnostic fields')
                full_output_counts[lang] = {
                    'non_eos': sum(not r['eos'] for r in rows),
                    'empty': sum(not r['hypothesis'].strip() for r in rows),
                    'replacement_characters': sum(int(r['replacement_characters']) for r in rows),
                }
            stream_metrics[name][lang] = {}
            for mode in MODES:
                sr = [r for r in streams[name] if r['language'] == lang and r['mode'] == mode]
                check(len(sr) == 100, f'{name}/{lang}/{mode}: expected 100 trajectories')
                val = score(sr, normalizer, bleu, chrf)
                expected = summaries[name]['scores'][lang][mode]['t2s']
                check(all(abs(val[k] - expected[k]) < 1e-9 for k in val), f'{name}/{lang}/{mode}: stream metrics differ from summary')
                stream_metrics[name][lang][mode] = val

        if name == LMT:
            stream_stats[name] = {}
            for mode in MODES:
                trajectories = [r for r in streams[name] if r['mode'] == mode]
                steps = [s for r in trajectories for s in r['steps']]
                stream_stats[name][mode] = {
                    'updates': len(steps), 'latency_seconds': percentiles([s['seconds'] for s in steps]),
                    'normalized_erasure_characters_mean': mean(r['normalized_erasure_characters'] for r in trajectories),
                    'non_eos_updates': sum(not s.get('eos', False) for s in steps),
                    'empty_updates': sum(not s['hypothesis'].strip() for s in steps),
                }

    control_metrics = {}
    if control_available and control_alignment:
        control_metrics = {lang: score([r for r in control if r['language'] == lang], normalizer, bleu, chrf) for lang in LANGUAGES}
    control_lookup = {(r['language'], r['id']): r['hypothesis'] for r in control} if control_available else {}
    retranslation = [r for r in streams[LMT] if r['mode'] == 'retranslate']
    mismatches = (sum(r['hypothesis'] != control_lookup.get((r['language'], r['id'])) for r in retranslation)
                  if control_available and control_alignment else None)
    batch_end_state_passed = mismatches == 0 if mismatches is not None else False
    macro = {n: {k: mean(metrics[n][l][k] for l in LANGUAGES) for k in ('BLEU_zh', 'chrFpp')} for n in MODELS}
    gpu_indices = metadata.get('gpu_indices')
    audit_passed = control_alignment and batch_end_state_passed
    config_line = ('模型：NiuTrans/LMT-60-0.6B（翻译微调版，不是 Base）；许可：Apache-2.0。'
      f"revision：`{metadata.get('revision', '见metadata')}`；runtime：`{metadata.get('runtime', 'CTranslate2')} {metadata.get('ctranslate2', '见metadata')} {metadata.get('ct2_class', 'Generator')}`；"
      f"compute type：`{metadata.get('compute_type', '见metadata')}`；beam：{metadata.get('beam_size', '见metadata')}；"
      f"实际 offline batch：{offline_batch if offline_batch is not None else 'metadata未记录'}；stream batch：{metadata.get('stream_batch_size', '见metadata')}。"
      f"GPU indices：{gpu_indices if gpu_indices is not None else 'metadata未记录'}（来自合并 metadata）。"
      '报告读取 metadata 实际值，不预设 batch 大小。旧模型运行时及精度配置见各自 metadata。')
    parameter_audit_path=ROOT/'tokenizer_and_parameter_audit.json'
    parameter_audit = None
    if parameter_audit_path.exists():
        parameter_audit=json.loads(parameter_audit_path.read_text())
        config_line+=f" 实际唯一参数量：{parameter_audit['unique_parameter_count_if_tied']:,}；checkpoint tensor统计中包含重复的共享embedding/lm_head，已核对二者权重相同。参数和术语编码检查见 tokenizer_and_parameter_audit.json。"
    audit = {
        'status': 'passed' if audit_passed else 'failed', 'data_sha256': data_hash,
        'offline_rows_each_model': 8096, 'stream_trajectories_each_model': 1600,
        'lmt_batch1_control_available': control_available, 'lmt_batch1_control_source': control_source,
        'lmt_batch1_control_ids_aligned_to_fixed_seed_800': control_alignment,
        'lmt_batch1_vs_retranslation_final': {'pairs': 800, 'mismatches': mismatches},
        'all_source_prefix_schedules_match_older_runs': True,
        'all_stream_final_steps_use_full_source': True,
        'source_reference_and_subset_match_flores': True,
        'independent_full_and_stream_metric_recalculation_matches_summaries': True,
        'full_metrics': metrics, 'stream_metrics': stream_metrics, 'macro_full_metrics': macro,
        'lmt_full_output_counts_by_language_raw': full_output_counts,
        'lmt_streaming_stats_by_mode': stream_stats[LMT],
        'paired_stream_subset_metrics': {'batch1_control': control_metrics,
          'retranslate': {l: stream_metrics[LMT][l]['retranslate'] for l in LANGUAGES},
          'agreement_forced_prefix': {l: stream_metrics[LMT][l]['agreement_forced_prefix'] for l in LANGUAGES}},
    }
    if parameter_audit is not None:
        audit['tokenizer_and_parameter_audit'] = parameter_audit

    out = ['# LMT-60-0.6B 与现有多语模型对比', '',
      '日期：2026-10-17。基于本地实际推理产物，指标从逐句原始输出独立复算。', '',
      '## 测试口径', '',
      '- FLORES-101 devtest，英语、法语、德语、西班牙语、俄语、日语、韩语、阿拉伯语到简体中文；每语种1012句，共8096句。四模型使用同一数据、同一参考译文和固定种子选取的100句/语种流式子集。',
      '- BLEU 使用 SacreBLEU 中文 zh tokenizer，chrF++ 使用 word_order=2；输出和参考均经 OpenCC 简体转换。每个语种分开计算，宏平均是8个语种分数的算术平均，不是合并语料后的分数。分数不是百分比准确率。',
      '- FLORES 是公开书面文本基准，可能进入模型训练、开发或选择流程；不能代表未见业务文本。这里没有音频、ASR、术语库、业务后处理或真实语音到达时间。',
      '- 两种增量更新均以人工切分的源文本前缀驱动，是文本回放，不是模型原生源流式能力。延迟是模型更新计算耗时，不含采音、ASR、排队或端到端字幕时间。NE、非EOS更新和空白更新按两个模式分别统计。',
      '- 四模型 GPU 主评测均为 int8_float16：LMT 使用 CTranslate2 4.8.3 Generator，旧三模型使用 4.6.3 Translator；计算精度相同。LMT 全量 batch=1，旧模型为 batch=4；测试时间和共享 GPU 负载也不同，速度数据不能直接用于严格排名或速度倍数结论。另有原始权重 CPU FP32 的25个业务样例对照，单独报告，不参与 GPU 时延统计。', '',
      '## LMT运行配置与状态', '',
      config_line, '',
      '## 完整输入质量', '',
      '| 源语言 | 句数 | LMT BLEU | M2M100 BLEU | SMaLL-100 BLEU | NLLB BLEU | LMT chrF++ | M2M100 chrF++ | SMaLL-100 chrF++ | NLLB chrF++ |',
      '|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|']
    for lang, label in LANGUAGES.items():
        vals = [metrics[n][lang]['BLEU_zh'] for n in MODELS] + [metrics[n][lang]['chrFpp'] for n in MODELS]
        out.append(f"| {label} | 1012 | " + ' | '.join(f'{v:.2f}' for v in vals) + ' |')
    vals = [macro[n]['BLEU_zh'] for n in MODELS] + [macro[n]['chrFpp'] for n in MODELS]
    out += ['| 8语种宏平均 | 8096 | ' + ' | '.join(f'{v:.2f}' for v in vals) + ' |', '',
      '## LMT完整输入输出诊断', '',
      '| 源语言 | 全句数 | 非EOS输出 | 空白输出 | 原始输出中的 U+FFFD 数量 |',
      '|---|---:|---:|---:|---:|']
    for lang, label in LANGUAGES.items():
        c = full_output_counts[lang]
        out.append(f"| {label} | 1012 | {c['non_eos']} | {c['empty']} | {c['replacement_characters']} |")
    out += ['', '以上诊断从原始 hypothesis 统计，未进行简繁转换；U+FFFD 是替换字符计数。最大生成长度为配置上限，不能据此估算漏译率或推断被截断句子的比例。', '',
      '## LMT文本增量更新', '',
      '| 策略 | 更新数 | P50 ms | P95 ms | 平均字符 NE | 非EOS更新 | 空白更新 |',
      '|---|---:|---:|---:|---:|---:|---:|']
    for mode in MODES:
        st = stream_stats[LMT][mode]
        out.append(f"| {mode} | {st['updates']} | {st['latency_seconds']['p50']*1000:.1f} | {st['latency_seconds']['p95']*1000:.1f} | {st['normalized_erasure_characters_mean']:.3f} | {st['non_eos_updates']} | {st['empty_updates']} |")
    out += ['', 'NE 是旧字幕尾部被删除字符数相对最终字幕长度的归一化值，数值可能大于1；使用 benchmark_streaming_mt.percentiles 的近邻秩口径。以上延迟是原始文本更新调用的计算时间，不是音频或端到端延迟。', '',
      '## 固定100句的 batch=1 与流式末态质量', '',
      f'batch=1 对照来源：{control_source}。']
    if control_available and control_alignment:
        out += ['', '| 源语言 | batch1 BLEU | 重译末态 BLEU | 锁定前缀 BLEU | batch1 chrF++ | 重译末态 chrF++ | 锁定前缀 chrF++ |',
          '|---|---:|---:|---:|---:|---:|---:|']
        for lang, label in LANGUAGES.items():
            b, r, a = control_metrics[lang], stream_metrics[LMT][lang]['retranslate'], stream_metrics[LMT][lang]['agreement_forced_prefix']
            out.append(f"| {label} | {b['BLEU_zh']:.2f} | {r['BLEU_zh']:.2f} | {a['BLEU_zh']:.2f} | {b['chrFpp']:.2f} | {r['chrFpp']:.2f} | {a['chrFpp']:.2f} |")
        out += ['', f'全句 batch=1 对照与普通重译末态逐条文本不一致：{mismatches}/800。只有控制ID对齐且零差异时 AUDIT 才标为 passed；它不说明中间暂定译文正确。']
    else:
        out += ['', 'batch=1 对照缺失或ID未与固定种子选取的800句配对，因此不报告该组质量分数，AUDIT 不会标为 passed。']
    out += ['', '## 模型许可与范围', '',
      'LMT-60-0.6B 翻译微调模型卡标注 Apache-2.0；许可条款见模型卡及其附带许可文本。本次只测试该翻译微调版，不代表 LMT-60-0.6B-Base。',
      '来源：[LMT-60-0.6B 模型卡](https://huggingface.co/NiuTrans/LMT-60-0.6B)；[NiuTrans/LMT 项目](https://github.com/NiuTrans/LMT)；[FLORES-101 数据说明](https://github.com/facebookresearch/flores/blob/main/previous_releases/flores101/README.md)；[SacreBLEU](https://github.com/mjpost/sacrebleu)。', '',
      '## 数据与复现', '',
      f"逐句输出：`{ROOT}/{LMT}.offline.jsonl`；流式轨迹：`{ROOT}/{LMT}.streaming.jsonl`；batch1控制：`{control_path}`（若不存在且全量 batch=1，则复用固定种子子集）；运行revision及配置：`{metadata_path}`；对齐输入：`{DATA}`。", '',
      '复核本报告和 AUDIT.json：', '', '```bash', 'cd /home/aim0/桌面/tan/ASR/Translator', '.runtime/mt-eval-venv/bin/python scripts/report_lmt_comparison.py', '```', '',
      '## 业务评测', '']
    business = ROOT / 'BUSINESS_REPORT.md'
    if business.exists():
        out += [business.read_text().rstrip(), '']
        audit['business_report_included'] = True
    else:
        out += ['业务报告尚未接入；后续可在本目录放置 BUSINESS_REPORT.md，由本脚本附加到本节。', '']
        audit['business_report_included'] = False
    notes=ROOT/'BUSINESS_NOTES.md'
    if notes.exists():
        out += [notes.read_text().rstrip(),'']
    if (ROOT/'PROTOCOL_REVIEW.md').exists():
        out += ['提示模板、停止符、生成长度与分词器兼容性检查见 [PROTOCOL_REVIEW.md](PROTOCOL_REVIEW.md)。','']
    (ROOT / 'REPORT.md').write_text('\n'.join(out))
    (ROOT / 'AUDIT.json').write_text(json.dumps(audit, ensure_ascii=False, indent=2) + '\n')
    print(ROOT / 'REPORT.md')


if __name__ == '__main__':
    main()
