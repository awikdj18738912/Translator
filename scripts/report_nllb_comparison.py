"""Independently audit and report NLLB-200 600M comparison artifacts."""
import hashlib
import json
from pathlib import Path
from statistics import mean

ROOT = Path('results/nllb600m_comparison_20261017')
OLD = Path('results/multilingual_mt_accuracy_20261017')
NLLB = 'nllb200_600M'
OLD_MODELS = ('m2m100_418M', 'small100')
MODELS = (NLLB,) + OLD_MODELS
LANGUAGES = {'en':'英语','fr':'法语','de':'德语','es':'西班牙语','ru':'俄语','ja':'日语','ko':'韩语','ar':'阿拉伯语'}
MODES = ('retranslate', 'agreement_forced_prefix')
DATA = Path('data/eval/flores101/aligned_devtest.jsonl')


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
    summaries = {n: json.loads(((ROOT if n == NLLB else OLD) / f'{n}.summary.json').read_text()) for n in MODELS}
    metadata_path = ROOT / f'{NLLB}.metadata.json'
    metadata = json.loads(metadata_path.read_text()) if metadata_path.exists() else summaries[NLLB]
    offline = {n: read_jsonl((ROOT if n == NLLB else OLD) / f'{n}.offline.jsonl') for n in MODELS}
    streams = {n: read_jsonl((ROOT if n == NLLB else OLD) / f'{n}.streaming.jsonl') for n in MODELS}
    control = read_jsonl(ROOT / 'batch1_control' / f'{NLLB}.offline.jsonl')
    data_hash = hashlib.sha256(DATA.read_bytes()).hexdigest()
    gold_rows = read_jsonl(DATA)
    gold = {(r['language'], r['id']): r for r in gold_rows}
    official_subset = {(r['language'], r['id']) for r in gold_rows if r['streaming_subset']}

    check(summaries[NLLB]['status'] == 'completed', 'NLLB summary is not completed')
    check(summaries[NLLB]['data_sha256'] == data_hash, 'NLLB summary data hash mismatch')
    for name in MODELS:
        check(len(offline[name]) == 8096, f'{name}: expected 8096 offline rows')
        check(len({(r['language'], r['id']) for r in offline[name]}) == 8096, f'{name}: duplicate offline IDs')
        check(len(streams[name]) == 1600, f'{name}: expected 1600 stream trajectories')
        check(len({(r['language'], r['id'], r['mode']) for r in streams[name]}) == 1600, f'{name}: duplicate stream trajectories')
    check(len(control) == 800, 'batch1 control must contain 800 rows')
    control_ids = {(r['language'], r['id']) for r in control}
    check(len(control_ids) == 800, 'duplicate batch1 control IDs')

    for name in MODELS:
        rows_to_check = offline[name] + streams[name] + (control if name == NLLB else [])
        for row in rows_to_check:
            ref = gold.get((row['language'], row['id']))
            check(ref is not None, f"{name}: unknown pair {(row['language'], row['id'])}")
            check(row['source'] == ref['source'] and row['reference'] == ref['reference'], f"{name}: source/reference mismatch for {row['id']}")
            check(row['streaming_subset'] == ref['streaming_subset'], f"{name}: subset flag mismatch for {row['id']}")

    # Ensure all systems, both stream strategies, and the explicit control use exactly the official paired 100 IDs.
    paired_ids = {}
    for lang in LANGUAGES:
        expected = {i for l, i in official_subset if l == lang}
        check(len(expected) == 100, f'{lang}: official streaming subset is not 100 rows')
        paired_ids[lang] = expected
        controls = {r['id'] for r in control if r['language'] == lang}
        check(controls == expected, f'{lang}: batch1 IDs differ from official stream subset')
        for name in MODELS:
            for mode in MODES:
                ids = {r['id'] for r in streams[name] if r['language'] == lang and r['mode'] == mode}
                check(len(ids) == 100 and ids == expected, f'{name}/{lang}/{mode}: IDs differ from official stream subset')

    # Prefix schedules must match the older runs trajectory by trajectory; the final step must be the full source.
    old_lookup = {n: {(r['language'], r['id'], r['mode']): r for r in streams[n]} for n in OLD_MODELS}
    nllb_lookup = {(r['language'], r['id'], r['mode']): r for r in streams[NLLB]}
    for key, row in nllb_lookup.items():
        steps = row['steps']
        check(bool(steps), f'{key}: empty stream steps')
        check(steps[-1]['source'] == row['source'], f'{key}: final step is not full source')
        check(all(b['source'].startswith(a['source']) for a, b in zip(steps, steps[1:])), f'{key}: source prefixes are not monotonic')
        if row['mode'] == 'agreement_forced_prefix':
            check(steps[-1]['committed_text'] == row['hypothesis'], f'{key}: final committed text differs from hypothesis')
            check(all(b['committed_text'].startswith(a['committed_text']) for a, b in zip(steps, steps[1:])), f'{key}: committed text is not monotonic')
        for name in OLD_MODELS:
            old = old_lookup[name][key]
            check([s['source'] for s in steps] == [s['source'] for s in old['steps']], f'{name} and NLLB source prefix mismatch for {key}')

    metrics, stream_metrics, control_metrics, stream_stats = {}, {}, {}, {}
    for name in MODELS:
        metrics[name] = {}
        stream_metrics[name] = {}
        for lang in LANGUAGES:
            rows = [r for r in offline[name] if r['language'] == lang]
            check(len(rows) == 1012, f'{name}/{lang}: expected 1012 offline rows')
            val = score(rows, normalizer, bleu, chrf)
            expected = summaries[name]['scores'][lang]['offline_full']['t2s']
            check(all(abs(val[k] - expected[k]) < 1e-9 for k in val), f'{name}/{lang}: offline metrics differ from summary')
            metrics[name][lang] = val
            if name == NLLB:
                full_rows = rows
                metrics[name][lang]['output_counts'] = {
                    'non_eos': sum(not r.get('eos', False) for r in full_rows),
                    'empty': sum(not r['hypothesis'].strip() for r in full_rows),
                    'unknown_token_outputs': sum(r.get('unknown_tokens', 0) > 0 for r in full_rows),
                    'unknown_token_ids': sum(int(r.get('unknown_tokens', 0)) for r in full_rows),
                }
            stream_metrics[name][lang] = {}
            if name == NLLB:
                control_rows = [r for r in control if r['language'] == lang]
                check(len(control_rows) == 100, f'{lang}: expected 100 control rows')
                control_metrics[lang] = score(control_rows, normalizer, bleu, chrf)
            for mode in MODES:
                sr = [r for r in streams[name] if r['language'] == lang and r['mode'] == mode]
                check(len(sr) == 100, f'{name}/{lang}/{mode}: expected 100 trajectories')
                val = score(sr, normalizer, bleu, chrf)
                stream_metrics[name][lang][mode] = val
                expected = summaries[name]['scores'][lang][mode]['t2s']
                check(all(abs(val[k] - expected[k]) < 1e-9 for k in val), f'{name}/{lang}/{mode}: stream metrics differ from summary')
        if name == NLLB:
            stream_stats[name] = {}
            for mode in MODES:
                trajectories = [r for r in streams[name] if r['mode'] == mode]
                steps = [s for r in trajectories for s in r['steps']]
                latency = percentiles([s['seconds'] for s in steps])
                stream_stats[name][mode] = {
                    'updates': len(steps), 'latency_seconds': latency,
                    'normalized_erasure_characters_mean': mean(r['normalized_erasure_characters'] for r in trajectories),
                    'non_eos_updates': sum(not s['eos'] for s in steps),
                    'empty_updates': sum(not s['hypothesis'].strip() for s in steps),
                }

    control_lookup = {(r['language'], r['id']): r['hypothesis'] for r in control}
    retranslation = [r for r in streams[NLLB] if r['mode'] == 'retranslate']
    mismatches = sum(r['hypothesis'] != control_lookup[(r['language'], r['id'])] for r in retranslation)
    check(mismatches == 0, f'batch1 vs retranslation final mismatch count is {mismatches}/800')

    macro = {n: {k: mean(metrics[n][l][k] for l in LANGUAGES) for k in ('BLEU_zh', 'chrFpp')} for n in MODELS}
    audit = {
        'status': 'passed', 'data_sha256': data_hash, 'offline_rows_each_model': 8096,
        'stream_trajectories_each_model': 1600, 'nllb_batch1_control_rows': 800,
        'official_stream_subset_ids_match_all_models_and_control': True,
        'all_source_prefix_schedules_match_older_runs': True,
        'all_stream_final_steps_use_full_source': True,
        'source_reference_and_subset_match_flores': True,
        'independent_full_and_stream_metric_recalculation_matches_summaries': True,
        'full_metrics': metrics, 'stream_metrics': stream_metrics, 'macro_full_metrics': macro,
        'nllb_full_output_counts_by_language': {l: metrics[NLLB][l]['output_counts'] for l in LANGUAGES},
        'nllb_streaming_stats_by_mode': stream_stats[NLLB],
        'nllb_batch1_vs_retranslation_final': {'pairs': 800, 'mismatches': mismatches},
        'paired_stream_subset_metrics': {'batch1_control': control_metrics,
          'retranslate': {l: stream_metrics[NLLB][l]['retranslate'] for l in LANGUAGES},
          'agreement_forced_prefix': {l: stream_metrics[NLLB][l]['agreement_forced_prefix'] for l in LANGUAGES}},
    }

    out = ['# NLLB-200 600M 与现有多语模型对比', '',
      '日期：2026-10-17。基于本地实际推理产物，指标从逐句原始输出独立复算。', '',
      '## 测试口径', '',
      '- FLORES-101 devtest，英语、法语、德语、西班牙语、俄语、日语、韩语、阿拉伯语到简体中文；每语种1012句，共8096句。三模型使用同一数据、同一参考译文。',
      '- BLEU 使用 SacreBLEU 的中文 zh tokenizer，chrF++ 使用 word_order=2；输出和参考均经 OpenCC 简体转换。报告为语料分数，不是句子准确率或业务正确率。',
      '- 每个分数由8个语言方向独立计算。宏平均是8个语种分数的算术平均，不是把语料合并后重算。',
      '- FLORES 是公开书面文本基准，可能已进入模型训练、开发或选择流程；结果不代表未见业务文本。数据集没有音频、ASR、语音到达时间、术语库或业务后处理。',
      '- 延迟是文本更新的模型服务计算耗时，回放没有按真实语音时钟等待，不包含采音、ASR、排队、端到端字幕延迟。',
      '- 本次 NLLB 的延迟是在单条文本更新下测得；旧模型是在另一时间、共享 GPU 环境测得，不能据此计算严格速度倍数或排名。',
      '- 本对比不将指标解释为“准确率百分比”。商用前应另做业务样本和人工事实审查。', '',
      '## 运行配置与输出状态', '',
      f"NLLB revision：`{metadata.get('revision', '见metadata')}`；推理引擎：`{metadata.get('runtime', 'CTranslate2')}` {metadata.get('ctranslate2', '')}；GPU：NVIDIA RTX 3090（设备 {metadata.get('gpu_index', 1)}，24 GiB）；计算类型：`{metadata.get('compute_type', 'int8_float16')}`；beam：{metadata.get('beam_size', 3)}；offline batch：{metadata.get('offline_batch_size', 4)}；stream batch：{metadata.get('stream_batch_size', 1)}；最大生成长度：{metadata.get('max_decoding_length', 256)}。旧模型使用不同时间的共享GPU测量，延迟不可据此严格比较。", '',
      '| 源语言 | 全句非EOS输出 | 空白输出 | 含UNK输出 | UNK实际token ID数 |',
      '|---|---:|---:|---:|---:|']
    for lang, label in LANGUAGES.items():
        c = metrics[NLLB][lang]['output_counts']
        out.append(f"| {label} | {c['non_eos']} | {c['empty']} | {c['unknown_token_outputs']} | {c['unknown_token_ids']} |")
    out += ['', 'UNK统计按实际生成 token ID 记录；跳过特殊符号的可读文本可能看不到UNK。旧模型没有保存真实UNK ID计数，因此这里不作跨模型UNK对比。', '',
      '## 完整输入质量', '',
      '| 源语言 | 句数 | NLLB BLEU | M2M100 BLEU | SMaLL-100 BLEU | NLLB chrF++ | M2M100 chrF++ | SMaLL-100 chrF++ |',
      '|---|---:|---:|---:|---:|---:|---:|---:|']
    for lang, label in LANGUAGES.items():
        vals = [metrics[n][lang]['BLEU_zh'] for n in MODELS] + [metrics[n][lang]['chrFpp'] for n in MODELS]
        out.append(f"| {label} | 1012 | " + ' | '.join(f'{v:.2f}' for v in vals) + ' |')
    vals = [macro[n]['BLEU_zh'] for n in MODELS] + [macro[n]['chrFpp'] for n in MODELS]
    out += ['| 8语种宏平均 | 8096 | ' + ' | '.join(f'{v:.2f}' for v in vals) + ' |', '',
      '在本次固定数据和配置下，NLLB 的 BLEU 在八个方向均低于两个现有模型；宏平均 chrF++ 也较低。因此本次测试不支持将 NLLB-600M 作为质量升级替换。这个结论只覆盖本次八个语言方向和已测试配置，不代表所有语种、领域或解码配置。', '',
      '## NLLB 单条文本增量更新', '',
      '| 策略 | 更新数 | P50 ms | P95 ms | 平均字符 NE | 非EOS更新 | 空白更新 |',
      '|---|---:|---:|---:|---:|---:|---:|']
    for mode in MODES:
        st = stream_stats[NLLB][mode]
        out.append(f"| {mode} | {st['updates']} | {st['latency_seconds']['p50']*1000:.1f} | {st['latency_seconds']['p95']*1000:.1f} | {st['normalized_erasure_characters_mean']:.3f} | {st['non_eos_updates']} | {st['empty_updates']} |")
    out += ['', 'NE 是旧字幕尾部被删除字符数相对最终字幕长度的归一化值，可能大于1。按模式分别统计；采用 benchmark_streaming_mt.percentiles 的近邻秩口径。延迟来自文本更新计算，不是语音端到端延迟。', '',
      '## 配对100句的 NLLB 策略质量', '',
      '| 源语言 | batch1 BLEU | 重译最终 BLEU | 锁定前缀 BLEU | batch1 chrF++ | 重译最终 chrF++ | 锁定前缀 chrF++ |',
      '|---|---:|---:|---:|---:|---:|---:|']
    for lang, label in LANGUAGES.items():
        b, r, a = control_metrics[lang], stream_metrics[NLLB][lang]['retranslate'], stream_metrics[NLLB][lang]['agreement_forced_prefix']
        out.append(f"| {label} | {b['BLEU_zh']:.2f} | {r['BLEU_zh']:.2f} | {a['BLEU_zh']:.2f} | {b['chrFpp']:.2f} | {r['chrFpp']:.2f} | {a['chrFpp']:.2f} |")
    out += ['', f'全句 batch=1 对照与普通重译末态逐条文本不一致：{mismatches}/800。只有零差异时脚本才会写出 AUDIT.json 的 passed 状态。这个检查不说明过程中的暂定译文都正确。', '',
      '## 模型许可', '',
      'NLLB-200 distilled 600M 模型卡标注 CC-BY-NC 4.0（非商业）；M2M100 418M 模型卡标注 MIT；SMaLL-100 上游仓库的 LICENSE 为 MIT。NLLB 的非商业许可限制使其不能直接作为商用候选；应按实际权重版本复核条款。',
      '来源：[NLLB-200 模型卡](https://huggingface.co/facebook/nllb-200-distilled-600M)；[M2M100 模型卡](https://huggingface.co/facebook/m2m100_418M)；[SMaLL-100 LICENSE](https://github.com/alirezamshi/small100/blob/main/LICENSE)；[FLORES-101 数据说明](https://github.com/facebookresearch/flores/blob/main/previous_releases/flores101/README.md)；[SacreBLEU](https://github.com/mjpost/sacrebleu)。', '',
      '## 数据与复现', '',
      f"逐句完整输入、输出和EOS/UNK状态：`{ROOT}/{NLLB}.offline.jsonl`；流式更新轨迹：`{ROOT}/{NLLB}.streaming.jsonl`；batch1控制：`{ROOT}/batch1_control/{NLLB}.offline.jsonl`；运行revision、GPU和解码设置：`{ROOT}/{NLLB}.metadata.json`；官方对齐输入：`{DATA}`。", '',
      '复核本报告和 AUDIT.json：', '', '```bash', 'cd /home/aim0/桌面/tan/ASR/Translator', '.runtime/mt-eval-venv/bin/python scripts/report_nllb_comparison.py', '```', '',
      '## 词表覆盖诊断', '']
    vocab_audit = ROOT / 'tokenizer_trade_vocabulary_audit.json'
    if vocab_audit.exists():
        out += ['Fasttokenizer 词表审计显示，简体“阀”以及“锈”没有独立词条；对应探测短语“阀门”和“不锈钢”分别切出 `<unk>`，解码可见文本分别变为“门”和“不钢”。繁体“閥”在词表中（ID 255790），所以不能据此说模型绝对无法输出“阀门”：它仍可能生成繁体“閥门”，之后再由 OpenCC 转成简体。这里的证据支持较窄的结论：直接以规范简体术语输出时存在词表覆盖障碍，且未知 token 会丢失字面信息；繁体路径是否会在翻译中稳定触发，需另行验证。',
          '原始 FP16 报价探针的 `SUS304门` 输出也带有未知 token，说明这个现象不只出现在量化译文中；探针只是少量句子的诊断，不能推断整集发生率。词表审计详情：[tokenizer_trade_vocabulary_audit.json](tokenizer_trade_vocabulary_audit.json)。', '']
        if (ROOT / 'tokenizer_independent_review.md').exists():
            out += ['原始 SentencePiece、slow/fast tokenizer 和 CT2 词表交叉核查见 [tokenizer_independent_review.md](tokenizer_independent_review.md)；该模型无字节回退，缺字不是转换时丢失。', '']
    out += ['## 业务评测', '']
    business = ROOT / 'BUSINESS_REPORT.md'
    if business.exists():
        out += [business.read_text().rstrip(), '']
        audit['business_report_included'] = True
    else:
        out += ['业务结果尚未接入；后续可在本目录放置 BUSINESS_REPORT.md，由本脚本附加到本节。', '']
        audit['business_report_included'] = False
    (ROOT / 'REPORT.md').write_text('\n'.join(out))
    (ROOT / 'AUDIT.json').write_text(json.dumps(audit, ensure_ascii=False, indent=2) + '\n')
    print(ROOT / 'REPORT.md')


if __name__ == '__main__':
    main()
