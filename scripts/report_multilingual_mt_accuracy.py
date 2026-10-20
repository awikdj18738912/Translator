"""Audit paired evaluation artifacts and report public-corpus quality, not percent accuracy."""
import json
import hashlib
from pathlib import Path
from statistics import mean

ROOT=Path('results/multilingual_mt_accuracy_20261017')
MODELS=('m2m100_418M','small100')
LANGUAGES={'en':'英语','fr':'法语','de':'德语','es':'西班牙语','ru':'俄语','ja':'日语','ko':'韩语','ar':'阿拉伯语'}


def jsonl(path):
    return [json.loads(s) for s in path.read_text().splitlines()]


def main():
    from opencc import OpenCC
    from sacrebleu.metrics import BLEU,CHRF
    normalizer=OpenCC('t2s')
    summaries={name:json.loads((ROOT/f'{name}.summary.json').read_text()) for name in MODELS}
    offline={name:jsonl(ROOT/f'{name}.offline.jsonl') for name in MODELS}
    streaming={name:jsonl(ROOT/f'{name}.streaming.jsonl') for name in MODELS}
    controls={name:jsonl(ROOT/'batch1_control'/f'{name}.offline.jsonl') for name in MODELS}
    data_path=Path('data/eval/flores101/aligned_devtest.jsonl')
    data_hash=hashlib.sha256(data_path.read_bytes()).hexdigest()
    gold={(r['language'],r['id']):r for r in jsonl(data_path)}
    for name in MODELS:
        assert summaries[name]['data_sha256']==data_hash
        for row in offline[name]+streaming[name]+controls[name]:
            reference=gold[row['language'],row['id']]
            assert row['source']==reference['source'] and row['reference']==reference['reference']
            assert row['streaming_subset']==reference['streaming_subset']
    control_metrics={}
    for name in MODELS:
        assert len(controls[name])==800
        lookup={(r['language'],r['id']):r['hypothesis'] for r in controls[name]}
        final_rows=[r for r in streaming[name] if r['mode']=='retranslate']
        mismatches=sum(r['hypothesis']!=lookup[(r['language'],r['id'])] for r in final_rows)
        control_metrics[name]={'full_input_batch1_vs_retranslation_final_mismatches':mismatches,'pairs':800}
        for language in LANGUAGES:
            rows=[r for r in controls[name] if r['language']==language]
            assert len(rows)==100
            hypotheses=[normalizer.convert(r['hypothesis']) for r in rows]
            references=[[normalizer.convert(r['reference']) for r in rows]]
            control_metrics[name][language]={'BLEU_zh':BLEU(tokenize='zh').corpus_score(hypotheses,references).score,
                                           'chrFpp':CHRF(word_order=2).corpus_score(hypotheses,references).score}
    for name in MODELS:
        summary=summaries[name]
        assert summary['status']=='completed'
        assert summary['offline_pairs']==8096 and summary['streaming_trajectories']==1600
        assert len(offline[name])==len({(r['language'],r['id']) for r in offline[name]})==8096
        assert len(streaming[name])==len({(r['language'],r['id'],r['mode']) for r in streaming[name]})==1600
        for language in LANGUAGES:
            assert sum(r['language']==language for r in offline[name])==1012
            for mode in ('retranslate','agreement_forced_prefix'):
                values=[r for r in streaming[name] if r['language']==language and r['mode']==mode]
                assert len(values)==100
                for row in values:
                    assert row['steps'][-1]['source']==row['source']
                    assert all(b['source'].startswith(a['source']) for a,b in zip(row['steps'],row['steps'][1:]))
                    if mode=='agreement_forced_prefix':
                        assert row['steps'][-1]['committed_text']==row['hypothesis']
                        assert all(b['committed_text'].startswith(a['committed_text']) for a,b in zip(row['steps'],row['steps'][1:]))
    for language in LANGUAGES:
        for mode in ('retranslate','agreement_forced_prefix'):
            first=[r for r in streaming[MODELS[0]] if r['language']==language and r['mode']==mode]
            second=[r for r in streaming[MODELS[1]] if r['language']==language and r['mode']==mode]
            assert [r['id'] for r in first]==[r['id'] for r in second]
            assert [[s['source'] for s in r['steps']] for r in first]==[[s['source'] for s in r['steps']] for r in second]
    # Independently recompute scores from raw hypotheses; do not trust saved aggregates.
    for name in MODELS:
        for language in LANGUAGES:
            rows=[r for r in offline[name] if r['language']==language]
            hypotheses=[normalizer.convert(r['hypothesis']) for r in rows]
            references=[[normalizer.convert(r['reference']) for r in rows]]
            b=BLEU(tokenize='zh').corpus_score(hypotheses,references).score
            c=CHRF(word_order=2).corpus_score(hypotheses,references).score
            expected=summaries[name]['scores'][language]['offline_full']['t2s']
            assert abs(b-expected['BLEU_zh'])<1e-9 and abs(c-expected['chrFpp'])<1e-9
            for mode in ('retranslate','agreement_forced_prefix'):
                values=[r for r in streaming[name] if r['language']==language and r['mode']==mode]
                b=BLEU(tokenize='zh').corpus_score([normalizer.convert(r['hypothesis']) for r in values],
                    [[normalizer.convert(r['reference']) for r in values]]).score
                c=CHRF(word_order=2).corpus_score([normalizer.convert(r['hypothesis']) for r in values],
                    [[normalizer.convert(r['reference']) for r in values]]).score
                expected=summaries[name]['scores'][language][mode]['t2s']
                assert abs(b-expected['BLEU_zh'])<1e-9 and abs(c-expected['chrFpp'])<1e-9
    text=['# 多语翻译模型：公开测试集质量与增量流式评测','',
          '日期：2026-10-17。模型已经下载到 /home/aim0/data/models/ASR。本文为真实GPU运行结果，不是论文分数。','',
          '## 范围与判定口径','',
          '- 两模型都用 CTranslate2 4.6.3、GPU、int8_float16、beam=3。采用同一解码长度上限256；不主动截断源文本。',
          '- FLORES-101 官方 devtest：每种源语言1012句，8种语言合计8096个源→中文平行对；每模型全部翻译。',
          '- 各语言使用同一组固定随机种子选出的100个对齐句；每句构造4个单调增长的文本前缀。模型间输入前缀逐字相同。每模型1600条完整流式轨迹（100句×8语言×2策略），通常6400次更新。',
          '- 重译策略：每次根据当前完整前缀重新生成，末次输入为全句。稳定策略：连续两次候选的共同目标前缀，保留一个token后退量，再强制已提交前缀。策略为实验实现，不是原生在线编码、完整NLLW或真实同传训练。',
          '- FLORES提供书面句对，没有音频或词级到达时间；这里没有ASR、语种识别、业务实体库、Refiner或真实语音延迟。',
          '- BLEU、chrF++是0–100范围的翻译质量分，不是句子正确率，也不能用“BLEU28=28%正确”解释。两模型都只使用一个简体中文参考译文。',
          '- 主表把输出和参考统一为简体后计分；原始简繁字形分数也保存在summary中。OpenCC只处理字形，不纠正内容。',
          '- 本次 BLEU 使用中文分词器zh，不是SMaLL论文中的spBLEU，不能直接与论文数字比较。',
          '- 该数据集是已公开多年的基准，可能参与过开发或模型选择，不视为全新保密测试。','',
          '## 完整输入：各语言到中文的质量','',
          '| 源语言 | 句数 | M2M100 BLEU↑ | SMaLL-100 BLEU↑ | M2M100 chrF++↑ | SMaLL-100 chrF++↑ |',
          '|---|---:|---:|---:|---:|---:|']
    for language,label in LANGUAGES.items():
        a=summaries[MODELS[0]]['scores'][language]['offline_full']['t2s']
        b=summaries[MODELS[1]]['scores'][language]['offline_full']['t2s']
        text.append(f"| {label} | 1012 | {a['BLEU_zh']:.2f} | {b['BLEU_zh']:.2f} | {a['chrFpp']:.2f} | {b['chrFpp']:.2f} |")
    macro={name:{metric:mean(summaries[name]['scores'][lang]['offline_full']['t2s'][metric] for lang in LANGUAGES)
                 for metric in ('BLEU_zh','chrFpp')} for name in MODELS}
    text.append(f"| 8语种宏平均 | 8096 | {macro[MODELS[0]]['BLEU_zh']:.2f} | {macro[MODELS[1]]['BLEU_zh']:.2f} | {macro[MODELS[0]]['chrFpp']:.2f} | {macro[MODELS[1]]['chrFpp']:.2f} |")
    text += ['', '宏平均是8个语种各自分数的算术平均，不是把所有句子拼接后重算的总体BLEU。','',
             '## 相同100句子集：流式最终质量是否下降','',
             '| 源语言 | 模型 | 全句单条对照 BLEU | 增量重译最终 BLEU | 锁定前缀最终 BLEU | 锁定前缀相对单条对照Δ |',
             '|---|---|---:|---:|---:|---:|']
    for language,label in LANGUAGES.items():
        for name in MODELS:
            score=summaries[name]['scores'][language]
            a=control_metrics[name][language]['BLEU_zh']
            b=score['retranslate']['t2s']['BLEU_zh']
            c=score['agreement_forced_prefix']['t2s']['BLEU_zh']
            text.append(f'| {label} | {name} | {a:.2f} | {b:.2f} | {c:.2f} | {c-a:+.2f} |')
    text += ['', '为排除batch=4与batch=1的计算差异，另对同一800个对齐样本运行了全句batch=1对照。上表采用单条对照作比较。普通重译最后一次看到完整源文，因此其最终分数应接近全句单条对照；这不能说明过程中显示的暂定译文都正确。锁定前缀会产生路径依赖，必须单独报告。',
             '单条全句对照与普通重译末次的逐条文本不一致数：'+ '；'.join(f"{name} {control_metrics[name]['full_input_batch1_vs_retranslation_final_mismatches']}/800" for name in MODELS)+'。','',
             '## 单条文本更新计算耗时与回改','',
             '下表P50/P95为各语言100句所有更新的服务耗时汇总；测试尽快回放而非按真实语音时钟到达。NE按原始输出字符统计旧字幕尾部被删除的数量，再除以最后字幕长度，未做简繁归一。表中为每轨迹NE的均值，可能大于1。生产字幕若统一简体，需另测展示回改。','',
             '| 模型 | 策略 | 更新数 | P50 ms | P95 ms | 平均字符NE↓ | 非EOS更新数 | 空白更新数 |',
             '|---|---|---:|---:|---:|---:|---:|---:|']
    from benchmark_streaming_mt import percentiles
    for name in MODELS:
        for mode in ('retranslate','agreement_forced_prefix'):
            rows=[r for r in streaming[name] if r['mode']==mode]
            steps=[s for r in rows for s in r['steps']]
            latency=percentiles([s['seconds'] for s in steps])
            text.append(f"| {name} | {mode} | {len(steps)} | {latency['p50']*1000:.1f} | {latency['p95']*1000:.1f} | {mean(r['normalized_erasure_characters'] for r in rows):.3f} | {sum(not s['eos'] for s in steps)} | {sum(not s['hypothesis'].strip() for s in steps)} |")
    text += ['', '不能把以上计算耗时当作语音字幕总延迟。实际还需等待足够源文、ASR确认、目标前缀稳定与GPU排队。测试期间GPU上存在其他任务，资源占用会变化；本次保持现有任务运行，所以此处是共享GPU测量。','',
             '## 外贸定向挑战样例','',
             '此前还测了8语种的报价、数量、型号、交期、改口与短应答。完整数据见 ../translation_streaming_20261017/REPORT.md。主要错误包括12.50美元→1250美元、阀门→风扇/门、以及部分否定或付款条件错误；原始FP16也出现这些问题。',
             '公开书面基准的分数不能替代这些关键事实检查。没有进行正式业务人审，不把BLEU或字面匹配包装成业务正确率。',
             '逐条报价字段诊断见 trade_field_diagnostics.trade.md 和对应JSON；这8条是同一报价模板的不同语言版本，不是8个独立抽样业务场景。CT2 beam=3的12.50美元金额+币种保留分别为M2M100 3/8、SMaLL-100 7/8；目标术语“阀”字命中分别为7/8和0/8。两种指标并不等于整句准确率。','',
             '## 结论边界','',
             '支持的工程形式：已知源语言文本持续增加，模型多次重译，当前尾部允许修改，结束时提交全句。两模型的源语编码/目标中文标记、尾部提交和稳定前缀不变量均已检查。',
             '不支持的即插即用形式：直接把不断变化的输入追加到一次generate调用，让模型原生保持在线编码状态并自动选择读写；本次都是外部调度，源文变化时重跑编码。',
             '质量选择应按语言方向判断；更快不代表商务事实更准确。只有中文译文作为输入的Refiner不能可靠修复上游漏译、错价或条件反转。','',
             '## 数据与复现','',
             '- data/eval/flores101/MANIFEST.json：官方数据URL、SHA256、语种映射及100句固定子集ID。',
             '- *.offline.jsonl：所有完整输入、参考中文、模型原始输出、输入/输出长度与EOS状态。',
             '- *.streaming.jsonl：每条轨迹的所有源前缀、候选中文、已确认前缀、更新计算耗时和最终译文。',
             '- *.protocol_audit.json：每种语言的实际模型输入token/ID与目标中文协议。',
             '- *.summary.json：逐语种质量、原始与简体归一分数、指标signature、版本和资源信息。','',
             '```bash',
             'cd /home/aim0/桌面/tan/ASR/Translator',
             'export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 TOKENIZERS_PARALLELISM=false',
             'export LD_LIBRARY_PATH=/home/aim0/data/conda/envs/vllm/lib/python3.12/site-packages/nvidia/cublas/lib:/home/aim0/data/conda/envs/vllm/lib/python3.12/site-packages/nvidia/cudnn/lib:${LD_LIBRARY_PATH:-}',
             '.runtime/mt-eval-venv/bin/python scripts/evaluate_multilingual_mt.py --model small100 --gpu 0 --batch-size 4 --output results/multilingual_mt_accuracy_rerun',
             '# 另一个模型使用 --model m2m100_418M，并选择空闲显存足够的GPU。',
             '```','',
             '## 来源','',
             '- [官方 FLORES-101 数据与说明](https://github.com/facebookresearch/flores/blob/main/previous_releases/flores101/README.md)',
             '- [SacreBLEU](https://github.com/mjpost/sacrebleu)',
             '- [M2M100 模型卡](https://huggingface.co/facebook/m2m100_418M)',
             '- [SMaLL-100 模型与作者分词器](https://github.com/alirezamshi/small100)','']
    (ROOT/'REPORT.md').write_text('\n'.join(text))
    (ROOT/'AUDIT.json').write_text(json.dumps({'status':'passed','offline_rows_each_model':8096,
        'stream_trajectories_each_model':1600,'same_ids_and_prefixes':True,'independently_recomputed_metrics_match':True,
        'source_and_reference_match_official_extracted_data':True,'data_sha256':data_hash,
        'macro_offline_t2s':macro,'batch1_control':control_metrics},ensure_ascii=False,indent=2)+'\n')
    print(ROOT/'REPORT.md')


if __name__=='__main__':
    main()
