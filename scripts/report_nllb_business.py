"""Compare the same constructed business cases; field hits are not accuracy."""
import hashlib
import json
import argparse
from pathlib import Path
from analyze_trade_fields import extract_amount
from opencc import OpenCC

CASES = Path('examples/streaming_mt_cases.json')
OLD_MODELS = {
    'm2m100_418M': Path('results/translation_streaming_20261017/beam3_realtime'),
    'small100': Path('results/translation_streaming_20261017/beam3_realtime'),
    'nllb200_600M': Path('results/nllb600m_comparison_20261017/business'),
}
LMT_ROOT = Path('results/lmt600m_comparison_20261017')
LMT_MODEL = 'LMT-60-0.6B'


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--lmt', action='store_true', help='include LMT business outputs and write report under LMT results')
    args = parser.parse_args()
    if args.lmt:
        root = LMT_ROOT
        models = {
            'm2m100_418M': OLD_MODELS['m2m100_418M'],
            'small100': OLD_MODELS['small100'],
            'nllb200_600M': OLD_MODELS['nllb200_600M'],
            LMT_MODEL: root / 'business',
        }
        fp16_model = LMT_MODEL
        fp16_title = 'LMT 原始FP16对照'
    else:
        root = Path('results/nllb600m_comparison_20261017')
        models = OLD_MODELS
        fp16_model = 'nllb200_600M'
        fp16_title = 'NLLB 原始FP16对照'

    gold = {r['id']: r for r in json.loads(CASES.read_text())['cases']}
    case_hash = hashlib.sha256(CASES.read_bytes()).hexdigest()
    normalizer = OpenCC('t2s')
    all_rows, diagnosis = {}, {}
    for name, directory in models.items():
        meta = json.loads((directory / f'{name}.summary.json').read_text())
        assert meta['cases_sha256'] == case_hash
        assert meta['stream_beam_size'] == 3 and meta['dtype'] == 'int8_float16'
        values = [json.loads(s) for s in (directory / f'{name}.cases.jsonl').read_text().splitlines()]
        rows = [r for r in values if r['mode'] == 'retranslate']
        assert len(rows) == len({r['case_id'] for r in rows}) == 25
        all_rows[name] = {r['case_id']: r for r in rows}
        detail = []
        for row in rows:
            case = gold[row['case_id']]
            source = case['prefixes'][-1] if 'prefixes' in case else ''.join(case['chunks'])
            assert row['source'] == source and row['reference'] == case['reference']
            if not row['case_id'].endswith('_order'):
                continue
            text = normalizer.convert(row['final_translation'])
            amount = extract_amount(text)
            detail.append({'case_id': row['case_id'], 'translation_t2s': text, 'amount': amount,
                'amount_currency_preserved': amount['status'] == 'explicit_arabic_amount_and_currency'
                    and len(amount['matches']) == 1 and amount['matches'][0]['normalized_amount'] == '12.5',
                'valve_term_hit': '阀' in text, 'model_literal_hit': 'SUS304' in text,
                'quantity_literal_hit': '500' in text})
        assert len(detail) == 8
        diagnosis[name] = {'orders': detail,
            'amount_currency_preserved': sum(r['amount_currency_preserved'] for r in detail),
            'valve_term_hit': sum(r['valve_term_hit'] for r in detail),
            'model_literal_hit': sum(r['model_literal_hit'] for r in detail),
            'quantity_literal_hit': sum(r['quantity_literal_hit'] for r in detail),
            'uncertain_amount_count': sum(r['amount']['status'].startswith('uncertain') for r in detail)}
    output = ['本节为同一组25个构造样例，含8语种报价、数量改口、否定和付款条件、短应答及1个ASR回改样例。每次输入间隔0.5秒为人工文本回放，无音频或ASR模型。', '',
        f'{len(models)}模型 CT2 int8_float16、beam=3；以下使用普通重译的最终输出，简繁统一。八条报价是同一模板的八语种版本，不是八个独立抽样业务场景。', '',
        '自动字段统计只作诊断：金额必须明确与美元币种相邻且可解析为12.50；中文数字或结构不明确时标为不确定。术语按“阀”字命中，型号/数量按字面匹配，均不等于业务或整句准确率。', '',
        '| 模型 | 12.50+美元保留 | 阀字命中 | SUS304字面命中 | 500字面命中 | 金额不确定 |',
        '|---|---:|---:|---:|---:|---:|']
    if args.lmt:
        output[3] += ' LMT使用stream-beam=3、realtime_interval=0.5秒；LMT原始FP16若存在，仅作整句对照。'
    for name, d in diagnosis.items():
        output.append(f"| {name} | {d['amount_currency_preserved']}/8 | {d['valve_term_hit']}/8 | {d['model_literal_hit']}/8 | {d['quantity_literal_hit']}/8 | {d['uncertain_amount_count']} |")
    output += ['', '以下保留全部最终译文，供人工核查否定、付款条件、日期、数量和金额；未把字面匹配自动当作语义正确。', '']
    for case_id, case in gold.items():
        output += [f"### {case_id}", '', f"参考：{case['reference']}", '']
        for name in models:
            row = all_rows[name][case_id]
            output.append(f"- {name}：{normalizer.convert(row['final_translation'])}")
        output.append('')
    fp16 = root / 'business_fp16' / f'{fp16_model}.cases.jsonl'
    if fp16.exists():
        output += [f'### {fp16_title}', '', '同样25个样例、beam=3，以下为整句FP16控制输出，用于定位量化或推理实现差异，不把不同结果直接归因为量化。', '']
        native_rows = [json.loads(s) for s in fp16.read_text().splitlines()]
        native_rows = [r for r in native_rows if r['mode'] == 'retranslate']
        assert len(native_rows) == 25
        for row in native_rows:
            output.append(f"- {row['case_id']}：{normalizer.convert(row['offline_beam3_translation'])}")
    cpu_control = root / 'native_fp32_cpu' / 'LMT-60-0.6B.cases.jsonl'
    if args.lmt and cpu_control.exists():
        meta_path=cpu_control.with_name('LMT-60-0.6B.summary.json')
        meta=json.loads(meta_path.read_text())
        assert meta['status']=='completed' and meta['device']=='cpu' and meta['dtype']=='float32'
        assert meta['cases_sha256']==case_hash and meta['beam_size']==3
        native_rows=[json.loads(s) for s in cpu_control.read_text().splitlines()]
        assert len(native_rows)==len({r['case_id'] for r in native_rows})==25
        output += ['### LMT 原始权重的 CPU FP32 质量对照', '',
            '原始权重以FP32在CPU运行，25例完整文本、beam=3、同一作者提示词。仅用于核查量化/引擎下的译文差异，CPU耗时不参与任何GPU延迟统计。差异同时包含数值精度和推理引擎因素，不能全部归因为量化。', '']
        for row in native_rows:
            assert row['source']==all_rows['LMT-60-0.6B'][row['case_id']]['source']
            output.append(f"- {row['case_id']}：{normalizer.convert(row['final_translation'])}")
    (root / 'BUSINESS_REPORT.md').write_text('\n'.join(output) + '\n')
    (root / 'business_field_diagnostics.json').write_text(json.dumps(
        {'cases_sha256': case_hash, 'models': diagnosis, 'scope': 'field diagnostics, not sentence accuracy'},
        ensure_ascii=False, indent=2) + '\n')
    print(json.dumps({name: {k:v for k,v in d.items() if k!='orders'} for name,d in diagnosis.items()}, ensure_ascii=False))


if __name__ == '__main__':
    main()
