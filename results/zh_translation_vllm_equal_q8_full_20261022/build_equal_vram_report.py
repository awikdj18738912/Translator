#!/usr/bin/env python3
"""Assemble fully scored results and paired intervals for memory-matched runs."""
from __future__ import annotations
import json, statistics
from pathlib import Path
import numpy as np

OUT=Path(__file__).resolve().parent/'calibrated'
ROOT=Path(__file__).resolve().parents[2]
QUANT=ROOT/'results/zh_translation_quantized_full_20261021'
RUNS=ROOT/'results/zh_translation_backends_full_20261021'
LANGS=('en','fr','de','es','ru','ja','ko','ar')
SYSTEMS=('q8_0','bf16_vllm','fp8_vllm','nvfp4_vllm')
NAMES={'q8_0':'Q8_0 / llama.cpp','bf16_vllm':'BF16 / vLLM','fp8_vllm':'FP8 / vLLM','nvfp4_vllm':'NVFP4 / vLLM'}

def read(path):return [json.loads(x) for x in Path(path).open(encoding='utf-8') if x.strip()]
def save(path,data):Path(path).write_text(json.dumps(data,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')

def bootstrap(a,b,n=10000,seed=20261022):
 ids=sorted({ident for _,ident in a},key=int)
 delta=np.asarray([[a[(language,ident)]-b[(language,ident)] for ident in ids] for language in LANGS])
 rng=np.random.default_rng(seed); draws=np.empty(n,dtype=np.float64)
 for start in range(0,n,200):
  count=min(200,n-start);ix=rng.integers(0,len(ids),size=(count,len(ids)))
  draws[start:start+count]=delta[:,ix].mean(axis=2).mean(axis=0)
 low,high=np.percentile(draws,[2.5,97.5])
 return {'delta':float(delta.mean()),'low':float(low),'high':float(high),'samples':n,'seed':seed}

def main():
 metrics=('comet','metricx','xcomet'); maps=json.loads((OUT/'scoring/mapping.json').read_text())
 unique=read(OUT/'scoring/unique_inputs.jsonl'); nscore={}
 for metric in metrics:
  values=read(OUT/'scoring/unique'/f'{metric}.scores.jsonl')
  assert len(values)==len(unique)==18353 and all(v['unique_id']==i for i,v in enumerate(values))
  nscore[metric]={r['unique_id']:r['score'] for r in values}
 all_lang={};macros={};pair_scores={};outdir=OUT/'scoring'
 for key in SYSTEMS:
  raw=read((OUT/'q8_0' if key=='q8_0' else OUT/key)/'predictions.jsonl');assert len(raw)==8096
  byid={(r['language'],str(r['id'])):r for r in raw};score_rows=[]
  (outdir/key).mkdir(parents=True,exist_ok=True)
  for r in maps[key]:
   z={'id':r['id'],'language':r['language'],'unique_id':r['unique_id'],**{m:nscore[m][r['unique_id']] for m in metrics}}
   score_rows.append(z)
  with (outdir/key/'neural_scores.jsonl').open('w',encoding='utf-8') as f:
   for r in score_rows:f.write(json.dumps(r,ensure_ascii=False)+'\n')
  neural={language:{metric:[] for metric in metrics} for language in LANGS}
  for r in score_rows:
   for m in metrics:neural[r['language']][m].append(float(r[m]))
  classic=json.loads(((OUT/'q8_0' if key=='q8_0' else OUT/key)/'summary.json').read_text())
  bleu=json.loads((outdir/'bleu13a'/key/'summary.json').read_text())
  lang={}
  for language in LANGS:
   q=classic['languages'][language]
   lang[language]={'spBLEU':q['spBLEU'],'chrFpp':q['chrFpp'],
    'BLEU13a':bleu['languages'].get(language,{}).get('BLEU'),
    **{'COMET':statistics.fmean(neural[language]['comet']),'MetricX':statistics.fmean(neural[language]['metricx']),'XCOMET':statistics.fmean(neural[language]['xcomet'])}}
  if key=='q8_0':
   meta=json.loads((OUT/'q8_0/metadata.json').read_text());speed={'sentences_per_second':meta['sentences_per_second'],'generated_tokens_per_second':meta['generated_tokens_per_second'],'language_results':meta['language_results']}
   old_path=QUANT/'q8_0/predictions.jsonl'
   eq=json.loads((OUT/'equal_vram_summary.json').read_text());memory=eq['q8_baseline']['q8_observed_peak_mib']
  else:
   eq=json.loads((OUT/'equal_vram_summary.json').read_text());speed=next(x for x in eq['models'] if x['model']==key);meta=json.loads((OUT/key/'metadata.json').read_text());speed['language_results']=meta['language_results'];
   old_path=RUNS/key/'predictions.jsonl';memory=speed['peak_gpu_memory_mib']
  old={(r['language'],str(r['id'])):r['hypothesis'] for r in read(old_path)}
  exact=sum(r['hypothesis']==old[(r['language'],str(r['id']))] for r in raw)
  macros[key]={m:statistics.fmean(lang[l][m] for l in LANGS) for m in ('spBLEU','chrFpp','COMET','MetricX','XCOMET')}
  macros[key]['BLEU13a']=statistics.fmean(bleu['languages'][l]['BLEU'] for l in bleu['languages'])
  macros[key].update(sentences_per_second=speed['sentences_per_second'],generated_tokens_per_second=speed['generated_tokens_per_second'],
   truncated=sum(bool(r.get('truncated')) for r in raw),memory_mib=memory,exact_previous=exact)
  all_lang[key]=lang
  for m in ('COMET','MetricX','XCOMET'):
   pair_scores.setdefault((key,m),{(r['language'],str(r['id'])):float(r[m.lower()]) for r in score_rows})
 pairs={}
 for key in SYSTEMS[1:]:pairs[key]={m:bootstrap(pair_scores[(key,m)],pair_scores[('q8_0',m)]) for m in ('COMET','MetricX','XCOMET')}
 xsummary=json.loads((outdir/'unique/xcomet.scores.summary.json').read_text())
 report=['# Q8_0 与 vLLM 在相同显存预算下的全量对照','','Q8_0 / llama.cpp 实测整卡峰值 6,695 MiB，按该值反馈校准三个 vLLM 版本；每组 8,096 条、八语各 1,012 条、128 并发。',
  'COMET 与 MetricX 对本轮 18,353 个不同输入全部新鲜评分；XCOMET 对本轮与历史译文不同的输入新评分，其余使用同 checkpoint 对完全相同输入的既有分数；覆盖全部 32,384 个系统输出。',
  f"XCOMET 共对 {xsummary['fresh_unique_inputs']:,} 个新输入评分，复用 {xsummary['reused_identical_inputs']:,} 个完全相同输入的分数。",'',
  '| 方案 | 整卡峰值 MiB / GiB | 句/秒 | token/秒 | spBLEU | BLEU-13a（6语） | chrF++ | COMET | MetricX ↓ | XCOMET | 截断 | 与原运行译文完全一致 |',
  '|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|']
 for key in SYSTEMS:
  m=macros[key];report.append(f"| {NAMES[key]} | {m['memory_mib']} / {m['memory_mib']/1024:.2f} | {m['sentences_per_second']:.2f} | {m['generated_tokens_per_second']:.2f} | {m['spBLEU']:.3f} | {m['BLEU13a']:.3f} | {m['chrFpp']:.3f} | {m['COMET']:.6f} | {m['MetricX']:.6f} | {m['XCOMET']:.6f} | {m['truncated']} | {m['exact_previous']}/8096 ({m['exact_previous']/8096:.1%}) |")
 report+=['','spBLEU、BLEU-13a、chrF++、COMET、XCOMET 越高越好；MetricX 越低越好。BLEU-13a 只报六语，其余宏平均为八语等权。分数不是正确率百分比。','',
  '## 分语种结果','']
 for key in SYSTEMS:
  report += [f"### {NAMES[key]}",'','| 语种 | spBLEU | BLEU-13a | chrF++ | COMET | MetricX | XCOMET | 句/秒 | token/秒 |','|---|---:|---:|---:|---:|---:|---:|---:|---:|']
  meta=json.loads((OUT/'q8_0/metadata.json' if key=='q8_0' else OUT/key/'metadata.json').read_text())
  for language in LANGS:
   z=all_lang[key][language];b='—' if z['BLEU13a'] is None else f"{z['BLEU13a']:.3f}";v=meta['language_results'][language]
   report.append(f"| {language} | {z['spBLEU']:.3f} | {b} | {z['chrFpp']:.3f} | {z['COMET']:.6f} | {z['MetricX']:.6f} | {z['XCOMET']:.6f} | {v['sentences_per_second']:.2f} | {v['generation_tokens_per_second']:.2f} |")
 report+=['','## vLLM 减 Q8_0：配对质量分差与 95% bootstrap 区间','',
  '以中文源句 ID 为抽样单位，每次连同该句八个语言一起抽取，10,000 次 bootstrap，seed=20261022。COMET、MetricX 为本轮全新句分；XCOMET 对完全相同输入复用相同 checkpoint 的先前句分。','']
 for key in SYSTEMS[1:]:
  report += [f"### {NAMES[key]} − Q8_0",'','| 指标 | Δ | 95%区间 |','|---|---:|---:|']
  for metric,v in pairs[key].items():report.append(f"| {metric} | {v['delta']:+.6f} | [{v['low']:+.6f}, {v['high']:+.6f}] |")
 report+=['','## 运行口径','',
  '- Q8_0 和 vLLM 共用同一 FLORES-101 devtest、提示词、贪心解码及 512 token 上限。Q8_0 使用 llama.cpp；vLLM 根据上一轮整卡峰值校准 `gpu_memory_utilization`，再按请求区间遥测反馈调整。','- BF16 的低显存运行与原高显存运行逐条译文一致率见表；FP8/NVFP4 也列出与原运行的一致率。所有译文变化都进入本轮质量统计。','- RTX 3090 上 FP8 和 NVFP4 使用 Marlin W8A16/W4A16，而不是原生 FP8/FP4 激活计算。每种版本实际显存峰值在 6.48–6.56 GiB。','- 显存为 `nvidia-smi` 整卡采样，Q8_0 379 个样本、三个 vLLM 各约 100 个样本。吞吐不含模型加载与预热。token/s 使用全量生成 token 数除以推理 wall time。','- XCOMET 对 15,770 个与此前版本完全相同的输入复用了同 checkpoint 分数；对剩余 2,583 个唯一输入新评分。COMET 和 MetricX 对全部 18,353 个唯一输入新鲜评分。','',
  '原始输出、全量显存遥测、唯一输入清单与评分映射见本目录的 `q8_0/`、各 vLLM 子目录及 `scoring/`。']
 (OUT/'EQUAL_MEMORY_COMPARISON.md').write_text('\n'.join(report)+'\n',encoding='utf-8')
 save(OUT/'equal_vram_quality_summary.json',{'systems':SYSTEMS,'macros':macros,'languages':all_lang,'paired_bootstrap':pairs,'xcomet_reuse':xsummary,'mapped_rows':len(SYSTEMS)*8096})
 print(OUT/'EQUAL_MEMORY_COMPARISON.md')
if __name__=='__main__':main()
