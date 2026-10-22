#!/usr/bin/env python3
"""Fresh full-corpus scoring of Q8_0 and VRAM-matched vLLM outputs."""
import argparse,hashlib,json,os,statistics,subprocess,sys
from collections import Counter,defaultdict
from pathlib import Path

OUT=Path(__file__).resolve().parent/'calibrated'
ROOT=Path(__file__).resolve().parents[2]
LANGS=('en','fr','de','es','ru','ja','ko','ar')
SYSTEMS=('q8_0','bf16_vllm','fp8_vllm','nvfp4_vllm')
BENCH='/home/aim0/data/venvs/translation-bench/bin/python'
COMET='/home/aim0/data/venvs/translation-comet/bin/python'
MX='/home/aim0/data/venvs/translation-metricx/bin/python'
SCORER=ROOT/'results/zh_translation_backends_full_20261021/score_unique_neural.py'
QUANT=ROOT/'results/zh_translation_quantized_full_20261021'
RUNS=ROOT/'results/zh_translation_backends_full_20261021'

def read(p):
 with Path(p).open(encoding='utf-8') as f:return [json.loads(x) for x in f if x.strip()]
def save(p,d):
 p=Path(p);p.parent.mkdir(parents=True,exist_ok=True);t=p.with_suffix(p.suffix+'.tmp');t.write_text(json.dumps(d,ensure_ascii=False,indent=2)+'\n');t.replace(p)
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def path_for(k):return (OUT/'q8_0' if k=='q8_0' else OUT/k)/'predictions.jsonl'

def prepare():
 rows_by={};unique=[];toid={};mapping={};base=None
 for k in SYSTEMS:
  rows=read(path_for(k))
  if len(rows)!=8096 or Counter(r['language'] for r in rows)!=Counter({l:1012 for l in LANGS}):raise ValueError(f'{k}: incomplete')
  keyed={(r['language'],str(r['id'])):r for r in rows}
  if len(keyed)!=8096:raise ValueError(f'{k}: non-unique IDs')
  if base is None:base=keyed
  elif keyed.keys()!=base.keys():raise ValueError(f'{k}: IDs mismatch')
  else:
   for ident,r in keyed.items():
    if (r['source'],r['reference'])!=(base[ident]['source'],base[ident]['reference']):raise ValueError(f'{k}: source/reference mismatch')
  rows_by[k]=rows;mp=[]
  for r in rows:
   ident=(r['source'],r['hypothesis'],r['reference'],r['language'])
   if ident not in toid:
    uid=len(unique);toid[ident]=uid
    unique.append({'unique_id':uid,'language':r['language'],'source':r['source'],'hypothesis':r['hypothesis'],'reference':r['reference']})
   mp.append({'id':r['id'],'language':r['language'],'unique_id':toid[ident]})
  mapping[k]=mp
 for k,rows in rows_by.items():
  folder=OUT/'scoring/bleu13a'/k;folder.mkdir(parents=True,exist_ok=True)
  with (folder/'predictions.jsonl').open('w',encoding='utf-8') as f:
   for r in rows:
    if r['language'] in ('en','fr','de','es','ru','ar'):f.write(json.dumps(r,ensure_ascii=False)+'\n')
 sc=OUT/'scoring';sc.mkdir(parents=True,exist_ok=True)
 with (sc/'unique_inputs.jsonl').open('w',encoding='utf-8') as f:
  for r in unique:f.write(json.dumps(r,ensure_ascii=False)+'\n')
 save(sc/'mapping.json',mapping)
 save(sc/'manifest.json',{'systems':SYSTEMS,'rows_per_system':8096,'mapped_rows':32384,'unique_exact_inputs':len(unique),'dedup_key':['source','hypothesis','reference','language'],'fresh_scores':True})
 print(f'Prepared {len(unique)} unique inputs mapped to 32,384 output rows.',flush=True)

def run_classic():
 for k in SYSTEMS:
  pred=path_for(k);out=pred.parent/'summary.json'
  if not out.is_file():subprocess.run([BENCH,str(ROOT/'scripts/benchmark_zh_translation.py'),'score','--predictions',str(pred),'--bleu-tokenizer','flores200'],cwd=ROOT,check=True)
  pred13=OUT/'scoring/bleu13a'/k/'predictions.jsonl';out13=pred13.parent/'summary.json'
  if not out13.is_file():subprocess.run([BENCH,str(ROOT/'scripts/benchmark_zh_translation.py'),'score','--predictions',str(pred13),'--bleu-tokenizer','13a'],cwd=ROOT,check=True)

def score_neural():
 env=os.environ.copy();env.update(CUDA_VISIBLE_DEVICES='1',HF_HOME='/home/aim0/data/hf-cache',HF_HUB_CACHE='/home/aim0/data/hf-cache/hub',HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1',PYTHONUNBUFFERED='1')
 for metric,exe,batch in [('comet',COMET,'256'),('metricx',MX,'128'),('xcomet',COMET,'16')]:
  out=OUT/'scoring/unique'/f'{metric}.scores.jsonl'
  if metric=='metricx':env['PYTHONPATH']='/home/aim0/data/metricx-source'
  subprocess.run([exe,str(SCORER),metric,'--input',str(OUT/'scoring/unique_inputs.jsonl'),'--output',str(out),'--batch-size',batch],cwd=ROOT,env=env,check=True)

def paired_bootstrap(a,b,metric,n=10000,seed=20261022):
 import numpy as np
 ids=sorted({ident for _,ident in a},key=int);delta=np.asarray([[a[(l,i)]-b[(l,i)] for i in ids] for l in LANGS])
 rng=np.random.default_rng(seed);z=np.empty(n)
 for st in range(0,n,200):
  q=min(200,n-st);samples=rng.integers(0,len(ids),size=(q,len(ids)));z[st:st+q]=delta[:,samples].mean(axis=2).mean(axis=0)
 lo,hi=np.percentile(z,[2.5,97.5]);return {'delta':float(delta.mean()),'low':float(lo),'high':float(hi),'samples':n,'seed':seed,'paired_by':'Chinese source ID, keeping all eight languages together'}

def collect_report():
    sc=OUT/'scoring';unique=read(sc/'unique_inputs.jsonl');mapping=json.loads((sc/'mapping.json').read_text());scores={}
    macros_by={};langs_by={}
    for metric in ('comet','metricx','xcomet'):
     values=read(sc/'unique'/f'{metric}.scores.jsonl')
     if len(values)!=len(unique) or any(v['unique_id']!=i for i,v in enumerate(values)):raise ValueError(f'{metric} coverage mismatch')
     scores[metric]={v['unique_id']:float(v['score']) for v in values}
    for k in SYSTEMS:
     with (sc/k/'neural_scores.jsonl').open('w',encoding='utf-8') as f:
      for r in mapping[k]:f.write(json.dumps({**r,**{m:scores[m][r['unique_id']] for m in scores}},ensure_ascii=False)+'\n')
     rows=read(path_for(k));classic=json.loads((path_for(k).parent/'summary.json').read_text());bleu=json.loads((sc/'bleu13a'/k/'summary.json').read_text());nr=read(sc/k/'neural_scores.jsonl')
     by={l:defaultdict(list) for l in LANGS}
     for r in nr:
      for m in ('comet','metricx','xcomet'):by[r['language']][m].append(r[m])
     lang={}
     for l in LANGS:
      q=classic['languages'][l];lang[l]={'spBLEU':q['spBLEU'],'chrFpp':q['chrFpp'],'BLEU13a':bleu['languages'].get(l,{}).get('BLEU'),
       'COMET':statistics.fmean(by[l]['comet']),'MetricX':statistics.fmean(by[l]['metricx']),'XCOMET':statistics.fmean(by[l]['xcomet'])}
     metadata=json.loads(((OUT/'q8_0' if k=='q8_0' else OUT/k)/'metadata.json').read_text())
     macros={m:statistics.fmean(lang[l][m] for l in LANGS) for m in ('spBLEU','chrFpp','COMET','MetricX','XCOMET')};macros['BLEU13a']=statistics.fmean(bleu['languages'][l]['BLEU'] for l in bleu['languages'])
     if k!='q8_0':
      equal=json.loads((OUT/'equal_vram_summary.json').read_text());speed=next(x for x in equal['models'] if x['model']==k)
     macros['sentences_per_second']=speed['sentences_per_second'];macros['generated_tokens_per_second']=speed['generated_tokens_per_second']
     macros['truncated']=sum(r.get('truncated',False) for r in rows)
     save(sc/f'{k}.summary.json',{'macros':macros,'languages':lang,'rows':len(rows),'truncated':macros['truncated'],'dataset_sha256':metadata['dataset_sha256']})
     # exact Q8 words from prior run, or vLLM words from its original high-memory pass
     orig=QUANT/'q8_0/predictions.jsonl' if k=='q8_0' else RUNS/k/'predictions.jsonl'
     prior={(r['language'],str(r['id'])):r['hypothesis'] for r in read(orig)}
     exact=sum(r['hypothesis']==prior[(r['language'],str(r['id']))] for r in rows)
     save(sc/f'{k}.consistency.json',{'exact_hypotheses_vs_previous_run':exact,'rows':len(rows),'fraction':exact/len(rows)})
     macros_by[k]=macros;langs_by[k]=lang
    comparisons={}
    for k in SYSTEMS[1:]:
     a={(str(r['language']),str(r['id'])):float(r['comet']) for r in read(sc/f'{k}/neural_scores.jsonl')};b={(str(r['language']),str(r['id'])):float(r['comet']) for r in read(sc/'q8_0/neural_scores.jsonl')}
     comparisons[k]={m:paired_bootstrap(
      {(r['language'],str(r['id'])):float(r[m.lower()]) for r in read(sc/f'{k}/neural_scores.jsonl')},
      {(r['language'],str(r['id'])):float(r[m.lower()]) for r in read(sc/'q8_0/neural_scores.jsonl')},m) for m in ('COMET','MetricX','XCOMET')}
    summary={'systems':SYSTEMS,'q8_baseline':json.loads((OUT/'q8_memory_budget.json').read_text()),'macros':macros_by,'languages':langs_by,'paired_bootstrap':comparisons}
    save(OUT/'equal_vram_quality_summary.json',summary)
    names={'q8_0':'Q8_0 / llama.cpp','bf16_vllm':'BF16 / vLLM','fp8_vllm':'FP8 / vLLM','nvfp4_vllm':'NVFP4 / vLLM'}
    report=['# Q8_0 与 vLLM 在相同显存预算下的全量对照','','Q8_0 先以相同 128 并发、8,096 条 FLORES-101 devtest 测得整卡占用峰值；三个 vLLM 版本将 `gpu_memory_utilization` 单独校准至接近这一峰值。每组均为 8,096 条、八语各 1,012 条，0 截断。',f"新鲜神经指标精确评分 {len(unique):,} 个不同输入，并映射回全部 32,384 个译文记录。",'','| 方案 | 整卡峰值 MiB / GiB | 句/秒 | token/秒 | spBLEU | BLEU-13a (6语) | chrF++ | COMET | MetricX ↓ | XCOMET | 截断 | 对应旧批次译文完全一致 |','|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|']
    for k in SYSTEMS:
     m=macros_by[k];mem=summary['q8_baseline']['q8_observed_peak_mib'] if k=='q8_0' else next(x for x in json.loads((OUT/'equal_vram_summary.json').read_text())['models'] if x['model']==k)['peak_gpu_memory_mib'];cons=json.loads((sc/f'{k}.consistency.json').read_text());report.append(f"| {names[k]} | {mem} / {mem/1024:.2f} | {m['sentences_per_second']:.2f} | {m['generated_tokens_per_second']:.2f} | {m['spBLEU']:.3f} | {m['BLEU13a']:.3f} | {m['chrFpp']:.3f} | {m['COMET']:.6f} | {m['MetricX']:.6f} | {m['XCOMET']:.6f} | {m['truncated']} | {cons['exact_hypotheses_vs_previous_run']}/{cons['rows']} ({cons['fraction']:.1%}) |")
    report+=['','## 语种级结果','','质量分：spBLEU / chrF++ / COMET / MetricX / XCOMET。BLEU-13a 只覆盖六语；句/token吞吐为各系统全量速度。','']
    for k in SYSTEMS:
     report += [f"### {names[k]}",'','| 语种 | spBLEU | BLEU-13a | chrF++ | COMET | MetricX | XCOMET | 句/秒 | token/秒 |','|---|---:|---:|---:|---:|---:|---:|---:|---:|']
     for l in LANGS:
      v=langs_by[k][l];bv='—' if v['BLEU13a'] is None else f"{v['BLEU13a']:.3f}";ls=metadata['language_results'][l] if k=='q8_0' else None
      if k!='q8_0':
       rr=next(x for x in json.loads((OUT/'equal_vram_summary.json').read_text())['models'] if x['model']==k);mm=json.loads(((OUT/'q8_0' if k=='q8_0' else OUT/k)/'metadata.json').read_text());ls=mm['language_results'][l]
      report.append(f"| {l} | {v['spBLEU']:.3f} | {bv} | {v['chrFpp']:.3f} | {v['COMET']:.6f} | {v['MetricX']:.6f} | {v['XCOMET']:.6f} | {ls['sentences_per_second']:.2f} | {ls['generation_tokens_per_second']:.2f} |")
    report+=['','## 配对差值与 95% bootstrap 区间','','差值为 vLLM 减 Q8_0，以中文源句 ID 成对抽样，每次保留该 ID 对应的八语，10,000 次重采样，seed=20261022。神经指标的差值基于这次全量新鲜评分。','']
    for k in SYSTEMS[1:]:
     report += [f"### {names[k]} − Q8_0",'','| 指标 | 差值 | 95%区间 |','|---|---:|---:|']
     for m in ('COMET','MetricX','XCOMET'):
      v=comparisons[k][m];report.append(f"| {m} | {v['delta']:+.6f} | [{v['low']:+.6f}, {v['high']:+.6f}] |")
    report+=['','## 口径与限制','','- Q8_0 与三个 vLLM 版本均使用同一数据、提示、贪心解码和 512 token 上限，客户端并发均为 128。Q8_0 使用 llama.cpp；vLLM 每种格式按观测整卡峰值反向校准显存占用。','- BF16 vLLM 旧高显存批次与此低显存批次译文完全一致率见表；若译文有变化，本表质量分全部取新鲜完整评分，不沿用旧分数。','- BLEU-13a 为六语宏平均，其余质量指标为八语宏平均。指标不是正确率百分比；COMET/XCOMET 越高越好，MetricX 越低越好。配对区间覆盖 0 表示本次样本无法确认差值脱离 0。','- 这批 vLLM FP8/NVFP4 在 RTX 3090 上分别采用 Marlin W8A16/W4A16，实际核对以启动日志为准。GPU1 空闲时的峰值占用没有受其他进程影响。','',f"原始数据及神经评分映射保存在 `results/zh_translation_vllm_equal_q8_full_20261022/calibrated/scoring/`。"]
    (OUT/'EQUAL_MEMORY_COMPARISON.md').write_text('\n'.join(report)+'\n',encoding='utf-8')
    save(OUT/'equal_vram_quality_summary.json',summary)

def main():
 ap=argparse.ArgumentParser();ap.add_argument('--stage',choices=('prepare','classic','neural','collect','report','all'),required=True);a=ap.parse_args()
 if a.stage=='prepare':prepare()
 elif a.stage=='classic':run_classic()
 elif a.stage=='neural':score_neural()
 elif a.stage=='collect':collect_report()
 elif a.stage=='report':collect_report()
 else:prepare();run_classic();score_neural();collect_report()
if __name__=='__main__':main()
