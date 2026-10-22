#!/usr/bin/env python3
"""Reuse sentence-identical XCOMET scores and prepare only changed inputs."""
import json
from pathlib import Path
from collections import Counter

ROOT=Path(__file__).resolve().parents[2]
OUT=Path(__file__).resolve().parent/'calibrated'
QUANT=ROOT/'results/zh_translation_quantized_full_20261021'
RUNS=ROOT/'results/zh_translation_backends_full_20261021'
SYSTEMS=('q8_0','bf16_vllm','fp8_vllm','nvfp4_vllm')

def read(p):return [json.loads(x) for x in Path(p).open(encoding='utf-8') if x.strip()]

def identity(row):return (row['source'],row['hypothesis'],row['reference'],row['language'])

def main():
  cache={}
  cached_values={}
  previous={}
  for key in SYSTEMS:
    if key=='q8_0':
      pred_path=QUANT/'q8_0/predictions.jsonl'
      scores_path=QUANT/'scoring/xcomet/index.scores.jsonl'
      score_field='xcomet_score'
      span_field='error_spans'
      score_rows=read(scores_path)
      score_map={(r['language'],str(r['id'])):r for r in score_rows}
    else:
      pred_path=RUNS/key/'predictions.jsonl'
      score_path=RUNS/'scoring'/key/'neural_scores.jsonl'
      span_path=RUNS/'scoring'/key/'xcomet_error_spans.jsonl'
      score_rows=read(score_path);span_rows=read(span_path)
      span_map={(r['language'],str(r['id'])):r for r in span_rows}
      score_map={(r['language'],str(r['id'])):r for r in score_rows}
      score_field='xcomet';span_field='error_spans'
    pred_rows=read(pred_path)
    previous[key]={(r['language'],str(r['id'])):r for r in pred_rows}
    for row in pred_rows:
      ident=(row['language'],str(row['id']))
      scored=score_map[ident]
      score=float(scored[score_field])
      spans=(scored.get(span_field,[]) if key=='q8_0' else span_map[ident].get('error_spans',[]))
      item=identity(row)
      cached_values.setdefault(item,[]).append({'score':score,'error_spans':spans or []})
  inconsistent={item for item,values in cached_values.items() if max(v['score'] for v in values)-min(v['score'] for v in values)>1e-5}
  cache={item:values[0] for item,values in cached_values.items() if item not in inconsistent}
  mapping={};unique=[];uid_by_identity={};counts=Counter()
  for key in SYSTEMS:
    current=read(OUT/('q8_0' if key=='q8_0' else key)/'predictions.jsonl')
    pairs=[]
    for row in current:
      ident=(row['language'],str(row['id']))
      prior=previous[key][ident]
      full_id=identity(row)
      if full_id not in inconsistent and full_id in cache:
        if full_id==identity(prior):continue
      counts[key]+=1
      if full_id not in cache and full_id not in uid_by_identity:
        uid=len(unique);uid_by_identity[full_id]=uid
        unique.append({'unique_id':uid,'language':row['language'],'source':row['source'],
          'hypothesis':row['hypothesis'],'reference':row['reference']})
      if full_id in uid_by_identity:pairs.append({'id':row['id'],'language':row['language'],'unique_id':uid_by_identity[full_id]})
    mapping[key]=pairs
  sc=OUT/'scoring';sc.mkdir(parents=True,exist_ok=True)
  with (sc/'xcomet_changed_inputs.jsonl').open('w',encoding='utf-8') as f:
    for row in unique:f.write(json.dumps(row,ensure_ascii=False)+'\n')
  Path(sc/'xcomet_changed_mapping.json').write_text(json.dumps(mapping,ensure_ascii=False,indent=2)+'\n')
  Path(sc/'xcomet_previous_exact_cache.json').write_text(json.dumps({json.dumps(k,ensure_ascii=False):v for k,v in cache.items()},ensure_ascii=False)+'\n')
  print(json.dumps({'cached_exact_inputs':len(cache),'changed_rows_by_system':dict(counts),
    'changed_unique_inputs':len(unique),'inconsistent_cached_inputs_rescored':len(inconsistent),'total_rows':len(SYSTEMS)*8096},ensure_ascii=False),flush=True)
if __name__=='__main__':main()
