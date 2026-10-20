"""Replay foreign text snapshots through GPU LMT and the real Codex-web pipeline."""
import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import sqlite3
import sys
from time import perf_counter, sleep

from lmt_runtime import LmtRuntime
from benchmark_streaming_mt import gpu_snapshot, process_gpu_memory, percentiles

ROOT=Path(__file__).resolve().parents[1]


def source_updates(case):
    if 'prefixes' in case:
        return case['prefixes']
    source,updates='',[]
    for chunk in case['chunks']:
        source+=chunk
        updates.append(source)
    return updates


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--codexweb-root',type=Path,default=ROOT.parent/'Codex-web')
    parser.add_argument('--translation-model',type=Path,default=Path('/home/aim0/data/models/ASR/LMT-60-0.6B'))
    parser.add_argument('--refiner-model',type=Path,default=Path('/home/aim0/data/models/ASR/AgenticASR-Refiner'))
    parser.add_argument('--gpu',type=int,default=1)
    parser.add_argument('--cases',type=Path,default=ROOT/'examples/streaming_mt_cases.json')
    parser.add_argument('--case-ids',default='en_order,fr_order,de_order,es_order,ru_order,ja_order,ko_order,ar_order,en_correction,en_asr_revision')
    parser.add_argument('--entity-db',type=Path,help='Copy an existing Codex-web SQLite entity database into this run')
    parser.add_argument('--interval',type=float,default=.5)
    parser.add_argument('--output',type=Path,default=ROOT/'results/codexweb_translation_replay_20261019')
    parser.add_argument('--max-new-tokens',type=int,default=256)
    args=parser.parse_args()
    if args.interval<0:
        parser.error('interval cannot be negative')
    sys.path.insert(0,str(args.codexweb_root.resolve()))
    from system.web_app import create_app
    from system.entity_store import EntityStore
    if (args.output/'events.jsonl').exists():
        raise FileExistsError('Run output already exists; use another --output directory')
    args.output.mkdir(parents=True,exist_ok=True)
    db=args.output/'entities.db'
    if args.entity_db:
        if not args.entity_db.is_file():
            raise FileNotFoundError(args.entity_db)
        with sqlite3.connect(args.entity_db.resolve().as_uri()+'?mode=ro',uri=True) as src,sqlite3.connect(db) as dst:
            src.backup(dst)
    store=EntityStore(db)
    # Demo rules only; never add domain assumptions to the user's original DB.
    for canonical,aliases,kind in [
        ('iPhone',('iphone','Iphone'),'PRODUCT'),('SUS304',(),'MODEL'),
        ('AcmeFlow',(),'BRAND'),('阀门',('阀門',),'TERM'),('不锈钢',('不鏽鋼','不銹鋼'),'TERM')]:
        if not args.entity_db:
            store.upsert_entity(canonical,aliases=aliases,entity_type=kind,domain='trade',normalization_policy='normalize')
    definitions=store.list_entities(domain='trade')
    snapshot=[{'canonical_text':d.canonical_text,'aliases':d.aliases,'entity_type':d.entity_type,
               'domain':d.domain,'normalization_policy':d.normalization_policy} for d in definitions]
    manifest=json.loads((args.translation_model/'DOWNLOAD_MANIFEST.json').read_text())
    metadata={'created_at':datetime.now(timezone.utc).isoformat(),'status':'running',
        'scope':'Offline text snapshot replay only; no audio, ASR model or browser/audio WebSocket integration',
        'codexweb_root':str(args.codexweb_root.resolve()),'translation_model':str(args.translation_model),
        'translation_revision':manifest['revision'],'translation_compute_type':'int8_float16',
        'translation_beam':3,'gpu':args.gpu,'refiner_model':str(args.refiner_model),
        'refiner_device':f'cuda:{args.gpu}','refiner_pipeline':'Codex-web create_app.refine_update + fixed_k3_context_v2',
        'gate':'off (exercise actual refiner on every ready window)','entity_fuzzy_mode':'off (verified exact rules only)',
        'entity_db_source':str(args.entity_db) if args.entity_db else 'isolated trade demo',
        'entity_snapshot':snapshot,'input_interval_seconds':args.interval,
        'backpressure':'Sequential, wait for each update; no unbounded task queue',
        'gpu_start':gpu_snapshot(),'codexweb_source_sha256':{
            name:hashlib.sha256((args.codexweb_root/'system'/name).read_bytes()).hexdigest()
            for name in ('web_app.py','offline_text_refinement.py','window_refinement.py','live_qwen_refiner.py')}}
    (args.output/'metadata.json').write_text(json.dumps(metadata,ensure_ascii=False,indent=2)+'\n')
    print(json.dumps({'event':'loading','gpu':args.gpu}),flush=True)
    app=create_app(args.refiner_model,f'cuda:{args.gpu}','http://127.0.0.1:1','Chinese',
        args.max_new_tokens,None,entity_db=db,entity_fuzzy_mode='off',refinement_gate_mode='off',
        entity_audio_repair=False,entity_asr_context=False,entity_ctc_mode='off')
    lmt=LmtRuntime(args.translation_model,args.gpu)
    import ctranslate2,torch
    metadata.update(ctranslate2=ctranslate2.__version__,torch=torch.__version__,
        gpu_process_loaded_mib=process_gpu_memory())
    warm,_=lmt.translate(['Please confirm the delivery date.'],'en',beams=3)
    if not warm[0]['eos']:
        raise RuntimeError('Translation warmup did not terminate')
    all_cases=json.loads(args.cases.read_text())['cases']
    ids=set(args.case_ids.split(','))
    cases=[r for r in all_cases if r['id'] in ids]
    if len(cases)!=len(ids):
        raise ValueError('Unknown or duplicate case IDs')
    # One English entity case and a multi-sentence stream to exercise ready
    # K=3 groups before final flush, rather than refining only at EOF.
    if args.case_ids==parser.get_default('case_ids'):
        cases.extend([
            {'id':'en_entity','language':'en','chunks':['We need 100 iphone 15 Pro Max units',
                ', 256GB each',', from AcmeFlow',', with delivery by Friday.'],
             'reference':'我们需要100台iPhone 15 Pro Max，每台256GB，来自AcmeFlow，周五前交货。'},
            {'id':'en_long_window','language':'en','chunks':[
                'We need 500 SUS304 valves. The unit price is 12.50 US dollars. Delivery is due by November 15. ',
                'Please confirm the order. The valves are made of stainless steel. ',
                'Please send the invoice to AcmeFlow. Do not ship before receiving the deposit.'],
             'reference':'我们需要500个SUS304阀门。单价12.50美元。11月15日前交货。请确认订单。阀门由不锈钢制成。请将发票发送给AcmeFlow。收到定金前不要发货。'}])
    metadata['cases_sha256']=hashlib.sha256(args.cases.read_bytes()).hexdigest()
    events,finals,traces=[],[],[]
    started=perf_counter()
    with (args.output/'events.jsonl').open('w') as file:
        for case in cases:
            session=app.state.new_text_refinement_session('trade')
            updates=source_updates(case)
            case_started=perf_counter()
            for index,source in enumerate(updates):
                sleep(max(0,case_started+index*args.interval-perf_counter()))
                begin=perf_counter()
                prediction,translation_seconds=lmt.translate([source],case['language'],beams=3,max_new_tokens=256)
                translated=prediction[0]['hypothesis']
                if not prediction[0]['eos'] or not translated.strip():
                    raise RuntimeError(f'Translation incomplete: {case["id"]}/{index}')
                before=len(session.model_calls)
                refine_started=perf_counter()
                result=session.update(translated,revision=index+1,final=index==len(updates)-1)
                refinement_seconds=perf_counter()-refine_started
                record={'case_id':case['id'],'language':case['language'],'step':index,
                    'revision':index+1,'final':index==len(updates)-1,'source':source,
                    'translation':translated,'clean_text':result['clean_text'],
                    'translation_seconds':translation_seconds,'refinement_update_seconds':refinement_seconds,
                    'update_seconds':perf_counter()-begin,'actual_call_traces':len(session.model_calls)-before,
                    'new_refiner_model_calls':sum(len(t.get('model_calls',[])) for t in session.model_calls[before:]),
                    'refinement':result}
                events.append(record)
                file.write(json.dumps(record,ensure_ascii=False)+'\n');file.flush()
                print(json.dumps({'event':'update','case_id':case['id'],'step':index,
                    'final':record['final'],'translation':translated,'clean_text':record['clean_text'],
                    'refiner_executed':result.get('refiner_executed'),
                    'refiner_accepted':result.get('refiner_accepted')},ensure_ascii=False),flush=True)
            finals.append({**record,'reference':case['reference']})
            traces.extend({'case_id':case['id'],**r} for r in session.model_calls)
    metadata.update(status='completed',cases=len(cases),updates=len(events),
        total_seconds=perf_counter()-started,gpu_end=gpu_snapshot(),
        gpu_process_end_mib=process_gpu_memory(),
        update_latency_seconds=percentiles([e['update_seconds'] for e in events]),
        nonfinal_refiner_updates=sum(not e['final'] and e['refinement'].get('refiner_executed',False) for e in events),
        final_accepted=sum(e['refinement'].get('refiner_accepted',False) for e in finals))
    for filename,data in [('metadata.json',metadata),('finals.json',finals),('refiner_traces.json',traces)]:
        (args.output/filename).write_text(json.dumps(data,ensure_ascii=False,indent=2)+'\n')
    lines=['# LMT → Codex-web 离线文本流式联调','','真实GPU模型：LMT INT8/FP16翻译 + Codex-web原始Refiner、固定K=3分窗、实体保护与守卫。',
        '输入是外语文本快照回放，没有运行音频转录/麦克风/WebSocket音频链路。每次重译的中文完整快照替换上一版，不追加旧译文。',
        '词库为本次独立副本/演示库；精修只看到中文，不能保证修复上游错译。','',
        f"{len(cases)}个场景、{len(events)}次更新；中间更新实际进入精修的次数：{metadata['nonfinal_refiner_updates']}。最终守卫接受{metadata['final_accepted']}/{len(cases)}，其余保留回退结果；接受不等于业务事实正确。",'',
        '| 场景 | 最终外语原文 | 翻译中文 | 精修输出 | 精修接受 |','|---|---|---|---|---|']
    for r in finals:
        vals=[r['case_id'],r['source'],r['translation'],r['clean_text'],str(r['refinement'].get('refiner_accepted'))]
        lines.append('| '+' | '.join(str(x).replace('|','\\|').replace('\n',' ') for x in vals)+' |')
    lines+=['','逐步事件：events.jsonl；模型输入输出及补丁痕迹：refiner_traces.json；配置和GPU占用：metadata.json。']
    (args.output/'REPORT.md').write_text('\n'.join(lines)+'\n')
    print(json.dumps({'event':'complete','output':str(args.output),'cases':len(cases),'updates':len(events)}),flush=True)


if __name__=='__main__':
    main()
