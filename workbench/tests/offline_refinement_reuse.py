"""Exercise real translation + refiner via in-process ASGI, without a server."""
import json,sys,time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from fastapi.testclient import TestClient
from app import create_workbench
OUT=ROOT/'results/refinement_reuse_offline';OUT.mkdir(exist_ok=True)
app=create_workbench(translation_model=Path('/home/aim0/data/models/ASR/LMT-60-0.6B'),refiner_model=Path('/home/aim0/data/models/ASR/AgenticASR-Refiner'),gpu=1,asr_url='http://127.0.0.1:1',data_dir=OUT/'data')
source_events=[json.loads(x) for x in (ROOT/'data/sessions/5e3c6b7fc8104280ab006b5668efaf83.jsonl').read_text().splitlines()]
source=[e['source_text'] for e in source_events if e['event']=='source'][-1]
es=[];start=time.monotonic()
with TestClient(app).websocket_connect('/ws/translation') as ws:
    ws.send_json({'event':'start','language':'en','text':source,'domain':'trade','chunk_chars':80,'interval_ms':500})
    while True:
        e=ws.receive_json();es.append(e)
        assert e['event']!='error',e
        if e['event']=='final':break
(OUT/'events.json').write_text(json.dumps(es,ensure_ascii=False,indent=2))
progress=[e for e in es if e['event']=='refinement_progress']
translations=[e for e in es if e['event']=='translation']
assert progress,'No completed group progress'
assert all('\n' not in e['translation'] for e in translations),'Artificial or model newline'
assert not es[-1]['has_pending_refinement']
final=es[-1]
summary={'passed':True,'elapsed_seconds':round(time.monotonic()-start,2),'group_progress_events':len(progress),'final_model_calls':final['model_calls'],'total_model_calls':final['total_model_calls'],'final_translation_model_calls':final['translation_stats']['model_calls'],'final_refinement_reprocessed_groups':final.get('window_cache_reprocessed_group_count'),'final_refinement_reused_groups':final.get('window_cache_reused_group_count'),'final_refinement_ms':round(final['refinement_ms']),'services_started':False,'original_audio_asr_rerun':False}
(OUT/'STATUS.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2));print(json.dumps(summary,ensure_ascii=False))
