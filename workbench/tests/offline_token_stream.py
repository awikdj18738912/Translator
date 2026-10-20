"""Real LMT and in-process WebSockets; no HTTP server, ASR or Refiner loaded."""
import json,sys,time,statistics
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from fastapi import FastAPI
from fastapi.testclient import TestClient
from lmt_runtime import LmtRuntime
from system.translation_web import install_translation_routes
from system.translation_windows import StableTranslationSnapshots
from system.window_refinement import StreamingRefinementDisplay

OUT=ROOT/'results/token_stream_offline'
OUT.mkdir(exist_ok=True)
runtime=LmtRuntime('/home/aim0/data/models/ASR/LMT-60-0.6B',1)
def quality(text,language):
    rows,_=runtime.translate([text],language,beams=3,max_new_tokens=512)
    assert rows[0]['eos'];return rows[0]['hypothesis']
def stream(text,language,on_text):
    rows,_=runtime.translate_stream(text,language,on_text)
    assert rows[0]['eos'];return rows[0]['hypothesis']
quality('Please confirm the delivery date.','en')
stream('Please confirm the delivery date.','en',lambda text:None)
class Session:
    def __init__(self):self.display=StreamingRefinementDisplay();self.model_calls=[]
    def update(self,text,*,revision,final=False):
        result={'raw_text':text,'clean_text':text,'refinement_source_spans':[
            {'source_text':text,'clean_text':text,'state':'committed' if final else 'active'}]}
        self.display.accept(result,revision);return result
app=FastAPI();app.state.new_text_refinement_session=lambda domain:Session()
install_translation_routes(app,quality,asr_url='http://asr.invalid',output_dir=OUT/'sessions',stream_translate=stream)
all_samples=json.loads((ROOT/'results/streaming_decode_benchmark/samples.json').read_text())
cases=[]
for language in ('en','fr','de','es','ru','ja','ko','ar'):
    cases.append(next(sample for sample in all_samples if sample['language']==language))
recorded=[json.loads(line) for line in (ROOT/'data/sessions/853ed4352c4f47528f0a4b29bd22dd1b.jsonl').read_text().splitlines()]
cases.append({'language':'en','source':[e['source_text'] for e in recorded if e['event']=='source'][-1],'case_id':'recorded_long'})
results=[]
for sample in cases:
    text=sample['source'];language=sample['language']
    baseline,_=StableTranslationSnapshots(quality,max_chars=240,group_sentences=1).update(text,language)
    events=[]
    with TestClient(app).websocket_connect('/ws/translation') as ws:
        ws.send_json({'event':'start','language':language,'text':text,'chunk_chars':min(500,max(4,len(text)//2)),'interval_ms':700})
        while True:
            event=ws.receive_json();events.append(event)
            assert event['event']!='error',event
            if event['event']=='final':break
    previews=[e for e in events if e['event']=='translation_preview' and e['translation'].strip()]
    assert previews,language
    assert all('\ufffd' not in e['translation'] for e in previews)
    revision=previews[0]['revision']
    source=next(e for e in events if e['event']=='source' and e['revision']==revision)
    complete=next(e for e in events if e['event']=='translation' and e['revision']==revision)
    first_ms=(previews[0]['server_time']-source['server_time'])*1000
    complete_ms=(complete['server_time']-source['server_time'])*1000
    assert first_ms<complete_ms,(language,first_ms,complete_ms)
    assert events[-1]['translation']==baseline,language
    result={'language':language,'previews':len(previews),'first_visible_from_source_ms':round(first_ms,2),'complete_snapshot_from_source_ms':round(complete_ms,2),'final_equal_original_beam3':True}
    results.append(result)
    (OUT/f"{sample.get('case_id',language)}.events.json").write_text(json.dumps(events,ensure_ascii=False,indent=2))
summary={'test':'passed','gpu_models_loaded':['LMT-60-0.6B'],'refiner':'test double','network_services_started':False,'cases':results,'mean_first_visible_ms':statistics.mean(r['first_visible_from_source_ms'] for r in results),'mean_complete_snapshot_ms':statistics.mean(r['complete_snapshot_from_source_ms'] for r in results),'final_beam3_exact_match':len(results)}
(OUT/'STATUS.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2));print(json.dumps(summary,ensure_ascii=False))
