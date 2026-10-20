"""Replay the latest failed recording's ASR text through real GPU text stages."""
import asyncio,json,time
from pathlib import Path
import websockets
ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'results/refiner_stall_replay'
OUT.mkdir(exist_ok=True)
original=[json.loads(line) for line in (ROOT/'data/sessions/cb56d7893c0342dfb6546b8ea8cf2cce.jsonl').read_text().splitlines()]
source=[e['source_text'] for e in original if e['event']=='source'][-1]
async def main():
    events=[];started=time.monotonic()
    async with websockets.connect('ws://127.0.0.1:8083/ws/translation',max_size=10000000) as ws:
        await ws.send(json.dumps({'event':'start','language':'en','domain':'trade','text':source,'chunk_chars':80,'interval_ms':500}))
        async for message in ws:
            event=json.loads(message);event['client_elapsed']=time.monotonic()-started;events.append(event)
            (OUT/'events.jsonl').write_text(''.join(json.dumps(e,ensure_ascii=False)+'\n' for e in events))
            if event['event']=='error':raise RuntimeError(event)
            if event['event']=='final':break
    assert events[-1]['event']=='final'
    updates=[e for e in events if e['event']=='update']
    visible=[e for e in updates if len(e.get('display_refined_text',''))>30]
    assert visible, 'No substantial intermediate refined prefix became visible'
    assert not events[-1]['has_pending_refinement']
    assert events[-1]['pipeline_stats']['max_pending_per_stage']==1
    summary={'result':'passed','source_characters':len(source),'elapsed_seconds':round(time.monotonic()-started,2),'updates':len(updates),'intermediate_refined_prefix_lengths':[len(e.get('display_refined_text','')) for e in updates],'final_characters':len(events[-1]['clean_text']),'final_pending':events[-1]['has_pending_refinement']}
    (OUT/'STATUS.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2));print(json.dumps(summary,ensure_ascii=False))
asyncio.run(asyncio.wait_for(main(),180))
