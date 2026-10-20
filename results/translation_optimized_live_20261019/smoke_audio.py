import asyncio, json, time
from pathlib import Path
import httpx, websockets

ROOT=Path('/home/aim0/桌面/tan/ASR/Translator')
OUT=ROOT/'results/translation_optimized_live_20261019'
async def main():
    with httpx.Client(timeout=30) as client:
        health=client.get('http://127.0.0.1:8081/api/translation/health').json()
        (OUT/'health.json').write_text(json.dumps(health,ensure_ascii=False,indent=2))
        with (ROOT/'results/codexweb_translation_frontend_20261019/english_probe.wav').open('rb') as audio:
            response=client.post('http://127.0.0.1:8081/api/translation/audio',files={'file':('english_probe.wav',audio,'audio/wav')},data={'language':'en','domain':'trade','chunk_seconds':'2','interval_ms':'0'})
        response.raise_for_status()
        job=response.json()
    started=time.monotonic()
    events=[]
    async with websockets.connect('ws://127.0.0.1:8081'+job['websocket'],open_timeout=20,max_size=10000000) as ws:
        async for message in ws:
            event=json.loads(message)
            event['_elapsed_seconds']=round(time.monotonic()-started,3)
            events.append(event)
            (OUT/'audio_events.jsonl').write_text(''.join(json.dumps(e,ensure_ascii=False)+'\n' for e in events))
            if event.get('event') in ('error','final'):
                print(json.dumps(event,ensure_ascii=False),flush=True)
                if event['event']=='error': raise RuntimeError(event)
                return
    raise RuntimeError('WebSocket closed without final')
asyncio.run(asyncio.wait_for(main(),timeout=180))
