import json
from pathlib import Path
from playwright.sync_api import sync_playwright
out=Path(__file__).resolve().parent
with sync_playwright() as p:
 browser=p.chromium.launch(headless=True,args=['--no-sandbox'])
 page=browser.new_page(viewport={'width':1440,'height':1200})
 errors=[]
 page.on('pageerror',lambda error:errors.append(str(error)))
 page.route('https://unpkg.com/**',lambda route:route.abort())
 page.goto('http://127.0.0.1:8081',wait_until='domcontentloaded')
 page.wait_for_function("!document.getElementById('translation-start').disabled",timeout=30000)
 page.locator('#translation-input').fill('We need 100 iPhone units. The price is 12.50 dollars each. Please deliver by Friday.')
 page.locator('#translation-start').click()
 page.wait_for_function("document.getElementById('translation-status').textContent.includes('完成')",timeout=120000)
 page.wait_for_function("!document.getElementById('translation-start').disabled")
 def state():
  return {key:page.locator('#'+key).inner_text() for key in ['translation-status','translation-source','translation-original','translation-clean','translation-calls','translation-guard']}
 text=state();assert 'iPhone' in text['translation-clean'] and '100' in text['translation-clean']
 page.locator('#translation-panel').screenshot(path=str(out/'browser_text.png'))
 (out/'browser_text.json').write_text(json.dumps({**text,'page_errors':errors},ensure_ascii=False,indent=2)+'\n')
 print('TEXT_OK '+json.dumps(text,ensure_ascii=False),flush=True)
 page.wait_for_function("!document.getElementById('translation-audio-tab').disabled",timeout=360000)
 page.locator('#translation-audio-tab').click()
 page.locator('#translation-audio-file').set_input_files(str(out/'english_probe.wav'))
 page.locator('#translation-start').click()
 page.wait_for_function("document.getElementById('translation-status').textContent.includes('完成') || document.getElementById('translation-status').classList.contains('error')",timeout=180000)
 audio=state()
 assert '完成' in audio['translation-status'],audio
 assert audio['translation-source'].strip(),audio
 assert any('\u3400'<=c<='\u9fff' for c in audio['translation-clean']),audio
 assert not errors,errors
 page.locator('#translation-panel').screenshot(path=str(out/'browser_audio.png'))
 (out/'browser_audio.json').write_text(json.dumps({**audio,'page_errors':errors},ensure_ascii=False,indent=2)+'\n')
 (out/'browser_audio_events.jsonl').write_text(page.locator('#translation-log').inner_text()+'\n')
 print('AUDIO_OK '+json.dumps(audio,ensure_ascii=False),flush=True)
 browser.close()
