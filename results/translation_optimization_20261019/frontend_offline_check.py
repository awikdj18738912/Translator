"""Exercise the real frontend JS with fake browser transports, no service."""
import json,re
from pathlib import Path
from playwright.sync_api import sync_playwright
root=Path(__file__).resolve().parents[2]
web=root.parent/'Codex-web/system/web'
html=re.sub(r'<script\b[^>]*>[\s\S]*?</script>','',(web/'index.html').read_text(),flags=re.I)
with sync_playwright() as p:
 browser=p.chromium.launch(headless=True,args=['--no-sandbox'])
 page=browser.new_page(accept_downloads=True)
 errors=[]
 page.on('pageerror',lambda error:errors.append(str(error)))
 page.route('http://translation.test/**',lambda route:route.fulfill(status=200,content_type='text/html',body=html))
 page.goto('http://translation.test/',wait_until='domcontentloaded')
 page.evaluate('''() => {
 window.fetch=async()=>({ok:true,json:async()=>({ready:true,audio_ready:true,languages:[{code:'en',name:'英语'}]})});
 window.fakeSockets=[];
 window.WebSocket=class extends EventTarget {
  static CLOSING=2;
  constructor(){super();this.readyState=0;window.fakeSockets.push(this);setTimeout(()=>{this.readyState=1;this.dispatchEvent(new Event('open'));},0);}
  send(x){this.lastSent=JSON.parse(x);}
  close(){this.readyState=3;this.dispatchEvent(new Event('close'));}
 };
 window.feed=x=>window.fakeSockets.at(-1).dispatchEvent(new MessageEvent('message',{data:JSON.stringify(x)}));
}''')
 page.add_script_tag(path=str(web/'translation.js'))
 page.wait_for_function("!document.getElementById('translation-start').disabled")
 page.locator('#translation-input').fill('test input')
 page.locator('#translation-start').click()
 page.wait_for_function('window.fakeSockets.length===1 && window.fakeSockets[0].lastSent')
 def feed(data):page.evaluate('(x)=>window.feed(x)',data)
 feed({'event':'source','revision':1,'source_text':'old source'})
 feed({'event':'source','revision':2,'source_text':'new source'})
 feed({'event':'translation','revision':1,'translation':'STALE TRANSLATION'})
 feed({'event':'final','revision':1,'clean_text':'STALE FINAL'})
 assert 'STALE' not in page.locator('#translation-original').inner_text()
 assert 'STALE' not in page.locator('#translation-clean').inner_text()
 assert page.locator('#translation-start').is_disabled()
 feed({'event':'translation','revision':2,'translation':'最新译文'})
 for i in range(100):feed({'event':'progress','marker':i})
 feed({'event':'final','revision':2,'clean_text':'最新精修','source_text':'new source','translation':'最新译文'})
 page.evaluate('window.fakeSockets.at(-1).close()')
 page.wait_for_function("!document.getElementById('translation-start').disabled")
 page.wait_for_function("document.getElementById('translation-log').textContent.includes('仅展示最近60条')")
 assert page.locator('#translation-clean').inner_text()=='最新精修'
 assert page.locator('#translation-original').inner_text()=='最新译文'
 lines=page.locator('#translation-log').inner_text().splitlines()
 assert len(lines)==61,len(lines)
 with page.expect_download() as info:
  page.locator('#translation-download').click()
 download=info.value
 records=json.loads(Path(download.path()).read_text())
 assert len(records)==106,len(records)
 assert not errors,errors
 result={'status':'passed','service_started':False,'stale_translation_and_final_ignored':True,
         'visible_log_events':60,'download_log_events':len(records),'browser_errors':errors}
 (Path(__file__).parent/'frontend_offline_check.json').write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n')
 print(json.dumps(result,ensure_ascii=False))
 browser.close()
