"""Render saved real-model events in an intercepted browser; no HTTP service."""
import json,time
from pathlib import Path
from playwright.sync_api import sync_playwright
ROOT=Path(__file__).resolve().parents[1]
WEB=ROOT/'system/web'
OUT=ROOT/'results/adaptive_workbench_offline'
events=json.loads((OUT/'events.json').read_text())
with sync_playwright() as p:
    browser=p.chromium.launch(headless=True,args=['--no-sandbox'])
    page=browser.new_page(viewport={'width':1440,'height':1100})
    errors=[];page.on('pageerror',lambda e:errors.append(str(e)))
    page.add_init_script('''window.WebSocket=class extends EventTarget{
      static CLOSING=2;
      constructor(){super();this.readyState=0;window.replaySocket=this;
        setTimeout(()=>{this.readyState=1;this.dispatchEvent(new Event('open'))},0);}
      send(){} close(){this.readyState=3;this.dispatchEvent(new Event('close'));}
    };''')
    def route(request):
        path=request.request.url.split('offline.workbench')[-1].split('?')[0]
        if path=='/api/translation/health':
            request.fulfill(json={'ready':True,'audio_ready':False,'refinement_gate_mode':'regex','languages':[{'code':'en','name':'英语'}]});return
        file=WEB/('index.html' if path=='/' else path.removeprefix('/static/'))
        request.fulfill(path=str(file))
    page.route('http://offline.workbench/**',route)
    page.goto('http://offline.workbench/',wait_until='networkidle')
    page.locator('#translation-text-tab').click()
    page.locator('#translation-start').click()
    page.wait_for_function('window.replaySocket.readyState===1')
    start=time.monotonic();first=events[0]['server_time']
    for event in events:
        delay=event['server_time']-first-(time.monotonic()-start)
        if delay>0:page.wait_for_timeout(delay*1000)
        page.evaluate('(e)=>window.replaySocket.dispatchEvent(new MessageEvent("message",{data:JSON.stringify(e)}))',event)
    page.wait_for_timeout(50)
    assert page.locator('#translation-calls').inner_text()=='6'
    assert page.locator('#translation-gate-skips').inner_text()=='10'
    assert 'S T A R S K I' in page.locator('#translation-original').inner_text()
    assert 'S T A R S K I' in page.locator('#translation-clean').inner_text()
    page.evaluate("document.querySelector('.subhead').textContent='离线真实模型事件回放；整体用时是本次浏览器回放用时'")
    page.screenshot(path=str(OUT/'desktop_metrics.png'),full_page=True)
    page.set_viewport_size({'width':390,'height':844})
    assert page.evaluate('document.documentElement.scrollWidth<=window.innerWidth')
    page.screenshot(path=str(OUT/'mobile_metrics.png'),full_page=True)
    assert not errors,errors
    print(json.dumps({'passed':True,'events_from_real_model_offline_test':True,'services_started':False,'browser_errors':errors}))
    browser.close()
