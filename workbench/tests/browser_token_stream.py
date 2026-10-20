"""Verify real incremental column mutations, including pending refinement."""
import json
from pathlib import Path
from playwright.sync_api import sync_playwright
ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'results/token_stream_live'
OUT.mkdir(parents=True,exist_ok=True)
with sync_playwright() as p:
    browser=p.chromium.launch(headless=True,args=['--no-sandbox'])
    page=browser.new_page(viewport={'width':1440,'height':1050},accept_downloads=True)
    errors=[];page.on('pageerror',lambda e:errors.append(str(e)))
    page.goto('http://127.0.0.1:8083/',wait_until='networkidle')
    page.wait_for_function("!document.getElementById('translation-start').disabled")
    page.evaluate('''() => {
      window.displayFrames=[];
      new MutationObserver(() => window.displayFrames.push({
        time:performance.now(),source:document.getElementById('translation-source').textContent,
        translation:document.getElementById('translation-original').textContent,
        refined:document.getElementById('translation-clean').textContent,
        pending:!!document.querySelector('#translation-clean .pending-refinement'),
        status:document.getElementById('translation-status').textContent
      })).observe(document.querySelector('.results'),{subtree:true,childList:true,characterData:true});
    }''')
    page.locator('#translation-text-tab').click()
    source=('We need three hundred steel valves. Please deliver them by Friday. '
            'The unit price is twelve dollars. Please confirm the total amount. '
            'We will pay after receiving the invoice. Keep the iPhone samples in a separate box. '
            'The shipping address is Shanghai. Please include the packing list.')
    page.locator('#translation-input').fill(source)
    page.locator('#translation-start').click()
    page.wait_for_selector('#translation-clean .pending-refinement',timeout=30000)
    page.screenshot(path=str(OUT/'pending.png'),full_page=True)
    page.wait_for_function("document.getElementById('translation-status').textContent==='翻译与中文精修完成。'",timeout=120000)
    page.wait_for_function("!document.getElementById('translation-start').disabled")
    frames=page.evaluate('window.displayFrames')
    assert any(f['translation'] and f['pending'] for f in frames),frames
    assert len({f['translation'] for f in frames if f['translation']})>=2,frames
    assert not page.locator('#translation-clean .pending-refinement').count()
    assert page.locator('#translation-source').inner_text()==source
    page.screenshot(path=str(OUT/'final.png'),full_page=True)
    with page.expect_download() as download:page.locator('#translation-download').click()
    events=json.loads(Path(download.value.path()).read_text())
    translations=[e for e in events if e['event']=='translation']
    previews=[e for e in events if e['event']=='translation_preview']
    assert len(previews)>=2
    assert len(translations)>=2
    assert all('pending_raw_text' in e for e in translations)
    assert events[-1]['event']=='final'
    assert not events[-1]['has_pending_refinement']
    assert events[-1]['pipeline_stats']['max_pending_per_stage']==1
    assert not errors,errors
    (OUT/'events.json').write_text(json.dumps(events,ensure_ascii=False,indent=2))
    (OUT/'frames.json').write_text(json.dumps(frames,ensure_ascii=False,indent=2))
    summary={'real_model_test':'passed','intermediate_translations':len(translations),'token_previews':len(previews),
             'pending_text_visible_before_final':True,'final_pending_tail_removed':True,
             'pending_queue_limit':1,'browser_errors':errors}
    (OUT/'STATUS.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2))
    print(json.dumps(summary,ensure_ascii=False));browser.close()
