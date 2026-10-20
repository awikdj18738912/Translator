"""Run against the real GPU service; requires Playwright and Chromium."""
import json
from pathlib import Path
from playwright.sync_api import sync_playwright

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'results/browser_smoke'
OUT.mkdir(parents=True,exist_ok=True)
with sync_playwright() as p:
    browser=p.chromium.launch(headless=True,args=['--no-sandbox'])
    page=browser.new_page(viewport={'width':1440,'height':1050},accept_downloads=True)
    errors=[]
    page.on('pageerror',lambda e:errors.append(str(e)))
    page.goto('http://127.0.0.1:8083/',wait_until='networkidle')
    page.wait_for_function("!document.getElementById('translation-start').disabled",timeout=30000)
    assert page.locator('#translation-audio-tab').get_attribute('aria-selected')=='true'
    assert page.locator('#translation-text-view').is_hidden()
    boxes=page.locator('.result-card').all()
    assert len(boxes)==3
    positions=[item.bounding_box() for item in boxes]
    assert len({round(item['y']) for item in positions})==1,positions
    assert page.locator('#quality-tag').inner_text()=='等待精修'
    page.locator('#manage-entities').click()
    page.wait_for_function("document.getElementById('entity-notice').textContent.includes('已加载')")
    assert page.locator('#entity-rows').inner_text().find('iPhone')>=0
    page.locator('#add-entity').click()
    assert page.locator('#entity-domain').input_value()=='trade'
    assert page.locator('#entity-policy').input_value()=='normalize'
    page.locator('#cancel-entity').click()
    page.locator('#close-entities').click()
    page.locator('#translation-text-tab').click()
    page.locator('#translation-input').fill('We need three hundred steel valves. Please deliver them by Friday.')
    page.locator('#translation-start').click()
    page.wait_for_function("document.getElementById('translation-status').textContent==='翻译与中文精修完成。'",timeout=90000)
    page.wait_for_function("!document.getElementById('translation-start').disabled")
    text={key:page.locator('#'+key).inner_text() for key in ('translation-source','translation-original','translation-clean','quality-tag')}
    assert '300' in text['translation-clean'] or '三百' in text['translation-clean'],text
    page.locator('#translation-audio-tab').click()
    page.locator('#translation-audio-file').set_input_files(str(ROOT/'tests/fixtures/english_probe.wav'))
    page.locator('#translation-start').click()
    page.wait_for_function("document.getElementById('translation-status').textContent==='翻译与中文精修完成。'",timeout=90000)
    page.wait_for_function("!document.getElementById('translation-start').disabled")
    audio={key:page.locator('#'+key).inner_text() for key in ('translation-source','translation-original','translation-clean','quality-tag','translation-calls')}
    assert 'Friday' in audio['translation-source'],audio
    assert audio['translation-original'] and audio['translation-clean'],audio
    page.screenshot(path=str(OUT/'desktop.png'),full_page=True)
    with page.expect_download() as info:page.locator('#translation-download').click()
    events=json.loads(Path(info.value.path()).read_text())
    assert any(e.get('event')=='final' for e in events)
    (OUT/'audio_events.json').write_text(json.dumps(events,ensure_ascii=False,indent=2))
    page.set_viewport_size({'width':390,'height':844})
    page.screenshot(path=str(OUT/'mobile.png'),full_page=True)
    assert page.evaluate('document.documentElement.scrollWidth <= window.innerWidth'), 'Horizontal overflow'
    positions=[item.bounding_box() for item in boxes]
    assert positions[0]['y']<positions[1]['y']<positions[2]['y'],positions
    assert not errors,errors
    result={'text':text,'audio':audio,'three_columns_desktop':True,'stacked_mobile':True,'entity_dialog_defaults_verified':True,'download_events':len(events),'browser_errors':errors}
    (OUT/'STATUS.json').write_text(json.dumps(result,ensure_ascii=False,indent=2))
    print(json.dumps(result,ensure_ascii=False))
    browser.close()
