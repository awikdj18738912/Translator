"""Exercise the live Chinese-first workbench, including actual audio ASR."""
import json,re
from pathlib import Path
from playwright.sync_api import sync_playwright
OUT=Path(__file__).resolve().parent
SOURCE='嗯，我们需要三百个不锈钢阀门。单价是十二点五美元。请在星期五之前发货。'
LANGUAGES=['en','fr','de','es','ru','ja','ko','ar']
SCRIPT={'ru':r'[\u0400-\u04FF]','ja':r'[\u3040-\u30FF]','ko':r'[\uAC00-\uD7AF]','ar':r'[\u0600-\u06FF]'}
with sync_playwright() as p:
    browser=p.chromium.launch(headless=True,args=['--no-sandbox'])
    page=browser.new_page(viewport={'width':1440,'height':1050},accept_downloads=True)
    errors=[];page.on('pageerror',lambda error:errors.append(str(error)))
    page.goto('http://127.0.0.1:8083/',wait_until='networkidle')
    page.wait_for_function("!document.getElementById('translation-start').disabled",timeout=30000)
    assert page.locator('.results .result-card h2').all_text_contents()==['中文转录','中文精修','外语翻译']
    assert page.locator('.session-summary > div').count()==5
    records=[]
    def collect(name):
        page.wait_for_function("document.getElementById('translation-status').textContent==='中文精修与外语翻译完成。'",timeout=120000)
        page.wait_for_function("!document.getElementById('translation-start').disabled")
        with page.expect_download() as info:page.locator('#translation-download').click()
        events=json.loads(Path(info.value.path()).read_text())
        assert events[-1]['event']=='final',events[-1]
        final=events[-1]
        assert final['translation_source_text']==final['clean_text'],final
        assert final['source_language']=='zh'
        assert final['pipeline_stats']['max_pending_per_stage']==1
        allowed=set()
        for event in events:
            if event.get('event') in {'refined','refinement_progress'}:
                allowed.add(event.get('committed_clean_text',''))
            if event.get('event') in {'translation','translation_preview','final'} and event.get('translation_source_text'):
                assert event['translation_source_text'] in allowed,(name,event)
        values={key:page.locator('#'+key).inner_text() for key in ['translation-source','translation-clean','translation-original','translation-calls','translation-gate-skips','translation-asr-time','translation-total-time']}
        assert values['translation-original']==final['translation'],(values,final['translation'])
        assert values['translation-clean']==final['clean_text'],(values,final['clean_text'])
        record={'name':name,'target_language':final['target_language'],'values':values,'final':final,
                'previews':sum(x['event']=='translation_preview' for x in events),
                'event_count':len(events)}
        (OUT/(name+'_events.json')).write_text(json.dumps(events,ensure_ascii=False,indent=2))
        records.append(record)
        print(json.dumps({'name':name,'source':final['source_text'],'clean':final['clean_text'],
                          'translation':final['translation'],'previews':record['previews']},ensure_ascii=False),flush=True)
        return final
    page.locator('#translation-text-tab').click()
    for language in LANGUAGES:
        page.locator('#translation-language').select_option(language)
        text=SOURCE if language=='en' else '我们需要300个产品。请在星期五发货。'
        page.locator('#translation-input').fill(text)
        page.locator('#translation-start').click()
        final=collect(language)
        assert final['target_language']==language
        assert final['source_text']==text
        assert final['translation'].strip() and final['translation']!=final['clean_text']
        if language in SCRIPT:assert re.search(SCRIPT[language],final['translation']),final['translation']
        if language=='en':
            assert '300' in final['clean_text'] and '12.5' in final['clean_text'],final['clean_text']
            page.screenshot(path=str(OUT/'text_english_desktop.png'),full_page=True)
    page.locator('#translation-audio-tab').click()
    page.locator('#translation-audio-language').select_option('en')
    page.locator('#translation-audio-file').set_input_files(str(OUT/'chinese_probe.wav'))
    page.locator('#translation-start').click()
    final=collect('chinese_audio_english')
    assert final['audio_total_seconds']==12.0
    assert final['asr_total_ms']>0
    assert re.search(r'[\u4E00-\u9FFF]',final['source_text']),final['source_text']
    assert re.search(r'[A-Za-z]{3,}',final['translation']),final['translation']
    page.screenshot(path=str(OUT/'audio_desktop.png'),full_page=True)
    page.set_viewport_size({'width':390,'height':844})
    assert page.evaluate('document.documentElement.scrollWidth <= window.innerWidth')
    page.screenshot(path=str(OUT/'audio_mobile.png'),full_page=True)
    assert not errors,errors
    (OUT/'browser_live.json').write_text(json.dumps({'passed':True,'records':records,'browser_errors':errors},ensure_ascii=False,indent=2))
    browser.close()
