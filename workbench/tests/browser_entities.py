"""Verify terminology UI mutations using a uniquely owned disposable entry."""
import json,uuid
from pathlib import Path
from playwright.sync_api import sync_playwright
ROOT=Path(__file__).resolve().parents[1]
name='WorkbenchUISmoke-'+uuid.uuid4().hex[:12]
with sync_playwright() as p:
    browser=p.chromium.launch(headless=True,args=['--no-sandbox'])
    page=browser.new_page(viewport={'width':1440,'height':1050})
    errors=[];page.on('pageerror',lambda e:errors.append(str(e)))
    page.goto('http://127.0.0.1:8083/',wait_until='networkidle')
    try:
        page.locator('#manage-entities').click()
        page.wait_for_function("document.getElementById('entity-notice').textContent.includes('已加载')")
        page.locator('#add-entity').click()
        page.locator('#entity-canonical').fill(name)
        page.locator('#entity-aliases').fill('Smoke Test Alias')
        page.locator('#entity-form [type=submit]').click()
        page.wait_for_function("!document.getElementById('entity-dialog').open")
        page.locator('#entity-search').fill(name)
        page.wait_for_function("document.getElementById('entity-rows').querySelectorAll('button[data-action=edit]').length===1")
        row=page.locator('#entity-rows')
        assert 'Smoke Test Alias' in row.inner_text()
        row.locator('button[data-action=edit]').click()
        assert page.locator('#entity-domain').input_value()=='trade'
        page.locator('#entity-priority').fill('9')
        page.locator('#entity-aliases').fill('Edited Smoke Alias')
        page.locator('#entity-form [type=submit]').click()
        page.wait_for_function("!document.getElementById('entity-dialog').open")
        page.wait_for_function("document.getElementById('entity-rows').textContent.includes('Edited Smoke Alias')")
        row.locator('button[data-action=toggle]').click()
        page.wait_for_function("document.getElementById('entity-rows').querySelector('.muted-row')!==null")
        entry=page.request.get('http://127.0.0.1:8083/api/entities',params={'query':name}).json()['entities'][0]
        assert entry['enabled'] is False and entry['priority']==9 and entry['domain']=='trade',entry
        assert entry['aliases']==['Edited Smoke Alias'],entry
        assert not errors,errors
        result={'create':'passed','edit':'passed','toggle':'passed','search':'passed','browser_errors':errors}
        (ROOT/'results/entity_ui_check.json').write_text(json.dumps(result,ensure_ascii=False,indent=2))
        print(json.dumps(result))
    finally:
        entries=page.request.get('http://127.0.0.1:8083/api/entities',params={'query':name}).json()['entities']
        for entry in entries:
            if entry['canonical_text']==name and entry['domain']=='trade':
                response=page.request.delete(f"http://127.0.0.1:8083/api/entities/{entry['entity_id']}")
                assert response.ok,response.text()
        browser.close()
