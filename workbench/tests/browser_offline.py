"""Exercise revision ordering and streaming rendering with fake browser transports."""
import json
import re
from pathlib import Path

from playwright.sync_api import sync_playwright

root = Path(__file__).resolve().parents[1]
web = root / "system/web"
html = re.sub(r"<script\b[^>]*>[\s\S]*?</script>", "", (web / "index.html").read_text(), flags=re.I)

with sync_playwright() as playwright:
    browser = playwright.chromium.launch(headless=True, args=["--no-sandbox"])
    page = browser.new_page(accept_downloads=True)
    errors = []
    page.on("pageerror", lambda error: errors.append(str(error)))
    page.route(
        "http://translation.test/**",
        lambda route: route.fulfill(
            status=200,
            content_type="text/css" if route.request.url.split("?", 1)[0].endswith(".css") else "text/html",
            body=(web / "style.css").read_text() if route.request.url.split("?", 1)[0].endswith(".css") else html,
        ),
    )
    page.goto("http://translation.test/", wait_until="domcontentloaded")
    layouts = {}
    for width in (1440, 800, 390):
        page.set_viewport_size({"width":width,"height":844})
        layout = page.locator(".session-summary").evaluate("""element => {
          const cards = [...element.children].map(card => card.getBoundingClientRect());
          return {
            rows:new Set(cards.map(rect => Math.round(rect.top))).size,
            columns:getComputedStyle(element).gridTemplateColumns.split(' ').length,
            scrollWidth:element.scrollWidth, clientWidth:element.clientWidth,
            documentWidth:document.documentElement.scrollWidth, innerWidth:innerWidth,
            firstWidth:cards[0].width, lastRight:cards.at(-1).right,
            containerRight:element.getBoundingClientRect().right
          };
        }""")
        layouts[width] = layout
        assert layout["rows"] == 1 and layout["columns"] == 5, (width, layout)
        assert layout["documentWidth"] <= layout["innerWidth"], (width, layout)
        assert layout["firstWidth"] >= 120, (width, layout)
        if width == 800:
            assert layout["lastRight"] <= layout["containerRight"] + 1, (width, layout)
            assert layout["scrollWidth"] == layout["clientWidth"], (width, layout)
        if width == 390:
            assert layout["scrollWidth"] > layout["clientWidth"], (width, layout)
    result_dir = root / "results/metrics_single_row_20261020"
    result_dir.mkdir(parents=True, exist_ok=True)
    for width in (800, 390):
        page.set_viewport_size({"width":width,"height":844})
        page.screenshot(path=str(result_dir / f"metrics_{width}.png"), full_page=True)
    (result_dir / "layout.json").write_text(json.dumps(layouts, ensure_ascii=False, indent=2) + "\n")
    page.set_viewport_size({"width":1280,"height":720})
    page.evaluate(
        """() => {
          window.fetch = async (url, options={}) => {
            if (String(url).includes('/api/translation/audio') && options.method === 'POST') {
              return {ok:true, json:async()=>({id:'fake-audio',websocket:'/ws/audio'})};
            }
            return {ok:true, json:async()=>({ready:true,audio_ready:true,languages:[{code:'en',name:'英语'}]})};
          };
          window.fakeSockets = [];
          window.WebSocket = class extends EventTarget {
            static CLOSING = 2;
            constructor() {
              super(); this.readyState = 0; window.fakeSockets.push(this);
              setTimeout(() => { this.readyState = 1; this.dispatchEvent(new Event('open')); }, 0);
            }
            send(value) { this.lastSent = JSON.parse(value); }
            close() { this.readyState = 3; this.dispatchEvent(new Event('close')); }
          };
          window.feed = value => window.fakeSockets.at(-1).dispatchEvent(
            new MessageEvent('message', {data:JSON.stringify(value)})
          );
        }"""
    )
    page.add_script_tag(path=str(web / "translation.js"))
    page.wait_for_function("!document.getElementById('translation-start').disabled")
    page.locator("#translation-text-tab").click()
    page.locator("#translation-input").fill("test input")
    page.locator("#translation-start").click()
    page.wait_for_function("window.fakeSockets.length===1 && window.fakeSockets[0].lastSent")
    page.wait_for_timeout(250)

    def feed(payload):
        page.evaluate("value => window.feed(value)", payload)

    def text(selector):
        return page.locator(selector).inner_text()

    def wait_text(selector, expected):
        expression = (
            f"document.querySelector({json.dumps(selector)}).textContent === "
            f"{json.dumps(expected, ensure_ascii=False)}"
        )
        page.wait_for_function(expression)

    assert text("#translation-total-time").endswith(" s")
    assert float(text("#translation-total-time")[:-2]) > 0, "active wallclock timer should advance"

    feed({"event": "source", "revision": 1, "source_text": "source 1"})
    feed({"event":"progress", "session_metrics":{"refiner_model_calls":4,"gate_skipped_groups":2}})
    assert text("#translation-calls") == "4"
    assert text("#translation-gate-skips") == "2"
    feed({"event":"progress", "session_metrics":{"refiner_model_calls":2,"gate_skipped_groups":1}})
    assert text("#translation-calls") == "4", "stale session metrics must not roll back calls"
    assert text("#translation-gate-skips") == "2", "stale session metrics must not roll back skips"
    feed({"event": "source", "revision": 2, "source_text": "source 2"})
    feed({
        "event": "translation", "revision": 1, "translation_revision": 1,
        "translation": "译文1", "display_revision": 0,
        "display_refined_text": "", "pending_raw_text": "译文1", "has_pending_refinement": True,
    })
    wait_text("#translation-original", "译文1")
    feed({"event": "source", "revision": 3, "source_text": "source 3"})
    assert text("#translation-source") == "source 3"
    assert text("#translation-original") == "译文1", "new source must not erase a visible translation"
    assert page.locator("#translation-clean .pending-refinement").inner_text() == "译文1"

    feed({
        "event": "translation", "revision": 2, "translation_revision": 2,
        "translation": "译文2", "display_revision": 0,
        "display_refined_text": "", "pending_raw_text": "译文2", "has_pending_refinement": True,
    })
    wait_text("#translation-original", "译文2")
    feed({
        "event": "translation", "revision": 1, "translation_revision": 1,
        "translation": "迟到译文1", "display_revision": 0,
        "display_refined_text": "", "pending_raw_text": "迟到译文1", "has_pending_refinement": True,
    })
    assert text("#translation-original") == "译文2", "older translation must not overwrite newer translation"
    assert text("#translation-source") == "source 3", "translation events must not rewrite the source column"

    clean_before_preview = text("#translation-clean")
    feed({
        "event": "translation_preview", "revision": 3, "preview_sequence": 1,
        "translation": "预览译文3（第一步）", "provisional": True,
    })
    wait_text("#translation-original", "预览译文3（第一步）")
    assert text("#translation-phase-label") == "翻译中"
    assert text("#translation-clean") == clean_before_preview, "preview must not touch refinement output"
    feed({
        "event": "translation_preview", "revision": 3, "preview_sequence": 4,
        "translation": "预览译文3（更新）", "provisional": True,
    })
    wait_text("#translation-original", "预览译文3（更新）")
    feed({
        "event": "translation_preview", "revision": 3, "preview_sequence": 3,
        "translation": "较旧预览3", "provisional": True,
    })
    assert text("#translation-original") == "预览译文3（更新）", "preview sequence must be monotonic within a revision"
    feed({
        "event": "update", "revision": 3, "translation_revision": 2,
        "translation": "迟到的已完成译文2",
    })
    assert text("#translation-original") == "预览译文3（更新）", "older completed update must not replace a newer preview"
    feed({
        "event": "translation", "revision": 3, "translation_revision": 3,
        "translation": "片段完成译文3",
    })
    wait_text("#translation-original", "片段完成译文3")
    assert text("#translation-phase-label") == "片段完成"
    feed({
        "event": "translation_preview", "revision": 3, "preview_sequence": 99,
        "translation": "完成后迟到的预览", "provisional": True,
    })
    assert text("#translation-original") == "片段完成译文3", "completed phase must reject later previews"

    feed({
        "event": "update", "revision": 1, "translation_revision": 2, "refinement_revision": 1,
        "display_revision": 1, "translation": "迟到译文1",
        "display_refined_text": "已确认前缀", "pending_raw_text": "译文2待精修",
        "display_text": "已确认前缀译文2待精修", "has_pending_refinement": True,
        "refiner_executed": True, "refiner_accepted": True, "refinement_state": "window_refined",
    })
    assert text("#translation-original") == "片段完成译文3"
    assert text("#translation-clean") == "已确认前缀译文2待精修"
    assert page.locator("#translation-clean .pending-refinement").inner_text() == "译文2待精修"
    assert text("#quality-tag") == "部分待精修", "an accepted prefix must not imply the pending tail is accepted"

    feed({
        "event": "update", "revision": 1, "translation_revision": 1, "refinement_revision": 1,
        "display_revision": 1, "translation": "旧译文", "display_refined_text": "旧精修回退",
        "pending_raw_text": "旧尾", "has_pending_refinement": True,
    })
    assert text("#translation-original") == "片段完成译文3"
    assert text("#translation-clean") == "已确认前缀译文2待精修", "older refinement must not roll back newer display"
    assert text("#translation-source") == "source 3"

    feed({
        "event": "final", "revision": 2, "translation_revision": 2, "refinement_revision": 2,
        "display_revision": 2, "translation": "陈旧最终译文", "clean_text": "陈旧最终结果",
        "display_refined_text": "陈旧最终结果", "pending_raw_text": "", "has_pending_refinement": False,
    })
    assert page.locator("#translation-start").is_disabled(), "a stale final must not finish the active run"
    assert text("#translation-clean") == "已确认前缀译文2待精修"

    feed({"event":"refinement_progress", "revision":2,"translation_revision":2,
        "display_revision":2,"display_refined_text":"分组完成前缀", "pending_raw_text":"待后续精修",
        "has_pending_refinement":True,"refinement_state":"refining"})
    assert text("#translation-clean")=="分组完成前缀待后续精修"
    assert text("#translation-original")=="片段完成译文3", "group progress must not roll back token preview"

    before_finalizing = text("#translation-original")
    feed({"event":"translation_finalizing", "revision":3})
    wait_text("#translation-phase-label", "最终翻译中")
    assert text("#translation-original") == before_finalizing

    for marker in range(100):
        feed({"event": "progress", "marker": marker})
    feed({
        "event": "final", "revision": 3, "translation_revision": 3, "refinement_revision": 3,
        "display_revision": 3, "source_text": "不能通过final回写source",
        "translation": "译文3", "display_refined_text": "最终确认全文", "pending_raw_text": "",
        "display_text": "最终确认全文", "has_pending_refinement": False,
        "refiner_executed": True, "refiner_accepted": True, "refinement_state": "complete",
    })
    assert text("#translation-source") == "source 3", "translation/final events must not rewrite the source column"
    wait_text("#translation-original", "译文3")
    assert text("#translation-phase-label") == "最终译文"
    assert text("#translation-clean") == "最终确认全文"
    assert page.locator("#translation-clean .pending-refinement").count() == 0, "final pending tail must disappear"
    assert text("#quality-tag") == "已接受"
    frozen_final_time = text("#translation-total-time")
    page.wait_for_timeout(250)
    assert text("#translation-total-time") == frozen_final_time, "final event must freeze elapsed time"

    page.evaluate("window.fakeSockets.at(-1).close()")
    page.wait_for_function("!document.getElementById('translation-start').disabled")
    page.wait_for_function("document.getElementById('translation-log').textContent.includes('仅展示最近60条')")
    lines = page.locator("#translation-log").text_content().splitlines()
    assert len(lines) == 61, len(lines)
    with page.expect_download() as download_info:
        page.locator("#translation-download").click()
    records = json.loads(Path(download_info.value.path()).read_text())
    assert len(records) == 120, len(records)
    assert not errors, errors
    result = {
        "status": "passed",
        "service_started": False,
        "streaming_revisions": "passed",
        "stale_final_ignored": True,
        "pending_tail_cleared_on_final": True,
        "visible_log_events": 60,
        "download_log_events": len(records),
        "browser_errors": errors,
    }
    page.locator("#translation-start").click()
    page.wait_for_function("window.fakeSockets.length===2 && window.fakeSockets[1].lastSent")
    feed({"event":"refining", "revision":9})
    assert "正在精修第 9 版" in text("#translation-status")
    page.evaluate("window.fakeSockets.at(-1).close()")
    assert text("#quality-tag")=="连接中断"
    frozen_disconnect_time = text("#translation-total-time")
    page.wait_for_timeout(200)
    assert text("#translation-total-time") == frozen_disconnect_time, "disconnect must freeze elapsed time"
    assert "当前任务未完成" in text("#translation-guard")
    result["unexpected_disconnect_visible"] = True
    result["translation_preview_sequence"] = "passed"
    result["completed_phase_rejects_late_preview"] = True
    result["final_replaces_preview"] = True
    page.locator("#translation-start").click()
    page.wait_for_function("window.fakeSockets.length===3 && window.fakeSockets[2].lastSent")
    feed({"event":"source", "revision":1,"source_text":"Clean."})
    feed({"event":"final","revision":1,"translation_revision":1,"display_revision":1,
        "translation":"请按时发货。","display_refined_text":"请按时发货。","pending_raw_text":"",
        "has_pending_refinement":False,"refiner_executed":False,"refinement_gate_mode":"regex",
        "refinement_gate_skipped_segments":1})
    assert text("#quality-tag")=="规则保留"
    assert text("#translation-guard")=="规则未命中，保留译文"
    page.evaluate("window.fakeSockets.at(-1).close()")
    result["regex_keep_is_terminal_not_pending"] = True
    page.locator("#translation-start").click()
    page.wait_for_function("window.fakeSockets.length===4 && window.fakeSockets[3].lastSent")
    page.locator("#translation-stop").click()
    frozen_cancel_time = text("#translation-total-time")
    page.wait_for_timeout(200)
    assert text("#translation-total-time") == frozen_cancel_time, "cancel must freeze elapsed time"
    assert text("#translation-total-time") != frozen_final_time, "new run must reset elapsed timer"
    page.locator("#translation-audio-tab").click()
    page.locator("#translation-audio-file").set_input_files({"name":"sample.wav","mimeType":"audio/wav","buffer":b"fake"})
    page.locator("#translation-start").click()
    page.wait_for_function("window.fakeSockets.length===5")
    assert text("#translation-asr-time") == "—", "new audio run must clear ASR time"
    feed({"event":"asr_chunk", "asr_ms":1200, "asr_total_ms":1200})
    assert text("#translation-asr-time") == "1.2 s"
    feed({"event":"asr_chunk", "asr_ms":1200, "asr_total_ms":1200})
    assert text("#translation-asr-time") == "1.2 s", "repeated cumulative total must not be added twice"
    feed({"event":"asr_chunk", "asr_ms":250, "asr_total_ms":900})
    assert text("#translation-asr-time") == "1.2 s", "older cumulative total must not roll back or add per-event time"
    feed({"event":"asr_chunk", "asr_ms":650, "asr_total_ms":1850})
    assert text("#translation-asr-time") == "1.9 s"
    page.locator("#translation-stop").click()
    assert text("#translation-asr-time") == "1.9 s", "cancel must retain the last ASR total"

    page.locator("#translation-start").click()
    page.wait_for_function("window.fakeSockets.length===6")
    assert text("#translation-asr-time") == "—", "new audio task must reset ASR time"
    feed({"event":"asr_chunk", "asr_ms":399, "asr_total_ms":399})
    assert text("#translation-asr-time") == "0.4 s"
    page.evaluate("window.fakeSockets.at(-1).close()")
    page.wait_for_function("!document.getElementById('translation-start').disabled")
    assert text("#translation-asr-time") == "0.4 s", "disconnect must retain the last ASR total"
    page.locator("#translation-audio-file").set_input_files({"name":"replacement.wav","mimeType":"audio/wav","buffer":b"fake"})
    assert text("#translation-asr-time") == "—", "selecting another file must clear the previous ASR total"
    page.locator("#translation-text-tab").click()
    assert text("#translation-asr-time") == "不适用"
    page.locator("#translation-audio-tab").click()
    assert text("#translation-asr-time") == "—", "switching back to audio must clear the prior ASR total"
    result["asr_cumulative_cancel_disconnect_reset_text_mode"] = "passed"
    print(json.dumps(result, ensure_ascii=False))
    browser.close()
