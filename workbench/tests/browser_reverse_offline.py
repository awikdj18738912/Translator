"""Verify the Chinese-first ASR/refinement/translation UI with fake transports."""
import json
import re
from pathlib import Path

from playwright.sync_api import sync_playwright


root = Path(__file__).resolve().parents[1]
web = root / "system/web"
html = re.sub(r"<script\b[^>]*>[\s\S]*?</script>", "", (web / "index.html").read_text(), flags=re.I)

with sync_playwright() as playwright:
    browser = playwright.chromium.launch(headless=True, args=["--no-sandbox"])
    page = browser.new_page()
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
    page.set_viewport_size({"width": 1280, "height": 900})
    page.evaluate(
        """() => {
          window.fetch = async (url, options={}) => {
            if (String(url).includes('/api/translation/audio') && options.method === 'POST') {
              window.audioForm = options.body;
              return {ok:true, json:async()=>({id:'fake-audio',websocket:'/ws/audio'})};
            }
            return {ok:true, json:async()=>({ready:true,audio_ready:true,pipeline_direction:'zh_to_foreign',
              languages:[{code:'en',name:'英语'},{code:'fr',name:'法语'}]})};
          };
          window.fakeSockets = [];
          window.WebSocket = class extends EventTarget {
            static CLOSING = 2;
            constructor(url) {
              super(); this.url=url; this.readyState=0; window.fakeSockets.push(this);
              setTimeout(() => { this.readyState=1; this.dispatchEvent(new Event('open')); }, 0);
            }
            send(value) { this.lastSent=JSON.parse(value); }
            close() { this.readyState=3; this.dispatchEvent(new Event('close')); }
          };
          window.feed = value => window.fakeSockets.at(-1).dispatchEvent(
            new MessageEvent('message', {data:JSON.stringify(value)})
          );
        }"""
    )
    page.add_script_tag(path=str(web / "translation.js"))
    page.wait_for_function("!document.getElementById('translation-start').disabled")

    def text(selector):
        return page.locator(selector).inner_text()

    def feed(payload):
        page.evaluate("value => window.feed(value)", payload)

    def wait_text(selector, expected):
        page.wait_for_function(
            "args => document.querySelector(args.selector).textContent === args.expected",
            arg={"selector": selector, "expected": expected},
        )

    result = {"status": "running", "service_started": False, "inference_run": False}

    columns = page.locator(".results > article").evaluate_all(
        "items => items.map(item => item.querySelector('h2').textContent.trim())"
    )
    assert columns == ["中文转录", "中文精修", "外语翻译"], columns
    assert page.locator(".session-summary .headline-metric").count() == 5
    assert text("#source-language-label") == "中文"
    target_values = page.locator("#translation-language option").evaluate_all(
        "options => options.map(option => option.value)"
    )
    assert target_values == ["en", "fr", "de", "es", "ru", "ja", "ko", "ar"], target_values
    assert "中文" in page.locator("#translation-input").get_attribute("placeholder")

    page.locator("#translation-text-tab").click()
    page.locator("#translation-language").select_option("de")
    page.locator("#translation-input").fill("请把订单在周五前发出，并确认运费。")
    page.locator("#translation-start").click()
    try:
        page.wait_for_function("window.fakeSockets.length===1 && window.fakeSockets[0].lastSent", timeout=5000)
    except Exception as error:
        raise AssertionError({
            "page_errors": errors,
            "status": text("#translation-status"),
            "button_disabled": page.locator("#translation-start").is_disabled(),
            "sockets": page.evaluate("window.fakeSockets.length"),
        }) from error
    sent = page.evaluate("window.fakeSockets[0].lastSent")
    assert sent["source_language"] == "zh", sent
    assert sent["target_language"] == "de", sent
    assert "language" not in sent, sent

    feed({"event": "ready", "source_language": "zh", "target_language": "de", "pipeline": "fake"})
    feed({"event": "source", "revision": 1, "source_text": "请把订单在周五前发出。"})
    wait_text("#translation-source", "请把订单在周五前发出。")
    feed({
        "event": "refinement_progress", "revision": 1, "refinement_revision": 1,
        "display_refined_text": "请于周五前安排订单发货，",
        "clean_text": "请于周五前安排订单发货，运费待确认。",
        "pending_raw_text": "运费待确认。", "has_pending_refinement": True,
        "refinement_state": "refining", "session_metrics": {"refiner_model_calls": 1, "gate_skipped_groups": 0},
        "refiner_executed": True, "refiner_accepted": True,
    })
    assert text("#translation-clean") == "请于周五前安排订单发货，运费待确认。"
    assert page.locator("#translation-clean .pending-refinement").inner_text() == "运费待确认。"
    assert text("#translation-original") == ""
    feed({
        "event": "refined", "revision": 1, "refinement_revision": 1,
        "display_refined_text": "请于周五前安排订单发货，",
        "clean_text": "请于周五前安排订单发货，运费请另行确认。",
        "pending_raw_text": "", "has_pending_refinement": False,
        "refinement_state": "completed", "session_metrics": {"refiner_model_calls": 2, "gate_skipped_groups": 0},
        "refiner_executed": True, "refiner_accepted": True,
    })
    wait_text("#translation-clean", "请于周五前安排订单发货，")
    feed({"event": "translation_preview", "revision": 1, "translation_revision": 1, "preview_sequence": 1,
        "translation": "Bitte versenden Sie die Bestellung bis Freitag.", "provisional": True})
    wait_text("#translation-original", "Bitte versenden Sie die Bestellung bis Freitag.")
    assert "Bitte" not in text("#translation-clean")
    feed({
        "event": "translation", "revision": 1, "translation_revision": 2,
        "translation": "Bitte versenden Sie die Bestellung bis Freitag und bestätigen Sie die Versandkosten.",
        "clean_text": "请于周五前安排订单发货，运费待核实。",
        "display_refined_text": "请于周五前安排订单发货，",
        "pending_raw_text": "运费待核实。", "has_pending_refinement": True,
    })
    wait_text("#translation-original", "Bitte versenden Sie die Bestellung bis Freitag und bestätigen Sie die Versandkosten.")
    assert page.locator("#translation-clean .pending-refinement").inner_text() == "运费待核实。"
    assert "Bitte" not in text("#translation-clean"), "foreign translation leaked into Chinese refinement"
    assert text("#translation-calls") == "2"
    assert text("#translation-gate-skips") == "0"

    # A second refinement group on the same ASR revision gets its own translation revision.
    feed({
        "event": "refinement_progress", "revision": 1, "refinement_revision": 2,
        "display_refined_text": "请于周五前安排订单发货，并确认运费，",
        "clean_text": "请于周五前安排订单发货，并确认运费，金额待确认。",
        "pending_raw_text": "金额待确认。", "has_pending_refinement": True,
        "refinement_state": "refining",
    })
    feed({"event": "translation_preview", "revision": 1, "translation_revision": 3, "preview_sequence": 1,
        "translation": "Bitte versenden und die Versandkosten bestätigen.", "provisional": True})
    wait_text("#translation-original", "Bitte versenden und die Versandkosten bestätigen.")

    feed({"event": "source", "revision": 2, "source_text": "请于周五前发货，并确认运费金额。"})
    wait_text("#translation-source", "请于周五前发货，并确认运费金额。")
    feed({
        "event": "refinement_progress", "revision": 2, "refinement_revision": 2,
        "display_refined_text": "请于周五前安排发货，",
        "clean_text": "请于周五前安排发货，运费金额请予确认。",
        "pending_raw_text": "运费金额请予确认。", "has_pending_refinement": True,
        "refinement_state": "refining",
    })
    wait_text("#translation-clean", "请于周五前安排发货，运费金额请予确认。")
    feed({
        "event": "refined", "revision": 1, "refinement_revision": 1,
        "display_refined_text": "旧中文精修回滚", "clean_text": "旧中文精修回滚",
        "pending_raw_text": "旧尾句", "has_pending_refinement": True,
    })
    feed({"event": "translation", "revision": 1, "translation_revision": 1,
        "translation": "alte deutsche Übersetzung", "clean_text": "旧中文精修回滚"})
    assert text("#translation-source") == "请于周五前发货，并确认运费金额。"
    assert text("#translation-clean") == "请于周五前安排发货，运费金额请予确认。"
    assert "alte" not in text("#translation-clean")
    assert text("#translation-original") == "Bitte versenden und die Versandkosten bestätigen."

    feed({"event": "translation_preview", "revision": 2, "translation_revision": 4, "preview_sequence": 1,
        "translation": "Bitte bis Freitag versenden.", "provisional": True})
    wait_text("#translation-original", "Bitte bis Freitag versenden.")
    feed({"event": "translation", "revision": 2, "translation_revision": 5,
        "translation": "Bitte versenden Sie die Bestellung bis Freitag und bestätigen Sie die Frachtkosten."})
    wait_text("#translation-original", "Bitte versenden Sie die Bestellung bis Freitag und bestätigen Sie die Frachtkosten.")
    feed({"event": "translation", "revision": 1, "translation_revision": 3,
        "translation": "迟到的旧外语翻译", "clean_text": "迟到的旧中文精修"})
    assert text("#translation-original") == "Bitte versenden Sie die Bestellung bis Freitag und bestätigen Sie die Frachtkosten."
    assert text("#translation-clean") == "请于周五前安排发货，运费金额请予确认。"

    feed({
        "event": "final", "revision": 2, "translation_revision": 5, "refinement_revision": 2,
        "translation": "Bitte versenden Sie die Bestellung bis Freitag und bestätigen Sie die Frachtkosten.",
        "source_text": "请于周五前发货，并确认运费金额。",
        "clean_text": "请于周五前安排发货，并确认运费金额。",
        "display_refined_text": "请于周五前安排发货，并确认运费金额。",
        "pending_raw_text": "", "has_pending_refinement": False,
        "refiner_executed": True, "refiner_accepted": True, "refinement_state": "complete",
    })
    wait_text("#translation-phase-label", "最终译文")
    wait_text("#translation-clean", "请于周五前安排发货，并确认运费金额。")
    assert text("#translation-source") == "请于周五前发货，并确认运费金额。"
    assert text("#quality-tag") == "已接受"
    assert text("#translation-status") == "中文精修与外语翻译完成。"
    assert text("#translation-calls") == "2"
    result["text_ws_payload"] = sent
    result["pipeline_events"] = ["ready", "source", "refinement_progress", "refined", "translation_preview", "translation", "final"]
    result["pending_refinement_stayed_chinese"] = True
    result["stale_refinement_and_translation_rejected"] = True
    result["five_metric_cards"] = page.locator(".session-summary .headline-metric").count()

    # Exercise audio submission fields, ASR timing, and Arabic output direction.
    page.evaluate("window.fakeSockets.at(-1).close()")
    page.wait_for_function("!document.getElementById('translation-start').disabled")
    page.locator("#translation-audio-tab").click()
    page.locator("#translation-audio-language").select_option("ar")
    assert text("#target-language-label") == "阿拉伯语"
    assert page.locator("#translation-original").get_attribute("dir") == "auto"
    page.locator("#translation-audio-file").set_input_files(
        {"name": "fake.wav", "mimeType": "audio/wav", "buffer": b"offline-audio"}
    )
    page.locator("#translation-start").click()
    page.wait_for_function("window.fakeSockets.length===2")
    form = page.evaluate("() => Object.fromEntries(window.audioForm.entries())")
    assert form["target_language"] == "ar", form
    assert "language" not in form, form
    assert text("#translation-asr-time") == "—"
    feed({"event": "ready", "source_language": "zh", "target_language": "ar", "pipeline": "fake"})
    feed({"event": "asr_chunk", "asr_ms": 1250, "asr_total_ms": 1250})
    assert text("#translation-asr-time") == "1.3 s"
    feed({"event": "asr_chunk", "asr_ms": 400, "asr_total_ms": 1650})
    assert text("#translation-asr-time") == "1.6 s"
    result["audio_form_target_language"] = form["target_language"]
    result["arabic_direction"] = page.locator("#translation-original").get_attribute("dir")
    result["asr_total_ms_display"] = text("#translation-asr-time")

    assert not errors, errors
    result["browser_errors"] = errors
    result["status"] = "passed"
    result_dir = root.parent / "results/chinese_first_pipeline_20261020"
    result_dir.mkdir(parents=True, exist_ok=True)
    (result_dir / "frontend_offline.json").write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps(result, ensure_ascii=False))
    browser.close()
