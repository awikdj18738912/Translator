"""CPU-only integration coverage for the Chinese-first translation route."""
from __future__ import annotations

import sys
import threading
from pathlib import Path

import numpy as np
import pytest

WEB_ROOT = Path(__file__).resolve().parents[1]
if str(WEB_ROOT) not in sys.path:
    sys.path.insert(0, str(WEB_ROOT))

from fastapi import FastAPI
from fastapi.testclient import TestClient

from system import translation_web, web_app


class _ChineseSession:
    """Refiner stub exposing the same guarded source-span shape as production."""

    def __init__(self, *, gate=None):
        self.calls = []
        self.model_calls = []
        self.gate = gate
        self.entered = threading.Event()

    @staticmethod
    def clean(text):
        text = text.replace('呃，', '').replace('呃', '')
        return text.replace('三点五', '3.5').replace('三点', '3')

    def update(self, text, *, revision, final=False, on_progress=None):
        self.calls.append((text, revision, final))
        if self.gate is not None and len(self.calls) == 1:
            self.entered.set()
            assert self.gate.wait(10), 'test did not release blocked refiner'
        clean = self.clean(text)
        span = {'source_text': text, 'clean_text': clean,
                'state': 'committed' if final else 'active'}
        return {'raw_text': text, 'clean_text': clean, 'final': final,
                'refinement_source_spans': [span]}


def _make_app(tmp_path, *, translate, session_factory=None):
    app = FastAPI()
    sessions = []

    def new_session(domain):
        session = session_factory() if session_factory else _ChineseSession()
        sessions.append((domain, session))
        return session

    app.state.new_text_refinement_session = new_session
    calls = []

    def recording_translate(text, language):
        calls.append((text, language))
        return translate(text, language)

    translation_web.install_translation_routes(
        app, recording_translate, asr_url='http://asr.invalid', output_dir=tmp_path,
        pipeline_direction='zh_to_foreign',
    )
    return app, calls, sessions


@pytest.mark.parametrize(('target', 'expected'), [('ja', 'JA'), ('en', 'EN')])
def test_chinese_text_is_refined_before_translation_and_final_keeps_chinese(tmp_path, target, expected):
    source = '呃，三点五公斤。'
    app, translate_calls, sessions = _make_app(
        tmp_path, translate=lambda text, language: f'{language.upper()}:{text}')

    with TestClient(app).websocket_connect('/ws/translation') as ws:
        ws.send_json({'event': 'start', 'source_language': 'zh', 'target_language': target,
                      'text': source, 'chunk_chars': 40, 'interval_ms': 0})
        events = []
        while not events or events[-1]['event'] not in {'error', 'final'}:
            events.append(ws.receive_json())

    assert events[-1]['event'] == 'final', events[-1]
    assert sessions[0][1].calls == [(source, 1, True)]
    assert translate_calls == [('3.5公斤。', target)]
    final = events[-1]
    assert final['source_text'] == source
    assert final['clean_text'] == '3.5公斤。'
    assert final['translation'] == f'{expected}:3.5公斤。'
    assert final['source_language'] == 'zh'
    assert final['target_language'] == target


def test_pending_refinement_is_not_translated_until_final(tmp_path):
    source = '呃，三点五公斤。'
    app, translate_calls, sessions = _make_app(
        tmp_path, translate=lambda text, language: f'{language}:{text}')

    with TestClient(app).websocket_connect('/ws/translation') as ws:
        ws.send_json({'event': 'start', 'source_language': 'zh', 'target_language': 'ja',
                      'text': source, 'chunk_chars': 6, 'interval_ms': 0})
        events = []
        while not events or events[-1]['event'] not in {'error', 'final'}:
            events.append(ws.receive_json())

    assert events[-1]['event'] == 'final', events[-1]
    assert any(call[2] is False for call in sessions[0][1].calls)
    assert sessions[0][1].calls[-1][2] is True
    assert translate_calls == [('3.5公斤。', 'ja')]


def test_audio_target_language_ja_uses_chinese_asr_and_reports_asr_time(tmp_path, monkeypatch):
    monkeypatch.setattr(translation_web, 'decode_audio', lambda path: np.zeros(8000, dtype='<f4'))
    requests = []

    def fake_asr(url, endpoint, session_id=None, data=b'', params=None):
        requests.append((endpoint, params))
        if endpoint == '/stream/start':
            return {'session_id': 'stub-asr'}
        if endpoint == '/stream/chunk':
            return {'text': '呃，三点五公斤。'}
        if endpoint == '/stream/finish':
            return {'text': '呃，三点五公斤。'}
        return {'cancelled': True}

    monkeypatch.setattr(web_app, '_stream_request', fake_asr)
    app, translate_calls, _ = _make_app(
        tmp_path, translate=lambda text, language: f'{language}:{text}')

    with TestClient(app) as client:
        upload = client.post('/api/translation/audio',
            files={'file': ('stub.wav', b'audio')}, data={'target_language': 'ja', 'interval_ms': '0'})
        assert upload.status_code == 200, upload.text
        with client.websocket_connect(upload.json()['websocket']) as ws:
            events = []
            while not events or events[-1]['event'] not in {'error', 'final'}:
                events.append(ws.receive_json())

    assert events[-1]['event'] == 'final', events[-1]
    start = next(params for endpoint, params in requests if endpoint == '/stream/start')
    assert start['language'] == 'Chinese'
    assert translate_calls == [('3.5公斤。', 'ja')]
    assert 'asr_total_ms' in events[-1]
    assert events[-1]['asr_total_ms'] >= 0


def test_asr_finishes_while_refiner_is_blocked_and_latest_final_text_is_translated(
    tmp_path, monkeypatch,
):
    import time

    release = threading.Event()
    session_holder = []

    def make_session():
        session = _ChineseSession(gate=release)
        session_holder.append(session)
        return session

    samples = np.zeros(24000, dtype='<f4')
    monkeypatch.setattr(translation_web, 'decode_audio', lambda path: samples)
    chunk_texts = iter(['呃，三点', '呃，三点五公斤', '呃，三点五公斤。'])
    asr_calls = []

    def fake_asr(url, endpoint, session_id=None, data=b'', params=None):
        asr_calls.append(endpoint)
        if endpoint == '/stream/start':
            return {'session_id': 'blocking-asr'}
        if endpoint == '/stream/chunk':
            return {'text': next(chunk_texts)}
        if endpoint == '/stream/finish':
            return {'text': '呃，三点五公斤。'}
        return {'cancelled': True}

    monkeypatch.setattr(web_app, '_stream_request', fake_asr)
    app, translate_calls, _ = _make_app(
        tmp_path, translate=lambda text, language: f'{language}:{text}',
        session_factory=make_session)

    with TestClient(app) as client:
        upload = client.post('/api/translation/audio', files={'file': ('stub.wav', b'audio')},
            data={'target_language': 'en', 'chunk_seconds': '0.5', 'interval_ms': '0'})
        assert upload.status_code == 200, upload.text
        with client.websocket_connect(upload.json()['websocket']) as ws:
            events = []
            try:
                while not session_holder or not session_holder[0].entered.is_set():
                    event = ws.receive_json()
                    events.append(event)
                    assert event['event'] != 'error', event
                deadline = time.monotonic() + 5
                while '/stream/finish' not in asr_calls and time.monotonic() < deadline:
                    time.sleep(.01)
                assert '/stream/finish' in asr_calls, 'ASR waited for blocked refinement'
                release.set()
                while not events or events[-1]['event'] not in {'error', 'final'}:
                    events.append(ws.receive_json())
            finally:
                release.set()

    final = events[-1]
    assert final['event'] == 'final', final
    assert final['source_text'] == '呃，三点五公斤。'
    assert final['translation_source_text'] == '3.5公斤。'
    assert final['translation'] == 'en:3.5公斤。'
    assert final['pipeline_stats']['max_pending_per_stage'] == 1
    assert translate_calls[-1] == ('3.5公斤。', 'en')


@pytest.mark.parametrize('start', [
    {'source_language': 'en', 'target_language': 'ja'},
    {'source_language': 'zh', 'target_language': 'xx'},
])
def test_rejects_non_chinese_source_and_unsupported_target(tmp_path, start):
    app, translate_calls, sessions = _make_app(
        tmp_path, translate=lambda text, language: f'{language}:{text}')

    with TestClient(app).websocket_connect('/ws/translation') as ws:
        ws.send_json({'event': 'start', 'text': '中文原文', **start})
        events = []
        while not events or events[-1]['event'] != 'error':
            events.append(ws.receive_json())

    assert events[-1]['event'] == 'error'
    assert translate_calls == []
    assert sessions == []


def test_translation_error_is_sent_and_releases_busy_lock_for_next_request(tmp_path):
    attempts = []

    def translate(text, language):
        attempts.append((text, language))
        if len(attempts) == 1:
            raise RuntimeError('stub translation failed')
        return f'{language}:{text}'

    app, _, _ = _make_app(tmp_path, translate=translate)
    client = TestClient(app)

    with client.websocket_connect('/ws/translation') as ws:
        ws.send_json({'event': 'start', 'source_language': 'zh', 'target_language': 'ja',
                      'text': '三点五公斤。', 'chunk_chars': 40, 'interval_ms': 0})
        events = []
        while not events or events[-1]['event'] not in {'error', 'final'}:
            events.append(ws.receive_json())
    assert events[-1]['event'] == 'error'
    assert 'stub translation failed' in events[-1]['detail']
    assert not app.state.translation_busy.locked()

    with client.websocket_connect('/ws/translation') as ws:
        ws.send_json({'event': 'start', 'source_language': 'zh', 'target_language': 'ja',
                      'text': '三点五公斤。', 'chunk_chars': 40, 'interval_ms': 0})
        events = []
        while not events or events[-1]['event'] not in {'error', 'final'}:
            events.append(ws.receive_json())
    assert events[-1]['event'] == 'final', events[-1]
    assert len(attempts) == 2


@pytest.mark.parametrize('blocked_stage', ['refinement', 'translation'])
def test_disconnect_keeps_busy_lock_until_blocked_model_thread_exits(tmp_path, blocked_stage):
    release = threading.Event()
    entered = threading.Event()

    session_holder = []

    def make_session():
        session = _ChineseSession(gate=release if blocked_stage == 'refinement' else None)
        session_holder.append(session)
        return session

    def translate(text, language):
        if blocked_stage == 'translation':
            entered.set()
            assert release.wait(5), 'test did not release blocked translation'
        return f'{language}:{text}'

    app, _, _ = _make_app(tmp_path, translate=translate, session_factory=make_session)
    client = TestClient(app)
    with client.websocket_connect('/ws/translation') as ws:
        ws.send_json({'event': 'start', 'source_language': 'zh', 'target_language': 'en',
                      'text': '三点五公斤。', 'chunk_chars': 40, 'interval_ms': 0})
        if blocked_stage == 'refinement':
            # Receive through the refining event; session.update is now blocked in its thread.
            assert _wait_for(lambda: bool(session_holder) and session_holder[0].entered.is_set())
        else:
            assert entered.wait(3), 'translation did not enter its blocking stub'

        closer = threading.Thread(target=ws.close, daemon=True)
        closer.start()
        try:
            # Allow the server to observe disconnect and begin cancellation while the
            # to_thread call remains alive. The lock must still protect model ownership.
            closer.join(.2)
            assert app.state.translation_busy.locked()
            release.set()
            closer.join(3)
            assert not closer.is_alive(), 'websocket close did not finish after releasing model stub'
        finally:
            release.set()

    assert _wait_for(lambda: not app.state.translation_busy.locked())


def _wait_for(predicate, timeout=3):
    import time
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if predicate():
            return True
        time.sleep(.01)
    return predicate()
