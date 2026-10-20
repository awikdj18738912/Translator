import threading
import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient
from system.translation_web import install_translation_routes
from test_translation_web import _FakeSession


@pytest.mark.parametrize("chunk_chars", [4, 40])
def test_previews_arrive_before_decode_finishes_and_never_enter_refinement(tmp_path,chunk_chars):
    app=FastAPI();sessions=[];release=threading.Event()
    def session(domain):
        value=_FakeSession();sessions.append(value);return value
    app.state.new_text_refinement_session=session
    quality_calls=[]
    def quality(source,lang):
        quality_calls.append(source);return '最终译文。'
    def stream(source,lang,on_text):
        on_text('预览文字')
        assert release.wait(3)
        return '片段译文。'
    install_translation_routes(app,quality,asr_url='http://asr.invalid',output_dir=tmp_path,stream_translate=stream)
    events=[]
    with TestClient(app).websocket_connect('/ws/translation') as ws:
        ws.send_json({'event':'start','language':'en','text':'abcdefgh','chunk_chars':chunk_chars,'interval_ms':100})
        try:
            while True:
                e=ws.receive_json();events.append(e)
                assert e['event']!='error',e
                if e['event']=='translation_preview':break
            assert events[-1]['translation']=='预览文字'
            assert events[-1]['provisional'] is True
            assert sessions[0].updates==[]
        finally:release.set()
        while events[-1]['event']!='final':
            e=ws.receive_json();events.append(e);assert e['event']!='error',e
    assert quality_calls[-1]=='abcdefgh'
    assert quality_calls.count('abcdefgh')==1
    assert all('预览文字' not in value[0] for value in sessions[0].updates)
    assert events[-1]['translation']=='最终译文。'
