import asyncio
from dataclasses import replace
from pathlib import Path
import time

from fastapi.testclient import TestClient
import pytest

from translator.app import create_app
from translator.config import ModelConfig, Settings
from translator.providers import ModelError
from translator.schemas import EntityInput
from translator.store import QueueFull, Store


class Chat:
    def __init__(self, settings, client):
        pass

    async def generate(self, config, system, user):
        if config.model == "translation":
            return "请在10天内交付20件。"
        return user

    async def readiness(self, config):
        return {"ready":True,"model":config.model,"message":"测试模型"}


class ASR:
    def __init__(self, settings):
        pass

    async def transcribe(self, path, language):
        assert path.exists()
        return {"text":"Deliver 20 items within 10 days.","language":"en","segments":[],"duration":3}

    def status(self):
        return {"installed":True,"loaded":True,"model":"test"}


@pytest.fixture
def settings(tmp_path):
    return Settings(data_dir=tmp_path, translation=ModelConfig(model="translation"),
                    refiner=ModelConfig(model="refiner"), max_upload_mb=1)


def wait_done(client, job_id):
    deadline = time.monotonic() + 5
    while time.monotonic() < deadline:
        response = client.get(f"/api/jobs/{job_id}")
        assert response.status_code == 200
        job = response.json()
        if job["status"] not in {"queued", "running"}:
            return job
        time.sleep(.01)
    pytest.fail("Job did not finish within five seconds")


def test_text_job_export_history_and_glossary_persist(settings):
    with TestClient(create_app(settings,chat_factory=Chat,asr_factory=ASR)) as client:
        assert client.get("/").status_code == 200
        assert client.get("/static/app.js").status_code == 200
        assert client.get("/api/health").json()["mode"] == "local"
        assert client.get("/api/readiness").json()["translation"]["ready"]
        response = client.post("/api/jobs/text",json={"text":"Deliver 20 items within 10 days.","language":"en"})
        assert response.status_code == 202
        job_id = response.json()["id"]
        job = wait_done(client, job_id)
        assert job["status"] == "completed"
        assert job["result"]["refined_text"] == "请在10天内交付20件。"
        assert not job["result"]["needs_review"]
        assert "精修结果" in client.get(f"/api/jobs/{job_id}/export?format=txt").text
        assert client.get(f"/api/jobs/{job_id}/export").json()["id"] == job_id
        assert client.get(f"/api/jobs/{job_id}/export?format=csv").status_code == 422
        entity = {"canonical_zh":"最低起订量","source_aliases":["MOQ"],"language":"en"}
        entity_id = client.post("/api/entities",json=entity).json()["id"]
        assert client.post("/api/entities",json=entity).json()["id"] == entity_id
        assert len(client.get("/api/entities").json()) == 1
    with TestClient(create_app(settings,chat_factory=Chat,asr_factory=ASR)) as client:
        assert len(client.get("/api/entities").json()) == 1
        assert client.get("/api/jobs").json()[0]["id"] == job_id
        assert client.delete(f"/api/entities/{entity_id}").status_code == 200
        assert client.delete(f"/api/entities/{entity_id}").status_code == 404


def test_audio_job_cleanup_and_private_path_not_exposed(settings):
    with TestClient(create_app(settings,chat_factory=Chat,asr_factory=ASR)) as client:
        response = client.post("/api/jobs/audio",files={"file":("../customer.wav",b"test-audio","audio/wav")},data={"language":"auto"})
        assert response.status_code == 202
        assert "audio_path" not in response.json()["request"]
        job = wait_done(client,response.json()["id"])
        assert job["status"] == "completed"
        assert job["request"]["filename"] == "customer.wav"
        assert job["result"]["asr"]["duration"] == 3
        assert not list((settings.data_dir / "uploads").iterdir())


def test_invalid_uploads_and_inputs(settings):
    with TestClient(create_app(settings,chat_factory=Chat,asr_factory=ASR)) as client:
        assert client.post("/api/jobs/audio",files={"file":("x.exe",b"x")}).status_code == 415
        assert client.post("/api/jobs/audio",files={"file":("x.wav",b"")}).status_code == 422
        assert client.post("/api/jobs/audio",files={"file":("x.wav",b"x")},data={"language":"xx"}).status_code == 422
        assert client.post("/api/jobs/audio",files={"file":("x.wav",b"x"*(1024*1024+1))}).status_code == 413
        assert client.post("/api/jobs/text",json={"text":"  "}).status_code == 422
        assert client.post("/api/jobs/text",json={"text":"__TR_unsafe"}).status_code == 422
        assert client.post("/api/jobs/text",json={"text":"Hello","language":"xx"}).status_code == 422
        assert client.get("/api/jobs/unknown").status_code == 404
        assert not list((settings.data_dir / "uploads").iterdir())


def test_loopback_browser_security_and_body_limit(settings):
    with TestClient(create_app(settings,chat_factory=Chat,asr_factory=ASR)) as client:
        assert client.post("/api/jobs/text",json={"text":"Hello"},headers={"Origin":"https://remote.example"}).status_code == 403
        assert client.get("/api/health",headers={"Host":"evil.example"}).status_code == 400
        assert client.post("/api/jobs/text",content=b"{}",headers={"Content-Length":str(3*1024*1024)}).status_code == 413
        assert "frame-ancestors 'none'" in client.get("/").headers["content-security-policy"]


def test_audio_failure_preserves_error_and_cleans_file(settings):
    class BrokenASR(ASR):
        async def transcribe(self, path, language):
            raise ModelError("无法识别录音")
    with TestClient(create_app(settings,chat_factory=Chat,asr_factory=BrokenASR)) as client:
        response = client.post("/api/jobs/audio",files={"file":("x.wav",b"x")})
        job = wait_done(client,response.json()["id"])
        assert job["status"] == "failed" and job["error"] == "无法识别录音"
        assert not list((settings.data_dir / "uploads").iterdir())


def test_bulk_import_is_validated_before_any_write(settings):
    with TestClient(create_app(settings,chat_factory=Chat,asr_factory=ASR)) as client:
        response = client.post("/api/entities/import",json={"entities":[{"canonical_zh":"有效"},{"canonical_zh":""}]})
        assert response.status_code == 422
        assert client.get("/api/entities").json() == []
        assert client.post("/api/entities/import",json={"entities":[{"canonical_zh":"有效"}]}).status_code == 200


def test_queue_limit_is_atomic_and_restart_marks_running_interrupted(tmp_path):
    store = Store(tmp_path / "jobs.sqlite3")
    job = store.create_job({"text":"x"},1)
    with pytest.raises(QueueFull):
        store.create_job({"text":"y"},1)
    assert store.claim_next()["id"] == job["id"]
    store.update_job(job["id"],result={"source_text":"retained"})
    store.recover()
    recovered = store.get_job(job["id"])
    assert recovered["status"] == "failed" and recovered["stage"] == "interrupted"
    assert recovered["result"]["source_text"] == "retained"
    assert store.create_job({"text":"y"},1)["status"] == "queued"

