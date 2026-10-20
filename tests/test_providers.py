import asyncio
import json

import httpx
import pytest

from translator.config import ModelConfig, Settings
from translator.providers import LocalChat, ModelError


def test_ollama_request_is_local_cpu_non_streaming_and_non_thinking():
    async def run():
        def handle(request):
            assert str(request.url) == "http://127.0.0.1:11434/api/chat"
            body = json.loads(request.content)
            assert body["think"] is False and body["stream"] is False
            assert body["options"]["num_gpu"] == 0
            return httpx.Response(200, json={"message":{"content":"译文"},"done":True,"done_reason":"stop"})
        async with httpx.AsyncClient(transport=httpx.MockTransport(handle)) as client:
            return await LocalChat(Settings(), client).generate(ModelConfig(), "system", "source")
    assert asyncio.run(run()) == "译文"


@pytest.mark.parametrize("body", [
    {"message":{"content":"不完整"},"done":True,"done_reason":"length"},
    {"message":{"content":"<think>内部推理</think>译文"},"done":True},
    {"message":{"content":""},"done":True},
    {"unexpected":True},
])
def test_malformed_or_truncated_model_output_is_rejected(body):
    async def run():
        async with httpx.AsyncClient(transport=httpx.MockTransport(lambda _: httpx.Response(200,json=body))) as client:
            await LocalChat(Settings(), client).generate(ModelConfig(), "system", "source")
    with pytest.raises(ModelError):
        asyncio.run(run())


def test_server_error_does_not_disclose_upstream_content_or_key():
    async def run():
        async with httpx.AsyncClient(transport=httpx.MockTransport(lambda _: httpx.Response(500,text="SECRET CUSTOMER TEXT"))) as client:
            await LocalChat(Settings(), client).generate(ModelConfig(api_key="SECRET_KEY"), "system", "source")
    with pytest.raises(ModelError) as error:
        asyncio.run(run())
    assert "500" in str(error.value)
    assert "SECRET" not in str(error.value)


def test_compatible_provider_and_auth():
    async def run():
        def handle(request):
            assert request.url.path == "/v1/chat/completions"
            assert request.headers["authorization"] == "Bearer local-key"
            return httpx.Response(200,json={"choices":[{"message":{"content":"译文"},"finish_reason":"stop"}]})
        async with httpx.AsyncClient(transport=httpx.MockTransport(handle)) as client:
            return await LocalChat(Settings(), client).generate(ModelConfig(backend="compatible",base_url="http://localhost:8080/v1",api_key="local-key"),"system","source")
    assert asyncio.run(run()) == "译文"


def test_default_settings_reject_remote_model_service():
    with pytest.raises(ValueError, match="remote"):
        Settings(translation=ModelConfig(base_url="https://example.com")).validate()
    Settings(translation=ModelConfig(base_url="https://example.com"), allow_remote_models=True).validate()
    assert "local-key" not in repr(ModelConfig(api_key="local-key"))

