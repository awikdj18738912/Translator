"""Index-Translate client for the local OpenAI-compatible vLLM server."""
from __future__ import annotations

import json
from time import perf_counter

import httpx


LANGUAGES = {
    "en": "英语",
    "fr": "法语",
    "de": "德语",
    "es": "西班牙语",
    "ru": "俄语",
    "ja": "日语",
    "ko": "韩语",
    "ar": "阿拉伯语",
}


class IndexVllmRuntime:
    """Small synchronous bridge that preserves the workbench translator API."""

    def __init__(self, base_url: str, model: str, *, timeout: float = 600.0):
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.client = httpx.Client(timeout=timeout, trust_env=False)

    @staticmethod
    def prompt(source: str, language: str) -> str:
        try:
            target = LANGUAGES[language]
        except (KeyError, TypeError):
            raise ValueError(f"Unsupported target language: {language!r}") from None
        return f"请将以下文本翻译为{target}，直接输出翻译结果，不要进行任何解释。\n\n{source}"

    def _payload(self, source: str, language: str, max_new_tokens: int, *, stream: bool):
        return {
            "model": self.model,
            "messages": [{"role": "user", "content": self.prompt(source, language)}],
            "temperature": 0,
            "top_p": 1,
            "max_tokens": max_new_tokens,
            "stop_token_ids": [248044, 248046],
            "chat_template_kwargs": {"enable_thinking": False},
            "stream": stream,
        }

    def _translate_one(self, source: str, language: str, max_new_tokens: int):
        response = self.client.post(
            f"{self.base_url}/chat/completions",
            json=self._payload(source, language, max_new_tokens, stream=False),
        )
        response.raise_for_status()
        result = response.json()
        choice = result["choices"][0]
        text = (choice.get("message", {}).get("content") or "").strip()
        usage = result.get("usage") or {}
        return {
            "hypothesis": text,
            "body_ids": [],
            "source_tokens": usage.get("prompt_tokens", 0),
            "target_tokens": usage.get("completion_tokens", 0),
            "unknown_tokens": 0,
            "eos": choice.get("finish_reason") == "stop",
            "replacement_characters": text.count("\ufffd"),
        }

    def translate(self, texts, language, committed=None, beams=1, max_new_tokens=512,
                  callback=None, streamer=None, source_language=None, target_language=None):
        if source_language not in (None, "zh"):
            raise ValueError("Index workbench translation source must be Chinese")
        if target_language not in (None, language):
            raise ValueError("Conflicting target language")
        if committed:
            raise ValueError("Index vLLM runtime does not accept committed token IDs")
        started = perf_counter()
        rows = [self._translate_one(text, language, max_new_tokens) for text in texts]
        return rows, perf_counter() - started

    def translate_stream(self, source, language, on_text, *, max_new_tokens=512,
                         source_language=None, target_language=None):
        if source_language not in (None, "zh"):
            raise ValueError("Index workbench translation source must be Chinese")
        if target_language not in (None, language):
            raise ValueError("Conflicting target language")
        started = perf_counter()
        body = self._payload(source, language, max_new_tokens, stream=True)
        assembled = []
        finish_reason = None
        usage = {}
        with self.client.stream("POST", f"{self.base_url}/chat/completions", json=body) as response:
            response.raise_for_status()
            for line in response.iter_lines():
                if not line or not line.startswith("data:"):
                    continue
                data = line[5:].strip()
                if data == "[DONE]":
                    break
                event = json.loads(data)
                usage.update(event.get("usage") or {})
                choices = event.get("choices") or []
                if not choices:
                    continue
                choice = choices[0]
                finish_reason = choice.get("finish_reason") or finish_reason
                delta = choice.get("delta") or {}
                chunk = delta.get("content")
                if isinstance(chunk, str) and chunk:
                    assembled.append(chunk)
                    current = "".join(assembled)
                    if "\ufffd" not in current:
                        on_text(current)

        text = "".join(assembled).strip()
        row = {
            "hypothesis": text,
            "body_ids": [],
            "source_tokens": usage.get("prompt_tokens", 0),
            "target_tokens": usage.get("completion_tokens", 0),
            "unknown_tokens": 0,
            "eos": finish_reason == "stop",
            "replacement_characters": text.count("\ufffd"),
        }
        return [row], perf_counter() - started
