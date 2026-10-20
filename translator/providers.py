import asyncio
import importlib.util
from pathlib import Path
from threading import Lock

import httpx

from .config import ModelConfig, Settings


class ModelError(Exception):
    """User-visible message; never include endpoint response bodies or credentials."""


class LocalChat:
    def __init__(self, settings: Settings, client: httpx.AsyncClient):
        self.settings, self.client = settings, client

    async def generate(self, config: ModelConfig, system: str, user: str):
        messages = [{"role": "system", "content": system}, {"role": "user", "content": user}]
        headers = {"Authorization": f"Bearer {config.api_key}"} if config.api_key else {}
        if config.backend == "ollama":
            path, payload = "/api/chat", {
                "model": config.model, "messages": messages, "stream": False, "think": False,
                "keep_alive": "5m", "options": {"temperature": 0, "num_predict": 2048,
                    "num_ctx": self.settings.ollama_num_ctx, "num_gpu": self.settings.ollama_num_gpu},
            }
        else:
            path, payload = "/chat/completions", {
                "model": config.model, "messages": messages, "stream": False,
                "temperature": 0, "max_tokens": 2048,
            }
        try:
            response = await self.client.post(config.base_url.rstrip("/") + path, json=payload, headers=headers)
            response.raise_for_status()
            body = response.json()
            if config.backend == "ollama":
                if body.get("done_reason") == "length" or body.get("done") is not True:
                    raise ModelError("模型输出被截断，请减小 CHUNK_CHARS。")
                content = body["message"]["content"]
            else:
                choice = body["choices"][0]
                if choice.get("finish_reason") != "stop":
                    raise ModelError("模型未正常完成输出，请检查上下文或生成长度。")
                content = choice["message"]["content"]
            if not isinstance(content, str) or not content.strip() or len(content) > 16000:
                raise ModelError("模型返回了空白或过长的输出。")
            if "<think>" in content or "</think>" in content:
                raise ModelError("模型返回了思考过程，请使用支持关闭思考的指令模型。")
            return content.strip()
        except httpx.TimeoutException as exc:
            raise ModelError("本地模型处理超时，可缩短输入或换用更小的模型。") from exc
        except httpx.HTTPStatusError as exc:
            raise ModelError(f"本地模型接口返回 HTTP {exc.response.status_code}；请检查服务与模型名称。") from exc
        except httpx.RequestError as exc:
            raise ModelError("无法连接本地模型服务，请启动 Ollama 并下载配置的模型。") from exc
        except (KeyError, IndexError, TypeError, ValueError) as exc:
            raise ModelError("模型接口响应格式不符合配置的协议。") from exc

    async def readiness(self, config):
        path = "/api/tags" if config.backend == "ollama" else "/models"
        headers = {"Authorization": f"Bearer {config.api_key}"} if config.api_key else {}
        try:
            response = await self.client.get(config.base_url.rstrip("/") + path, headers=headers, timeout=3)
            response.raise_for_status()
            body = response.json()
            names = [m["name"] for m in body["models"]] if config.backend == "ollama" else [m["id"] for m in body["data"]]
            ready = config.model in names or config.model + ":latest" in names
            return {"ready": ready, "model": config.model, "message": "模型已就绪" if ready else "服务可连接，但未找到配置的模型"}
        except (httpx.HTTPError, ValueError, KeyError, TypeError):
            return {"ready": False, "model": config.model, "message": "模型服务未就绪"}


class WhisperASR:
    def __init__(self, settings: Settings):
        self.settings, self.model = settings, None
        self.lock = Lock()

    def status(self):
        return {"installed": importlib.util.find_spec("faster_whisper") is not None,
                "loaded": self.model is not None, "model": self.settings.asr_model,
                "device": self.settings.asr_device, "compute_type": self.settings.asr_compute_type}

    async def transcribe(self, path: Path, language: str):
        return await asyncio.to_thread(self._transcribe, path, language)

    def _transcribe(self, path, language):
        with self.lock:
            try:
                from faster_whisper import WhisperModel
            except ImportError as exc:
                raise ModelError('未安装 ASR 依赖，请执行 pip install -e ".[asr]"。') from exc
            # Decode incrementally: WebM microphone recordings often have no duration metadata.
            try:
                import av
                import numpy as np
                arrays, samples = [], 0
                with av.open(str(path)) as container:
                    duration = container.duration / av.time_base if container.duration is not None else None
                    if duration and duration > self.settings.max_audio_seconds:
                        raise ModelError(f"音频超过 {self.settings.max_audio_seconds} 秒上限，请分段上传。")
                    resampler = av.AudioResampler(format="fltp", layout="mono", rate=16000)
                    def append_frames(frames):
                        nonlocal samples
                        for frame in frames:
                            samples += frame.samples
                            if samples > self.settings.max_audio_seconds * 16000:
                                raise ModelError("解码后的音频时长超过限制，请分段上传。")
                            arrays.append(frame.to_ndarray().reshape(-1))
                    for frame in container.decode(audio=0):
                        append_frames(resampler.resample(frame))
                    append_frames(resampler.resample(None))
                if not arrays:
                    raise ModelError("音频没有可解码的采样。")
                audio = np.concatenate(arrays)
            except ModelError:
                raise
            except Exception as exc:
                raise ModelError("无法解码音频，请检查文件是否为有效录音。") from exc
            try:
                if self.model is None:
                    self.model = WhisperModel(self.settings.asr_model, device=self.settings.asr_device,
                        compute_type=self.settings.asr_compute_type, cpu_threads=self.settings.asr_threads,
                        num_workers=1, local_files_only=self.settings.asr_local_only)
                segments, info = self.model.transcribe(audio, language=None if language == "auto" else language,
                    task="transcribe", vad_filter=True, beam_size=3, condition_on_previous_text=False)
                rows = [{"start": s.start, "end": s.end, "text": s.text.strip(),
                         "avg_logprob": s.avg_logprob, "no_speech_prob": s.no_speech_prob} for s in segments]
                separator = "" if info.language in {"zh", "ja"} else " "
                text = separator.join(r["text"] for r in rows if r["text"])
                return {"text": text, "language": info.language, "segments": rows,
                        "language_probability": info.language_probability, "duration": len(audio) / 16000}
            except Exception as exc:
                raise ModelError("本地 ASR 推理失败，请检查模型权重、依赖及可用内存。") from exc

