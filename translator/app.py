import asyncio
from contextlib import asynccontextmanager
import logging
from pathlib import Path
from uuid import uuid4

from dotenv import load_dotenv
from fastapi import FastAPI, File, Form, HTTPException, Request, UploadFile
from fastapi.responses import FileResponse, JSONResponse, Response
from fastapi.staticfiles import StaticFiles
import httpx
from pydantic import ValidationError
from starlette.middleware.trustedhost import TrustedHostMiddleware

from .config import Settings
from .pipeline import Pipeline
from .providers import LocalChat, ModelError, WhisperASR
from .schemas import EntityBatch, EntityInput, LANGUAGES, TextRequest
from .store import QueueFull, Store

logger = logging.getLogger(__name__)
STATIC = Path(__file__).parent / "static"
EXTENSIONS = {".wav", ".mp3", ".m4a", ".mp4", ".ogg", ".opus", ".webm", ".flac", ".aac"}


class BodyLimit:
    def __init__(self, app, limit):
        self.app, self.limit = app, limit

    async def __call__(self, scope, receive, send):
        if scope["type"] != "http":
            return await self.app(scope, receive, send)
        headers = dict(scope["headers"])
        try:
            size = int(headers.get(b"content-length", b"0"))
        except ValueError:
            size = self.limit + 1
        if size > self.limit:
            return await JSONResponse({"detail": "请求体超过上传上限"}, status_code=413)(scope, receive, send)
        received = 0
        async def limited_receive():
            nonlocal received
            message = await receive()
            received += len(message.get("body", b""))
            if received > self.limit:
                raise HTTPException(413, "请求体超过上传上限")
            return message
        await self.app(scope, limited_receive, send)


class Worker:
    def __init__(self, store, pipeline):
        self.store, self.pipeline = store, pipeline
        self.wake, self.closing = asyncio.Event(), False

    async def run(self):
        while not self.closing:
            self.wake.clear()
            job = self.store.claim_next()
            if job is None:
                try:
                    await asyncio.wait_for(self.wake.wait(), timeout=1)
                except asyncio.TimeoutError:
                    pass
                continue
            try:
                await self.pipeline.run(job)
            except ModelError as exc:
                self.store.update_job(job["id"], status="failed", error=str(exc))
            except Exception as exc:
                # Avoid logging customer content, endpoint URLs or upstream error bodies.
                logger.error("Job %s failed (%s)", job["id"], type(exc).__name__)
                self.store.update_job(job["id"], status="failed", error="处理发生内部错误；已保存阶段结果，请检查服务日志。")
            finally:
                if job["request"].get("audio_path"):
                    Path(job["request"]["audio_path"]).unlink(missing_ok=True)


def create_app(settings=None, *, chat_factory=LocalChat, asr_factory=WhisperASR):
    if settings is None:
        load_dotenv()
        settings = Settings.from_env()
    settings.validate()

    @asynccontextmanager
    async def lifespan(app):
        settings.data_dir.mkdir(parents=True, exist_ok=True)
        uploads = settings.data_dir / "uploads"
        uploads.mkdir(exist_ok=True)
        store = Store(settings.data_dir / "translator.sqlite3")
        store.recover()
        active = store.active_audio_paths()
        for path in uploads.iterdir():
            if path.is_file() and str(path.resolve()) not in active:
                path.unlink()
        async with httpx.AsyncClient(timeout=settings.model_timeout_seconds, trust_env=False, follow_redirects=False) as client:
            chat, asr = chat_factory(settings, client), asr_factory(settings)
            pipeline = Pipeline(settings, store, chat, asr)
            worker = Worker(store, pipeline)
            app.state.store, app.state.worker, app.state.chat, app.state.asr = store, worker, chat, asr
            task = asyncio.create_task(worker.run())
            try:
                yield
            finally:
                worker.closing = True
                worker.wake.set()
                await task  # Let the current CPU inference finish before closing its HTTP client.

    app = FastAPI(title="外贸语音译站", version="0.1.0", lifespan=lifespan)
    app.add_middleware(BodyLimit, limit=settings.max_upload_mb * 1024 * 1024 + 1024 * 1024)
    app.add_middleware(TrustedHostMiddleware, allowed_hosts=["localhost", "127.0.0.1", "[::1]", "testserver"])

    @app.middleware("http")
    async def local_browser_guard(request: Request, call_next):
        # Prevent a remote website from posting to this unauthenticated local service.
        origin = request.headers.get("origin")
        if request.method not in {"GET", "HEAD", "OPTIONS"} and origin and origin.rstrip("/") != str(request.base_url).rstrip("/"):
            return JSONResponse({"detail": "仅接受同源操作"}, status_code=403)
        response = await call_next(request)
        response.headers["X-Content-Type-Options"] = "nosniff"
        response.headers["Referrer-Policy"] = "no-referrer"
        response.headers["Content-Security-Policy"] = "default-src 'self'; script-src 'self'; style-src 'self'; media-src 'self' blob:; connect-src 'self'; img-src 'self' data:; frame-ancestors 'none'"
        return response

    @app.exception_handler(QueueFull)
    async def queue_full(request, exc):
        return JSONResponse({"detail": "本地任务队列已满，请等待当前任务完成。"}, status_code=429)

    @app.get("/", include_in_schema=False)
    async def index():
        return FileResponse(STATIC / "index.html")

    @app.get("/api/health")
    async def health(request: Request):
        # Independent from model readiness, useful for process supervision.
        return {"status": "ok", "mode": "local", "refiner_enabled": settings.refiner_enabled,
                "max_audio_seconds": settings.max_audio_seconds, "max_upload_mb": settings.max_upload_mb,
                "languages": LANGUAGES}

    @app.get("/api/readiness")
    async def readiness(request: Request):
        translation, refiner = await asyncio.gather(
            request.app.state.chat.readiness(settings.translation), request.app.state.chat.readiness(settings.refiner))
        return {"translation": translation, "refiner": refiner, "refiner_enabled": settings.refiner_enabled,
                "asr": request.app.state.asr.status()}

    @app.post("/api/jobs/text", status_code=202)
    async def text_job(body: TextRequest, request: Request):
        if "__TR_" in body.text:
            raise HTTPException(422, "输入含有保留的内部标记 __TR_")
        job = request.app.state.store.create_job({"kind": "text", **body.model_dump()}, settings.max_pending_jobs)
        request.app.state.worker.wake.set()
        return job

    @app.post("/api/jobs/audio", status_code=202)
    async def audio_job(request: Request, file: UploadFile = File(...), language: str = Form("auto"), domain: str = Form("general")):
        try:
            metadata = TextRequest(text="audio", language=language, domain=domain)
        except ValidationError:
            raise HTTPException(422, "语言或领域参数无效") from None
        suffix = Path(file.filename or "").suffix.lower()
        if suffix not in EXTENSIONS:
            raise HTTPException(415, "不支持此音频格式")
        path = (settings.data_dir / "uploads" / f"{uuid4().hex}{suffix}").resolve()
        accepted = False
        try:
            size = 0
            with path.open("xb") as output:
                while data := await file.read(1024 * 1024):
                    size += len(data)
                    if size > settings.max_upload_mb * 1024 * 1024:
                        raise HTTPException(413, "音频文件超过上传上限")
                    output.write(data)
            if not size:
                raise HTTPException(422, "音频文件为空")
            job = request.app.state.store.create_job({"kind": "audio", "audio_path": str(path),
                "filename": Path(file.filename).name, "language": metadata.language, "domain": metadata.domain}, settings.max_pending_jobs)
            accepted = True
            request.app.state.worker.wake.set()
            return job
        finally:
            await file.close()
            if not accepted:
                path.unlink(missing_ok=True)

    @app.get("/api/jobs")
    async def jobs(request: Request):
        return request.app.state.store.list_jobs()

    def find_job(request, job_id):
        job = request.app.state.store.get_job(job_id)
        if job is None:
            raise HTTPException(404, "任务不存在")
        return job

    @app.get("/api/jobs/{job_id}")
    async def get_job(job_id: str, request: Request):
        return find_job(request, job_id)

    @app.get("/api/jobs/{job_id}/export")
    async def export_job(job_id: str, request: Request, format: str = "json"):
        job = find_job(request, job_id)
        if format == "json":
            return JSONResponse(job, headers={"Content-Disposition": f'attachment; filename="{job["id"]}.json"'})
        if format != "txt":
            raise HTTPException(422, "导出格式只能为 json 或 txt")
        r = job["result"]
        text = "原文\n" + r.get("source_text", "") + "\n\n中文译文\n" + r.get("translation", "") + "\n\n精修结果\n" + r.get("refined_text", "")
        if r.get("warnings"):
            text += "\n\n待核对\n" + "\n".join(r["warnings"])
        if job.get("error"):
            text += "\n\n任务错误\n" + job["error"]
        return Response(text, media_type="text/plain; charset=utf-8", headers={"Content-Disposition": f'attachment; filename="{job["id"]}.txt"'})

    @app.get("/api/entities")
    async def list_entities(request: Request, domain: str | None = None):
        return request.app.state.store.list_entities(domain)

    @app.post("/api/entities")
    async def add_entity(body: EntityInput, request: Request):
        return {"id": request.app.state.store.upsert_entities([body])[0]}

    @app.post("/api/entities/import")
    async def import_entities(body: EntityBatch, request: Request):
        return {"ids": request.app.state.store.upsert_entities(body.entities)}

    @app.delete("/api/entities/{entity_id}")
    async def delete_entity(entity_id: int, request: Request):
        if not request.app.state.store.delete_entity(entity_id):
            raise HTTPException(404, "术语不存在")
        return {"deleted": True}

    app.mount("/static", StaticFiles(directory=STATIC), name="static")
    return app

