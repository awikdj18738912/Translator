from contextlib import contextmanager
from datetime import datetime, timezone
import json
from pathlib import Path
import sqlite3
from uuid import uuid4

from .schemas import EntityInput


def now():
    return datetime.now(timezone.utc).isoformat()


class QueueFull(Exception):
    pass


class Store:
    def __init__(self, path: Path):
        path.parent.mkdir(parents=True, exist_ok=True)
        self.path = path
        with self.connect() as db:
            db.execute("PRAGMA journal_mode=WAL")
            db.executescript("""
                CREATE TABLE IF NOT EXISTS entities (
                    id INTEGER PRIMARY KEY, canonical_zh TEXT NOT NULL,
                    domain TEXT NOT NULL, language TEXT NOT NULL, payload TEXT NOT NULL,
                    UNIQUE(canonical_zh, domain, language)
                );
                CREATE TABLE IF NOT EXISTS jobs (
                    id TEXT PRIMARY KEY, status TEXT NOT NULL, stage TEXT NOT NULL,
                    created_at TEXT NOT NULL, updated_at TEXT NOT NULL,
                    request TEXT NOT NULL, result TEXT NOT NULL DEFAULT '{}', error TEXT
                );
                CREATE INDEX IF NOT EXISTS idx_job_status_created ON jobs(status, created_at);
            """)

    @contextmanager
    def connect(self):
        db = sqlite3.connect(self.path, timeout=10)
        db.row_factory = sqlite3.Row
        try:
            with db:
                yield db
        finally:
            db.close()

    def list_entities(self, domain=None):
        with self.connect() as db:
            rows = db.execute("SELECT id, payload FROM entities ORDER BY id").fetchall()
        items = [{"id": r["id"], **json.loads(r["payload"])} for r in rows]
        return [e for e in items if domain is None or e["domain"] in {"general", domain}]

    def upsert_entities(self, entities: list[EntityInput]):
        ids = []
        with self.connect() as db:
            for e in entities:
                row = db.execute("""
                    INSERT INTO entities(canonical_zh, domain, language, payload) VALUES(?,?,?,?)
                    ON CONFLICT(canonical_zh,domain,language) DO UPDATE SET payload=excluded.payload
                    RETURNING id
                """, (e.canonical_zh, e.domain, e.language, e.model_dump_json())).fetchone()
                ids.append(row[0])
        return ids

    def delete_entity(self, entity_id):
        with self.connect() as db:
            return db.execute("DELETE FROM entities WHERE id=?", (entity_id,)).rowcount > 0

    def create_job(self, request, max_pending):
        job_id, timestamp = uuid4().hex, now()
        with self.connect() as db:
            db.execute("BEGIN IMMEDIATE")
            count = db.execute("SELECT COUNT(*) FROM jobs WHERE status IN ('queued','running')").fetchone()[0]
            if count >= max_pending:
                raise QueueFull()
            db.execute("INSERT INTO jobs(id,status,stage,created_at,updated_at,request) VALUES(?,?,?,?,?,?)",
                       (job_id, "queued", "queued", timestamp, timestamp, json.dumps(request, ensure_ascii=False)))
        return self.get_job(job_id)

    def get_job(self, job_id, private=False):
        with self.connect() as db:
            row = db.execute("SELECT * FROM jobs WHERE id=?", (job_id,)).fetchone()
        if not row:
            return None
        value = dict(row)
        value["request"] = json.loads(value["request"])
        value["result"] = json.loads(value["result"])
        if not private:
            value["request"].pop("audio_path", None)
        return value

    def list_jobs(self):
        with self.connect() as db:
            return [dict(r) for r in db.execute(
                "SELECT id,status,stage,created_at,updated_at,error FROM jobs ORDER BY created_at DESC LIMIT 50"
            )]

    def update_job(self, job_id, *, status=None, stage=None, result=None, error=None):
        values = {"updated_at": now()}
        for key, value in (("status", status), ("stage", stage), ("result", result), ("error", error)):
            if value is not None:
                values[key] = json.dumps(value, ensure_ascii=False) if key == "result" else value
        with self.connect() as db:
            db.execute(f"UPDATE jobs SET {','.join(f'{k}=?' for k in values)} WHERE id=?", (*values.values(), job_id))

    def claim_next(self):
        with self.connect() as db:
            db.execute("BEGIN IMMEDIATE")
            row = db.execute("SELECT id FROM jobs WHERE status='queued' ORDER BY created_at LIMIT 1").fetchone()
            if not row:
                return None
            db.execute("UPDATE jobs SET status='running',updated_at=? WHERE id=?", (now(), row[0]))
        return self.get_job(row[0], private=True)

    def recover(self):
        with self.connect() as db:
            db.execute("""UPDATE jobs SET status='failed',stage='interrupted',error=?,updated_at=?
                          WHERE status='running'""", ("服务中断，请重新提交；已完成阶段的结果仍保留。", now()))

    def active_audio_paths(self):
        with self.connect() as db:
            rows = db.execute("SELECT request FROM jobs WHERE status='queued'").fetchall()
        return {json.loads(r[0]).get("audio_path") for r in rows}

