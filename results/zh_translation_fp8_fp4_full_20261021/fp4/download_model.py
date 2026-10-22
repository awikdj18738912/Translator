#!/usr/bin/env python3
"""Anonymous, revision-pinned Hugging Face downloader with resumable partials."""
from __future__ import annotations
import hashlib, json, os, time, urllib.error, urllib.parse, urllib.request
from datetime import datetime, timezone
from pathlib import Path

REPO = os.environ.get("HF_REPO", "IndexTeam/Index-Translate-2B-FP4")
REVISION = os.environ.get("HF_REVISION", "bf49b475ce2b75307f4da835a9d2e06758a6f040")
TARGET = Path(os.environ.get("HF_TARGET", "/home/aim0/data/models/ASR/Index-Translate-2B-FP4"))
HERE = Path(__file__).resolve().parent
LOG = HERE / "download.log"
MANIFEST = HERE / "download_manifest.json"
API = f"https://huggingface.co/api/models/{REPO}/tree/{REVISION}?recursive=true&blobs=true"
CHUNK = 8 * 1024 * 1024

def now(): return datetime.now(timezone.utc).isoformat()
def log(s):
    line=f"{now()} {s}"
    print(line, flush=True)
    with LOG.open("a",encoding="utf-8") as f: f.write(line+"\n"); f.flush(); os.fsync(f.fileno())
def request(url, headers=None):
    return urllib.request.urlopen(urllib.request.Request(url, headers={"User-Agent":"Index-Translate-Anonymous-Downloader/1.0", **(headers or {})}), timeout=120)
def git_blob_sha1(path):
    h=hashlib.sha1(); size=path.stat().st_size; h.update(f"blob {size}\0".encode())
    with path.open("rb") as f:
        for b in iter(lambda:f.read(CHUNK),b""): h.update(b)
    return h.hexdigest()
def sha256(path):
    h=hashlib.sha256()
    with path.open("rb") as f:
        for b in iter(lambda:f.read(CHUNK),b""): h.update(b)
    return h.hexdigest()
def write_manifest(status, files, **extra):
    obj={"repository":REPO,"revision":REVISION,"api_url":API,"anonymous":True,"token_used":False,"target":str(TARGET),"status":status,"updated_utc":now(),"files":files,**extra}
    tmp=MANIFEST.with_suffix(".json.tmp"); tmp.write_text(json.dumps(obj,indent=2,ensure_ascii=False)+"\n",encoding="utf-8"); tmp.replace(MANIFEST)
def get_listing():
    with request(API) as r: data=json.load(r)
    files=[x for x in data if x.get("type")=="file"]
    if not files: raise RuntimeError("API returned no files")
    return files
def fetch_file(entry, records):
    rel=entry["path"]
    if rel.startswith("/") or ".." in Path(rel).parts: raise RuntimeError(f"Unsafe repository path: {rel}")
    dest=TARGET/rel; dest.parent.mkdir(parents=True,exist_ok=True)
    expected=int(entry["size"]); lfs=entry.get("lfs") or {}; expected_sha=lfs.get("oid")
    # For ordinary Git blobs verify the API oid using Git's blob hash convention.
    expected_oid=entry.get("oid")
    rec={"path":rel,"size_bytes":expected,"api_oid":expected_oid,"lfs_sha256":expected_sha,"status":"pending"}
    records[rel]=rec; write_manifest("downloading",records)
    if dest.exists():
        if dest.stat().st_size != expected: raise RuntimeError(f"Existing target differs in size; refusing overwrite: {dest}")
        valid=sha256(dest)==expected_sha if expected_sha else git_blob_sha1(dest)==expected_oid
        if not valid: raise RuntimeError(f"Existing target differs from pinned content; refusing overwrite: {dest}")
        rec.update(status="verified_existing",actual_size_bytes=expected,actual_sha256=sha256(dest),actual_git_blob_oid=git_blob_sha1(dest)); log(f"Already verified: {rel}"); write_manifest("downloading",records); return
    partial=dest.with_name(dest.name+".partial")
    offset=partial.stat().st_size if partial.exists() else 0
    if offset>expected: raise RuntimeError(f"Partial exceeds expected size; preserved: {partial}")
    url=f"https://huggingface.co/{REPO}/resolve/{REVISION}/"+urllib.parse.quote(rel,safe="/")
    tries=0
    while offset<expected:
        tries+=1
        try:
            headers={"Range":f"bytes={offset}-"} if offset else {}
            with request(url,headers) as r:
                if offset and r.status==206:
                    if not r.headers.get("Content-Range","").startswith(f"bytes {offset}-"): raise RuntimeError("Unexpected Content-Range")
                    mode="ab"
                elif r.status==200: offset=0; mode="wb"
                else: raise RuntimeError(f"Unexpected HTTP status {r.status}")
                got=offset; last=time.monotonic()
                with partial.open(mode) as out:
                    while True:
                        b=r.read(CHUNK)
                        if not b: break
                        out.write(b); got+=len(b)
                        if time.monotonic()-last>=15:
                            out.flush(); os.fsync(out.fileno()); rec.update(status="downloading",partial_size_bytes=got); write_manifest("downloading",records); log(f"{rel}: {got}/{expected} bytes ({got/expected:.1%})"); last=time.monotonic()
                    out.flush(); os.fsync(out.fileno())
                offset=got
            if offset>expected: raise RuntimeError(f"Transfer exceeded expected size for {rel}")
        except (urllib.error.URLError,TimeoutError,OSError) as e:
            offset=partial.stat().st_size if partial.exists() else 0
            rec.update(status="retrying",partial_size_bytes=offset,last_error=type(e).__name__); write_manifest("downloading",records)
            if tries>=10: raise RuntimeError(f"Failed after {tries} attempts for {rel}; partial preserved") from None
            delay=min(tries*5,60); log(f"{rel}: transfer interrupted ({type(e).__name__}); retry in {delay}s"); time.sleep(delay)
    actual=partial.stat().st_size
    actual_sha=sha256(partial); actual_oid=git_blob_sha1(partial)
    valid=(actual==expected and (actual_sha==expected_sha if expected_sha else actual_oid==expected_oid))
    rec.update(actual_size_bytes=actual,actual_sha256=actual_sha,actual_git_blob_oid=actual_oid)
    if not valid:
        rec["status"]="verification_failed"; write_manifest("verification_failed",records); raise RuntimeError(f"Verification failed for {rel}; partial preserved")
    # Publish atomically without replacing any existing path (checked above).
    if dest.exists(): raise RuntimeError(f"Target appeared during download; preserving both files: {dest}")
    partial.rename(dest); rec["status"]="verified"; write_manifest("downloading",records); log(f"Verified and published: {rel} ({actual} bytes)")
def main():
    TARGET.mkdir(parents=True,exist_ok=True)
    files=get_listing(); records={}
    # Persist full API metadata before downloading.
    for f in files:
        records[f["path"]]={"path":f["path"],"size_bytes":f.get("size"),"api_oid":f.get("oid"),"lfs_sha256":(f.get("lfs") or {}).get("oid"),"status":"queued"}
    write_manifest("metadata_fetched",records,file_count=len(files))
    total=sum(int(f["size"]) for f in files); log(f"Anonymous metadata fetched: {len(files)} files, {total} bytes; revision={REVISION}")
    for f in files: fetch_file(f,records)
    write_manifest("complete",records,file_count=len(files),total_size_bytes=total)
    log(f"COMPLETE: {len(files)} files verified in {TARGET}")
if __name__=="__main__":
    try: main()
    except Exception as e:
        log(f"FAILED: {type(e).__name__}: {e}")
        raise SystemExit(1)
