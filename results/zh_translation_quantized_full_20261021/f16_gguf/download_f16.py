#!/usr/bin/env python3
"""Download the pinned official F16 GGUF anonymously and verify it."""

from __future__ import annotations

import hashlib
import json
import os
import time
import urllib.error
import urllib.request
from datetime import datetime, timezone
from pathlib import Path


REVISION = "3a0eca6ce931d88d00614d427ab3ed8ff0743902"
REPO = "IndexTeam/Index-Translate-2B-GGUF"
FILENAME = "Index-Translate-2B.f16.gguf"
EXPECTED_SIZE = 3_897_387_520
EXPECTED_SHA256 = "f9ab1279d1169ad18655a7b8085a9111bef12a0e2eb733aaa2234186241e5ad7"
URL = f"https://huggingface.co/{REPO}/resolve/{REVISION}/{FILENAME}"

HERE = Path(__file__).resolve().parent
TARGET = Path("/home/aim0/data/models/ASR/Index-Translate-2B-GGUF") / FILENAME
PARTIAL = TARGET.with_name(TARGET.name + ".partial")
LOG = HERE / "download.log"
MANIFEST = HERE / "download_manifest.json"
CHUNK = 8 * 1024 * 1024


def log(message: str) -> None:
    stamp = datetime.now(timezone.utc).isoformat()
    line = f"{stamp} {message}"
    print(line, flush=True)
    with LOG.open("a", encoding="utf-8") as stream:
        stream.write(line + "\n")
        stream.flush()
        os.fsync(stream.fileno())


def write_manifest(status: str, **extra: object) -> None:
    record: dict[str, object] = {
        "repository": REPO,
        "revision": REVISION,
        "filename": FILENAME,
        "url": URL,
        "anonymous": True,
        "token_used": False,
        "expected_size_bytes": EXPECTED_SIZE,
        "expected_sha256": EXPECTED_SHA256,
        "target": str(TARGET),
        "partial": str(PARTIAL),
        "status": status,
        "updated_utc": datetime.now(timezone.utc).isoformat(),
    }
    record.update(extra)
    tmp = MANIFEST.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(record, indent=2) + "\n", encoding="utf-8")
    tmp.replace(MANIFEST)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(CHUNK), b""):
            digest.update(block)
    return digest.hexdigest()


def download() -> None:
    TARGET.parent.mkdir(parents=True, exist_ok=True)
    if TARGET.exists():
        size = TARGET.stat().st_size
        digest = sha256(TARGET)
        if size == EXPECTED_SIZE and digest == EXPECTED_SHA256:
            log("Existing target already passes size and SHA-256 verification.")
            write_manifest("verified_existing", actual_size_bytes=size, actual_sha256=digest)
            return
        raise RuntimeError("Target already exists but does not match expected size/SHA-256; refusing to overwrite it.")

    offset = PARTIAL.stat().st_size if PARTIAL.exists() else 0
    if offset > EXPECTED_SIZE:
        raise RuntimeError("Partial file exceeds expected size; preserving it and stopping.")
    log(f"Starting anonymous download (token=False); existing partial bytes={offset}.")
    write_manifest("downloading", partial_size_bytes=offset)

    attempts = 0
    while offset < EXPECTED_SIZE:
        attempts += 1
        headers = {"User-Agent": "Index-Translate-F16-Downloader/1.0"}
        if offset:
            headers["Range"] = f"bytes={offset}-"
        request = urllib.request.Request(URL, headers=headers)
        try:
            with urllib.request.urlopen(request, timeout=90) as response:
                status = response.status
                if offset and status == 206:
                    content_range = response.headers.get("Content-Range", "")
                    if not content_range.startswith(f"bytes {offset}-"):
                        raise RuntimeError("Server returned an unexpected Content-Range; partial preserved.")
                    mode = "ab"
                elif status == 200:
                    # Some mirrors ignore Range. Restart only the partial file, never the target.
                    offset = 0
                    mode = "wb"
                else:
                    raise RuntimeError(f"Unexpected HTTP status {status}; partial preserved.")

                received = offset
                last_report = time.monotonic()
                with PARTIAL.open(mode) as output:
                    while True:
                        block = response.read(CHUNK)
                        if not block:
                            break
                        output.write(block)
                        received += len(block)
                        if time.monotonic() - last_report >= 15:
                            output.flush()
                            os.fsync(output.fileno())
                            log(f"Download progress: {received}/{EXPECTED_SIZE} bytes ({received / EXPECTED_SIZE:.1%}).")
                            write_manifest("downloading", partial_size_bytes=received)
                            last_report = time.monotonic()
                    output.flush()
                    os.fsync(output.fileno())
                offset = received
                log(f"HTTP transfer segment complete; partial bytes={offset}.")
        except (urllib.error.URLError, TimeoutError, OSError) as error:
            offset = PARTIAL.stat().st_size if PARTIAL.exists() else 0
            write_manifest("retrying", partial_size_bytes=offset, last_error_type=type(error).__name__)
            if attempts >= 8:
                raise RuntimeError(f"Download failed after {attempts} attempts ({type(error).__name__}); partial preserved.") from None
            delay = min(attempts * 5, 30)
            log(f"Transfer interrupted ({type(error).__name__}); partial bytes={offset}; retrying in {delay}s.")
            time.sleep(delay)

    actual_size = PARTIAL.stat().st_size
    actual_digest = sha256(PARTIAL)
    log(f"Transfer finished; verifying size={actual_size} and SHA-256.")
    if actual_size != EXPECTED_SIZE or actual_digest != EXPECTED_SHA256:
        write_manifest(
            "verification_failed",
            actual_size_bytes=actual_size,
            actual_sha256=actual_digest,
            partial_size_bytes=actual_size,
        )
        raise RuntimeError("Size or SHA-256 mismatch; verified-failure partial preserved.")

    PARTIAL.replace(TARGET)
    write_manifest(
        "verified",
        actual_size_bytes=actual_size,
        actual_sha256=actual_digest,
        final_path=str(TARGET),
    )
    log("Download verified and atomically moved into the requested target path.")


if __name__ == "__main__":
    try:
        download()
    except Exception as exc:
        log(f"FAILED: {exc}")
        raise SystemExit(1)
