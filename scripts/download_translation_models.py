"""Download pinned translation checkpoints and verify Hub LFS SHA256 digests."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import time
import urllib.request

from huggingface_hub import HfApi


def fetch(url, destination, size=None):
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists() and size is not None and destination.stat().st_size == size:
        return
    partial = destination.with_name(destination.name + ".partial")
    for attempt in range(4):
        offset = partial.stat().st_size if partial.exists() else 0
        headers = {"User-Agent":"translation-model-evaluation/1.0"}
        if offset:
            headers["Range"] = f"bytes={offset}-"
        try:
            with urllib.request.urlopen(urllib.request.Request(url, headers=headers), timeout=120) as response:
                append = offset > 0 and response.status == 206
                with partial.open("ab" if append else "wb") as output:
                    while chunk := response.read(4 * 1024 * 1024):
                        output.write(chunk)
            if size is not None and partial.stat().st_size != size:
                raise RuntimeError(f"Incomplete file: {destination.name}: {partial.stat().st_size}/{size}")
            partial.replace(destination)
            return
        except Exception as exc:
            print(json.dumps({"event":"retry","file":destination.name,"attempt":attempt+1,"error_type":type(exc).__name__}),flush=True)
            if attempt == 3:
                raise
            time.sleep(2 * (attempt + 1))

MODELS = [
    ("facebook/m2m100_418M", "55c2e61bbf05dfb8d7abccdc3fae6fc8512fd636", "m2m100_418M", "pytorch_model.bin"),
    ("alirezamsh/small100", "8ab680e26a596d2e3d2d2d17ae0f68df1037328c", "small100", "model.safetensors"),
    ("facebook/nllb-200-distilled-600M", "f8d333a098d19b4fd9a8b18f94170487ad3f821d", "nllb200_600M", "pytorch_model.bin"),
    ("NiuTrans/LMT-60-0.6B", "dd189845cdc73346cef33c7a94f4b8bd8efdd4eb", "LMT-60-0.6B", "model.safetensors"),
]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path("/home/aim0/data/models/ASR"))
    parser.add_argument("--model", choices=[m[2] for m in MODELS])
    args = parser.parse_args()
    os.environ.setdefault("HF_HUB_DISABLE_XET", "1")
    for repo, revision, name, weights in MODELS:
        if args.model and name != args.model:
            continue
        destination = args.root / name
        info = HfApi().model_info(repo, revision=revision, files_metadata=True)
        files = ["README.md", "config.json", "generation_config.json", "sentencepiece.bpe.model",
                 "special_tokens_map.json", "tokenizer_config.json", "vocab.json", weights]
        if name == "nllb200_600M":
            files.append("tokenizer.json")
        if name == "LMT-60-0.6B":
            files.extend(["tokenizer.json", "merges.txt", "added_tokens.json", "chat_template.jinja"])
        if name == "small100":
            files.append("tokenization_small100.py")
        print(json.dumps({"event":"download_start", "repo":repo, "revision":revision, "destination":str(destination)}, ensure_ascii=False), flush=True)
        started = time.monotonic()
        # Direct GET handles proxies that omit Hub-specific HEAD metadata; LFS digest
        # verification below remains mandatory for model weights.
        from concurrent.futures import ThreadPoolExecutor
        entries = [e for e in info.siblings if e.rfilename in files]
        with ThreadPoolExecutor(max_workers=3) as pool:
            futures = [pool.submit(fetch, f"https://huggingface.co/{repo}/resolve/{revision}/{e.rfilename}",
                                   destination / e.rfilename, e.size) for e in entries]
            for future in futures:
                future.result()
        records = []
        for entry in info.siblings:
            if entry.rfilename not in files:
                continue
            path = destination / entry.rfilename
            digest = hashlib.sha256()
            with path.open("rb") as source:
                for chunk in iter(lambda: source.read(8 * 1024 * 1024), b""):
                    digest.update(chunk)
            expected = entry.lfs.sha256 if entry.lfs else None
            actual = digest.hexdigest()
            if expected and expected != actual:
                raise RuntimeError(f"SHA256 mismatch: {path}")
            records.append({"file":entry.rfilename, "size_bytes":path.stat().st_size,
                            "sha256":actual, "hub_lfs_sha256":expected, "lfs_verified":bool(expected)})
        manifest = {"repo_id":repo, "revision":revision, "files":records,
                    "download_seconds":round(time.monotonic()-started, 2)}
        (destination / "DOWNLOAD_MANIFEST.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2)+"\n")
        print(json.dumps({"event":"download_verified", "repo":repo, "bytes":sum(r["size_bytes"] for r in records),
                          "seconds":manifest["download_seconds"]}), flush=True)


if __name__ == "__main__":
    main()
