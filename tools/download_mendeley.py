"""Download the original Mendeley dataset using documented public ZIP endpoints.

Fails closed: never substitutes test fixtures, HTML error pages or simulation.
"""
from __future__ import annotations
import argparse
import hashlib
import json
import os
import time
import urllib.request
import zipfile
from pathlib import Path

DATASET = "zt2j5j4v44"
VERSION = 3
API = f"https://api.data.mendeley.com/datasets/{DATASET}"
HEADERS = {"User-Agent": "PhotonIQ-reproducible-research/1.0", "Accept": "application/json"}

def request_json(url: str) -> dict:
    req = urllib.request.Request(url, headers=HEADERS)
    with urllib.request.urlopen(req, timeout=90) as resp:
        return json.load(resp)

def stream_download(url: str, output: Path, limit_bytes: int = 1_500_000_000) -> None:
    req = urllib.request.Request(url, headers={"User-Agent": HEADERS["User-Agent"]})
    output.parent.mkdir(parents=True, exist_ok=True)
    total = 0
    with urllib.request.urlopen(req, timeout=120) as resp, output.open("wb") as f:
        while True:
            part = resp.read(1024 * 1024)
            if not part:
                break
            total += len(part)
            if total > limit_bytes:
                raise ValueError("Source archive exceeds the 1.5 GB download limit")
            f.write(part)
    if not zipfile.is_zipfile(output):
        raise ValueError("Dataset endpoint returned non-ZIP content; refusing to train")

def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for b in iter(lambda: f.read(1024 * 1024), b""):
            h.update(b)
    return h.hexdigest()

def download(output: Path) -> dict:
    meta_url = f"{API}/zip?version={VERSION}"
    last_error = None
    info = {}
    # The archive may initially be in creation; poll the documented metadata endpoint.
    for attempt in range(8):
        try:
            info = request_json(meta_url)
            url = info.get("url")
            if url:
                stream_download(url, output)
                break
            last_error = f"ZIP not ready: {info.get('status')}"
        except Exception as exc:
            last_error = f"{type(exc).__name__}: {exc}"
        if attempt < 7:
            time.sleep(15)
    else:
        # Redirecting archive endpoint is a documented alternative.
        redirect_url = f"{API}/zip/file_downloaded?version={VERSION}"
        try:
            stream_download(redirect_url, output)
        except Exception as exc:
            raise RuntimeError(f"Cannot retrieve original dataset. metadata: {last_error}; redirect: {exc}") from exc
    observed = sha256(output)
    expected = info.get("sha256_hash")
    if expected and expected.lower() != observed.lower():
        output.unlink(missing_ok=True)
        raise ValueError("Downloaded archive SHA256 differs from Mendeley metadata")
    return {"dataset_doi": f"10.17632/{DATASET}.{VERSION}",
            "downloaded_bytes": output.stat().st_size, "sha256": observed,
            "server_sha256": expected, "archive_source": meta_url}

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--output", type=Path, required=True)
    args = p.parse_args()
    result = download(args.output)
    print(json.dumps(result, indent=2))
    args.output.with_suffix(".provenance.json").write_text(json.dumps(result, indent=2))
