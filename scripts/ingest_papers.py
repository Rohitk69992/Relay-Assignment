"""
Ingest and verify the exact arXiv paper versions required by The Relay assignment:
1. Attention Is All You Need (arXiv:1706.03762v7)
2. Moshi: a speech-text foundation model for real-time dialogue (arXiv:2410.00037v1)
3. Dynamic Chunking for End-to-End Hierarchical Sequence Modeling (arXiv:2507.07955v2)
"""

import hashlib
import json
import os
import sys
import time
import urllib.request
from datetime import datetime, timezone
from pathlib import Path

PAPERS = [
    {
        "id": "1706.03762v7",
        "title": "Attention Is All You Need",
        "version": 7,
        "url": "https://arxiv.org/pdf/1706.03762v7",
        "abs_url": "https://arxiv.org/abs/1706.03762v7",
        "filename": "1706.03762v7.pdf"
    },
    {
        "id": "2410.00037v1",
        "title": "Moshi: a speech-text foundation model for real-time dialogue",
        "version": 1,
        "url": "https://arxiv.org/pdf/2410.00037v1",
        "abs_url": "https://arxiv.org/abs/2410.00037v1",
        "filename": "2410.00037v1.pdf"
    },
    {
        "id": "2507.07955v2",
        "title": "Dynamic Chunking for End-to-End Hierarchical Sequence Modeling",
        "version": 2,
        "url": "https://arxiv.org/pdf/2507.07955v2",
        "abs_url": "https://arxiv.org/abs/2507.07955v2",
        "filename": "2507.07955v2.pdf"
    }
]

def download_file(url: str, dest_path: Path, max_retries: int = 3) -> None:
    headers = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) TheRelayResearch/1.0"}
    req = urllib.request.Request(url, headers=headers)
    for attempt in range(1, max_retries + 1):
        try:
            print(f"Downloading {url} (attempt {attempt}/{max_retries})...")
            with urllib.request.urlopen(req, timeout=30) as response, open(dest_path, "wb") as out_file:
                chunk_size = 65536
                while True:
                    chunk = response.read(chunk_size)
                    if not chunk:
                        break
                    out_file.write(chunk)
            print(f"Saved to {dest_path} ({dest_path.stat().st_size:,} bytes)")
            return
        except Exception as e:
            print(f"Attempt {attempt} failed: {e}")
            if attempt == max_retries:
                raise
            time.sleep(2)

def calculate_sha256(file_path: Path) -> str:
    sha = hashlib.sha256()
    with open(file_path, "rb") as f:
        for block in iter(lambda: f.read(65536), b""):
            sha.update(block)
    return sha.hexdigest()

def main():
    base_dir = Path(__file__).resolve().parent.parent
    papers_dir = base_dir / "data" / "papers"
    papers_dir.mkdir(parents=True, exist_ok=True)
    metadata_path = base_dir / "data" / "metadata.json"

    metadata = {
        "ingestion_timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "papers": []
    }

    for paper in PAPERS:
        dest_pdf = papers_dir / paper["filename"]
        if not dest_pdf.exists() or dest_pdf.stat().st_size == 0:
            download_file(paper["url"], dest_pdf)
        else:
            print(f"File exists: {dest_pdf} ({dest_pdf.stat().st_size:,} bytes)")

        sha256_hash = calculate_sha256(dest_pdf)
        paper_meta = {
            **paper,
            "local_path": str(dest_pdf.relative_to(base_dir)).replace("\\", "/"),
            "file_size_bytes": dest_pdf.stat().st_size,
            "sha256": sha256_hash,
            "downloaded_at_utc": datetime.now(timezone.utc).isoformat()
        }
        metadata["papers"].append(paper_meta)
        print(f"Paper: {paper['title']}")
        print(f"  URL: {paper['url']}")
        print(f"  SHA-256: {sha256_hash}")
        print(f"  Size: {paper_meta['file_size_bytes']:,} bytes\n")

    with open(metadata_path, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)

    print(f"Successfully generated metadata at {metadata_path}")

if __name__ == "__main__":
    main()
