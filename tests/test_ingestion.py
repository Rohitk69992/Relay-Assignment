"""Unit tests for research paper ingestion, checksum validation, and text extraction."""

import json
from pathlib import Path
import pytest
from src.relay.config import settings
from src.relay.ingestion.extractor import PaperExtractor


def test_metadata_file_exists():
    metadata_path = settings.DATA_DIR / "metadata.json"
    assert metadata_path.exists(), "metadata.json must exist"

    with open(metadata_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    assert "papers" in data
    assert len(data["papers"]) == 3

    expected_ids = {"1706.03762v7", "2410.00037v1", "2507.07955v2"}
    found_ids = {p["id"] for p in data["papers"]}
    assert expected_ids == found_ids, f"Expected {expected_ids}, got {found_ids}"


def test_downloaded_papers_exist_and_non_empty():
    papers_dir = settings.DATA_DIR / "papers"
    expected_files = ["1706.03762v7.pdf", "2410.00037v1.pdf", "2507.07955v2.pdf"]

    for fname in expected_files:
        pdf_path = papers_dir / fname
        assert pdf_path.exists(), f"Paper file {fname} must exist"
        assert pdf_path.stat().st_size > 100_000, f"Paper file {fname} size is too small"


def test_paper_extractor():
    extractor = PaperExtractor(settings.DATA_DIR / "papers")
    pdf_path = settings.DATA_DIR / "papers" / "1706.03762v7.pdf"

    pages = extractor.extract_paper(pdf_path, "1706.03762v7", "Attention Is All You Need")
    assert len(pages) == 15
    assert pages[0].page_number == 1
    assert "Attention Is All You Need" in pages[0].text
    # Verify section detected on subsequent pages
    sections = [p.dominant_section for p in pages]
    assert any("Attention" in s or "Model" in s or "Background" in s for s in sections)
