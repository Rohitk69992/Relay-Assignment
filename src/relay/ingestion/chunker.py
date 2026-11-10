"""Configurable document chunking preserving page-level citation metadata."""

import math
import re
from typing import List
from src.relay.ingestion.extractor import ExtractedPage
from src.relay.schemas import DocumentChunk


class DocumentChunker:
    """Partitions extracted pages into overlapping text chunks while retaining page and section metadata."""

    def __init__(self, chunk_size: int = 600, chunk_overlap: int = 100):
        if chunk_overlap >= chunk_size:
            raise ValueError(f"Overlap {chunk_overlap} must be smaller than chunk size {chunk_size}")
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.step_size = chunk_size - chunk_overlap

    def chunk_pages(self, pages: List[ExtractedPage]) -> List[DocumentChunk]:
        all_chunks: List[DocumentChunk] = []

        for page in pages:
            text = page.text.strip()
            if not text:
                continue

            # If the page text fits in one chunk, emit directly
            if len(text) <= self.chunk_size:
                chunk_id = f"{page.paper_id}_p{page.page_number}_c0"
                estimated_tokens = max(1, math.ceil(len(text) / 4.0))
                all_chunks.append(
                    DocumentChunk(
                        chunk_id=chunk_id,
                        paper_id=page.paper_id,
                        paper_title=page.paper_title,
                        page_number=page.page_number,
                        section_name=page.dominant_section,
                        text=text,
                        char_count=len(text),
                        estimated_tokens=estimated_tokens
                    )
                )
                continue

            # Sliding window with boundary awareness
            start = 0
            chunk_idx = 0
            text_len = len(text)

            while start < text_len:
                end = min(start + self.chunk_size, text_len)

                # Try not to break in the middle of a sentence or word if near end
                if end < text_len:
                    # Look for sentence boundary
                    boundary = text.rfind('. ', start + self.chunk_size // 2, end)
                    if boundary != -1:
                        end = boundary + 1
                    else:
                        space = text.rfind(' ', start + self.chunk_size // 2, end)
                        if space != -1:
                            end = space

                chunk_text = text[start:end].strip()
                if chunk_text:
                    chunk_id = f"{page.paper_id}_p{page.page_number}_c{chunk_idx}"
                    estimated_tokens = max(1, math.ceil(len(chunk_text) / 4.0))
                    all_chunks.append(
                        DocumentChunk(
                            chunk_id=chunk_id,
                            paper_id=page.paper_id,
                            paper_title=page.paper_title,
                            page_number=page.page_number,
                            section_name=page.dominant_section,
                            text=chunk_text,
                            char_count=len(chunk_text),
                            estimated_tokens=estimated_tokens
                        )
                    )
                    chunk_idx += 1

                if end >= text_len:
                    break
                start = max(start + 1, end - self.chunk_overlap)

        return all_chunks
