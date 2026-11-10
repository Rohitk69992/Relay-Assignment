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

            # If the entire page fits in one chunk, emit directly
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

            # Split page text into sentences preserving scientific punctuation
            raw_sentences = [s.strip() for s in re.split(r'(?<=[.!?])\s+', text) if s.strip()]
            if not raw_sentences:
                raw_sentences = [text]

            # Merge standalone short captions (e.g. "Figure 2: (left) ...") with subsequent sentence
            merged_sentences: List[str] = []
            i = 0
            while i < len(raw_sentences):
                s = raw_sentences[i]
                if (s.startswith("Figure ") or s.startswith("Table ")) and len(s) < 120 and (i + 1 < len(raw_sentences)):
                    merged_sentences.append(f"{s} {raw_sentences[i+1]}")
                    i += 2
                else:
                    merged_sentences.append(s)
                    i += 1

            chunk_idx = 0
            sent_idx = 0
            n_sents = len(merged_sentences)

            while sent_idx < n_sents:
                current_sents: List[str] = []
                current_len = 0
                idx = sent_idx

                while idx < n_sents:
                    s = merged_sentences[idx]
                    s_len = len(s)

                    # If a single sentence exceeds chunk_size, split by space
                    if s_len > self.chunk_size and not current_sents:
                        current_sents.append(s[:self.chunk_size])
                        idx += 1
                        break

                    added_len = s_len if not current_sents else (s_len + 1)
                    if current_len + added_len > self.chunk_size and current_sents:
                        break

                    current_sents.append(s)
                    current_len += added_len
                    idx += 1

                chunk_text = " ".join(current_sents).strip()
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

                if idx >= n_sents:
                    break

                # Calculate overlap: rewind by sentences that fit within chunk_overlap
                overlap_len = 0
                rewind_count = 0
                for back_s in reversed(current_sents):
                    if overlap_len + len(back_s) <= self.chunk_overlap and rewind_count < len(current_sents) - 1:
                        overlap_len += len(back_s) + 1
                        rewind_count += 1
                    else:
                        break

                sent_idx = max(sent_idx + 1, idx - rewind_count)

        return all_chunks
