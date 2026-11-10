"""Research paper retrieval engine and tool interface."""

import json
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from src.relay.config import settings
from src.relay.ingestion.extractor import PaperExtractor, ExtractedPage
from src.relay.ingestion.chunker import DocumentChunker
from src.relay.retrieval.indexer import DocumentIndex
from src.relay.retrieval.reranker import CrossScoringReranker
from src.relay.schemas import DocumentChunk, RetrievalResult, RetrievalSetting


class ResearchRetriever:
    """Manages paper extraction, chunk indexing, and high-performance retrieval."""

    def __init__(self, data_dir: Optional[Path] = None):
        self.data_dir = Path(data_dir or settings.DATA_DIR)
        self.papers_dir = self.data_dir / "papers"
        self.metadata_path = self.data_dir / "metadata.json"
        self.extractor = PaperExtractor(self.papers_dir)
        self.reranker = CrossScoringReranker()

        # Cached raw extracted pages per paper
        self._raw_pages: Dict[str, List[ExtractedPage]] = {}
        # Cached indices keyed by chunk_size
        self._index_cache: Dict[int, DocumentIndex] = {}

        self._load_and_extract()

    def _load_and_extract(self):
        """Extract pages from all verified papers."""
        if not self.metadata_path.exists():
            raise FileNotFoundError(f"Metadata file not found at {self.metadata_path}. Run ingest_papers.py first.")

        with open(self.metadata_path, "r", encoding="utf-8") as f:
            meta = json.load(f)

        for paper in meta.get("papers", []):
            pdf_path = self.data_dir.parent / paper["local_path"]
            pages = self.extractor.extract_paper(pdf_path, paper["id"], paper["title"])
            self._raw_pages[paper["id"]] = pages

    def get_index(self, chunk_size: int, chunk_overlap: int = 100) -> DocumentIndex:
        """Returns or builds a BM25 index for the requested chunk size."""
        if chunk_size not in self._index_cache:
            chunker = DocumentChunker(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
            all_chunks: List[DocumentChunk] = []
            for pages in self._raw_pages.values():
                all_chunks.extend(chunker.chunk_pages(pages))
            self._index_cache[chunk_size] = DocumentIndex(all_chunks)
        return self._index_cache[chunk_size]

    def retrieve(
        self,
        query: str,
        setting: Optional[RetrievalSetting] = None
    ) -> RetrievalResult:
        """Explicit retrieval tool callable by the pipeline."""
        start_time = time.perf_counter()
        cfg = setting or RetrievalSetting()

        index = self.get_index(chunk_size=cfg.chunk_size, chunk_overlap=cfg.chunk_overlap)
        # Fetch candidate pool (2x top_k if reranking, else top_k)
        fetch_k = cfg.top_k * 2 if cfg.rerank else cfg.top_k
        candidates = index.search(query, top_k=fetch_k)

        if cfg.rerank:
            final_scored = self.reranker.rerank(query, candidates, top_k=cfg.top_k)
        else:
            final_scored = candidates[:cfg.top_k]

        chunks = [c for c, _ in final_scored]
        elapsed_ms = (time.perf_counter() - start_time) * 1000.0

        # Evaluate sufficiency threshold: does top chunk score above minimal threshold?
        is_sufficient = True
        reason = "Passages retrieved with sufficient match confidence."
        if not chunks or (final_scored and final_scored[0][1] < 1.0):
            is_sufficient = False
            reason = "Retrieved candidate scores fall below relevance threshold."

        return RetrievalResult(
            chunks=chunks,
            query=query,
            retrieval_time_ms=elapsed_ms,
            is_sufficient=is_sufficient,
            refinement_performed=False,
            sufficiency_reason=reason
        )
