"""Re-ranking model for candidate document chunks."""

import re
from typing import List, Tuple
from src.relay.schemas import DocumentChunk
from src.relay.retrieval.indexer import tokenize


class CrossScoringReranker:
    """Re-ranks retrieved candidates based on exact term density, n-gram matching, and section relevance."""

    def __init__(self):
        pass

    def rerank(
        self,
        query: str,
        candidates: List[Tuple[DocumentChunk, float]],
        top_k: int
    ) -> List[Tuple[DocumentChunk, float]]:
        if not candidates:
            return []

        query_tokens = tokenize(query)
        if not query_tokens:
            return candidates[:top_k]

        query_lower = query.lower()
        reranked: List[Tuple[DocumentChunk, float]] = []

        for chunk, base_score in candidates:
            text_lower = chunk.text.lower()
            section_lower = chunk.section_name.lower()
            bonus = 0.0

            # 1. Exact phrase match bonus
            if len(query_lower) > 5 and query_lower in text_lower:
                bonus += 5.0

            # 2. Bigram matches
            for i in range(len(query_tokens) - 1):
                bigram = f"{query_tokens[i]} {query_tokens[i+1]}"
                if bigram in text_lower:
                    bonus += 1.5

            # 3. Section heading relevance
            for token in query_tokens:
                if len(token) > 3 and token in section_lower:
                    bonus += 0.8

            # 4. Token coverage ratio
            chunk_tokens_set = set(tokenize(text_lower))
            matched_query_tokens = sum(1 for t in set(query_tokens) if t in chunk_tokens_set)
            coverage_ratio = matched_query_tokens / max(1, len(set(query_tokens)))
            bonus += coverage_ratio * 3.0

            final_score = base_score + bonus
            reranked.append((chunk, final_score))

        reranked.sort(key=lambda x: x[1], reverse=True)
        return reranked[:top_k]
