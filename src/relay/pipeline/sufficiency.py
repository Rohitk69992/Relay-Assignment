"""Stage 4 & 5: Evidence sufficiency evaluation and bounded retrieval refinement."""

from typing import List, Tuple
from src.relay.pipeline.intent import QueryPlan
from src.relay.schemas import DocumentChunk


class SufficiencyEvaluator:
    """Evaluates whether retrieved passages provide sufficient evidence to answer the query."""

    def __init__(self, min_token_match_ratio: float = 0.35):
        self.min_token_match_ratio = min_token_match_ratio

    def evaluate(
        self,
        query_plan: QueryPlan,
        chunks: List[DocumentChunk],
        refinement_count: int
    ) -> Tuple[bool, str, str]:
        """Evaluates retrieval sufficiency.

        Returns:
            (is_sufficient: bool, refined_query: str, reason: str)
        """
        if not chunks:
            if refinement_count == 0 and query_plan.expanded_search_terms:
                refined = " ".join(query_plan.expanded_search_terms[:4])
                return False, refined, "No passages found; attempting broadened query with key terms."
            return False, query_plan.original_query, "No passages retrieved from research library."

        # Compute term coverage across all retrieved chunks
        combined_text = " ".join(c.text.lower() for c in chunks)
        search_terms = query_plan.expanded_search_terms
        if not search_terms:
            return True, query_plan.original_query, "Sufficient context found."

        matched_terms = [t for t in search_terms if t in combined_text]
        coverage_ratio = len(matched_terms) / max(1, len(search_terms))

        if coverage_ratio < self.min_token_match_ratio:
            if refinement_count == 0:
                # Construct refined query focusing on unmatched key terms
                unmatched = [t for t in search_terms if t not in combined_text]
                refined_query = f"{query_plan.original_query} {' '.join(unmatched[:3])}".strip()
                return False, refined_query, f"Low evidence term overlap ({coverage_ratio:.1%}). Refining query once."
            else:
                return False, query_plan.original_query, f"Evidence insufficient after refinement (term coverage: {coverage_ratio:.1%})."

        return True, query_plan.original_query, f"Evidence sufficient (term coverage: {coverage_ratio:.1%})."
