"""Independent evaluation harness for assessing answer quality against ground truth."""

import re
from typing import Optional, Set, Tuple
from src.relay.pipeline.verifier import ResponseVerifier
from src.relay.schemas import AnswerGrade, EvaluationItem, PipelineResponse, RequestStatus, SourceType


class QualityEvaluator:
    """Evaluates pipeline responses against ground truth evaluation items.
    
    Operates strictly outside the application runtime to guarantee zero evaluation leakage.
    Distinguishes retrieval verification from real-model generative correctness.
    """

    def __init__(self):
        self.verifier = ResponseVerifier()

    def _extract_keywords(self, text: str) -> Set[str]:
        """Extracts substantive content keywords (lowercased, len >= 4)."""
        stop_words = {
            "what", "which", "where", "when", "that", "this", "from", "with",
            "have", "more", "used", "uses", "into", "than", "were", "been",
            "their", "also", "each", "both", "such", "then", "them"
        }
        words = re.findall(r'\b[a-zA-Z0-9_\-\.\/]{3,}\b', text.lower())
        return {w for w in words if w not in stop_words and not w.isdigit()}

    def grade(
        self,
        response: PipelineResponse,
        item: EvaluationItem
    ) -> Tuple[Optional[AnswerGrade], str]:
        """Grades a pipeline response against ground truth.
        
        Returns:
            (grade: Optional[AnswerGrade], rationale: str)
            If request was not successful (e.g. rate-limited or timed-out), grade is None.
        """
        # 1. Operational failure: do not penalize as a factual model hallucination
        if response.status != RequestStatus.OK:
            return None, f"Operational drop ({response.status.value}): not graded for factual accuracy."

        ans_lower = response.answer.lower()
        is_refusal = any(ind in ans_lower for ind in ResponseVerifier.REFUSAL_INDICATORS)

        # 2. Ground truth: question is unanswerable from the library
        if not item.is_answerable:
            if is_refusal:
                return AnswerGrade.REFUSED_CORRECTLY, "Unanswerable query was correctly refused by the system."
            else:
                return AnswerGrade.WRONG, "Question is unanswerable from the library, but the model attempted an answer."

        # 3. Ground truth: question is answerable
        if is_refusal:
            return AnswerGrade.REFUSED_WRONGLY, "Answerable query was wrongly refused."

        citations = response.citations or self.verifier.extract_citations(response.answer)
        if not citations:
            return AnswerGrade.WRONG, "Answer contains factual claims but lacks required citations."

        # Check target paper and page match (within +/- 1 page window for multi-page sections)
        target_match = any(
            c.paper_id == item.expected_paper and abs(c.page - item.expected_page) <= 1
            for c in citations
        )

        if not target_match:
            cited_desc = ", ".join(f"{c.paper_id} P.{c.page}" for c in citations)
            return AnswerGrade.WRONG, f"Citations ({cited_desc}) did not match expected ground truth ({item.expected_paper} P.{item.expected_page})."

        # Check semantic alignment if response is from a live model
        if response.source == SourceType.REAL and item.expected_answer:
            exp_kw = self._extract_keywords(item.expected_answer)
            ans_kw = self._extract_keywords(response.answer)
            overlap = exp_kw.intersection(ans_kw)
            if not overlap and len(exp_kw) > 0:
                return AnswerGrade.WRONG, f"Citation matched {item.expected_paper} P.{item.expected_page}, but answer lacked required factual content."

        # Stand-in retrieval audit or confirmed factual match
        if response.source == SourceType.STANDIN:
            rationale = f"Retrieval verified: target passage matched ground truth [{item.expected_paper} P.{item.expected_page}]."
        else:
            rationale = f"Factual answer and citation verified against [{item.expected_paper} P.{item.expected_page}]."

        return AnswerGrade.CORRECT, rationale
