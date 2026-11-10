"""Independent evaluation harness for assessing answer quality against ground truth."""

from typing import Optional, Tuple
from src.relay.pipeline.verifier import ResponseVerifier
from src.relay.schemas import AnswerGrade, EvaluationItem, PipelineResponse, RequestStatus


class QualityEvaluator:
    """Evaluates pipeline responses against ground truth evaluation items.
    
    Operates strictly outside the application runtime to guarantee zero evaluation leakage.
    """

    def __init__(self):
        self.verifier = ResponseVerifier()

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

        if target_match:
            return AnswerGrade.CORRECT, f"Target citation matched {item.expected_paper} Page {item.expected_page}."
        else:
            cited_desc = ", ".join(f"{c.paper_id} P.{c.page}" for c in citations)
            return AnswerGrade.WRONG, f"Citations ({cited_desc}) did not match expected ground truth ({item.expected_paper} P.{item.expected_page})."
