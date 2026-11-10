"""Stage 7: Citation and factuality verification."""

import re
from typing import List, Optional, Tuple
from src.relay.schemas import AnswerGrade, Citation, DocumentChunk


class ResponseVerifier:
    """Audits generated answers for citation accuracy, grounding, and proper refusal adherence."""

    CITATION_REGEX = re.compile(
        r'\[(Attention Is All You Need|Moshi[^,]*|Dynamic Chunking[^,]*),\s*Page\s*(\d+)(?:,\s*Section\s*([^\]]+))?\]',
        re.IGNORECASE
    )

    REFUSAL_INDICATORS = [
        "refusal:",
        "does not establish",
        "not mentioned in the provided",
        "supplied research library does not establish",
        "no evidence",
        "cannot be answered from"
    ]

    def extract_citations(self, text: str) -> List[Citation]:
        citations: List[Citation] = []
        matches = self.CITATION_REGEX.findall(text)
        for title, page_str, sec in matches:
            try:
                page_num = int(page_str)
                paper_id = "1706.03762v7" if "Attention" in title else ("2410.00037v1" if "Moshi" in title else "2507.07955v2")
                citations.append(
                    Citation(
                        paper_id=paper_id,
                        paper_title=title.strip(),
                        page=page_num,
                        section=sec.strip() if sec else None
                    )
                )
            except ValueError:
                continue
        return citations

    def verify(
        self,
        answer: str,
        retrieved_chunks: List[DocumentChunk],
        is_known_answerable: Optional[bool] = None,
        expected_paper_id: Optional[str] = None,
        expected_page: Optional[int] = None
    ) -> Tuple[AnswerGrade, List[Citation], str]:
        """Audits answer correctness and citations.

        Returns:
            (grade: AnswerGrade, citations: List[Citation], explanation: str)
        """
        ans_lower = answer.lower()
        is_refusal = any(ind in ans_lower for ind in self.REFUSAL_INDICATORS)

        # 1. Evaluate Refusal Cases
        if is_known_answerable is not None:
            if not is_known_answerable:
                if is_refusal:
                    return AnswerGrade.REFUSED_CORRECTLY, [], "Unanswerable question was correctly refused."
                else:
                    return AnswerGrade.WRONG, [], "Question is unanswerable from the library, but the model attempted an answer."
            else:
                if is_refusal:
                    return AnswerGrade.REFUSED_WRONGLY, [], "Answerable question was wrongly refused."

        # 2. If it's a general refusal without prior label
        if is_refusal:
            return AnswerGrade.REFUSED_CORRECTLY, [], "Model indicated information absent from library."

        # 3. Citation Audit
        extracted_citations = self.extract_citations(answer)
        if not extracted_citations:
            return AnswerGrade.WRONG, [], "Answer contains factual claims but lacks required [Paper, Page, Section] citations."

        # Verify whether citations exist in retrieved chunks
        valid_citations = 0
        for cite in extracted_citations:
            match_found = any(
                c.page_number == cite.page and cite.paper_id in c.paper_id
                for c in retrieved_chunks
            )
            if match_found:
                valid_citations += 1

        if valid_citations == 0:
            return AnswerGrade.WRONG, extracted_citations, "Citations present but do not match any retrieved chunks (hallucinated citation)."

        # Check ground truth target match if provided
        if expected_paper_id and expected_page:
            target_match = any(
                c.paper_id == expected_paper_id and abs(c.page - expected_page) <= 1
                for c in extracted_citations
            )
            if not target_match:
                return AnswerGrade.WRONG, extracted_citations, f"Answer cited pages other than expected ground truth ({expected_paper_id} P.{expected_page})."

        return AnswerGrade.CORRECT, extracted_citations, f"Verified with {valid_citations} valid citation(s)."
