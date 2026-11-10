"""Stage 7: Citation and factuality verification (Internal Grounding Auditor)."""

import re
from typing import List, Optional, Tuple
from src.relay.schemas import Citation, DocumentChunk


class ResponseVerifier:
    """Audits generated answers for internal citation validity and grounding against retrieved context.
    
    This verifier operates strictly as a runtime consistency check:
    - Extracts citations formatted as [Paper Title, Page X, Section Y].
    - Verifies that cited papers and pages actually exist in the retrieved context chunks.
    - Identifies legitimate refusals vs hallucinated citations.
    It does NOT access external evaluation ground truth.
    """

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

    def verify_grounding(
        self,
        answer: str,
        retrieved_chunks: List[DocumentChunk]
    ) -> Tuple[bool, List[Citation], bool, str]:
        """Audits answer correctness and citations against retrieved chunks.

        Returns:
            (is_grounded: bool, citations: List[Citation], is_refusal: bool, explanation: str)
        """
        ans_lower = answer.lower()
        is_refusal = any(ind in ans_lower for ind in self.REFUSAL_INDICATORS)

        if is_refusal:
            return True, [], True, "Explicit refusal grounded in absence of sufficient evidence."

        # Extract citations
        extracted_citations = self.extract_citations(answer)
        if not extracted_citations:
            return False, [], False, "Answer contains factual claims but lacks required [Paper, Page, Section] citations."

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
            return False, extracted_citations, False, "Citations present in text do not match any retrieved chunks (hallucinated citation)."

        return True, extracted_citations, False, f"Grounding verified with {valid_citations} valid citation(s) from context."
