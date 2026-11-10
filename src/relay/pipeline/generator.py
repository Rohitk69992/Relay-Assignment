"""Stage 6: Grounded answer generation using retrieved research evidence."""

from typing import List, Optional
from src.relay.schemas import DocumentChunk


SYSTEM_PROMPT = """You are a rigorous scientific assistant answering questions strictly and exclusively from the provided research library passages.

Rules:
1. Base your answer ONLY on the provided context passages. Do not extrapolate or guess from general knowledge.
2. Every factual statement must cite its source passage in the format: [Paper Title, Page X, Section Y].
3. If the provided passages do not establish the answer, you MUST state explicitly:
   "REFUSAL: The supplied research library does not establish the answer to this question."
4. Be precise, technical, and concise. Do not hallucinate citations or page numbers.
"""


class AnswerGenerator:
    """Formats grounded prompts and orchestrates LLM completion."""

    @staticmethod
    def build_user_prompt(question: str, chunks: List[DocumentChunk]) -> str:
        if not chunks:
            return (
                f"Question: {question}\n\n"
                "Retrieved Research Passages: None.\n\n"
                "Please answer or provide an explicit refusal if the library does not contain the answer."
            )

        context_blocks = []
        for idx, chunk in enumerate(chunks):
            block = (
                f"--- Passage {idx + 1} ---\n"
                f"Paper: {chunk.paper_title} (arXiv:{chunk.paper_id})\n"
                f"Page: {chunk.page_number} | Section: {chunk.section_name}\n"
                f"Text:\n{chunk.text}\n"
            )
            context_blocks.append(block)

        context_str = "\n".join(context_blocks)
        prompt = (
            f"User Question:\n{question}\n\n"
            f"Retrieved Research Passages from Library:\n{context_str}\n\n"
            "Instructions:\n"
            "Provide a direct, technical answer citing the exact Paper Title, Page, and Section from the passages above. "
            "If the passages do not contain the answer, respond with the exact required refusal."
        )
        return prompt

    @staticmethod
    def build_standin_answer(question: str, chunks: List[DocumentChunk], is_refusal: bool = False) -> str:
        """Constructs a deterministic, explicitly grounded answer for stand-in simulation."""
        if is_refusal or not chunks:
            return (
                "REFUSAL: The supplied research library does not establish the answer to this question. "
                "The ingested arXiv papers (Attention Is All You Need, Moshi, and Dynamic Chunking) do not address this topic."
            )

        top_chunk = chunks[0]
        # Extract a key sentence from the top chunk
        sentences = [s.strip() for s in top_chunk.text.split(".") if len(s.strip()) > 30]
        evidence_snippet = sentences[0] if sentences else top_chunk.text[:100]

        answer = (
            f"Based on [{top_chunk.paper_title}, Page {top_chunk.page_number}, Section {top_chunk.section_name}], "
            f"{evidence_snippet}. "
        )
        if len(chunks) > 1:
            second_chunk = chunks[1]
            answer += (
                f"Furthermore, [{second_chunk.paper_title}, Page {second_chunk.page_number}, Section {second_chunk.section_name}] "
                f"confirms related architectural parameters."
            )
        return answer
