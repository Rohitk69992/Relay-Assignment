"""Stage 6: Grounded answer generation using retrieved research evidence."""

import re
from typing import List, Optional, Set, Tuple
from src.relay.schemas import DocumentChunk


SYSTEM_PROMPT = """You are a rigorous, technical scientific research assistant answering questions strictly and exclusively from the provided research paper passages.

Your response must follow this structure:
### Direct Answer
Directly answer the user's technical question immediately without introductory filler, preamble, or boilerplate.

### Explanation
Explain the concept, architecture, mathematical formulation, equations, or mechanism in clear, understandable technical language as established in the paper.

### Supporting Evidence & Citations
Connect the explanation directly to the retrieved passages. Every factual statement must cite its exact source passage in the bracketed format:
[Paper Title, Page X, Section Y]
(For example: [Attention Is All You Need, Page 4, Section 3.2.1] or [Moshi: a speech-text foundation model for real-time dialogue, Page 6, Section 3]).

### Limitations
Clearly state any part of the question or specific detail that the provided passages do not establish.

Strict Invariants:
1. Base your answer exclusively on the provided context passages. Do not extrapolate, guess, or use external knowledge.
2. If the provided passages do not contain sufficient evidence to answer the question, state explicitly:
   "REFUSAL: The supplied research library does not establish the answer to this question."
3. Do not start answers with "Based on...".
4. Do not merely repeat the passage heading or write generic statements about published results.
5. Do not produce placeholder bullets or fabricated citations.
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
            clean_sec = chunk.section_name.split("\n")[0].strip() if chunk.section_name else "Overview"
            block = (
                f"--- Passage {idx + 1} ---\n"
                f"Paper: {chunk.paper_title} (arXiv:{chunk.paper_id})\n"
                f"Page: {chunk.page_number} | Section: {clean_sec}\n"
                f"Text:\n{chunk.text.strip()}\n"
            )
            context_blocks.append(block)

        context_str = "\n".join(context_blocks)
        prompt = (
            f"User Question:\n{question}\n\n"
            f"Retrieved Research Passages from Library:\n{context_str}\n\n"
            "Instructions:\n"
            "Provide a direct, technical answer structured into Direct Answer, Explanation, Supporting Evidence & Citations, and Limitations. "
            "Cite the exact Paper Title, Page, and Section from the passages above in bracketed format: [Paper Title, Page X, Section Y]. "
            "If the passages do not establish the answer, output the required refusal: "
            "'REFUSAL: The supplied research library does not establish the answer to this question.'"
        )
        return prompt

    @staticmethod
    def build_standin_answer(question: str, chunks: List[DocumentChunk], is_refusal: bool = False) -> str:
        """Constructs a deterministic, evidence-based extractive response grounded strictly in retrieved passages.

        Rules implemented:
        1. Uses actual retrieved passages from the selected paper scope.
        2. Identifies sentences relevant to the submitted question using lexical overlap.
        3. Prioritizes sentences containing the strongest query-relevant evidence.
        4. Preserves the original wording of extracted evidence.
        5. Includes exact [Paper Title, Page X, Section Y] citations matching ResponseVerifier.
        6. Presents multiple supporting sentences where appropriate.
        7. Refuses when the retrieved passages cannot establish a useful response.
        8. Explicitly labels the response as an extractive stand-in simulation.
        """
        if is_refusal or not chunks:
            return (
                "REFUSAL: The supplied research library does not establish the answer to this question. "
                "The ingested arXiv papers do not contain sufficient evidence on this topic."
            )

        stop_words: Set[str] = {
            "what", "when", "where", "which", "how", "does", "explain", "describe", "paper",
            "the", "and", "for", "with", "from", "that", "this", "was", "were", "are",
            "all", "you", "need", "used", "tested", "show", "tell", "model", "papers",
            "about", "into", "than", "instead", "why", "their", "have", "been", "using",
            "role", "compare", "between", "versus", "approach", "determines", "divided",
            "can", "could", "should", "would", "also", "such", "each", "both", "across"
        }

        # Tokenize question content terms
        q_tokens = [
            w.lower() for w in re.findall(r'\b[a-zA-Z0-9_\-\.]{3,}\b', question)
            if w.lower() not in stop_words
        ]
        if not q_tokens:
            q_tokens = [w.lower() for w in re.findall(r'\b[a-zA-Z0-9_\-\.]{2,}\b', question)]

        # Target paper detection from query keywords
        paper_topic_keywords = {
            "1706.03762v7": ["transformer", "vaswani", "scaled dot-product", "attention is all you need", "multi-head", "d_k", "d_model"],
            "2410.00037v1": ["moshi", "mimi", "helium", "audio codec", "spoken dialogue"],
            "2507.07955v2": ["dynamic chunking", "boundary predictor", "h-net", "hierarchical sequence"]
        }
        q_lower = question.lower()
        detected_targets = {pid for pid, kws in paper_topic_keywords.items() if any(k in q_lower for k in kws)}
        is_comparative = any(w in q_lower for w in ["compare", "difference", "versus", "vs", "trade-off", "both"])

        # Extract and score candidate sentences across all retrieved chunks
        scored_candidates: List[Tuple[float, str, DocumentChunk]] = []

        for chunk_idx, chunk in enumerate(chunks):
            raw_text = chunk.text.replace("\r\n", " ").replace("\n", " ")
            raw_sentences = re.split(r'(?<=[.!?])\s+', raw_text)

            for s in raw_sentences:
                clean_s = s.strip()
                if len(clean_s) < 25 or len(clean_s) > 450:
                    continue
                if not any(c.isalpha() for c in clean_s):
                    continue

                s_lower = clean_s.lower()
                # Score based on query tokens matched
                score = sum(2.0 for t in q_tokens if t in s_lower)

                # Retrieval rank weighting (higher ranked chunks get a bonus)
                score += max(0.0, 3.0 - chunk_idx * 0.75)

                # Target paper alignment bonus (if query specifically targets this paper)
                if detected_targets and chunk.paper_id in detected_targets:
                    score += 5.0
                elif detected_targets and not is_comparative:
                    score -= 5.0

                # Technical phrase boosts
                key_phrases = [
                    "scaled dot-product", "dot product", "softmax", "gradient",
                    "audio codec", "boundary predictor", "dynamic chunking",
                    "helium", "mimi", "multi-head", "encoder", "decoder",
                    "downsample", "downsampling", "cosine similarity", "latent"
                ]
                for phrase in key_phrases:
                    if phrase in s_lower and any(t in phrase for t in q_tokens):
                        score += 3.5

                if score > 0.0:
                    scored_candidates.append((score, clean_s, chunk))

        if not scored_candidates:
            return (
                "REFUSAL: The supplied research library does not establish the answer to this question. "
                "The retrieved passages contain insufficient lexical evidence for this query."
            )

        # Sort by score descending
        scored_candidates.sort(key=lambda x: x[0], reverse=True)

        # Deduplicate and pick top distinct sentences
        selected: List[Tuple[str, DocumentChunk]] = []
        seen_texts: Set[str] = set()

        for score, sentence, chunk in scored_candidates:
            key = sentence[:45].lower()
            if key in seen_texts:
                continue
            seen_texts.add(key)
            selected.append((sentence, chunk))
            if len(selected) >= 3:
                break

        # If cross-paper comparison was requested and multiple papers exist in chunks, ensure representation
        if is_comparative:
            distinct_papers = {c.paper_id for _, c in selected}
            available_papers = {c.paper_id for c in chunks}
            if len(available_papers) > 1 and len(distinct_papers) < len(available_papers):
                missing_papers = available_papers - distinct_papers
                for score, sentence, chunk in scored_candidates:
                    if chunk.paper_id in missing_papers:
                        selected.append((sentence, chunk))
                        missing_papers.remove(chunk.paper_id)
                        if not missing_papers:
                            break

        # Format output
        header = "[Stand-in evidence preview — extractive simulation, not LLM-generated]\n"
        passages_text = []

        for idx, (sent, chunk) in enumerate(selected, 1):
            clean_sec = chunk.section_name.split("\n")[0].strip() if chunk.section_name else "Overview"
            citation = f"[{chunk.paper_title}, Page {chunk.page_number}, Section {clean_sec}]"
            passages_text.append(f'{idx}. "{sent}" — {citation}')

        body = "\n\n".join(passages_text)
        footer = (
            "\n\n(Notice: The Stand-In provider performs deterministic evidence extraction. "
            "For full natural-language synthesis and conceptual explanations, configure PROVIDER_MODE=groq with a valid API key.)"
        )

        return f"{header}\n{body}{footer}"
