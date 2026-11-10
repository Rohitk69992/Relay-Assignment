"""Stage 2: Question understanding and query planning."""

import re
from typing import Dict, List, Optional
from pydantic import BaseModel


class QueryPlan(BaseModel):
    original_query: str
    target_papers: List[str] = []
    expanded_search_terms: List[str] = []
    is_comparative: bool = False
    is_off_topic: bool = False


class IntentPlanner:
    """Analyzes question intent and plans the retrieval query strategy."""

    PAPER_TOPIC_MAP = {
        "1706.03762v7": ["transformer", "attention", "scaled dot-product", "multi-head", "encoder", "decoder", "vaswani", "bleu"],
        "2410.00037v1": ["moshi", "helium", "mimi", "speech-text", "audio codec", "duplex", "real-time dialogue", "kyutai", "delay"],
        "2507.07955v2": ["dynamic chunking", "hierarchical", "h-net", "dechunking", "boundary predictor", "cartesia", "hwang", "compression ratio"]
    }

    OFF_TOPIC_INDICATORS = [
        "quantum computing", "dna sequencing", "astronomy", "black holes", "french revolution", "bitcoin",
        "cricket", "cooking recipe", "weather forecast", "superconductivity"
    ]

    def plan_query(self, question: str) -> QueryPlan:
        q_lower = question.lower()
        target_papers: List[str] = []
        expanded_terms: List[str] = []

        # Check for target papers
        for paper_id, topics in self.PAPER_TOPIC_MAP.items():
            for t in topics:
                if t in q_lower:
                    if paper_id not in target_papers:
                        target_papers.append(paper_id)
                    expanded_terms.append(t)

        # Check if comparative
        is_comparative = any(w in q_lower for w in ["compare", "difference", "versus", "vs", "trade-off", "both"])
        if is_comparative and len(target_papers) < 2:
            target_papers = list(self.PAPER_TOPIC_MAP.keys())

        # Check for obvious off-topic queries
        is_off_topic = any(ot in q_lower for ot in self.OFF_TOPIC_INDICATORS)

        # Clean search terms from alphanumeric words
        clean_tokens = [w for w in re.findall(r'\b[a-zA-Z0-9_\-\.]{3,}\b', q_lower) if w not in {"what", "when", "where", "which", "how", "does", "explain", "describe", "paper"}]
        expanded_terms.extend(clean_tokens)

        return QueryPlan(
            original_query=question,
            target_papers=target_papers,
            expanded_search_terms=list(dict.fromkeys(expanded_terms)),
            is_comparative=is_comparative,
            is_off_topic=is_off_topic
        )
