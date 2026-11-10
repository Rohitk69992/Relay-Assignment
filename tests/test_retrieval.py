"""Unit tests for document chunking, indexing, re-ranking, and sufficiency evaluation."""

import pytest
from src.relay.ingestion.extractor import ExtractedPage
from src.relay.ingestion.chunker import DocumentChunker
from src.relay.retrieval.indexer import DocumentIndex
from src.relay.retrieval.reranker import CrossScoringReranker
from src.relay.retrieval.retriever import ResearchRetriever
from src.relay.pipeline.sufficiency import SufficiencyEvaluator
from src.relay.pipeline.intent import IntentPlanner, QueryPlan
from src.relay.schemas import DocumentChunk, RetrievalSetting


def test_chunker_boundaries():
    chunker = DocumentChunker(chunk_size=200, chunk_overlap=50)
    page = ExtractedPage(
        paper_id="test_paper",
        paper_title="Test Title",
        page_number=1,
        text="This is a test sentence for chunking. " * 15,
        dominant_section="Section 1"
    )

    chunks = chunker.chunk_pages([page])
    assert len(chunks) > 1
    for chunk in chunks:
        assert chunk.paper_id == "test_paper"
        assert chunk.page_number == 1
        assert chunk.section_name == "Section 1"
        assert len(chunk.text) <= 250  # Allow modest boundary allowance


def test_bm25_index_and_search():
    chunker = DocumentChunker(chunk_size=150, chunk_overlap=30)
    p1 = ExtractedPage(paper_id="p1", paper_title="Title1", page_number=1, text="The scaled dot-product attention computes query key dot products.", dominant_section="Sec 1")
    p2 = ExtractedPage(paper_id="p2", paper_title="Title2", page_number=1, text="Moshi utilizes the Mimi neural audio codec for continuous speech streaming.", dominant_section="Sec 2")

    chunks = chunker.chunk_pages([p1, p2])
    index = DocumentIndex(chunks)

    results = index.search("scaled dot-product attention", top_k=2)
    assert len(results) > 0
    top_chunk, score = results[0]
    assert top_chunk.paper_id == "p1"
    assert "scaled dot-product" in top_chunk.text.lower()


def test_reranker_scoring():
    reranker = CrossScoringReranker()
    c1 = ExtractedPage(paper_id="p1", paper_title="T1", page_number=1, text="A general paragraph about transformers and natural language processing.", dominant_section="Sec 1")
    c2 = ExtractedPage(paper_id="p2", paper_title="T2", page_number=1, text="The exact scaled dot-product attention formula scales by the square root of d_k.", dominant_section="3.2.1 Scaled Dot-Product Attention")

    chunker = DocumentChunker(chunk_size=300, chunk_overlap=50)
    chunks = chunker.chunk_pages([c1, c2])

    # Initial equal baseline scores
    candidates = [(chunks[0], 2.0), (chunks[1], 2.0)]
    reranked = reranker.rerank("scaled dot-product attention formula", candidates, top_k=2)

    assert reranked[0][0].paper_id == "p2", "Reranker must rank the exact phrase and matching section higher"


def test_retriever_end_to_end():
    retriever = ResearchRetriever()
    setting = RetrievalSetting(chunk_size=500, top_k=3, rerank=True)
    res = retriever.retrieve("What is the parameter size of the Helium text language model?", setting)

    assert len(res.chunks) == 3
    assert res.is_sufficient is True
    # The top chunk should reference Moshi or Helium
    top_chunk = res.chunks[0]
    assert "moshi" in top_chunk.paper_title.lower() or "helium" in top_chunk.text.lower()


def test_sufficiency_evaluator():
    evaluator = SufficiencyEvaluator(min_token_match_ratio=0.3)
    planner = IntentPlanner()
    plan = planner.plan_query("What is the scaled dot-product attention formula?")

    # Case 1: Unrelated chunks
    unrelated_chunk = DocumentChunk(
        chunk_id="c1", paper_id="p1", paper_title="T1", page_number=1,
        section_name="Intro", text="This is about general audio speech recording without neural networks.",
        char_count=80, estimated_tokens=20
    )
    is_suff, refined_query, reason = evaluator.evaluate(plan, [unrelated_chunk], refinement_count=0)
    assert is_suff is False
    assert "scaled" in refined_query.lower() or "dot-product" in refined_query.lower()

    # Case 2: Matching chunk
    matching_chunk = DocumentChunk(
        chunk_id="c2", paper_id="p2", paper_title="T2", page_number=1,
        section_name="Attention", text="We compute the scaled dot-product attention by multiplying query and key matrices.",
        char_count=90, estimated_tokens=25
    )
    is_suff2, _, _ = evaluator.evaluate(plan, [matching_chunk], refinement_count=0)
    assert is_suff2 is True
