"""Tests for the paper-first frontend API and evaluation dashboard endpoints."""

import pytest
from httpx import ASGITransport, AsyncClient
from src.relay.server.app import app


@pytest.mark.asyncio
async def test_frontend_home_page():
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        resp = await client.get("/")
        assert resp.status_code == 200
        assert "THE RELAY" in resp.text
        assert "Paper-First Research Assistant" in resp.text
        assert "Evaluation Dashboard" in resp.text


@pytest.mark.asyncio
async def test_api_papers_endpoint():
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        resp = await client.get("/api/papers")
        assert resp.status_code == 200
        data = resp.json()
        assert len(data) == 3
        ids = {p["id"] for p in data}
        assert ids == {"1706.03762v7", "2410.00037v1", "2507.07955v2"}


@pytest.mark.asyncio
async def test_api_query_single_paper_scope_enforcement():
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        # 1. Question matching the selected paper
        payload = {
            "question": "What is the number of heads (h) in the base Transformer?",
            "paper_scope": "1706.03762v7",
            "chunk_size": 600,
            "top_k": 4,
            "rerank": True,
            "stream": False
        }
        resp = await client.post("/api/query", json=payload)
        assert resp.status_code == 200
        data = resp.json()
        assert data["status"] == "ok"
        # Citations must be strictly from 1706.03762v7
        for c in data["citations"]:
            assert c["paper_id"] == "1706.03762v7"

        # 2. Question about Moshi while paper_scope is locked to Attention paper
        payload_mismatch = {
            "question": "What is the framerate of the Mimi neural audio codec in Moshi?",
            "paper_scope": "1706.03762v7",
            "chunk_size": 600,
            "top_k": 4,
            "rerank": True,
            "stream": False
        }
        resp_mismatch = await client.post("/api/query", json=payload_mismatch)
        assert resp_mismatch.status_code == 200
        data_mismatch = resp_mismatch.json()
        # Must refuse because Moshi is outside the selected paper scope
        assert "REFUSAL" in data_mismatch["answer"]
        assert len(data_mismatch["citations"]) == 0


@pytest.mark.asyncio
async def test_api_experiments_summary_endpoint():
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        resp = await client.get("/api/experiments/summary")
        assert resp.status_code == 200
        data = resp.json()
        assert data["total_requests"] == 184
        assert len(data["table_1_limits"]) >= 5
        assert len(data["table_2a_latency"]) == 5
        assert len(data["table_2b_outcomes"]) == 5
        assert len(data["table_3_retrieval"]) == 3
        assert "dominant_component" in data["slow_request"]


@pytest.mark.asyncio
async def test_api_experiments_records_filtering():
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        # All records
        resp = await client.get("/api/experiments/records")
        assert resp.status_code == 200
        data = resp.json()
        assert data["total_records"] == 184
        assert data["filtered_count"] == 184

        # Filter by status: ok
        resp_ok = await client.get("/api/experiments/records?status=ok")
        data_ok = resp_ok.json()
        assert data_ok["filtered_count"] > 0
        assert all(r["status"] == "ok" for r in data_ok["records"])

        # Filter by scenario: burst
        resp_burst = await client.get("/api/experiments/records?scenario=burst")
        data_burst = resp_burst.json()
        assert data_burst["filtered_count"] == 25
        assert all(r["scenario"] == "burst" for r in data_burst["records"])
