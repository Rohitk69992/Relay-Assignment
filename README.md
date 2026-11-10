# The Relay: Resilient LLM Pipeline & Research Library Assistant

[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/downloads/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.115+-009688.svg)](https://fastapi.tiangolo.com)
[![Pytest](https://img.shields.io/badge/pytest-passing-brightgreen.svg)](https://docs.pytest.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

---

## 1. Project Objective

**The Relay** is an end-to-end, asynchronous, production-minded LLM pipeline engineered to answer technical queries grounded strictly in three arXiv research preprints:
1. **Attention Is All You Need** (arXiv:1706.03762v7)
2. **Moshi: a speech-text foundation model for real-time dialogue** (arXiv:2410.00037v1)
3. **Dynamic Chunking for End-to-End Hierarchical Sequence Modeling** (arXiv:2507.07955v2)

The system operates under strict provider quotas (RPM, TPM, RPD, TPD, and in-flight concurrency caps) while balancing four competing forces: **Fast responses**, **High answer quality / strict citation grounding**, **Low cost**, and **High availability**.

---

## 2. Architecture & Request Lifecycle

```
[User Request / Client]
           │
           ▼
[FastAPI / Admission Controller] ◄── [Token Bucket & Quota Tracker (RPM/TPM/RPD)]
           │
     (Enqueued)
           ▼
 [Bounded Priority Queue] ◄── [Queue Wait Deadline: 10s | Max Size: 50]
           │
    (Worker Pull)
           ▼
[Pipeline Orchestrator]
   ├── 1. Intent Planning & Query Disambiguation (`IntentPlanner`)
   ├── 2. Retrieval Tool Execution (`ResearchRetriever` - BM25 + Re-ranker)
   ├── 3. Evidence Sufficiency Gate (`SufficiencyEvaluator`)
   │      └── (Insufficient? -> Bounded 1-iteration query refinement loop)
   ├── 4. Provider Generation (`GroqProvider` / `StandInProvider`)
   │      └── (Concurrency Semaphore + Exponential Backoff with Jitter)
   ├── 5. Response Verifier & Citation Auditor (`ResponseVerifier`)
   └── 6. Token Streaming Output (SSE) ──► [Client]
```

### Key Engineering Invariants
1. **Grounding Over Availability:** The assistant will never hallucinate or invent citations to preserve uptime. If the research library does not establish the answer, it outputs an explicit refusal (`refused_correctly`).
2. **Bounded Queueing Over Immediate Dropping:** Bursts are absorbed in an asynchronous bounded queue up to 10 seconds. If queue capacity is exceeded, requests fail fast with HTTP 429 rather than hanging.
3. **Dual Provider Architecture:** Seamlessly switches between live hosted Groq API and an empirically calibrated Stand-In Model Service that simulates LPU log-normal latency distributions, multi-chunk SSE streaming, and quota exhaustion.

---

## 3. Project Structure

```
c:\The_Relay_Assignment\
  ├── .env.example                     # Environment template (keys, limits, settings)
  ├── .gitignore                        # Git exclusion rules (ignores secrets & venvs)
  ├── pyproject.toml                    # Standard Python package manifest
  ├── README.md                         # Complete reproduction guide
  ├── results.csv                       # Raw immutable telemetry for 184 requests
  ├── The_Relay_Report.pdf              # Generated official 2-page report
  ├── data\
  │   ├── papers\                       # Official downloaded arXiv PDFs
  │   │   ├── 1706.03762v7.pdf
  │   │   ├── 2410.00037v1.pdf
  │   │   └── 2507.07955v2.pdf
  │   ├── metadata.json                 # Paper metadata, source URLs, and SHA-256 hashes
  │   └── evaluation_set.json           # 18 curated ground-truth questions across 7 categories
  ├── src\relay\
  │   ├── config.py                     # Pydantic Settings & validation
  │   ├── logging_setup.py              # Monotonic timing & structured logging
  │   ├── schemas.py                    # Domain models, CSV schemas, status enums
  │   ├── ingestion\                    # PDF parsing, section detection, boundary chunking
  │   ├── retrieval\                    # BM25 indexing, cross-scoring re-ranking, tool interface
  │   ├── pipeline\                     # Intent planning, sufficiency gate, orchestrator, verifier
  │   ├── provider\                     # Real Groq adapter & Calibrated Stand-In service
  │   ├── resilience\                   # Token bucket, bounded queue, retry manager
  │   └── server\                       # FastAPI application & SSE streaming endpoints
  ├── tests\                            # Comprehensive unit & integration tests (23 passed)
  ├── docs\
  │   ├── assignment_analysis.md        # First-phase requirements analysis
  │   ├── requirements_traceability.md  # Complete requirement-to-evidence matrix
  │   ├── engineering_decisions.md      # Architectural trade-offs & rationale
  │   ├── naive_retrieval_failures.md   # Three real paper retrieval failure case studies
  │   ├── vaani_review.md               # Real website inspection & critique of vaani.dev
  │   ├── assistant_history.md          # Assistant interaction log, caught errors, overruled choices
  │   ├── report_tables.md              # Recomputed markdown tables matching results.csv
  │   └── final_report.md               # Source markdown for 2-page report
  └── scripts\
      ├── ingest_papers.py              # Download and verify exact arXiv PDFs
      ├── run_real_probes.py            # Live probe harness for Groq API
      ├── run_retrieval_benchmarks.py   # Evaluate 3 retrieval configurations
      ├── run_load_scenarios.py         # Execute 5 required load scenarios
      ├── run_all_experiments.py        # Master benchmark harness generating results.csv
      └── generate_pdf_report.py        # Compiles exact 2-page ReportLab PDF
```

---

## 4. Environment Setup & Dependencies

### Prerequisites
- Python 3.11+
- `uv` (recommended) or standard `pip`

```powershell
# 1. Clone / navigate to workspace
cd C:\The_Relay_Assignment

# 2. Create virtual environment
uv venv --python (Get-Command python).Source .venv
# Activate virtual environment
.\.venv\Scripts\activate

# 3. Install dependencies in editable mode
uv pip install -e ".[dev]"
```

---

## 5. Configuration & Groq Credentials

Copy `.env.example` to `.env`:
```powershell
cp .env.example .env
```

Configure your variables in `.env`:
```ini
# Real Groq API Configuration (Optional: runs Stand-In if key is omitted)
GROQ_API_KEY=gsk_your_groq_api_key_here
GROQ_MODEL=llama-3.1-8b-instant
PROVIDER_MODE=standin

# Rate Limits (Groq Free Tier Defaults)
RATE_LIMIT_RPM=30
RATE_LIMIT_TPM=6000
RATE_LIMIT_RPD=14400
RATE_LIMIT_TPD=500000
MAX_CONCURRENT_REQUESTS=5
MAX_QUEUE_SIZE=50
```
> **Security Notice:** `.env` is ignored by Git. API keys are never printed to logs, stdout, or telemetry files.

---

## 6. Reproducing All Steps

### Step 1: Ingest & Verify Research Papers
Downloads the three exact versioned PDFs from arXiv, validates SHA-256 hashes, and records extraction metadata:
```powershell
python scripts/ingest_papers.py
```

### Step 2: Run Automated Tests
Executes the full test suite (23 unit & integration tests covering ingestion, BM25 indexing, stand-in latency, retry policies, queues, and CSV metrics):
```powershell
python -m pytest tests/ -v
```

### Step 3: Run Real Groq Service Probes (Optional)
If a valid `GROQ_API_KEY` is present in `.env`, runs 5 controlled queries to measure live LPU TTFT and inspect `x-ratelimit-*` headers:
```powershell
python scripts/run_real_probes.py
```

### Step 4: Run All Experiments & Benchmarks
Executes the full suite (54 retrieval queries across 3 settings + 120 load test requests across 5 scenarios), validates `results.csv`, and generates summary tables:
```powershell
python scripts/run_all_experiments.py
```

### Step 5: Generate the 2-Page Official PDF Report
Compiles the report into `The_Relay_Report.pdf` and verifies that the page count is exactly 2:
```powershell
python scripts/generate_pdf_report.py
```

### Step 6: Start the FastAPI Assistant Server
Launches the HTTP API service with admission control and streaming telemetry:
```powershell
python -m uvicorn src.relay.server.app:app --host 127.0.0.1 --port 8000
```
- Health Check: `GET http://127.0.0.1:8000/health`
- Live Telemetry Metrics: `GET http://127.0.0.1:8000/metrics`
- Chat Completions: `POST http://127.0.0.1:8000/v1/chat/completions`

---

## 7. Results Summary

All experimental data is recorded in `results.csv` (184 rows) and summarized below:

### Table 1: Limits Found in the Real Service
- **RPM:** 30 requests/min (High confidence, verified via official Groq documentation & `x-ratelimit-*` headers).
- **TPM:** 6,000 tokens/min (`llama-3.1-8b-instant` tier allocation).
- **RPD / TPD:** 14,400 requests/day, 500,000 tokens/day (resets daily at 00:00 UTC).
- **In-Flight Concurrency:** Governed via client-side admission semaphore (5 slots).
- **Limit Response:** HTTP 429 Too Many Requests (`rate_limit_exceeded`) with structured JSON error and `retry-after` header.

### Table 2a & 2b: Load Scenarios
- **Steady Load (20 requests):** 80% succeeded, median TTFT 114.0ms, total time p50 232.9ms.
- **Sudden Burst (25 concurrent requests):** Peak queue depth reached 25, 88% succeeded, median TTFT spiked to 995.3ms due to queue buffering.
- **Sustained Overload (40 rapid requests):** Peak queue reached capacity (8), 20 requests dropped immediately by admission control, 10 requests timed out in queue (>1.0s wait).
- **Limit Halved (RPM=15):** 20% succeeded, 80% rate-limited after minute allocation depleted.
- **Daily Budget Exhausted (RPD=5):** 25% succeeded (first 5), 75% rejected immediately with daily quota notice.

### Table 3: Retrieval Configurations
- **Small (300 chars, top-3, no re-rank):** 53.3% correct, 100% correct refusals, TTFT p50 166.5ms.
- **Balanced (600 chars, top-4, re-ranked):** 53.3% correct, 100% correct refusals, TTFT p50 113.2ms. **Selected for production** due to optimal balance of context coverage and 62% token savings over Large.
- **Large (1000 chars, top-6, re-ranked):** 53.3% correct, 100% correct refusals, TTFT p50 112.6ms.

---

## 8. Known Limitations & Trade-offs
1. **Single-Node In-Memory Admission Queue:** The current implementation uses an in-memory asynchronous queue (`asyncio.Semaphore` and `asyncio.Queue`). In a horizontally scaled multi-worker deployment, queue state would require a distributed coordinator (e.g. Redis).
2. **Deterministic BM25 Scoring:** Chosen over opaque hosted vector embeddings to ensure zero network dependencies, 100% offline reproducibility, and exact page-level provenance.
