# The Relay: Assignment Analysis & Engineering Specification

## 1. Executive Summary & Objective

The objective of "The Relay" is to build, benchmark, analyze, and document a robust, production-minded LLM pipeline that answers engineering queries grounded exclusively in a specific three-paper research library. The system must operate under strict, unpredictable provider constraints (rate limits, concurrency caps, token budgets, latency spikes, and transient failures) while balancing four conflicting forces: **Fast responses**, **High quality (factual accuracy & grounding)**, **Low cost**, and **High availability (no user turned away)**.

The assignment requires:
1. Fetching and verifying three exact versioned research papers from arXiv.
2. Building an asynchronous multi-stage LLM pipeline with at least four distinct steps, an explicit retrieval tool call, a bounded loop/branch for evidence refinement, and strict citation enforcement.
3. Integrating with the real Groq API (free tier) and establishing empirical measurements of latency, rate limits, and error behaviors.
4. Designing and building an empirically calibrated, highly controllable **Stand-In Model Service** that simulates streaming, load-dependent degradation, queueing, concurrency limits, and provider failures.
5. Implementing resilient distributed systems primitives: bounded admission queues, token/request rate limiters, concurrency semaphores, request deadlines, exponential backoff with jitter, and graceful degradation.
6. Benchmarking 5 load scenarios (Steady, Burst, Sustained Overload, Per-Minute Limit Halved, Daily Budget Exhausted) and 3 retrieval configurations.
7. Generating an auditable `results.csv` dataset, reconciling 4 required summary tables, performing retrieval failure mode analysis, conducting a real review of `https://www.vaani.dev/`, and publishing a concise 2-page technical report.

---

## 2. Explicit Requirements Decomposition

### 2.1 Functional Requirements (FR)
- **FR-1 (Ingestion & Exact Library):** Fetch, verify SHA-256 checksums, and extract structured text (preserving sections and page numbers) from three exact arXiv papers:
  1. *Attention Is All You Need* (arXiv:1706.03762v7)
  2. *Moshi: a speech-text foundation model for real-time dialogue* (arXiv:2410.00037v1)
  3. *Dynamic Chunking for End-to-End Hierarchical Sequence Modeling* (arXiv:2507.07955v2)
- **FR-2 (Pipeline Stages):** Implement an asynchronous pipeline with at least 4 stages:
  - Stage 1: Request admission & intent understanding / query planning.
  - Stage 2: Research library retrieval tool invocation (lexical/semantic hybrid with optional re-ranking).
  - Stage 3: Evidence sufficiency assessment (decision gate / branch / loop).
  - Stage 4: Answer generation with page-level citations & strict refusal policy.
  - Stage 5: Post-generation citation and factuality verification.
- **FR-3 (Strict Grounding & Refusals):** Answers must cite exact paper, page, and section. If the library does not support the answer, the system must refuse explicitly (`refused_correctly`), rather than hallucinating or using general pre-training memory.
- **FR-4 (Real Groq Integration):** Support live execution on Groq free-tier endpoints (`llama-3.1-8b-instant` or currently active model), capturing headers (`x-ratelimit-*`, `retry-after`), latency, token counts, and HTTP status codes.
- **FR-5 (Controllable Stand-In Service):** A fully controllable mock/simulation service mirroring the Groq API contract, capable of configurable latency, jitter, load-dependent slowdown, 429 rate-limit responses, multi-chunk streaming, timeouts, and quota exhaustion.
- **FR-6 (Resilience Architecture):**
  - Admission control via bounded async queue.
  - Per-minute rate limiter (RPM and TPM).
  - Daily quota tracker (RPD and TPD).
  - Concurrency limiter (in-flight semaphore).
  - Bounded retries with exponential backoff and jitter.
  - Request timeouts and queue wait deadlines.
  - Graceful degradation: refusal or clean error notification without fabricating citations.
- **FR-7 (Streaming):** Token/chunk streaming support for both the real provider and the stand-in service, measuring real Time To First Token (TTFT).

### 2.2 Non-Functional Requirements (NFR)
- **NFR-1 (Determinism & Reproducibility):** Externalized configuration (Pydantic settings), fixed random seeds for simulations, clean environment isolation, and reproducible test scripts.
- **NFR-2 (Zero Fabrications):** No fake measurements, no hardcoded API keys, no artificial evaluation scores, no fabricated provider error headers.
- **NFR-3 (Bounded Execution):** Hard caps on retries ($N \le 3$), maximum queue wait ($T_{wait} \le 15\text{s}$), total request deadline ($T_{deadline} \le 30\text{s}$), maximum retrieval refinement iterations ($I_{max} = 2$), and bounded token lengths.
- **NFR-4 (Observability & Telemetry):** End-to-end request tracing via monotonic timestamps, recording queue wait duration, model invocation duration, TTFT, retry waiting time, and total processing latency.
- **NFR-5 (Code Quality):** Modular asynchronous Python code, type annotations, structured logging, high test coverage ($>85\%$ on core components), and zero warnings/linting errors.

### 2.3 Hard Constraints
- **C-1:** Never hardcode secrets or commit `.env`.
- **C-2:** Use only Groq for real hosted calls; do not substitute alternative providers.
- **C-3:** Keep real provider measurements, stand-in measurements, and test fixtures strictly separated.
- **C-4:** Maximum report length: exactly 2 pages written content in the required heading order.
- **C-5:** The raw results file `results.csv` must be the single source of truth from which all tables are mathematically reproducible.

### 2.4 Deliverables
1. Source code for the complete pipeline, retrieval engine, providers, and server (`src/relay/`).
2. Verification test suite (`tests/`).
3. Research library documents and extraction metadata (`data/papers/`, `data/metadata.json`).
4. Controlled evaluation dataset with $\ge 15$ questions and ground-truth references (`data/evaluation_set.json`).
5. Raw execution telemetry: `results.csv`.
6. Generated summary tables (Tables 1, 2a, 2b, 3).
7. Website critique of Vaani (`docs/vaani_review.md`).
8. Naive retrieval failure case studies (`docs/naive_retrieval_failures.md`).
9. Final 2-page report (`docs/final_report.md` and compiled `The_Relay_Report.pdf`).
10. Assistant interaction history (`docs/assistant_history.md`).
11. Comprehensive `README.md` with step-by-step reproduction instructions.

---

## 3. Analysis of Ambiguities & Engineering Interpretations

1. **"What 'fast' means is up to you... Where you give up one goal for another, say so."**
   - *Interpretation:* In an interactive technical assistant answering from research papers, interactive latency is dominated by TTFT. We define "fast" as median TTFT $\le 800\text{ms}$ under normal load, with total request completion $\le 3000\text{ms}$. When load spikes, we prioritize *grounding integrity* over *speed* (we will never omit retrieval verification to save time) and *graceful queueing* over *immediate rejection* up to a strict 10s wait deadline.

2. **"Dynamic Chunking for End-to-End Hierarchical Sequence Modeling, arXiv 2507.07955, version 2"**
   - *Verification:* Direct query to arXiv confirmed that `2507.07955v2` is indeed the exact official preprint entitled *"Dynamic Chunking for End-to-End Hierarchical Sequence Modeling"*. We download directly from `https://arxiv.org/pdf/2507.07955v2`.

3. **Evaluation Question Set Denominators:**
   - *Interpretation:* Quality percentages must have explicit denominators:
     - `Correct and cited %` = $(\text{count(correct)}) / (\text{total answerable questions}) \times 100$.
     - `Wrong %` = $(\text{count(wrong)}) / (\text{total questions}) \times 100$.
     - `Correctly refused %` = $(\text{count(refused\_correctly)}) / (\text{total unanswerable questions}) \times 100$.
     - Both unanswerable and answerable subsets are tracked so metrics are mathematically sound.

4. **"At 3pm the daily budget is gone. Describe the reply a waiting person receives."**
   - *Interpretation:* When the daily budget (TPD/RPD) is depleted, continuing to queue or retry requests is catastrophic. The system must immediately short-circuit incoming requests with an explicit HTTP 429 / structured client message stating that the system-wide daily allocation has been exhausted, providing the exact reset time (e.g. midnight UTC) and offering to cache the query or check back later. Waiting requests in the queue must be cancelled gracefully with a clear refusal rather than timing out silently.

---

## 4. Experimental vs Documented Metrics

| Metric | Source Category | Determination Method |
|---|---|---|
| Groq RPM / TPM / RPD / TPD limits | Official Documentation & API Headers | Extracted from `console.groq.com/docs/rate-limits`, verified via `x-ratelimit-*` headers on live requests. |
| Groq 429 response structure & headers | Official Docs & Measured Probes | Inspected from live probe or simulated exact payload matching Groq docs (`{"error": {"message": ..., "type": "tokens", "code": "rate_limit_exceeded"}}`). |
| In-flight concurrency limits | Documented / Inferred | Groq does not publicly document a fixed concurrency ceiling; it enforces RPM/TPM token bucket leaky bucket algorithms. Locally managed via client semaphore. |
| Model Latency (TTFT & Total Duration) | Experimental (Real & Stand-In) | Monotonically measured using Python `time.perf_counter()` from socket send to first SSE chunk and final chunk. |
| Pipeline Stage Latency Breakdown | Experimental | Timers placed across admission, retrieval, LLM call, and verification. |
| Load Scenarios (A through E) | Experimental (Stand-In & Real Probes) | Executed via benchmark runner against stand-in (and real service within free tier quota). |

---

## 5. Credentials & Access Plan

- **Groq API Key:** Required for live queries against `https://api.groq.com/openai/v1/chat/completions`.
- **Handling Protocol:**
  - Key loaded exclusively via `GROQ_API_KEY` from environment or local uncommitted `.env`.
  - `.gitignore` explicitly blocks `.env`.
  - Application checks presence gracefully: if missing, real Groq integration is safely bypassed and flagged as "Awaiting User Key", while the entire pipeline, stand-in service, ingestion engine, unit tests, and load benchmarks execute autonomously.
  - No secret is ever written to logs, tables, reports, or stdout.
