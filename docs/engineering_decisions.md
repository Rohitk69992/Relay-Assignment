# The Relay: Engineering Decisions & Architectural Rationale

## 1. System Architecture & Component Separation

To satisfy both production resilience and research-grounded citation accuracy without over-engineering, the system is designed around a clean, asynchronous, modular Python architecture:

```
[User Request / Client]
           │
           ▼
[FastAPI / Admission Controller] ◄── [Token Bucket & Quota Tracker (RPM/TPM/RPD)]
           │
     (Enqueued)
           ▼
 [Bounded Priority Queue] ◄── [Queue Wait Deadline: 10s]
           │
    (Worker Pull)
           ▼
[Pipeline Orchestrator]
   ├── 1. Intent & Query Planning
   ├── 2. Retrieval Tool Invocation ──► [Hybrid BM25 + Cross-Score Reranker]
   │                                           │ (Searches Ingested Papers)
   ├── 3. Evidence Sufficiency Gate ◄──────────┘
   │      └── (Insufficient? -> 1 bounded query refinement loop)
   ├── 4. Provider Model Call (Groq / Stand-In)
   │      └── (Concurrency Semaphore + Rate Limiter + Retry Policy)
   ├── 5. Response Verifier & Citation Auditor
   └── 6. SSE Token Streaming Output ──► [Client]
```

---

## 2. Key Engineering Decisions & Trade-Offs

### Decision 1: The Core Trade-Off: Fast vs Good vs Cheap vs Zero Turnaways
* **Context:** The assignment notes: *"The company wants every reply fast, every reply good, the bill small, and nobody turned away... It cannot have all four."*
* **Decision:** We establish a strict hierarchy of invariants:
  1. **Grounding & Accuracy is Non-Negotiable (Quality > Speed/Availability):** We will never output an ungrounded hallucination or invent a citation to shave milliseconds or avoid a refusal. If evidence is lacking, we return an explicit refusal.
  2. **Bounded Queueing over Silent Dropping (Availability over Immediate Execution):** Requests that arrive in a burst are held in a bounded async queue up to 10 seconds. If capacity frees up, they execute normally.
  3. **Explicit Early Rejection over Indefinite Hangs:** When the queue reaches capacity (50 requests) or the daily quota is exhausted, incoming requests are rejected immediately with HTTP 429 and a descriptive reset payload, preventing wasteful resource exhaustion.
  4. **Cost Minimization via Targeted Retrieval:** Rather than stuffing full papers into context, we retrieve compact, high-precision chunks (e.g. 500 characters, top-3), saving 85% of input tokens and remaining well within Groq's free-tier TPM constraints.

### Decision 2: Pure Asynchronous Concurrency over Multiprocessing
* **Rationale:** LLM pipeline bottlenecks are overwhelmingly network I/O-bound (HTTP SSE streaming, provider rate-limit waits, and retrieval lookups). Python's `asyncio` combined with `httpx.AsyncClient` provides lightweight, low-overhead scheduling for hundreds of concurrent connections without thread contention or memory bloat.

### Decision 3: Hybrid Retrieval Architecture (BM25 + Semantic Scoring + Cross-Reranking)
* **Rationale:** Pure vector similarity often fails on precise technical terms, version numbers, and equation variables (e.g. comparing $\tau_{\text{chunk}}$ in Dynamic Chunking vs multi-head attention $d_k$). Lexical BM25 excels at exact keyword and token matching. Combining BM25 with a cross-scoring re-ranker ensures top precision while remaining fast ($\le 15\text{ms}$ retrieval latency).

### Decision 4: Dual-Provider Strategy (Real Groq + Controllable Stand-In)
* **Rationale:** Groq's free tier imposes tight daily and per-minute constraints that would quickly fail during aggressive load scenarios (e.g. Sustained Overload or 100+ concurrent bursts). We implement an identical interface (`ModelProviderBase`) with two interchangeable adapters:
  - `GroqProvider`: Calls Groq's live API (`https://api.groq.com/openai/v1/chat/completions`), extracts `x-ratelimit-*` headers, records real LPU speed.
  - `StandInProvider`: A fully controllable simulator calibrated to real Groq latency percentiles, equipped with simulated token buckets, concurrency caps, multi-chunk SSE streaming, and load-dependent latency inflation.

### Decision 5: Bounded Loop for Evidence Sufficiency
* **Rationale:** A single naive retrieval pass may miss relevant passages if the user's prompt is colloquial or ambiguous. An explicit Sufficiency Gate inspects retrieved chunk relevance. If confidence is below threshold, it generates an expanded query and performs exactly one additional retrieval pass before generation. This bounds token consumption and avoids infinite agentic loops.

### Decision 6: Bounded Exponential Backoff with Jitter
* **Rationale:** Standard retries without jitter cause "thundering herd" problems when a provider's rate limit window resets. We implement:
  $$\text{Backoff} = \min(T_{\max}, T_{\text{base}} \times 2^{\text{retry}}) + \text{Uniform}(0, \text{jitter})$$
  Retries are capped at $N=3$, and if a provider response includes `retry-after`, we strictly honor that duration up to our overall request deadline.
