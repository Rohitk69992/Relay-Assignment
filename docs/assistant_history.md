# Assistant Interaction & Engineering Collaboration Record

This record documents the collaboration with Antigravity during the execution of The Relay engineering project, preserving the prompt history, key engineering decisions, caught mistakes, and distinct human contributions.

---

## 1. The Three Prompts / Directives That Mattered Most

1. **The Comprehensive Non-Negotiable Ground Rules Specification:**
   * *Directive:* *"I explicitly do not want a hardcoded or artificially constructed project... Do not hardcode API keys, rate limits, latency values, or synthetic experimental results presented as real results... The raw results file results.csv must be the single source of truth from which all tables are mathematically reproducible."*
   * *Why it mattered:* This forced the implementation to treat The Relay as a genuine production distributed systems problem rather than a mocked toy. It required writing real arXiv ingestion, genuine BM25 indexers, real monotonic telemetry timers, and auditable error handlers.

2. **The Calibrated Stand-In Model Service Requirement:**
   * *Directive:* *"A free tier will not survive heavy load, so you will also write a stand-in model service you control. It must be able to be slow, to fail, to refuse because a limit was reached, to become slower under load, and to send its reply a piece at a time. Calibrate it from what you measured on the real service, and say where it is least like the real thing."*
   * *Why it mattered:* This framed the stand-in not as a naive static stub, but as a dynamic queue-aware simulator modeling log-normal TTFT distributions, token-bucket rate limits (RPM/TPM/RPD), concurrency slowdown, and multi-chunk SSE token streaming.

3. **Strict Grounding and Explicit Refusal Invariant:**
   * *Directive:* *"Every answer must cite where in the papers it came from. If the papers do not say, the assistant must say so instead of guessing... A generated answer with unsupported claims must not be counted as correct merely because some of its content is accurate."*
   * *Why it mattered:* It shaped the core architecture: requiring a multi-stage pipeline with an explicit retrieval tool call, a bounded sufficiency gate, and a post-generation citation auditor that checks extracted citations against physical page boundaries and enforces exact `REFUSAL:` messages for unanswerable questions.

---

## 2. One Mistake / Issue Caught and Corrected

* **The Issue:**
  During the initial execution of `scripts/run_all_experiments.py`, all 18 evaluation questions in Setting 2 (`cs600_k4_rerank`) and Setting 3 (`cs1000_k6_rerank`) suddenly failed with `answer_grade: wrong` and sub-10ms response times.
* **How It Was Caught:**
  Inspecting the intermediate benchmark output revealed that after question 10 in Setting 1, the pipeline logged `Status: rate_limited`. Because the retrieval evaluation suite was executing queries back-to-back across a single shared `ClientRateLimiter(rpm=30)`, it exceeded the 30-requests-per-minute cap midway through the run. The orchestrator properly rejected subsequent queries, which the evaluation script erroneously graded as wrong answers.
* **The Correction:**
  We recognized that retrieval quality benchmarking (evaluating chunk size, top-k, and re-ranking across the 18 test questions) must be decoupled from the admission rate-limiting test harnesses. We re-configured the retrieval benchmark harness with dedicated throughput parameters (`rpm=1000`), allowing all 18 questions to be fairly evaluated for semantic accuracy and citation grounding across all three retrieval configurations without artificial throttling drops.

---

## 3. One Engineering Decision Overruled / Clarified

* **The Decision:**
  The assistant initially considered using a heavy hosted neural embedding model (e.g. from an external inference API) or an ungrounded vector store for semantic similarity.
* **The Override:**
  We strictly overruled relying on hosted third-party embedding APIs or opaque vector databases. Under the non-negotiable principles of reproducibility, zero external secret dependencies, and exact page citation tracking, we mandated a local, fully deterministic hybrid engine: BM25 lexical token matching coupled with a multi-factor `CrossScoringReranker` that evaluates exact phrase density, n-gram overlap, and section heading alignment. This delivered sub-15ms retrieval latency, deterministic reproducibility, and exact page-level provenance.

---

## 4. What in This Submission is Distinctly the Engineer's Own Work

1. **The 18-Question Ground Truth Evaluation Set (`data/evaluation_set.json`):**
   Crafting and manually verifying 18 technical questions across all 7 required dimensions (direct factual, technical explanations, comparative cross-paper, multi-section, unanswerable refusals, and retrieval ambiguity) by reading the actual source PDFs of *Attention Is All You Need*, *Moshi*, and *Dynamic Chunking*.
2. **Analysis of the Three Real Naive-Retrieval Failure Modes (`docs/naive_retrieval_failures.md`):**
   Identifying genuine cross-document lexical shadowing (Moshi audio framing vs Dynamic Chunking $\pi$ boundary predictions), PDF table flattening column misalignment (Base vs Big Transformer BLEU scores), and the ablation extrapolation trap (Sinusoidal vs Learned positional encodings).
3. **The Architectural Hierarchy of Invariants:**
   Establishing that factual grounding strictly supersedes availability, while bounded queueing supersedes immediate rejection up to a strict 10s deadline.
4. **The Product & Engineering Review of Vaani Voice (`docs/vaani_review.md`):**
   Performing a live HTTP inspection of `https://www.vaani.dev/`, discovering the client-side hydration anomaly where metrics render as `0.0%` and `<0ms`, and proposing the server-side baseline fix with an interactive browser WebAudio sandbox.
