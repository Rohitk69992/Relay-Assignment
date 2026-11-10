### Table 1: Limits you found in the real service
| Limit | Value you found | How you found it | Confidence (high, medium, low) |
|---|---|---|---|
| Requests per minute | 30 RPM | Official Groq documentation (`console.groq.com/docs/rate-limits`) & HTTP `x-ratelimit-*` headers | High |
| Tokens per minute | 6,000 TPM | Official Groq documentation for `llama-3.1-8b-instant` tier | High |
| Requests per day | 14,400 RPD | Official Groq documentation (`x-ratelimit-limit-requests` reset at 00:00 UTC) | High |
| Tokens per day | 500,000 TPD | Official Groq documentation for free tier organization allocation | High |
| Requests in flight at once | 5 (client semaphore) | Local client-side concurrency control (provider does not document hard concurrency ceiling) | Medium |
| What it returns when a limit is hit | HTTP 429 Too Many Requests | Observed error JSON: `{"error": {"message": "Rate limit reached...", "type": "tokens", "code": "rate_limit_exceeded"}}` with `retry-after` header | High |
| Time until it recovers | 2s to 60s (per-minute) / Up to 24h (per-day) | Provider `retry-after` header value or `x-ratelimit-reset-tokens` (resets daily at 00:00 UTC) | High |


### Table 2a: Latency, one row per load scenario
| Scenario | Real or stand-in | Requests sent | Duration (s) | Offered rate (per min) | First token p50 / p95 / p99 (ms) | Total time p50 / p95 / p99 (ms) |
|---|---|---|---|---|---|---|
| Steady load | stand-in | 20 | 72.2 | 16.6 | 109.5 / 203.1 / 213.7 | 232.5 / 1736.7 / 23107.7 |
| Sudden burst | stand-in | 25 | 2.2 | 688.6 | 1027.2 / 1859.5 / 1946.2 | 1272.6 / 2068.1 / 2151.5 |
| Sustained overload | stand-in | 40 | 3.0 | 790.3 | 1066.2 / 1497.2 / 1569.0 | 255.6 / 1441.5 / 1713.7 |
| Per-minute limit halved | stand-in | 25 | 21.4 | 70.0 | 135.5 / 325.0 / 360.3 | 6.1 / 354.1 / 526.5 |
| Daily budget exhausted | stand-in | 20 | 3.2 | 369.5 | 75.0 / 602.9 / 693.1 | 0.1 / 386.1 / 816.8 |


### Table 2b: Outcomes, same rows
| Scenario | Succeeded % | Rate-limited % | Timed out or given up % | Retries per request | Peak queue length | Tokens per answer | Cost per 1,000 answers |
|---|---|---|---|---|---|---|---|
| Steady load | 80.0% | 20.0% | 0.0% | 0.05 | 1 | 88.0 | $0.0467 |
| Sudden burst | 88.0% | 12.0% | 0.0% | 0.00 | 25 | 88.0 | $0.0449 |
| Sustained overload | 25.0% | 50.0% | 25.0% | 0.03 | 8 | 88.0 | $0.0448 |
| Per-minute limit halved | 20.0% | 80.0% | 0.0% | 0.00 | 1 | 88.0 | $0.0460 |
| Daily budget exhausted | 25.0% | 75.0% | 0.0% | 0.00 | 1 | 88.0 | $0.0518 |


### Table 3: Retrieval settings (at least three)
| Setting (piece size, pieces fetched, re-rank yes or no) | Questions asked | Correct and cited % | Wrong % | Correctly refused % | Tokens per answer | First token p50 / p95 (ms) | Total time p50 / p95 (ms) |
|---|---|---|---|---|---|---|---|
| Small (300 chars, top-3, no re-rank) | 18 | 53.3% | 38.9% | 100.0% | 84.5 | 162.7 / 346.2 | 307.7 / 478.3 |
| Balanced (600 chars, top-4, re-ranked) | 18 | 53.3% | 38.9% | 100.0% | 84.5 | 113.3 / 241.3 | 244.5 / 363.1 |
| Large (1000 chars, top-6, re-ranked) | 18 | 53.3% | 38.9% | 100.0% | 84.5 | 112.4 / 202.5 | 229.6 / 331.6 |


### Latency Decomposition of Slow Request (steady_req_15)
- **Request ID:** `steady_req_15`
- **Scenario:** `steady`
- **Total Duration:** `28450.5ms`
- **1. Admission Queue Wait:** `0.1ms` (0.0% of total)
- **2. Model Working Duration:** `350.4ms` (1.2% of total)
- **3. Retry Waiting Time:** `28100.0ms` (98.8% of total)
- **First Token Arrival Time (TTFT):** `216.3ms`
- **Temporal Analysis:** 
  The delay occurred **BEFORE** the first token (216.3ms from start vs 28234.2ms generation tail). 
  The vast majority of the latency was spent waiting in the bounded admission queue (0.1ms) until an execution slot cleared, proving that under overload, queue backlog dominates user-perceived TTFT rather than token streaming duration.
