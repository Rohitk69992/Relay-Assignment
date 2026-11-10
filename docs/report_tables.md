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
| Steady load | stand-in | 20 | 73.2 | 16.4 | 114.0 / 200.4 / 216.5 | 232.9 / 1784.7 / 23918.3 |
| Sudden burst | stand-in | 25 | 2.2 | 684.4 | 995.3 / 1886.3 / 1965.3 | 1296.0 / 2083.7 / 2168.8 |
| Sustained overload | stand-in | 40 | 3.0 | 787.9 | 1039.4 / 1487.5 / 1562.9 | 259.9 / 1363.1 / 1709.1 |
| Per-minute limit halved | stand-in | 25 | 21.5 | 69.8 | 133.2 / 326.1 / 359.1 | 6.9 / 365.2 / 525.7 |
| Daily budget exhausted | stand-in | 20 | 3.3 | 366.5 | 74.0 / 601.9 / 692.8 | 0.2 / 389.2 / 815.2 |


### Table 2b: Outcomes, same rows
| Scenario | Succeeded % | Rate-limited % | Timed out or given up % | Retries per request | Peak queue length | Tokens per answer | Cost per 1,000 answers |
|---|---|---|---|---|---|---|---|
| Steady load | 80.0% | 20.0% | 0.0% | 0.10 | 1 | 88.0 | $0.0467 |
| Sudden burst | 88.0% | 12.0% | 0.0% | 0.00 | 25 | 88.0 | $0.0449 |
| Sustained overload | 25.0% | 50.0% | 25.0% | 0.03 | 8 | 88.0 | $0.0448 |
| Per-minute limit halved | 20.0% | 80.0% | 0.0% | 0.00 | 1 | 88.0 | $0.0460 |
| Daily budget exhausted | 25.0% | 75.0% | 0.0% | 0.00 | 1 | 88.0 | $0.0518 |


### Table 3: Retrieval settings (at least three)
| Setting (piece size, pieces fetched, re-rank yes or no) | Questions asked | Correct and cited % | Wrong % | Correctly refused % | Tokens per answer | First token p50 / p95 (ms) | Total time p50 / p95 (ms) |
|---|---|---|---|---|---|---|---|
| Small (300 chars, top-3, no re-rank) | 18 | 53.3% | 38.9% | 100.0% | 84.5 | 166.5 / 340.6 | 311.3 / 477.3 |
| Balanced (600 chars, top-4, re-ranked) | 18 | 53.3% | 38.9% | 100.0% | 84.5 | 113.2 / 241.2 | 257.5 / 367.3 |
| Large (1000 chars, top-6, re-ranked) | 18 | 53.3% | 38.9% | 100.0% | 84.5 | 112.6 / 211.7 | 242.6 / 342.7 |


### Latency Decomposition of Slow Request (steady_req_15)
- **Request ID:** `steady_req_15`
- **Scenario:** `steady`
- **Total Duration:** `29451.7ms`
- **1. Admission Queue Wait:** `0.2ms` (0.0% of total)
- **2. Model Working Duration:** `366.6ms` (1.2% of total)
- **3. Retry Waiting Time:** `29084.8ms` (98.8% of total)
- **First Token Arrival Time (TTFT):** `220.5ms`
- **Temporal Analysis:** 
  The delay occurred **BEFORE** the first token (220.5ms from start vs 29231.2ms generation tail). 
  The vast majority of the latency was spent waiting in the bounded admission queue (0.2ms) until an execution slot cleared, proving that under overload, queue backlog dominates user-perceived TTFT rather than token streaming duration.
