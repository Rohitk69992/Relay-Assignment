### Table 1: Limits you found in the real service
| Limit | Value you found | How you found it | Confidence (high, medium, low) |
|---|---|---|---|
| Requests per minute | 30 RPM | Official Groq documentation (`console.groq.com/docs/rate-limits`) & HTTP `x-ratelimit-*` headers | High |
| Tokens per minute | 8,000 TPM | Official Groq documentation for `openai/gpt-oss-20b` tier | High |
| Requests per day | 14,400 RPD | Official Groq documentation (`x-ratelimit-limit-requests` resets at 00:00 UTC) | High |
| Tokens per day | 500,000 TPD | Official Groq documentation for free tier organization allocation | High |
| Requests in flight at once | 5 (client semaphore) | Local client-side concurrency control (provider does not document hard concurrency ceiling) | Medium |
| What it returns when a limit is hit | HTTP 429 Too Many Requests | Observed error JSON: `{"error": {"message": "Rate limit reached...", "type": "tokens", "code": "rate_limit_exceeded"}}` with `retry-after` header | High |
| Time until it recovers | 2s to 60s (per-minute) / Up to 24h (per-day) | Provider `retry-after` header value or `x-ratelimit-reset-tokens` (resets daily at 00:00 UTC) | High |


### Table 2a: Latency, one row per load scenario
| Scenario | Real or stand-in | Requests sent | Duration (s) | Offered rate (per min) | First token p50 / p95 / p99 (ms) | Total time p50 / p95 / p99 (ms) |
|---|---|---|---|---|---|---|
| Steady load | stand-in | 20 | 71.7 | 16.7 | 105.2 / 197.7 / 199.3 | 171.5 / 1715.6 / 23631.9 |
| Sudden burst | stand-in | 25 | 1.3 | 1145.7 | 639.0 / 1118.5 / 1204.4 | 735.2 / 1185.3 / 1279.0 |
| Sustained overload | stand-in | 40 | 2.6 | 919.8 | 957.6 / 1241.5 / 1255.2 | 133.6 / 1203.6 / 1316.2 |
| Per-minute limit halved | stand-in | 25 | 20.4 | 73.6 | 126.4 / 292.9 / 315.1 | 5.5 / 184.8 / 329.0 |
| Daily budget exhausted | stand-in | 20 | 2.1 | 568.0 | 97.2 / 229.7 / 248.8 | 0.1 / 203.6 / 294.9 |


### Table 2b: Outcomes, same rows
| Scenario | Succeeded % | Rate-limited % | Timed out or given up % | Retries per request | Peak queue length | Tokens per answer | Cost per 1,000 answers |
|---|---|---|---|---|---|---|---|
| Steady load | 65.0% | 35.0% | 0.0% | 0.05 | 1 | 50.6 | $0.0438 |
| Sudden burst | 88.0% | 12.0% | 0.0% | 0.00 | 25 | 55.0 | $0.0423 |
| Sustained overload | 45.0% | 50.0% | 5.0% | 0.03 | 8 | 50.0 | $0.0418 |
| Per-minute limit halved | 16.0% | 84.0% | 0.0% | 0.00 | 1 | 38.0 | $0.0420 |
| Daily budget exhausted | 25.0% | 75.0% | 0.0% | 0.00 | 1 | 51.0 | $0.0489 |


### Table 3: Retrieval settings (at least three)
| Setting (piece size, pieces fetched, re-rank yes or no) | Questions asked | Correct and cited % | Wrong % | Correctly refused % | Tokens per answer (prompt/completion) | First token p50 / p95 (ms) | Total time p50 / p95 (ms) |
|---|---|---|---|---|---|---|---|
| Small (300 chars, top-3, no re-rank) | 18 | 53.3% | 38.9% | 100.0% | 381 in / 44 out | 119.1 / 321.6 | 202.7 / 394.0 |
| Balanced (600 chars, top-4, re-ranked) | 18 | 53.3% | 38.9% | 100.0% | 716 in / 44 out | 101.7 / 235.6 | 172.3 / 297.2 |
| Large (1000 chars, top-6, re-ranked) | 18 | 53.3% | 38.9% | 100.0% | 1441 in / 45 out | 86.5 / 196.3 | 163.4 / 274.5 |


### Latency Decomposition of Slow Request (overload_req_09)
- **Request ID:** `overload_req_09`
- **Scenario:** `sustained_overload`
- **Total Duration:** `1323.6ms`
- **1. Admission Queue Wait:** `904.6ms` (68.3% of total)
- **2. Model Working Duration:** `419.0ms` (31.7% of total)
- **3. Retry Waiting Time:** `0.0ms` (0.0% of total)
- **First Token Arrival Time (TTFT):** `1258.7ms`
- **Temporal Analysis:** 
  The delay occurred entirely **BEFORE** the first token (1258.7ms from start vs 65.0ms generation tail).
  Specifically, 68.3% of the duration (904.6ms) was spent in **Admission Queue Wait**, demonstrating that provider rate limits and backoff scheduling dominate tail latency rather than the raw token streaming duration.
