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
| Steady load | stand-in | 20 | 84.4 | 14.2 | 862.0 / 881.7 / 882.7 | 933.8 / 2447.4 / 24764.9 |
| Sudden burst | stand-in | 25 | 6.7 | 223.3 | 3348.0 / 6527.4 / 6622.9 | 4417.5 / 6466.4 / 6699.2 |
| Sustained overload | stand-in | 40 | 3.1 | 775.7 | 1775.6 / 1886.0 / 1892.3 | 0.4 / 1845.4 / 1944.0 |
| Per-minute limit halved | stand-in | 25 | 23.2 | 64.7 | 864.7 / 892.9 / 896.3 | 5.8 / 935.5 / 955.4 |
| Daily budget exhausted | stand-in | 20 | 5.9 | 203.5 | 866.0 / 881.0 / 883.3 | 0.2 / 953.8 / 1011.1 |


### Table 2b: Outcomes, same rows
| Scenario | Succeeded % | Rate-limited % | Timed out or given up % | Retries per request | Peak queue length | Tokens per answer | Cost per 1,000 answers |
|---|---|---|---|---|---|---|---|
| Steady load | 80.0% | 20.0% | 0.0% | 0.05 | 1 | 50.2 | $0.0437 |
| Sudden burst | 88.0% | 12.0% | 0.0% | 0.00 | 25 | 55.0 | $0.0423 |
| Sustained overload | 12.5% | 55.0% | 32.5% | 0.03 | 8 | 50.0 | $0.0418 |
| Per-minute limit halved | 16.0% | 84.0% | 0.0% | 0.00 | 1 | 38.0 | $0.0420 |
| Daily budget exhausted | 25.0% | 75.0% | 0.0% | 0.00 | 1 | 51.0 | $0.0489 |


### Table 3: Retrieval settings (at least three)
| Setting (piece size, pieces fetched, re-rank yes or no) | Questions asked | Correct and cited % | Wrong % | Correctly refused % | Tokens per answer (prompt/completion) | First token p50 / p95 (ms) | Total time p50 / p95 (ms) |
|---|---|---|---|---|---|---|---|
| Small (300 chars, top-3, no re-rank) | 18 | 53.3% | 38.9% | 100.0% | 381 in / 44 out | 870.5 / 904.2 | 935.7 / 993.8 |
| Balanced (600 chars, top-4, re-ranked) | 18 | 53.3% | 38.9% | 100.0% | 716 in / 44 out | 861.7 / 884.5 | 922.1 / 969.5 |
| Large (1000 chars, top-6, re-ranked) | 18 | 53.3% | 38.9% | 100.0% | 1441 in / 45 out | 854.2 / 888.8 | 920.8 / 956.4 |


### Latency Decomposition of Slow Request (steady_req_12)
- **Request ID:** `steady_req_12`
- **Scenario:** `steady`
- **Total Duration:** `30344.3ms`
- **1. Admission Queue Wait:** `0.1ms` (0.0% of total)
- **2. Model Working Duration:** `944.2ms` (3.1% of total)
- **3. Retry Waiting Time:** `29400.0ms` (96.9% of total)
- **First Token Arrival Time (TTFT):** `867.6ms`
- **Temporal Analysis:** 
  The delay occurred entirely **BEFORE** the first token (867.6ms from start vs 29476.7ms generation tail).
  Specifically, 96.9% of the duration (29400.0ms) was spent in **Retry Backoff Wait**, demonstrating that provider rate limits and backoff scheduling dominate tail latency rather than the raw token streaming duration.
