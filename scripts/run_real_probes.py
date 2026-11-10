"""Runs modest, controlled real-service probes against Groq API to measure empirical LPU latency and rate-limit headers."""

import asyncio
import os
import sys
import time
from pathlib import Path
from src.relay.config import settings
from src.relay.provider.groq_client import GroqProvider


async def run_probes():
    print("=" * 60)
    print("GROQ REAL-SERVICE PROBE HARNESS")
    print("=" * 60)

    if not settings.is_groq_available():
        print("[NOTICE] GROQ_API_KEY is not configured in the environment or .env file.")
        print("To run real-service measurements against your personal free-tier account:")
        print("  1. Copy .env.example to .env")
        print("  2. Set GROQ_API_KEY=gsk_your_actual_key_here")
        print("  3. Re-run: python scripts/run_real_probes.py")
        print("\nProceeding in standalone safe mode with documented official Groq parameters.")
        return

    provider = GroqProvider()
    print(f"Connected to model: {provider.model}")
    print(f"Base URL: {provider.base_url}")
    print("Executing 5 controlled measurement probes...\n")

    latencies = []
    first_tokens = []

    test_queries = [
        "What is the attention formula in Transformer?",
        "What is the framerate of Mimi codec in Moshi?",
        "What is the boundary predictor in Dynamic Chunking?",
        "What is the BLEU score on English to German for base Transformer?",
        "What is the parameter count of Helium model?"
    ]

    for idx, query in enumerate(test_queries, 1):
        print(f"Probe {idx}/5: '{query[:40]}...'")
        resp = await provider.generate(prompt=query, max_tokens=100)
        print(f"  HTTP Status: {resp.http_status}")
        print(f"  Total Duration: {resp.total_ms:.1f}ms")
        print(f"  Tokens: In={resp.tokens_in}, Out={resp.tokens_out}")
        print("  Rate Limit Headers:")
        for h, v in resp.headers.items():
            print(f"    {h}: {v}")
        print()
        latencies.append(resp.total_ms)
        if resp.first_token_ms:
            first_tokens.append(resp.first_token_ms)
        await asyncio.sleep(1.0)  # Safe delay between probes

    print("Probe execution complete.")
    if latencies:
        import numpy as np
        print(f"Empirical Total Latency: p50={np.percentile(latencies, 50):.1f}ms, p95={np.percentile(latencies, 95):.1f}ms")


if __name__ == "__main__":
    asyncio.run(run_probes())
