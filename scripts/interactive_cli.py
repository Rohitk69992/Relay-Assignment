"""Interactive CLI for manually testing The Relay pipeline in real-time."""

import asyncio
import sys
from pathlib import Path

# Ensure project root is in sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.relay.config import settings
from src.relay.pipeline.orchestrator import PipelineOrchestrator
from src.relay.provider.groq_client import GroqProvider
from src.relay.provider.standin_client import StandInProvider
from src.relay.schemas import (
    AnswerGrade,
    PipelineRequest,
    RetrievalSetting,
    SourceType
)


async def main():
    print("=" * 70)
    print("  THE RELAY: INTERACTIVE MANUAL TESTING CONSOLE")
    print("=" * 70)
    print("Grounded exclusively in 3 research papers:")
    print("  1. Attention Is All You Need (arXiv:1706.03762v7)")
    print("  2. Moshi: a speech-text foundation model (arXiv:2410.00037v1)")
    print("  3. Dynamic Chunking for Sequence Modeling (arXiv:2507.07955v2)")
    print("-" * 70)

    # Choose provider
    if settings.is_groq_available() and settings.PROVIDER_MODE == "groq":
        provider = GroqProvider()
        source = SourceType.REAL
        print(f"Provider: REAL GROQ API (Model: {provider.model})")
    else:
        provider = StandInProvider()
        source = SourceType.STANDIN
        print("Provider: CALIBRATED STAND-IN SIMULATOR (LPU Latency Model)")

    orchestrator = PipelineOrchestrator(provider=provider)
    print("System ready. Type your question or 'quit' / 'exit' to stop.")
    print("=" * 70 + "\n")

    counter = 1
    while True:
        try:
            prompt_text = input(f"[{counter:02d}] Enter Question > ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nExiting.")
            break

        if not prompt_text:
            continue
        if prompt_text.lower() in {"quit", "exit", "q"}:
            print("Goodbye!")
            break

        req = PipelineRequest(
            request_id=f"manual_{counter:03d}",
            question=prompt_text,
            scenario="manual_test",
            source=source,
            retrieval_setting=RetrievalSetting(chunk_size=600, top_k=4, rerank=True)
        )

        print("  -> Processing through 7-stage pipeline...")
        resp = await orchestrator.execute_request(req)

        print("\n" + "-" * 70)
        print("PIPELINE RESULT:")
        print(f"Status:       {resp.status.value.upper()}")
        print(f"Answer Grade: {resp.answer_grade.value.upper()}")
        print(f"Latency:      TTFT: {resp.first_token_ms or 0.0:0.1f}ms | Total: {resp.total_ms:0.1f}ms | Queue Wait: {resp.queue_wait_ms:0.1f}ms")
        print(f"Tokens:       In={resp.tokens_in}, Out={resp.tokens_out}")
        print("-" * 70)
        print("ANSWER:")
        print(resp.answer)
        print("-" * 70)

        if resp.citations:
            print("AUDITED CITATIONS:")
            for cite in resp.citations:
                sec_str = f" | Section: {cite.section}" if cite.section else ""
                print(f"  * [{cite.paper_title}] Page {cite.page}{sec_str}")
            print("-" * 70)

        print("\n")
        counter += 1


if __name__ == "__main__":
    asyncio.run(main())
