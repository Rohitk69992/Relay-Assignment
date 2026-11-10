"""Script to execute and document real Groq service probes."""

import asyncio
import sys
sys.stdout.reconfigure(encoding='utf-8')
from pathlib import Path

# Add project root to sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.relay.config import settings
from src.relay.provider.groq_prober import execute_probe_suite


async def main():
    print("=" * 60)
    print("REAL GROQ SERVICE PROBE HARNESS")
    print("=" * 60)
    output_dir = settings.DATA_DIR
    records = await execute_probe_suite(output_dir=output_dir, num_probes=3)
    print(f"\nProbe execution finished. Total records captured: {len(records)}")


if __name__ == "__main__":
    asyncio.run(main())
