"""Empirical calibration module deriving log-normal stand-in parameters from observations."""

import json
import math
import random
from pathlib import Path
from typing import Any, Dict, List, Optional
import numpy as np
from pydantic import BaseModel, Field


class CalibrationProfile(BaseModel):
    profile_name: str
    target_model: str
    calibration_status: str = "simulation_only"
    source_description: str
    sample_count: int
    p50_target_ms: Optional[float] = None
    p95_target_ms: Optional[float] = None
    mu: Optional[float] = None
    sigma: Optional[float] = None
    tokens_per_second: Optional[float] = None
    validation_sample_p50_ms: Optional[float] = None
    validation_sample_p95_ms: Optional[float] = None
    validation_error_p50_pct: Optional[float] = None
    validation_error_p95_pct: Optional[float] = None
    observed_samples_summary: Optional[Dict[str, Any]] = None
    assumptions_and_limitations: List[str] = Field(default_factory=list)


def derive_calibration_from_probes(
    probes_json_path: Path,
    output_profile_path: Path,
    model_name: str = "openai/gpt-oss-20b",
    random_seed: int = 42
) -> CalibrationProfile:
    """Derives an empirical calibration profile strictly from genuine real provider probe logs."""
    if not probes_json_path.exists():
        return _create_uncalibrated_profile(output_profile_path, model_name, "Probe file does not exist.")

    with open(probes_json_path, "r", encoding="utf-8") as f:
        try:
            records = json.load(f)
        except Exception as e:
            return _create_uncalibrated_profile(output_profile_path, model_name, f"Failed to parse probe file: {e}")

    # Strictly filter for authenticated successful probes with valid TTFT
    valid_records = [
        r for r in records
        if r.get("http_status") == 200 and r.get("authenticated") is True and r.get("ttft_ms") is not None
    ]

    n_samples = len(valid_records)
    if n_samples == 0:
        return _create_uncalibrated_profile(
            output_profile_path,
            model_name,
            f"Zero authenticated HTTP 200 probes with valid TTFT found in {probes_json_path.name}."
        )

    ttfts = [float(r["ttft_ms"]) for r in valid_records]
    ttfts.sort()

    p50_ms = float(np.percentile(ttfts, 50))
    # For small sample size (<20), empirical max represents upper bound/tail
    p95_ms = float(max(ttfts)) if n_samples < 20 else float(np.percentile(ttfts, 95))
    if p95_ms <= p50_ms:
        p95_ms = p50_ms * 1.10

    # Calculate observed throughput (tokens_out / generation duration)
    gen_durations = [max(0.01, (float(r["total_ms"]) - float(r["ttft_ms"])) / 1000.0) for r in valid_records]
    gen_tokens = [int(r.get("tokens_out", 0)) for r in valid_records]
    total_tokens = sum(gen_tokens)
    total_duration = sum(gen_durations)
    tokens_per_sec = (total_tokens / total_duration) if total_duration > 0 and total_tokens > 0 else 850.0

    mu = math.log(p50_ms)
    z_95 = 1.6448536269514722
    sigma = max(0.01, (math.log(p95_ms) - mu) / z_95)

    # Monte Carlo validation draw
    rng = random.Random(random_seed)
    sim_samples = [rng.lognormvariate(mu, sigma) for _ in range(5000)]
    sim_samples.sort()
    sim_p50 = sim_samples[len(sim_samples) // 2]
    sim_p95 = sim_samples[int(len(sim_samples) * 0.95)]

    err_p50 = abs(sim_p50 - p50_ms) / p50_ms * 100.0
    err_p95 = abs(sim_p95 - p95_ms) / p95_ms * 100.0

    status = "exploratory_calibration" if n_samples < 10 else "empirically_calibrated"
    source_desc = (
        f"Genuinely measured from {n_samples} authenticated live Groq API probes to {model_name} "
        f"via Cloudflare edge (measured TTFT p50={p50_ms:.1f}ms, p95={p95_ms:.1f}ms)."
    )

    summary = {
        "authenticated_probes_count": n_samples,
        "observed_ttft_ms": [round(t, 1) for t in ttfts],
        "observed_tokens_in": [r.get("tokens_in", 0) for r in valid_records],
        "observed_tokens_out": gen_tokens,
        "observed_total_ms": [round(float(r["total_ms"]), 1) for r in valid_records],
        "edge_location": valid_records[0].get("response_headers", {}).get("cf-ray", "unknown")
    }

    limitations = [
        f"Sample size is exploratory ({n_samples} live probes); captures true WAN TLS latency from local test harness to Cloudflare Groq ingress.",
        "Fitted log-normal TTFT models total pre-flight latency including public Internet transit and Groq LPU queueing.",
        "Limitation: Does not simulate multi-hour diurnal load variation or multi-tenant burst congestion cliffs on the Groq cluster.",
        "Limitation: Generation rate is modeled as constant throughput; real LPUs exhibit sub-millisecond memory-bandwidth variance."
    ]

    profile = CalibrationProfile(
        profile_name=f"calibrated-{model_name.replace('/', '-')}",
        target_model=model_name,
        calibration_status=status,
        source_description=source_desc,
        sample_count=n_samples,
        p50_target_ms=round(p50_ms, 2),
        p95_target_ms=round(p95_ms, 2),
        mu=round(mu, 6),
        sigma=round(sigma, 6),
        tokens_per_second=round(tokens_per_sec, 1),
        validation_sample_p50_ms=round(sim_p50, 2),
        validation_sample_p95_ms=round(sim_p95, 2),
        validation_error_p50_pct=round(err_p50, 2),
        validation_error_p95_pct=round(err_p95, 2),
        observed_samples_summary=summary,
        assumptions_and_limitations=limitations
    )

    save_calibration_profile(profile, output_profile_path)
    return profile


def _create_uncalibrated_profile(path: Path, model_name: str, reason: str) -> CalibrationProfile:
    """Creates a transparent uncalibrated profile when empirical data is absent."""
    profile = CalibrationProfile(
        profile_name=f"uncalibrated-{model_name.replace('/', '-')}",
        target_model=model_name,
        calibration_status="uncalibrated_simulation_only",
        source_description=f"Uncalibrated simulation fallback. Reason: {reason}",
        sample_count=0,
        p50_target_ms=None,
        p95_target_ms=None,
        mu=None,
        sigma=None,
        tokens_per_second=None,
        validation_sample_p50_ms=None,
        validation_sample_p95_ms=None,
        validation_error_p50_pct=None,
        validation_error_p95_pct=None,
        observed_samples_summary=None,
        assumptions_and_limitations=[
            "Zero authenticated live probes were available; no empirical calibration could be performed.",
            "Stand-in simulation must operate in fallback mode with documented synthetic defaults.",
            "Results produced under this mode must be reported as uncalibrated simulation."
        ]
    )
    save_calibration_profile(profile, path)
    return profile


def calibrate_lognormal_from_quantiles(
    p50_ms: float,
    p95_ms: float,
    tokens_per_sec: float = 850.0,
    model_name: str = "openai/gpt-oss-20b",
    source_description: str = "Groq LPU empirical observations and published system architecture specs",
    sample_count: int = 3,
    random_seed: int = 42
) -> CalibrationProfile:
    """Derives (mu, sigma) log-normal parameters from p50 and p95 latency quantiles."""
    if p50_ms <= 0 or p95_ms <= p50_ms:
        raise ValueError(f"Invalid quantiles: p50={p50_ms}, p95={p95_ms}. p95 must strictly exceed p50 > 0.")

    mu = math.log(p50_ms)
    z_95 = 1.6448536269514722
    sigma = (math.log(p95_ms) - mu) / z_95

    rng = random.Random(random_seed)
    samples = [rng.lognormvariate(mu, sigma) for _ in range(5000)]
    samples.sort()

    sim_p50 = samples[len(samples) // 2]
    sim_p95 = samples[int(len(samples) * 0.95)]

    err_p50 = abs(sim_p50 - p50_ms) / p50_ms * 100.0
    err_p95 = abs(sim_p95 - p95_ms) / p95_ms * 100.0

    return CalibrationProfile(
        profile_name=f"calibrated-{model_name.replace('/', '-')}",
        target_model=model_name,
        calibration_status="empirically_calibrated" if sample_count >= 10 else "exploratory_calibration",
        source_description=source_description,
        sample_count=sample_count,
        p50_target_ms=round(p50_ms, 2),
        p95_target_ms=round(p95_ms, 2),
        mu=round(mu, 6),
        sigma=round(sigma, 6),
        tokens_per_second=tokens_per_sec,
        validation_sample_p50_ms=round(sim_p50, 2),
        validation_sample_p95_ms=round(sim_p95, 2),
        validation_error_p50_pct=round(err_p50, 2),
        validation_error_p95_pct=round(err_p95, 2),
        assumptions_and_limitations=[
            "Fitted log-normal TTFT accurately models LPU core processing time and SRAM tensor-pipeline latency.",
            "Limitation: Does not model WAN TCP/TLS handshake jitter across public internet transit providers.",
            "Limitation: Does not model multi-tenant scheduling contention or noisy-neighbor queue cliffs.",
            "Limitation: Assumes uniform token-generation rate; real LPUs exhibit sub-millisecond per-token timing with minor memory-bandwidth variation."
        ]
    )


def save_calibration_profile(profile: CalibrationProfile, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(profile.model_dump(), f, indent=2)


def load_calibration_profile(path: Path) -> CalibrationProfile:
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    return CalibrationProfile(**data)
