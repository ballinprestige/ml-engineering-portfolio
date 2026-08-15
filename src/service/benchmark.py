"""Reproducible single-host HTTP benchmark for the portfolio scoring service.

This measures the complete FastAPI request path from a client process. It is intentionally
small and dependency-free; it does not claim distributed-system or production traffic scale.

Run after starting the service::

    python -m service.benchmark --requests 500 --concurrency 16 \
        --json-output benchmarks/local.json
"""

from __future__ import annotations

import argparse
import json
import os
import platform
import statistics
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import UTC, datetime
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

DEFAULT_PAYLOAD = {
    "pregnancies": 2,
    "glucose": 120.0,
    "blood_pressure": 70.0,
    "skin_thickness": 25.0,
    "insulin": 90.0,
    "bmi": 28.5,
    "diabetes_pedigree": 0.45,
    "age": 35,
}


def _percentile(values: list[float], percentile: float) -> float:
    """Return a linearly interpolated percentile for non-empty values."""
    if not values:
        raise ValueError("cannot calculate a percentile from an empty sequence")
    ordered = sorted(values)
    position = (len(ordered) - 1) * percentile
    lower = int(position)
    upper = min(lower + 1, len(ordered) - 1)
    fraction = position - lower
    return ordered[lower] + (ordered[upper] - ordered[lower]) * fraction


def _json_request(url: str, *, payload: dict | None = None, timeout: float = 10.0) -> dict:
    data = None if payload is None else json.dumps(payload).encode("utf-8")
    request = Request(url, data=data, headers={"Content-Type": "application/json"})
    with urlopen(request, timeout=timeout) as response:  # noqa: S310 - caller controls localhost URL
        return json.load(response)


def _timed_prediction(base_url: str, timeout: float) -> tuple[float | None, str | None]:
    start = time.perf_counter()
    try:
        _json_request(f"{base_url}/predict", payload=DEFAULT_PAYLOAD, timeout=timeout)
    except (HTTPError, URLError, TimeoutError, json.JSONDecodeError) as exc:
        return None, f"{type(exc).__name__}: {exc}"
    return (time.perf_counter() - start) * 1000, None


def summarize(
    *,
    latencies_ms: list[float],
    errors: list[str],
    elapsed_seconds: float,
    requests: int,
    concurrency: int,
    model: dict,
) -> dict:
    """Build the machine-readable benchmark report."""
    latency = None
    if latencies_ms:
        latency = {
            "mean_ms": round(statistics.fmean(latencies_ms), 3),
            "p50_ms": round(_percentile(latencies_ms, 0.50), 3),
            "p95_ms": round(_percentile(latencies_ms, 0.95), 3),
            "p99_ms": round(_percentile(latencies_ms, 0.99), 3),
            "max_ms": round(max(latencies_ms), 3),
        }

    return {
        "measured_at_utc": datetime.now(UTC).isoformat(),
        "scope": "single-host HTTP benchmark; not a distributed-scale or production-traffic claim",
        "requests": requests,
        "concurrency": concurrency,
        "succeeded": len(latencies_ms),
        "failed": len(errors),
        "elapsed_seconds": round(elapsed_seconds, 3),
        "throughput_requests_per_second": round(len(latencies_ms) / elapsed_seconds, 3),
        "latency": latency,
        "model": {
            key: model.get(key)
            for key in ("model_id", "version", "git_sha", "artifact_sha256")
            if model.get(key) is not None
        },
        "runtime": {
            "python": platform.python_version(),
            "platform": platform.platform(),
            "machine": platform.machine(),
            "processor": platform.processor(),
            "logical_cpu_count": os.cpu_count(),
        },
        "errors": errors[:10],
    }


def run_benchmark(
    *,
    base_url: str,
    requests: int,
    concurrency: int,
    warmup: int,
    timeout: float,
) -> dict:
    """Check readiness, warm the service, then measure concurrent prediction requests."""
    base_url = base_url.rstrip("/")
    ready = _json_request(f"{base_url}/ready", timeout=timeout)
    if not ready.get("ready"):
        raise RuntimeError("service reported that it was not ready")
    model = _json_request(f"{base_url}/model", timeout=timeout)

    for _ in range(warmup):
        _, error = _timed_prediction(base_url, timeout)
        if error:
            raise RuntimeError(f"warmup request failed: {error}")

    latencies: list[float] = []
    errors: list[str] = []
    started = time.perf_counter()
    with ThreadPoolExecutor(max_workers=concurrency) as pool:
        futures = [pool.submit(_timed_prediction, base_url, timeout) for _ in range(requests)]
        for future in as_completed(futures):
            latency, error = future.result()
            if latency is not None:
                latencies.append(latency)
            if error is not None:
                errors.append(error)
    elapsed = time.perf_counter() - started

    return summarize(
        latencies_ms=latencies,
        errors=errors,
        elapsed_seconds=elapsed,
        requests=requests,
        concurrency=concurrency,
        model=model,
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-url", default="http://127.0.0.1:8000")
    parser.add_argument("--requests", type=int, default=500)
    parser.add_argument("--concurrency", type=int, default=16)
    parser.add_argument("--warmup", type=int, default=20)
    parser.add_argument("--timeout", type=float, default=10.0)
    parser.add_argument("--json-output", type=Path)
    args = parser.parse_args(argv)

    if args.requests < 1 or args.concurrency < 1 or args.warmup < 0:
        parser.error("requests/concurrency must be positive and warmup must be non-negative")

    report = run_benchmark(
        base_url=args.base_url,
        requests=args.requests,
        concurrency=args.concurrency,
        warmup=args.warmup,
        timeout=args.timeout,
    )
    rendered = json.dumps(report, indent=2)
    print(rendered)
    if args.json_output:
        args.json_output.parent.mkdir(parents=True, exist_ok=True)
        args.json_output.write_text(rendered + "\n", encoding="utf-8")
    return 0 if report["failed"] == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
