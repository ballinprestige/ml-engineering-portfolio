# Engineering evidence

This page maps the repository's important engineering claims to inspectable code and checks.
It is an evidence index, not a claim that this demonstration has production users or distributed
scale.

## Claim-to-proof ledger

| Claim | Implementation | Enforcement or reproduction |
|---|---|---|
| The API refuses to start without a valid model artifact. | [`src/service/app.py`](src/service/app.py) loads the latest registered model during application lifespan. | [`tests/test_service.py`](tests/test_service.py) covers startup, readiness, input validation, and predictions; the container job also hits live endpoints. |
| A corrupted artifact is detected before serving. | [`src/service/registry.py`](src/service/registry.py) stores and verifies a SHA-256 checksum. | [`tests/test_registry.py`](tests/test_registry.py) changes artifact bytes and asserts that loading fails. |
| Interrupted publication cannot become the latest release. | The registry writes to a temporary directory, atomically renames it, then publishes `latest.json` last. | [`tests/test_registry.py`](tests/test_registry.py) simulates an interrupted release and verifies that the prior artifact stays visible. |
| Requests have a strict data contract. | [`src/service/schemas.py`](src/service/schemas.py) bounds fields and rejects extras with Pydantic. | [`tests/test_service.py`](tests/test_service.py) verifies that invalid and extra input receives HTTP 422. |
| The container has a narrow runtime posture. | [`Dockerfile`](Dockerfile) uses a digest-pinned base, hash-locked serving dependencies, a non-root UID, and a read-only baked model artifact. | [`.github/workflows/ci.yml`](.github/workflows/ci.yml) builds and runs the image, checks its user, validates commit lineage, and confirms the artifact is not writable. |
| Headline ML results are measured, not hand-entered. | The demos emit structured result objects for temporal validation, calibration, feature selection, drift, and a real-data pipeline. | [`tests/test_demos.py`](tests/test_demos.py) asserts the documented findings; CI reruns the demos on Python 3.11 and 3.12. |
| HTTP performance can be measured reproducibly. | [`src/service/benchmark.py`](src/service/benchmark.py) checks readiness and model identity, then records success count, latency percentiles, throughput, runtime, and failures. | Start the service and run the command below. [`tests/test_benchmark.py`](tests/test_benchmark.py) locks the report's evidence contract. |

## Reproduce the HTTP benchmark

Train and start the service in one terminal:

```bash
python -m service.train
uvicorn service.app:app --host 127.0.0.1 --port 8000
```

Run the benchmark in another terminal:

```bash
python -m service.benchmark \
  --base-url http://127.0.0.1:8000 \
  --requests 500 \
  --concurrency 16 \
  --json-output benchmarks/local.json
```

The report identifies the model artifact and runtime alongside p50, p95, and p99 latency,
throughput, and failures. It deliberately labels its scope as a **single-host HTTP benchmark**.
Results depend on hardware and should be compared only when the environment is recorded.

A checked-in [Windows reference run](benchmarks/windows-local-20260814.json) completed 500 of 500
requests with no failures at concurrency 16. It records the exact model checksum, Python version,
operating system, latency percentiles, and throughput so the result can be challenged or rerun.

## Deliberate limits

This repository does **not** prove:

- production users, revenue, uptime, or business impact;
- cloud deployment, Kubernetes, infrastructure as code, or distributed queues;
- a distributed or multi-writer artifact registry;
- team code review, on-call ownership, or cross-functional delivery.

Those are important hiring signals, but presenting them without evidence would make the rest of
the portfolio less credible.
