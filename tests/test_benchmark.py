from service.benchmark import _percentile, summarize


def test_percentile_interpolates_sorted_values():
    assert _percentile([40.0, 10.0, 30.0, 20.0], 0.50) == 25.0
    assert _percentile([40.0, 10.0, 30.0, 20.0], 0.95) == 38.5


def test_summary_reports_scope_failures_and_model_lineage():
    report = summarize(
        latencies_ms=[10.0, 20.0, 30.0],
        errors=["timeout"],
        elapsed_seconds=1.0,
        requests=4,
        concurrency=2,
        model={
            "model_id": "model-123",
            "version": 7,
            "git_sha": "abc123",
            "artifact_sha256": "deadbeef",
            "ignored": "not part of the evidence contract",
        },
    )

    assert report["succeeded"] == 3
    assert report["failed"] == 1
    assert report["throughput_requests_per_second"] == 3.0
    assert report["latency"]["p50_ms"] == 20.0
    assert "single-host" in report["scope"]
    assert "distributed-scale" in report["scope"]
    assert report["model"] == {
        "model_id": "model-123",
        "version": 7,
        "git_sha": "abc123",
        "artifact_sha256": "deadbeef",
    }
