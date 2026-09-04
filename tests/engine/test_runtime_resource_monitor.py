from __future__ import annotations

from src.engine.runtime_resource_monitor import RuntimeResourceMonitor


def test_cpu_snapshot_reports_rss_in_bytes_and_host_os() -> None:
    metrics = RuntimeResourceMonitor(device="cpu").snapshot()

    assert metrics["runtime_ram_rss_bytes"] > 0
    assert (
        metrics["runtime_ram_rss_peak_sampled_bytes"]
        >= metrics["runtime_ram_rss_bytes"]
    )
    assert metrics["runtime_ram_unit"] == "bytes"
    assert metrics["runtime_ram_source_os"] in {"macOS", "Linux"}


def test_cpu_snapshot_marks_gpu_measurement_unavailable() -> None:
    metrics = RuntimeResourceMonitor(device="cpu").snapshot()

    assert metrics["runtime_gpu_memory_available"] is False
    assert metrics["runtime_gpu_memory_unit"] == "bytes"
