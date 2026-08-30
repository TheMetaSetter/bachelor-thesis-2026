from __future__ import annotations

"""Small process-RAM and CUDA-memory measurements for offline runs."""

import platform
from typing import Any

import psutil
import torch


def _host_os_name() -> str:
    system_name = platform.system()
    if system_name == "Darwin":
        return "macOS"
    if system_name == "Linux":
        return "Linux"
    return system_name


class RuntimeResourceMonitor:
    """Sample process RSS and PyTorch CUDA allocator memory."""

    def __init__(self, device: str) -> None:
        self.device = str(device)
        self._cuda_device = torch.device(self.device)
        self._cuda_enabled = (
            self._cuda_device.type == "cuda" and torch.cuda.is_available()
        )
        self._ram_rss_peak_sampled_bytes = 0

    def reset(self) -> None:
        self._ram_rss_peak_sampled_bytes = 0
        if self._cuda_enabled:
            torch.cuda.reset_peak_memory_stats(self._cuda_device)

    def snapshot(self) -> dict[str, Any]:
        rss_bytes = int(psutil.Process().memory_info().rss)
        self._ram_rss_peak_sampled_bytes = max(
            self._ram_rss_peak_sampled_bytes,
            rss_bytes,
        )
        metrics: dict[str, Any] = {
            "runtime_ram_rss_bytes": rss_bytes,
            "runtime_ram_rss_peak_sampled_bytes": self._ram_rss_peak_sampled_bytes,
            "runtime_ram_unit": "bytes",
            "runtime_ram_source": "psutil_process_rss",
            "runtime_ram_source_os": _host_os_name(),
            "runtime_gpu_memory_available": False,
            "runtime_gpu_memory_unit": "bytes",
            "runtime_gpu_memory_source": "torch_cuda_allocator",
        }
        if not self._cuda_enabled:
            return metrics

        torch.cuda.synchronize(self._cuda_device)
        metrics.update(
            {
                "runtime_gpu_memory_available": True,
                "runtime_gpu_memory_allocated_bytes": int(
                    torch.cuda.memory_allocated(self._cuda_device)
                ),
                "runtime_gpu_memory_reserved_bytes": int(
                    torch.cuda.memory_reserved(self._cuda_device)
                ),
                "runtime_gpu_peak_memory_allocated_bytes": int(
                    torch.cuda.max_memory_allocated(self._cuda_device)
                ),
                "runtime_gpu_peak_memory_reserved_bytes": int(
                    torch.cuda.max_memory_reserved(self._cuda_device)
                ),
            }
        )
        return metrics
