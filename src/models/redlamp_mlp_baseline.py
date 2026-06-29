from __future__ import annotations

"""Compatibility shim for the renamed RedLamp baseline module.

Prefer importing `RedLampBaseline` from `src.models.redlamp_baseline`.
"""

from src.models.redlamp_baseline import RedLampBaseline, SimpleWindowCnnEncoder


RedLampMLPBaseline = RedLampBaseline

__all__ = ["RedLampBaseline", "RedLampMLPBaseline", "SimpleWindowCnnEncoder"]
