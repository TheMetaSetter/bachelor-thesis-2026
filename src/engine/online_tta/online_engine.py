from __future__ import annotations

"""Public THESIS online TTA entrypoint.

The implementation is split across smaller helper modules so every source file
stays within the repository line limit.
"""

from src.engine.online_tta.online_engine_run import (  # noqa: F401
    _build_dry_run_online_context,
    _build_online_execution_context,
    _build_runtime_online_context,
    _finalize_online_execution,
    _process_online_window,
    _run_online_execution_sequences,
    _run_online_sequence,
    run_thesis_online_tta_experiment,
)
from src.engine.online_tta.online_calibration import (
    build_online_stream as _build_online_stream,
)  # noqa: F401
from src.engine.online_tta.online_engine_shared import *  # noqa: F401,F403
from src.engine.online_tta.online_engine_step import execute_online_tta_step
from src.engine.online_tta.online_engine_window_core import *  # noqa: F401,F403
from src.engine.online_tta.online_engine_window_metrics import *  # noqa: F401,F403
