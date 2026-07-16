import sys as _sys

from scripts.experiments import (
    run_comparative_smd_experiments as _implementation_module,
)

_sys.modules[__name__] = _implementation_module


if __name__ == "__main__":
    raise SystemExit(_implementation_module.main())
