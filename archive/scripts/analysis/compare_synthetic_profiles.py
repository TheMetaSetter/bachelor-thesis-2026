import sys as _sys

from scripts.analysis import compare_synthetic_profiles as _implementation_module

_sys.modules[__name__] = _implementation_module


if __name__ == "__main__":
    raise SystemExit(_implementation_module.main())
