import sys as _sys

from src.models.online_impl import online_adaptation as _implementation_module

_sys.modules[__name__] = _implementation_module
