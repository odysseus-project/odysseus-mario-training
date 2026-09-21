"""Super Mario Land training environment, imported only when selected."""

from importlib import import_module

__all__ = ["PyBoySuperMarioLandEnv", "BaseEnv"]

_MODULES = {
    "PyBoySuperMarioLandEnv": ".super_mario_land",
    "BaseEnv": ".base_env",
}


def __getattr__(name):
    if name not in _MODULES:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    value = getattr(import_module(_MODULES[name], __name__), name)
    globals()[name] = value
    return value
