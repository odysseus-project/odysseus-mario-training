"""Super Mario Land environment with a lazy public export."""

from importlib import import_module

__all__ = ["PyBoySuperMarioLandEnv"]

_MODULES = {
    "PyBoySuperMarioLandEnv": ".pyboy_super_mario_land_env",
}


def __getattr__(name):
    if name not in _MODULES:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    value = getattr(import_module(_MODULES[name], __name__), name)
    globals()[name] = value
    return value
