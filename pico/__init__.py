from importlib import import_module
from typing import Any

__all__ = ["PICO"]


def __getattr__(name: str) -> Any:
    if name == "PICO":
        return import_module(".pico", __name__).PICO
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
