"""Minimal cloudpickle stubs used by Configgle tests."""

from collections.abc import Callable
from typing import Any


def dumps(
    obj: object,
    protocol: int | None = ...,
    buffer_callback: Callable[[Any], object] | None = ...,
) -> bytes: ...
