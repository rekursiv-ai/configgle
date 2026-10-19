"""Tests for ``configgle.custom_types``."""

from __future__ import annotations

import pytest

from configgle.custom_types import LateBound


def test_a_late_bound_object_must_implement_bind() -> None:
    with pytest.raises(NotImplementedError):
        LateBound().bind(object())


if __name__ == "__main__":
    from configgle.lib.testing.runner import test_main

    test_main(__file__)
