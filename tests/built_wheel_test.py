"""The configgle wheel, built and imported outside the checkout."""

from __future__ import annotations

import pytest

from configgle.lib.testing.wheel import built_wheel


pytestmark = pytest.mark.cli_uv


def test_the_wheel_imports_outside_the_checkout() -> None:
    wheel = built_wheel("configgle", anchor=__file__)

    assert "site-packages" in wheel.run("import configgle; print(configgle.__file__)")


def test_the_wheel_ships_its_typing_marker_and_ty_extensions() -> None:
    """``ty_extensions`` ships so that a checker without ty resolves the import."""
    names = built_wheel("configgle", anchor=__file__).names()

    assert {
        "configgle/__init__.py",
        "configgle/py.typed",
        "ty_extensions/__init__.py",
    } <= names


if __name__ == "__main__":
    from configgle.lib.testing.runner import test_main

    test_main(__file__)
