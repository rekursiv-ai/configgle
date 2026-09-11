"""Test whether basedpyright understands basedtyping.Intersection.

To run:
  uv run basedpyright docs/test_basedtyping_intersection.py

tl;dr: It does not. basedtyping.Intersection is a runtime component for
basedmypy, not basedpyright. basedpyright treats it as Unknown.

Tested with:
  basedpyright 1.37.3 (based on pyright 1.1.408)
"""

from typing import reveal_type

import sys


if "pytest" in sys.modules:
    import pytest

    pytest.skip(
        "Static basedpyright probe, not a runtime pytest test.", allow_module_level=True
    )

from basedtyping import Intersection


class A:
    x: int = 1


class B:
    y: str = "hi"


def foo(val: Intersection[A, B]) -> None:
    """Reveal what basedpyright infers for an intersection and its members.

    Args:
      val: A value typed as both ``A`` and ``B``.

    """
    reveal_type(val)  # Unknown.
    reveal_type(val.x)  # Unknown.
    reveal_type(val.y)  # Unknown.


if __name__ == "__main__":
    from configgle.lib.testing.main import test_main

    test_main(__file__)
