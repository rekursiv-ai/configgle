"""Test whether basedpyright understands basedtyping.Intersection.

To run:
  uv run basedpyright docs/test_basedtyping_intersection.py

tl;dr: It does not. basedtyping.Intersection is a runtime component for
basedmypy, not basedpyright. basedpyright treats it as Unknown.

Tested with:
  basedpyright 1.37.3 (based on pyright 1.1.408)
"""

from typing import reveal_type

from basedtyping import Intersection


class A:
    x: int = 1


class B:
    y: str = "hi"


def foo(
    val: Intersection[A, B],  # pyright: ignore[reportGeneralTypeIssues,reportUnknownParameterType]
) -> None:
    reveal_type(val)  # Unknown
    reveal_type(val.x)  # pyright: ignore[reportUnknownMemberType]  # Unknown
    reveal_type(val.y)  # pyright: ignore[reportUnknownMemberType]  # Unknown
