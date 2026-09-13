"""Probe whether basedpyright understands basedtyping.Intersection.

To run:
  uv run basedpyright docs/basedtyping_intersection_probe.py

tl;dr: It does not. basedtyping.Intersection is a runtime component for
basedmypy, not basedpyright. basedpyright treats it as Unknown.

Tested with:
  basedpyright 1.37.3 (based on pyright 1.1.408)
"""

from typing import reveal_type

from basedtyping import Intersection


class A:
    """Example class A with attribute x."""

    x: int = 1


class B:
    """Example class B with attribute y."""

    y: str = "hi"


# Every diagnostic below IS the finding this probe records; each is suppressed
# with its reason rather than excluding the file from the checkers.
def foo(
    val: Intersection[A, B],  # ty: ignore[invalid-type-form] -- Probe subject. # pyright: ignore[reportGeneralTypeIssues, reportUnknownParameterType] -- Probe subject.
) -> None:
    """Reveal what basedpyright infers for an intersection and its members.

    Args:
      val: A value typed as both ``A`` and ``B``.

    """
    reveal_type(val)  # Unknown.
    reveal_type(val.x)  # pyright: ignore[reportUnknownMemberType] -- Unknown is the finding.
    reveal_type(val.y)  # pyright: ignore[reportUnknownMemberType] -- Unknown is the finding.
