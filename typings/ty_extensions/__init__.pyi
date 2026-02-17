# Minimal stub so basedpyright can resolve ty_extensions.Intersection.
# ty has this module built-in; this stub bridges the gap for other type checkers.
# Intersection[A, B] approximates as A since basedpyright lacks native intersection
# support. See: https://github.com/astral-sh/ruff/tree/main/crates/ty_vendored/ty_extensions
from typing import TypeVar
from typing_extensions import TypeAliasType

_First = TypeVar("_First")
_Second = TypeVar("_Second")
Intersection = TypeAliasType("Intersection", _First, type_params=(_First, _Second))
