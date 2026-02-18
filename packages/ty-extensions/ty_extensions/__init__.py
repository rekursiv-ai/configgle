# Polyfill for ty_extensions.Intersection.
# ty has this module built-in (vendored into typeshed/stdlib); this polyfill
# provides the same API for other type checkers and at runtime.
# Intersection[A, B] approximates as A since basedpyright lacks native
# intersection support.
# See: https://github.com/astral-sh/ruff/tree/main/crates/ty_vendored/ty_extensions
from typing import TypeVar


_First = TypeVar("_First")
_Second = TypeVar("_Second")
type Intersection[_First, _Second] = _First
