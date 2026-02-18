"""Custom types for config module."""

# ty type system feature overview: https://github.com/astral-sh/ty/issues/1889

from __future__ import annotations

from typing import (
    ClassVar,
    Protocol,
    Self,
    runtime_checkable,
)
from typing_extensions import TypeVar

import dataclasses


__all__ = [
    "Configurable",
    "DataclassLike",
    "Finalizeable",
    "HasConfig",
    "HasRelaxedConfig",
    "Makeable",
    "RelaxedConfigurable",
    "RelaxedMakeable",
]


@runtime_checkable
class Finalizeable(Protocol):
    """Non-generic protocol for isinstance checks in _finalize_value.

    Using the generic Makeable protocol in isinstance causes basedpyright to
    leak Unknown into the negative branch. This simple non-generic protocol
    avoids that issue.
    """

    def finalize(self) -> Self: ...


_T_co = TypeVar("_T_co", covariant=True, default=object)
_T = TypeVar("_T")


# class _DataclassParamsProtocol(Protocol):
#     """Protocol for dataclasses._DataclassParams (private, Python 3.10+)."""
#
#     init: bool
#     repr: bool
#     eq: bool
#     order: bool
#     unsafe_hash: bool
#     frozen: bool
#     match_args: bool  # Python 3.10+
#     kw_only: bool  # Python 3.10+
#     slots: bool  # Python 3.10+
#     weakref_slot: bool  # Python 3.11+


@runtime_checkable
class DataclassLike(Protocol):
    """Protocol for objects that behave like dataclasses."""

    __dataclass_fields__: ClassVar[dict[str, dataclasses.Field[object]]]
    # __dataclass_params__: ClassVar[_DataclassParamsProtocol]  # Python 3.10+
    # __match_args__: ClassVar[tuple[str, ...]]  # When match_args=True (default)


@runtime_checkable
class Makeable(Protocol[_T_co]):
    """Protocol for objects with make(), finalize(), and update() methods."""

    _finalized: bool

    # Ideally this would be a read-only class attribute -- covariant and
    # accessible on both class and instance. Python's type system has no such
    # concept, so we pick @property (covariant, instance-only) over the
    # alternatives:
    #   - ClassVar: class-accessible but invariant (breaks Makeable[Derived]
    #     assignable to Makeable[Base]).
    #   - Custom descriptor with __get__: works on concrete classes but pyright
    #     doesn't resolve __get__ during protocol structural matching.
    #   - Final: can't combine with ClassVar or use in protocols.
    # The tradeoff is that type[Makeable[X]].parent_class doesn't work
    # through protocol-typed variables. Use a cast for that rare case.
    @property
    def parent_class(self) -> type[_T_co] | None: ...

    def make(self) -> _T_co: ...
    def finalize(self) -> Self: ...
    def update(
        self,
        source: DataclassLike | Makeable[object] | None = None,
        *,
        skip_missing: bool = False,
        **kwargs: object,
    ) -> Self: ...


Configurable = Makeable


@runtime_checkable
class HasConfig(Protocol[_T]):
    """Protocol for classes with a typed Config nested class."""

    # Spec-illegal (PEP 526: no TypeVars in ClassVar) but semantically correct.
    Config: ClassVar[type[Makeable[_T]]]  # pyright: ignore[reportGeneralTypeIssues]


@runtime_checkable
class RelaxedMakeable(Makeable[_T_co], Protocol):  # pyright: ignore[reportInvalidTypeVarUse]
    """Makeable with dynamic field access.

    Extends Makeable with __init__ and __getattr__ to support
    dynamic field access without requiring suppressions in user code.
    """

    # Semantically correct but spec-illegal: PEP 526 forbids type variables
    # inside ClassVar. We need a class-level attribute whose type varies per
    # parameterization — a concept the type system can't express. Alternatives:
    #   - Drop ClassVar: loses the "class attribute" semantic in the Protocol.
    #   - @property: covariant but instance-only (no Cls.parent_class access).
    # Suppressed in both checkers as a deliberate design choice.
    parent_class: ClassVar[type[_T_co] | None]  # pyright: ignore[reportGeneralTypeIssues,reportIncompatibleMethodOverride]  # ty: ignore[invalid-type-form]

    def __init__(self, *args: object, **kwargs: object) -> None: ...
    def __getattr__(self, name: str) -> object: ...


RelaxedConfigurable = RelaxedMakeable


@runtime_checkable
class HasRelaxedConfig(Protocol[_T]):
    """Protocol for classes decorated with @autofig."""

    # Spec-illegal (PEP 526: no TypeVars in ClassVar) but semantically correct.
    Config: ClassVar[type[RelaxedMakeable[_T]]]  # pyright: ignore[reportGeneralTypeIssues]
