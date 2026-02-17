"""Demo to show things we tried to preserve types.

To run:
  uv run basedpyright demo.py

tl;dr: Unless python has an intersection type you have to choose between:
    1. Status quo:
       - User specifically states the parent class type (current configgle).
       - Failing this, make returns Any (again, current configgle).
    2. Fantasy land:
       - Python supports an intersection operator.
"""

from __future__ import annotations

from typing import (
    Annotated,
    Any,
    Generic,
    Protocol,
    overload,
    reveal_type,
)
from typing_extensions import TypeIs, TypeVar, override


_T = TypeVar("_T")
_ParentT = TypeVar("_ParentT", default=Any)


class Maker(Generic[_ParentT]):
    def make(self) -> _ParentT:
        raise NotImplementedError


class MakerMeta1(type):
    # The following is the same as not having it at all but is here so you can
    # see the symmetry with other approaches.
    def __get__(
        cls: type[_T],
        obj: object,
        owner: type[_ParentT],
    ) -> type[_T]:
        return cls


class MakerMeta2(type):
    def __get__(
        cls: type[_T],
        obj: object,
        owner: type[_ParentT],
    ) -> type[Maker[_ParentT]]:
        return cls  # pyright: ignore[reportReturnType]  # ty: ignore[invalid-return-type]


class MakerMeta3(type):
    def __get__(
        cls: type[_T],
        obj: object,
        owner: type[_ParentT],
    ) -> type[_T | Maker[_ParentT]]:
        return cls


# Symmetrically speaking the only one missing is
# `type[_T & Maker[_ParentT]]` where `&` is an intersection (merge)
# operator and in fact that is exactly what we need.
# ...Too bad it doesn't exist.

# Note: Fig1 is the current choice of configgle.


class Fig1(Maker[_ParentT], metaclass=MakerMeta1):
    pass


class Fig2(Maker[_ParentT], metaclass=MakerMeta2):
    pass


class Fig3(Maker[_ParentT], metaclass=MakerMeta3):
    pass


# -------------------------------------------------------------------------------
# Fully works! ...But requires effort from user.
#
# Type of "Foo.Config" is "type[Config]"
# Type of "Foo.Config().x" is "int"
# Type of "Foo.Config().make()" is "Foo"


class Foo:
    class Config(Fig1["Foo"]):
        x: int = 0


reveal_type(Foo.Config)
reveal_type(Foo.Config().x)
reveal_type(Foo.Config().make())


# -------------------------------------------------------------------------------
# No errors or warnings! ...But loses type from make.
#
# Type of "Foo1.Config" is "type[Config]"
# Type of "Foo1.Config().x" is "int"
# Type of "Foo1.Config().make()" is "Any"


class Foo1:
    class Config(Fig1):
        x: int = 0


reveal_type(Foo1.Config)
reveal_type(Foo1.Config().x)
reveal_type(Foo1.Config().make())


# -------------------------------------------------------------------------------
# Loses Config type.
#
# Type of "Foo2.Config" is "type[Maker[Foo2]]"
# Type of "Foo2.Config().x" is "Unknown"
# Type of "Foo2.Config().make()" is "Foo2"


class Foo2:
    class Config(Fig2):
        x: int = 0


reveal_type(Foo2.Config)
reveal_type(Foo2.Config().x)  # pyright: ignore[reportAttributeAccessIssue,reportUnknownMemberType]  # ty: ignore[unresolved-attribute]
reveal_type(Foo2.Config().make())


# -------------------------------------------------------------------------------
# Loses Config type and make type.
#
# Type of "Foo3.Config" is "type[Config] | type[Maker[Foo3]]"
# Type of "Foo3.Config().x" is "int | Unknown"
# Type of "Foo3.Config().make()" is "Foo3 | Any"


class Foo3:
    class Config(Fig3):
        x: int = 0


reveal_type(Foo3.Config)
reveal_type(Foo3.Config().x)  # pyright: ignore[reportAttributeAccessIssue,reportUnknownMemberType]  # ty: ignore[possibly-missing-attribute]
reveal_type(Foo3.Config().make())


# -------------------------------------------------------------------------------
# IDEA 4: Protocol with __getattr__ to fake merge
#
# Result: make() works, but x becomes Any via __getattr__
#
# (This is basically the RelaxedConfig idea.)
#
# Type of "Foo4.Config" is "type[ConfigProtocol4[Config, Foo4]]"
# Type of "Foo4.Config().x" is "Any"
# Type of "Foo4.Config().make()" is "Foo4"


class ConfigProtocol4(  # pyright: ignore[reportInvalidTypeVarUse]
    Protocol[_T, _ParentT],
):
    """Protocol that has both Config fields (via __getattr__) and make()."""

    def make(self) -> _ParentT: ...
    def __getattr__(self, name: str) -> Any: ...


class MakerMeta4(type):
    def __get__(
        cls: type[_T],
        obj: object,
        owner: type[_ParentT],
    ) -> type[ConfigProtocol4[_T, _ParentT]]:
        return cls  # pyright: ignore[reportReturnType]


class Fig4(Maker[_ParentT], metaclass=MakerMeta4):
    pass


class Foo4:
    class Config(Fig4):
        x: int = 0


reveal_type(Foo4.Config)
reveal_type(Foo4.Config().x)
reveal_type(Foo4.Config().make())


# -------------------------------------------------------------------------------
# IDEA 5: @overload on __get__
#
# Result: Same as Idea 1 - overload doesn't help
#
# Type of "Foo5.Config" is "type[Config]"
# Type of "Foo5.Config().x" is "int"
# Type of "Foo5.Config().make()" is "Any"


class MakerMeta5(type):
    @overload
    def __get__(
        cls: type[_T],
        obj: None,
        owner: type[_ParentT],
    ) -> type[_T]: ...
    @overload
    def __get__(
        cls: type[_T],
        obj: object,
        owner: type[_ParentT],
    ) -> type[Maker[_ParentT]]: ...
    def __get__(
        cls: type[_T],
        obj: object | None,
        owner: type[_ParentT],
    ) -> type[_T | Maker[_ParentT]]:
        return cls


class Fig5(Maker[_ParentT], metaclass=MakerMeta5):
    pass


class Foo5:
    class Config(Fig5):
        x: int = 0


reveal_type(Foo5.Config)
reveal_type(Foo5.Config().x)
reveal_type(Foo5.Config().make())


# -------------------------------------------------------------------------------
# IDEA 6: Annotated to carry extra type info
#
# Result: Annotated is ignored for type inference - same as Idea 1
#
# Type of "Foo6.Config" is "type[Config]"
# Type of "Foo6.Config().x" is "int"
# Type of "Foo6.Config().make()" is "Any"


class ParentTypeMarker(Generic[_ParentT]):
    """Marker to carry parent type info in Annotated."""


class MakerMeta6(type):
    def __get__(
        cls: type[_T],
        obj: object,
        owner: type[_ParentT],
    ) -> Annotated[type[_T], ParentTypeMarker[_ParentT]]:
        return cls


class Fig6(Maker[_ParentT], metaclass=MakerMeta6):
    pass


class Foo6:
    class Config(Fig6):
        x: int = 0


reveal_type(Foo6.Config)
reveal_type(Foo6.Config().x)
reveal_type(Foo6.Config().make())


# -------------------------------------------------------------------------------
# IDEA 7: Manual override of make() in each Config
#
# Result: WORKS! But requires even _more_ manual effort from user than option 1.
#
# Type of "Foo7.Config" is "type[Config]"
# Type of "Foo7.Config().x" is "int"
# Type of "Foo7.Config().make()" is "Foo7"


class MakerProtocol(Protocol[_ParentT]):  # pyright: ignore[reportInvalidTypeVarUse]
    def make(self) -> _ParentT: ...


class MakerMeta7(type):
    def __get__(
        cls: type[_T],
        obj: object,
        owner: type[_ParentT],
    ) -> type[_T]:
        return cls


class Fig7Base:
    def make(self) -> Any:
        raise NotImplementedError


class Fig7(Fig7Base, metaclass=MakerMeta7):
    pass


class Foo7:
    class Config(Fig7):
        x: int = 0

        @override
        def make(self) -> Foo7:  # Manual override
            raise NotImplementedError


reveal_type(Foo7.Config)
reveal_type(Foo7.Config().x)
reveal_type(Foo7.Config().make())


# -------------------------------------------------------------------------------
# IDEA 8: TypeIs for narrowing
#
# Result: TypeIs doesn't help - can't narrow class types this way
#
# Type of "Foo8.Config" is "type[Config]"
# Type of "Foo8.Config().x" is "int"
# Type of "Foo8.Config().make()" is "Any"


def is_creatable(
    cls: type[_T],
    parent: type[_ParentT],
) -> TypeIs[type[Maker[_ParentT]]]:  # pyright: ignore[reportGeneralTypeIssues]  # ty: ignore[invalid-type-guard-definition]
    del cls, parent
    return True


class Foo8:
    class Config(Fig1):
        x: int = 0


reveal_type(Foo8.Config)
reveal_type(Foo8.Config().x)
reveal_type(Foo8.Config().make())

# With TypeIs narrowing:
if is_creatable(Foo8.Config, Foo8):
    reveal_type(Foo8.Config)
    reveal_type(Foo8.Config().make())


# -------------------------------------------------------------------------------
# IDEA 9: Decorator on parent class
#
# Result: make() works! But x becomes Any (same as Idea 4)
#
# Type of "Foo9" is "type[Configurable9[Foo9]]"
# Type of "Foo9.Config" is "type[ConfigFor9[Foo9]]"
# Type of "Foo9.Config().x" is "Any"
# Type of "Foo9.Config().make()" is "Foo9"


class ConfigFor9(Generic[_T]):
    """Tells type checker make() returns _T."""

    def make(self) -> _T: ...  # ty: ignore[empty-body]
    def __getattr__(self, name: str) -> Any: ...


class Configurable9(Generic[_T]):
    """Base class with typed Config."""

    Config: type[ConfigFor9[_T]]  # pyright: ignore[reportUninitializedInstanceVariable]


def has_config(cls: type[_T]) -> type[Configurable9[_T]]:
    return cls  # pyright: ignore[reportReturnType]  # ty: ignore[invalid-return-type]


@has_config
class Foo9:
    class Config(Fig1):
        x: int = 0


reveal_type(Foo9)
reveal_type(Foo9.Config)  # pyright: ignore[reportGeneralTypeIssues]
reveal_type(Foo9.Config().x)  # pyright: ignore[reportGeneralTypeIssues]
reveal_type(Foo9.Config().make())  # pyright: ignore[reportGeneralTypeIssues]


# -------------------------------------------------------------------------------
# IDEA 10: TypeIs intersection trick
#
# PEP 742 specifies that TypeIs narrows to the intersection of the declared
# type and the TypeIs target. By using two successive TypeIs assertions, we
# can build an intersection type that has both Config fields and a typed
# make() return.
#
# Two requirements for this to work:
#   1. _ParentT must NOT default to Any (use object instead), because Any is
#      bidirectionally compatible and makes the type checker consider Config as
#      already satisfying Maker10[Foo10] — turning the narrow into a no-op.
#   2. Maker10[Foo10] must be asserted FIRST — pyright resolves method
#      conflicts using left-to-right precedence in intersections.
#
# See: https://discuss.python.org/t/best-way-to-emulate-type-hint-intersection-of-generics/104511/9
#
# Result: WORKS!
#
# Type of "o" is "<subclass of Maker10[Foo10] and Config>"
# Type of "o.x" is "int"
# Type of "o.make()" is "Foo10"


def supports_t(o: object, tp: type[_T]) -> TypeIs[_T]:
    """Unconditionally returns True; used purely for TypeIs narrowing."""
    del o, tp
    return True


_ParentT10 = TypeVar("_ParentT10", default=object)


class Maker10(Generic[_ParentT10]):
    def make(self) -> _ParentT10:
        raise NotImplementedError


class Fig10(Maker10[_ParentT10]):
    pass


class Foo10:
    class Config(Fig10):
        x: int = 0


# Baseline (without trick): make() returns object (not Any)
reveal_type(Foo10.Config)
reveal_type(Foo10.Config().x)
reveal_type(Foo10.Config().make())


# With TypeIs intersection trick (Maker10 FIRST for method precedence):
def intersect10(o: object) -> object:
    assert supports_t(o, Maker10[Foo10])  # noqa: S101
    assert supports_t(o, Foo10.Config)  # noqa: S101
    reveal_type(o)  # "<subclass of Maker10[Foo10] and Config>"
    reveal_type(o.x)  # "int"
    reveal_type(o.make())  # "Foo10"
    return o
