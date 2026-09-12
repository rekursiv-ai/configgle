"""Check ``Config.make()`` assignability and runtime return types.

The entire purpose of configgle is that ``SomeClass.Config(...).make()`` is
statically known to return ``SomeClass`` -- INCLUDING a *bare*
``class Config(Fig)`` with no ``Fig["Parent"]`` parameter. The annotated
assignments check assignability during type checking; runtime assertions check
the constructed class. Assignability alone does not reject ``Any`` or prove
that the checker inferred the exact parent type.

The load-bearing machinery lives in ``MakerMeta.__get__`` returning
``Intersection[_T, type[Maker[_ParentT]]]`` with a covariant ``_ParentT``: the
descriptor ``owner`` binds to the enclosing class, and the intersection injects
``Maker[owner]`` so ``make()`` narrows to the parent even for a bare ``Fig``.
An incompatible inferred return type makes the annotated assignments fail.
"""

from __future__ import annotations

from importlib import import_module
from typing import TypeAliasType

from configgle.fig import Fig, Makes


class Bare:
    """Bare ``Fig`` base -- no explicit parent parameter (the hard case)."""

    class Config(Fig):
        x: int = 0

    def __init__(self, config: Config) -> None:
        del config


class Explicit:
    """Explicit ``Fig["Explicit"]`` parameterization."""

    class Config(Fig["Explicit"]):
        x: int = 0

    def __init__(self, config: Config) -> None:
        del config


class Animal:
    class Config(Fig["Animal"]):
        name: str = "animal"

    def __init__(self, config: Config) -> None:
        del config


class Dog(Animal):
    """Inherited Config re-narrowed with ``Makes["Dog"]``."""

    class Config(Makes["Dog"], Animal.Config):
        breed: str = "mutt"


def test_intersection_polyfill_preserves_the_first_type() -> None:
    """Keep the runtime export used by checkers without intersection support."""
    polyfill = vars(import_module("ty_extensions"))["Intersection"]
    assert isinstance(polyfill, TypeAliasType)
    assert len(polyfill.__type_params__) == 2
    assert polyfill.__value__ is polyfill.__type_params__[0]


def test_bare_fig_make_returns_parent() -> None:
    bare: Bare = Bare.Config().make()
    assert isinstance(bare, Bare)


def test_bare_fig_fields_resolve() -> None:
    x: int = Bare.Config().x
    assert x == 0


def test_explicit_fig_make_returns_parent() -> None:
    explicit: Explicit = Explicit.Config().make()
    assert isinstance(explicit, Explicit)


def test_makes_reparameterizes_to_child() -> None:
    dog: Dog = Dog.Config().make()
    assert isinstance(dog, Dog)


if __name__ == "__main__":
    from configgle.lib.testing.main import test_main

    test_main(__file__)
