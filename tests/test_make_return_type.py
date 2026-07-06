"""Static type-inference regression tests for ``Config.make()`` return types.

The entire purpose of configgle is that ``SomeClass.Config(...).make()`` is
statically known to return ``SomeClass`` -- INCLUDING a *bare*
``class Config(Fig)`` with no ``Fig["Parent"]`` parameter. ``ty`` enforces this
file on every ``ty check`` run; the assignments below are compile-time
assertions (an assignment to a parent-typed variable fails to type-check unless
``make()`` is known to return the parent) that also execute as harmless no-ops
under pytest.

The load-bearing machinery lives in ``MakerMeta.__get__`` returning
``Intersection[_T, type[Maker[_ParentT]]]`` with a covariant ``_ParentT``: the
descriptor ``owner`` binds to the enclosing class, and the intersection injects
``Maker[owner]`` so ``make()`` narrows to the parent even for a bare ``Fig``.
If that regresses, the annotated assignments below become ``ty`` errors.
"""

from __future__ import annotations

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


def test_bare_fig_make_returns_parent() -> None:
    # The decree: a bare ``class Config(Fig)`` must have ``make() -> Bare``.
    # The annotated binding is the assertion -- it fails to type-check if
    # ``make()`` returns ``Any`` alone or an unrelated type.
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
