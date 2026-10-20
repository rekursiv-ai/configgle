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

from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path
from typing import Final, TypeAliasType, cast
from typing_extensions import TypeVar

from configgle.fig import Fig, Makes, Makes2


_CWD: Final = Path(__file__).resolve().parent

# The source, not the installed copy ``import ty_extensions`` finds first: only the
# source counts toward this package's coverage. Exported, the tests sit one level
# below the package root.
_POLYFILL: Final = next(
    candidate
    for root in (_CWD, _CWD.parent)
    if (
        candidate := root / "packages/ty-extensions/ty_extensions/__init__.py"
    ).is_file()
)


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


# A subclassable Config leaves its parent slot open with a covariant TypeVar
# defaulting to its own owner; each subclass then fills it with
# ``Base.Config["Sub"]``. Only a TypeVar keeps a class generic for checkers.
_AnimalT_co = TypeVar("_AnimalT_co", bound="Animal", default="Animal", covariant=True)
_DogT_co = TypeVar("_DogT_co", bound="Dog", default="Dog", covariant=True)


class Animal:
    class Config(Fig[_AnimalT_co]):
        name: str = "animal"

    def __init__(self, config: Config) -> None:
        del config


class Cat(Animal):
    """Inherited Config re-narrowed with ``Makes["Cat"]``."""

    class Config(Makes["Cat"], Animal.Config):
        indoor: bool = True


class Kitten(Cat):
    """Grandchild re-narrowed with ``Makes2`` over a ``Makes`` parent."""

    class Config(Makes2["Kitten"], Cat.Config):
        weeks: int = 8


class Dog(Animal):
    """Inherited Config narrowed by filling the open slot, kept open for ``Puppy``."""

    class Config(Animal.Config[_DogT_co]):
        breed: str = "mutt"


class Puppy(Dog):
    """Grandchild narrowed by filling ``Dog.Config``'s open slot."""

    class Config(Dog.Config["Puppy"]):
        age: int = 0


def test_intersection_polyfill_preserves_the_first_type() -> None:
    """Keep the runtime export used by checkers without intersection support."""
    spec = spec_from_file_location("ty_extensions_source", _POLYFILL)
    assert spec is not None
    assert spec.loader is not None
    module = module_from_spec(spec)
    spec.loader.exec_module(module)
    polyfill = cast(TypeAliasType, vars(module)["Intersection"])
    assert len(polyfill.__type_params__) == 2
    assert polyfill.__value__ is polyfill.__type_params__[0]


def test_bare_fig_make_returns_parent() -> None:
    bare: Bare = Bare.Config().make()  # pyright: ignore[reportAny] -- A bare `Fig` builds `Any`; the assertion is what pins the class.
    assert type(bare) is Bare


def test_bare_fig_fields_resolve() -> None:
    x: int = Bare.Config().x
    assert x == 0


def test_explicit_fig_make_returns_parent() -> None:
    explicit: Explicit = Explicit.Config().make()
    assert type(explicit) is Explicit


def test_makes_reparameterizes_to_child() -> None:
    cat: Cat = Cat.Config().make()
    assert type(cat) is Cat


def test_makes2_reparameterizes_to_grandchild() -> None:
    kitten: Kitten = Kitten.Config().make()
    assert type(kitten) is Kitten
    assert (kitten_config := Kitten.Config()).indoor
    assert kitten_config.weeks == 8


def test_open_slot_defaults_to_owner() -> None:
    animal: Animal = Animal.Config().make()
    assert type(animal) is Animal


def test_filled_slot_narrows_to_child() -> None:
    dog: Dog = Dog.Config().make()
    assert type(dog) is Dog


def test_grandchild_inherits_parent_type() -> None:
    puppy: Puppy = Puppy.Config().make()
    assert type(puppy) is Puppy


if __name__ == "__main__":
    from configgle.lib.testing.runner import test_main

    test_main(__file__)
