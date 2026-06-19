from __future__ import annotations

from collections.abc import Callable
from dataclasses import field
from typing import NamedTuple, Self, cast, override

import dataclasses
import io
import logging
import pickle

import cloudpickle
import pytest

from configgle import InlineConfig, Makeable, PartialConfig
from configgle.fig import (
    Fig,
    Maker,
    Makes,
    _DataclassParams,
    _Default,
)


class BaseConfig(Fig):
    a: float = 1.6180


class Parent:
    class Config(BaseConfig):
        b: float = 2.1783

        @override
        def finalize(self) -> Self:
            x = super().finalize()
            x.a = -1
            return x

    def __init__(self, config: Config):
        self.a = config.a
        self.b = config.b


class Child(Parent):
    class Config(Parent.Config):
        c: complex = 3.1415j

        @override
        def finalize(self) -> Self:
            x = super().finalize()
            x.a = -1.6180
            x.b = -2.1783
            return x

    def __init__(self, config: Config):
        super().__init__(config)
        self.c = config.c


class Mutable:
    class Config(Fig, slots=False):
        a: int = 1


def test_standalone():
    assert BaseConfig.__name__ == "BaseConfig"
    assert BaseConfig.parent_class is None


def test_cant_set_unknown_field():
    cfg = Parent.Config()
    name = "nonexistent_field"
    with pytest.raises(AttributeError):
        setattr(cfg, name, 1)
    assert isinstance(cfg.make(), Parent)


def test_cloudpickle():
    cfg = Child.Config()
    cfg_ = cast(Child.Config, pickle.loads(cloudpickle.dumps(cfg)))
    assert cfg == cfg_
    assert cfg.a == cfg_.a
    assert cfg.b == cfg_.b
    assert cfg.c == cfg_.c
    with pytest.raises(AttributeError):
        cfg.nonexistent_field = 1  # pyright: ignore[reportAttributeAccessIssue]  # ty: ignore[unresolved-attribute]


def test_pickle_parent_class_restored():
    """Test that parent_class is correctly restored after pickle."""
    cfg = Parent.Config()

    # Verify parent_class is set before pickling
    assert Parent.Config.parent_class is Parent

    # Pickle and unpickle
    cfg_ = pickle.loads(pickle.dumps(cfg))

    # Verify parent_class is restored after unpickling
    assert type(cfg_).parent_class is Parent  # pyright: ignore[reportUnknownMemberType]  # parent_class untyped
    assert cfg_.make().__class__ is Parent


def test_cloudpickle_parent_class_restored():
    """Test that parent_class is correctly restored after cloudpickle."""
    cfg = Child.Config()

    # Verify parent_class is set before pickling
    assert Child.Config.parent_class is Child

    # Cloudpickle and unpickle
    cfg_ = cast(Child.Config, pickle.loads(cloudpickle.dumps(cfg)))

    # Verify parent_class is restored after unpickling
    assert type(cfg_).parent_class is Child
    assert cfg_.make().__class__ is Child


def test_pickle_nested_class_with_parent():
    """Test pickling the parent class that contains the nested Config."""
    # When we pickle the parent class itself, Config should be preserved
    Parent_pickled = pickle.loads(pickle.dumps(Parent))

    # Verify the Config class is accessible
    assert hasattr(Parent_pickled, "Config")
    assert Parent_pickled.Config.parent_class is Parent_pickled

    # Verify we can create and use the config
    cfg = Parent_pickled.Config(a=3.14, b=2.71)
    instance = cfg.make()
    assert instance.__class__ is Parent_pickled
    # Note: Parent.Config.finalize() sets a=-1, so we check the finalized value
    assert instance.a == -1
    assert instance.b == 2.71


def test_cloudpickle_nested_class_with_parent():
    """Test cloudpickling the parent class that contains the nested Config."""
    # When we cloudpickle the parent class itself, Config should be preserved
    Child_pickled = cast(type[Child], pickle.loads(cloudpickle.dumps(Child)))

    # Verify the Config class is accessible
    assert hasattr(Child_pickled, "Config")
    assert Child_pickled.Config.parent_class is Child_pickled

    # Verify we can create and use the config
    cfg = Child_pickled.Config(a=1.0, b=2.0, c=3.0j)
    instance = cfg.make()
    assert instance.__class__ is Child_pickled


def test_mutable():
    cfg = Mutable.Config()
    name = "b"
    setattr(cfg, name, 2)
    assert getattr(cfg, name) == 2


def test_make_without_parent():
    """Test Maker.make() raises error when no parent class."""
    maker = Maker()
    with pytest.raises(
        ValueError,
        match="Maker must be nested in a parent class",
    ):
        maker.make()


def test_default_bool_and_repr():
    """Test _Default.__bool__ and __repr__."""
    # Test truthy value
    d_true = _Default(True)
    assert bool(d_true) is True
    assert repr(d_true) == "True"

    # Test falsy value
    d_false = _Default(False)
    assert bool(d_false) is False
    assert repr(d_false) == "False"

    # Test non-boolean values
    d_int = _Default(42)
    assert bool(d_int) is True
    assert repr(d_int) == "42"

    d_zero = _Default(0)
    assert bool(d_zero) is False
    assert repr(d_zero) == "0"


def test_dataclass_params_repr():
    """Test _DataclassParams.__repr__."""
    params = _DataclassParams(init=True, repr=False, eq=True, frozen=True)
    repr_str = repr(params)
    assert "_DataclassParams(" in repr_str
    assert "init=True" in repr_str
    assert "repr=False" in repr_str
    assert "eq=True" in repr_str
    assert "frozen=True" in repr_str


def test_dataclass_params_iter():
    """Test _DataclassParams.__iter__ with slot inheritance."""
    params = _DataclassParams()
    keys = list(params)
    # Should have all the standard dataclass params
    assert "init" in keys
    assert "repr" in keys
    assert "eq" in keys
    assert "order" in keys
    assert "frozen" in keys
    assert "slots" in keys


def test_dataclass_params_create():
    """Test _DataclassParams.create with defaults and overrides."""
    existing = _DataclassParams(init=True, repr=True, frozen=False)

    # Override with explicit values
    new = _DataclassParams.create(existing, init=False, frozen=True)
    assert new.init is False
    assert new.frozen is True
    # Should inherit from existing when not specified
    assert new.repr is True

    # Test with _Default values
    # When v is _Default, the condition (v is missing or isinstance(v, _Default))
    # is True, so it falls through to try existing
    new2 = _DataclassParams.create(
        existing,
        init=_Default(False),
        frozen=_Default(True),
    )
    # Should inherit from existing since _Default causes fallthrough
    assert new2.init is True  # From existing
    assert new2.frozen is False  # From existing


def test_fig_update():
    """Test Fig.update method."""

    class TestConfig(Fig):
        x: int = 1
        y: float = 2.0

    cfg = TestConfig()
    assert cfg.x == 1
    assert cfg.y == 2.0

    # Update with kwargs
    cfg.update(x=10, y=20.0)
    assert cfg.x == 10
    assert cfg.y == 20.0

    # Update from another dataclass
    cfg2 = TestConfig(x=100, y=200.0)
    cfg.update(cfg2)
    assert cfg.x == 100
    assert cfg.y == 200.0

    # Update with both source and kwargs (kwargs override)
    cfg4 = TestConfig()
    cfg4.update(cfg2, x=5)
    assert cfg4.x == 5  # From kwargs
    assert cfg4.y == 200.0  # From source


def test_fig_finalize():
    """Test Fig finalize mutates in place and returns self."""

    class TestConfig(Fig):
        x: int = 1

    cfg = TestConfig()
    finalized = cfg.finalize()

    # finalize is in-place: it returns the same object, now finalized.
    assert finalized is cfg
    assert getattr(cfg, "_finalized", False) is True


def test_finalize_is_in_place():
    """Finalize mutates the receiver; copy_tree().finalize() isolates it."""

    class TestConfig(Fig):
        x: int = 1

        @override
        def finalize(self) -> Self:
            self = super().finalize()
            self.x = 99
            return self

    # Direct finalize mutates the original.
    cfg = TestConfig()
    cfg.finalize()
    assert cfg.x == 99
    assert getattr(cfg, "_finalized", False) is True

    # copy_tree().finalize() leaves the original untouched.
    pristine = TestConfig()
    finalized = pristine.copy_tree().finalize()
    assert pristine.x == 1
    assert getattr(pristine, "_finalized", False) is False
    assert finalized.x == 99
    assert finalized is not pristine


def test_make_does_not_mutate_original():
    """make() finalizes a copy; the source config stays pristine."""

    class Thing:
        class Config(Fig["Thing"]):
            x: int = 1

            @override
            def finalize(self) -> Self:
                self = super().finalize()
                self.x = 42
                return self

        def __init__(self, config: Config) -> None:
            self.x = config.x

    cfg = Thing.Config()
    obj = cfg.make()
    assert obj.x == 42  # finalized value reached the instance
    assert cfg.x == 1  # original config untouched
    assert getattr(cfg, "_finalized", False) is False


def test_finalize_cascades_into_children_in_place():
    """Finalize finalizes nested children in place (no extra copy in finalize)."""

    class Inner(Fig):
        v: int = 0

        @override
        def finalize(self) -> Self:
            self = super().finalize()
            self.v = 7
            return self

    class Outer(Fig):
        inner: Inner = field(default_factory=Inner)

    cfg = Outer()
    finalized = cfg.finalize()
    # finalize is in place: root identity preserved, child finalized in place.
    assert finalized is cfg
    assert cfg.inner.v == 7
    assert getattr(cfg.inner, "_finalized", False) is True


def test_make_isolates_nested_children():
    """make() (copy_tree first) finalizes children without touching the source."""

    class Inner(Fig):
        v: int = 0

        @override
        def finalize(self) -> Self:
            self = super().finalize()
            self.v = 7
            return self

    class Outer:
        class Config(Fig["Outer"]):
            inner: Inner = field(default_factory=Inner)

        def __init__(self, config: Config) -> None:
            self.inner = config.inner

    cfg = Outer.Config()
    cfg.make()
    # Source's nested child is untouched: copy_tree isolated it.
    assert cfg.inner.v == 0
    assert getattr(cfg.inner, "_finalized", False) is False


def test_dataclass_params_iter_with_str_slots():
    """Test _DataclassParams.__iter__ with string slots (line 159)."""

    class CustomParams(_DataclassParams):
        """Custom params with single-slot string for testing."""

        # Note: In real Python, __slots__ should be tuple, but we're testing the branch

    # Create instance and test iteration
    params = CustomParams()
    keys = list(params)
    # Should have standard dataclass params
    assert len(keys) > 0


def test_dataclass_params_iter_skip_seen():
    """Test _DataclassParams.__iter__ skips already seen slots (line 162)."""
    params = _DataclassParams()
    keys = list(params)
    # Should not have duplicates
    assert len(keys) == len(set(keys))


def test_dataclass_params_create_missing_value():
    """Test _DataclassParams.create with missing values (line 184)."""
    existing = _DataclassParams(init=True, repr=True)

    # Create new params without specifying all fields
    # The function should skip fields where v is missing
    new = _DataclassParams.create(existing)

    # Should inherit from existing
    assert new.init is True
    assert new.repr is True


def test_fig_update_skip_missing():
    """Test Fig.update with skip_missing=True."""

    class TestConfig(Fig):
        x: int = 1
        y: float = 2.0

    cfg = TestConfig()

    # Update existing fields with skip_missing=True (should work)
    cfg.update(skip_missing=True, x=10, y=20)
    assert cfg.x == 10
    assert cfg.y == 20

    # Try to update non-existent field with skip_missing=True (should skip it)
    cfg.update(skip_missing=True, x=100, z=999)
    assert cfg.x == 100  # x was updated
    assert cfg.y == 20  # y unchanged
    assert not hasattr(cfg, "z")  # z was skipped


class Animal:
    class Config(Fig["Animal"]):
        name: str = "animal"

    def __init__(self, config: Config):
        self.name = config.name


class Dog(Animal):
    class Config(Makes["Dog"], Animal.Config):
        breed: str = "mutt"

    def __init__(self, config: Config):
        super().__init__(config)
        self.breed = config.breed


def test_inherited_config_make_returns_child():
    dog: Dog = Dog.Config(name="Rex", breed="labrador").make()
    assert isinstance(dog, Dog)
    assert isinstance(dog, Animal)
    assert dog.name == "Rex"
    assert dog.breed == "labrador"


def test_inherited_config_parent_class_tracks_child():
    assert Animal.Config.parent_class is Animal
    assert Dog.Config.parent_class is Dog


def test_inherited_config_base_still_makes_parent():
    animal: Animal = Animal.Config(name="Spot").make()
    assert type(animal) is Animal
    assert animal.name == "Spot"


def test_builds_not_in_mro():
    """Makes should not appear in the runtime MRO."""
    mro_names = [c.__name__ for c in Dog.Config.__mro__]
    assert "Makes" not in mro_names
    assert "Fig" in mro_names


# ---------------------------------------------------------------------------
# Patterns that exercise Makeable covariance + parent_class.
# Replicates the Base -> Derived -> Composite hierarchy from real usage.
# ---------------------------------------------------------------------------


class Base:
    """Base class with Config(Fig["Base"])."""

    class Config(Fig["Base"]):
        x: float = 1.0

    def __init__(self, config: Config) -> None:
        self.x = config.x


class Derived(Base):
    """Child whose Config uses Makes to re-bind parent_class."""

    class Config(Makes["Derived"], Base.Config):
        y: float = 2.0

    def __init__(self, config: Config) -> None:
        super().__init__(config)
        self.y = config.y


class Composite:
    """Container whose Config has a field typed Makeable[Derived].

    This is the pattern that breaks: Derived.Config inherits from
    Base.Config, so the inherited parent_class is type[Base],
    but Makeable[Derived] expects parent_class -> type[Derived].
    """

    class Config(Fig["Composite"]):
        part: Makeable[Derived] = field(default_factory=Derived.Config)

    def __init__(self, config: Config) -> None:
        self.part = config.part.make()


def test_makeable_field_with_inherited_config():
    """Makeable[Derived] field accepts Derived.Config (inherits from Base.Config)."""
    cfg = Composite.Config()
    obj = cfg.make()
    assert isinstance(obj.part, Derived)


def test_makeable_field_assignment():
    """Assigning Derived.Config to a Makeable[Derived] field works."""
    cfg = Composite.Config()
    cfg.part = Derived.Config(x=0.1, y=0.2)
    obj = cfg.make()
    assert obj.part.x == 0.1
    assert obj.part.y == 0.2


def test_inherited_config_parent_class_is_child():
    """Makes re-binds parent_class to the child, not the parent."""
    assert Base.Config.parent_class is Base
    assert Derived.Config.parent_class is Derived


def test_makeable_covariance():
    """Makeable[Base] should accept Derived.Config (Derived <: Base)."""
    cfg: Makeable[Base] = Derived.Config()
    result = cfg.make()
    assert isinstance(result, Derived)
    assert isinstance(result, Base)


def test_inline_config_satisfies_makeable():
    """InlineConfig/PartialConfig must satisfy Makeable (no parent_class needed)."""
    inline: Makeable[int] = InlineConfig(int, "42")
    assert inline.make() == 42

    partial_cfg: Makeable[Callable[..., int]] = PartialConfig(int, "42")
    assert partial_cfg.make()() == 42


def test_require_defaults_error_message():
    """Test that missing default raises TypeError with helpful message."""
    with pytest.raises(TypeError, match="has no default value"):

        class Bad(Fig, require_defaults=True):  # pyright: ignore[reportUnusedClass]
            x: int  # No default


def test_finalize_with_nested_dict():
    """Test finalize recursively walks mapping values."""

    class Inner(Fig):
        v: int = 0

    class Outer(Fig):
        d: dict[str, Inner] = field(default_factory=lambda: {"a": Inner(v=1)})

    cfg = Outer()
    finalized = cfg.finalize()
    inner = finalized.d["a"]
    assert getattr(inner, "_finalized", False) is True


def test_finalize_with_nested_set():
    """Test finalize recursively walks set values."""

    class Container(Fig):
        s: frozenset[int] = frozenset({1, 2, 3})

    cfg = Container()
    finalized = cfg.finalize()
    assert finalized.s == frozenset({1, 2, 3})


def test_finalize_with_namedtuple():
    """Test finalize recursively walks namedtuple elements."""

    class Point(NamedTuple):
        x: int
        y: int

    class Config(Fig):
        p: Point = Point(1, 2)

    cfg = Config()
    finalized = cfg.finalize()
    assert finalized.p == Point(1, 2)


def test_finalize_with_plain_tuple():
    """Test finalize recursively walks plain tuples."""

    class Inner(Fig):
        v: int = 0

    class Config(Fig):
        t: tuple[Inner, ...] = (Inner(),)

    cfg = Config()
    finalized = cfg.finalize()
    assert getattr(finalized.t[0], "_finalized", False) is True


def test_finalize_skips_non_data_objects():
    """Test finalize skips objects without __dataclass_fields__ or own __slots__."""

    class Config(Fig):
        logger: logging.Logger = logging.getLogger("test")

    cfg = Config()
    # Should not crash when encountering a logger
    finalized = cfg.finalize()
    assert finalized.logger is not None


def test_finalize_recurses_into_dataclass_objects():
    """Test finalize recurses into nested dataclass objects."""

    @dataclasses.dataclass  # check-dataclass: ignore[kw_only,slots]
    class PlainDC:
        value: int = 0

    class Config(Fig):
        nested: PlainDC = field(default_factory=PlainDC)

    cfg = Config()
    finalized = cfg.finalize()
    assert finalized.nested.value == 0


def test_update_source_with_attribute_error():
    """Test update handles source where getattr raises AttributeError."""

    class BadSource:
        __slots__ = ("x", "y")  # pyright: ignore[reportUninitializedInstanceVariable]

        def __init__(self):
            self.x = 1
            # y is intentionally uninitialized

    class Config(Fig):
        x: int = 0
        y: int = 0

    cfg = Config()
    source = BadSource()
    # Should skip y since it raises AttributeError
    cfg.update(source, skip_missing=True)  # pyright: ignore[reportArgumentType]  # ty: ignore[invalid-argument-type]
    assert cfg.x == 1


def test_repr_pretty_normal():
    """Test _repr_pretty_ IPython hook normal path."""

    class Config(Fig):
        x: int = 42

    class MockPrinter:
        def __init__(self):
            self.text_calls: list[str] = []

        def text(self, text: str) -> None:
            self.text_calls.append(text)

    cfg = Config()
    p = MockPrinter()
    cfg._repr_pretty_(p, cycle=False)
    assert len(p.text_calls) == 1
    assert "Config" in p.text_calls[0]


def test_repr_pretty_cycle():
    """Test _repr_pretty_ IPython hook cycle path."""

    class Config(Fig):
        x: int = 42

    class MockPrinter:
        def __init__(self):
            self.text_calls: list[str] = []

        def text(self, text: str) -> None:
            self.text_calls.append(text)

    cfg = Config()
    p = MockPrinter()
    cfg._repr_pretty_(p, cycle=True)
    assert len(p.text_calls) == 1
    assert "..." in p.text_calls[0]


def test_dataclass_params_iter_with_string_slots():
    """Test _DataclassParams.__iter__ handles string __slots__."""

    class StringSlotParams(_DataclassParams):
        __slots__ = "extra"  # pyright: ignore[reportAssignmentType]  # noqa: PLC0205  # intentionally string for branch test

    params = StringSlotParams()
    params.extra = True  # ty: ignore[unresolved-attribute]
    keys = list(params)
    assert "extra" in keys


def test_finalize_with_uninitialized_slot():
    """Test finalize handles uninitialized slots gracefully."""

    class Config(Fig, slots=False):
        __slots__ = ("_lazy",)
        x: int = 1

    cfg = Config()
    # _lazy slot is not initialized — finalize should skip it
    finalized = cfg.finalize()
    assert finalized.x == 1


def test_update_source_skip_missing_filters_source_fields():
    """Test update with skip_missing filters source fields not in target."""

    class SmallConfig(Fig):
        x: int = 0

    class BigSource(Fig):
        x: int = 99
        extra: str = "not in target"

    cfg = SmallConfig()
    source = BigSource()
    cfg.update(source, skip_missing=True)
    assert cfg.x == 99
    assert not hasattr(cfg, "extra")


def test_dataclass_params_create_missing_from_both():
    """Test _DataclassParams.create when field missing from both existing and kwargs."""

    class SparseParams(_DataclassParams):
        __slots__ = ("custom_field",)  # pyright: ignore[reportUninitializedInstanceVariable]

    existing = SparseParams()
    # custom_field is NOT set on existing → getattr returns missing → continue
    new = _DataclassParams.create(existing)
    assert new.init is True  # Standard fields still work


def test_finalize_value_recurses_into_slotted_object_with_nested_config():
    """Test _finalize_value recurses into objects with own __slots__ containing configs."""

    class Inner(Fig):
        v: int = 0

    class Wrapper:
        __slots__ = ("child",)

        def __init__(self, child: object):
            self.child = child

    class Config(Fig):
        w: Wrapper = field(default_factory=lambda: Wrapper(Inner()))

    cfg = Config()
    finalized = cfg.finalize()
    assert getattr(finalized.w.child, "_finalized", False) is True


def test_finalize_value_slotted_object_with_uninitialized_slot():
    """Test _finalize_value skips uninitialized slots on non-dataclass objects."""

    class Wrapper:
        __slots__ = ("initialized", "uninitialized")  # pyright: ignore[reportUninitializedInstanceVariable]

        def __init__(self):
            self.initialized = 42
            # uninitialized is intentionally not set

    class Config(Fig):
        w: Wrapper = field(default_factory=Wrapper)

    cfg = Config()
    finalized = cfg.finalize()
    assert finalized.w.initialized == 42


def test_pformat_method():
    """Test the pformat method on Maker."""

    class MyClass:
        class Config(Fig):
            x: int = 0
            y: str = "hello"

        def __init__(self, config: Config):
            pass

    cfg = MyClass.Config(x=42)
    result = cfg.pformat()
    assert "Config" in result
    assert "x=42" in result


def test_pprint_method(capsys: pytest.CaptureFixture[str]):
    """Test the pprint method on Maker."""

    class MyClass:
        class Config(Fig):
            x: int = 0

        def __init__(self, config: Config):
            pass

    cfg = MyClass.Config(x=42)
    cfg.pprint()
    captured = capsys.readouterr()
    assert "x=42" in captured.out


def test_pprint_method_with_stream():
    """Test pprint method with a custom stream."""

    class MyClass:
        class Config(Fig):
            x: int = 0

        def __init__(self, config: Config):
            pass

    cfg = MyClass.Config(x=42)
    buf = io.StringIO()
    cfg.pprint(stream=buf)
    assert "x=42" in buf.getvalue()


def test_dataclass_params_create_missing_attr():
    """Test _DataclassParams.create skips keys missing from both kwargs and existing."""
    # Create an uninitialized _DataclassParams (no __init__ call, so slots are unset)
    existing = object.__new__(_DataclassParams)
    # Only set a subset of attributes
    existing.init = True
    existing.repr = True
    # All other slots are unset — getattr with default returns missing, so they're skipped
    result = _DataclassParams.create(existing)
    assert result.init is True
    assert result.repr is True


def test_make_twice_leaves_source_unmutated():
    """Two ``make()`` calls each finalize a fresh copy; the source stays pristine.

    ``make`` is ``copy_tree().finalize()``, so the source config is never
    mutated and each build applies derived defaults to its own copy.
    """

    class Thing:
        class Config(Fig["Thing"]):
            n: int = 0

            @override
            def finalize(self) -> Self:
                self = super().finalize()
                self.n += 1
                return self

        def __init__(self, config: Config) -> None:
            self.n = config.n

    cfg = Thing.Config()
    first = cfg.make()
    second = cfg.make()
    assert first.n == 1
    assert second.n == 1
    assert cfg.n == 0  # source config never mutated


def test_pformat_after_finalize_does_not_re_finalize():
    """``pformat`` skips an already-finalized config (its ``_finalized`` guard)."""

    class Cfg(Fig):
        n: int = 0

        @override
        def finalize(self) -> Self:
            self = super().finalize()
            self.n += 1
            return self

    cfg = Cfg()
    cfg.finalize()
    assert cfg.n == 1
    cfg.pformat()  # _try_to_finalize guards on _finalized; must not re-apply
    assert cfg.n == 1


def test_finalize_finalizes_hashable_fig_in_frozenset():
    """A hashable (eq=False) Fig held in a frozenset is finalized.

    Set members must be hashable; an ``eq=False`` Fig is, so a frozenset can
    legally hold one. ``_finalize_value`` must finalize it rather than skip the
    whole set.
    """

    class Leaf(Fig, eq=False):
        v: int = 0

        @override
        def finalize(self) -> Self:
            self = super().finalize()
            self.v = 99
            return self

    class Holder(Fig):
        s: frozenset[Leaf] = field(default_factory=frozenset[Leaf])

    holder = Holder(s=frozenset({Leaf()}))
    finalized = holder.copy_tree().finalize()
    member = next(iter(finalized.s))
    assert member.v == 99
    assert getattr(member, "_finalized", False) is True


def test_finalize_finalizes_hashable_fig_in_set():
    """A hashable (eq=False) Fig held in a mutable set is finalized."""

    class Leaf(Fig, eq=False):
        v: int = 0

        @override
        def finalize(self) -> Self:
            self = super().finalize()
            self.v = 99
            return self

    class Holder(Fig):
        s: set[Leaf] = field(default_factory=set[Leaf])

    holder = Holder(s={Leaf()})
    finalized = holder.copy_tree().finalize()
    member = next(iter(finalized.s))
    assert member.v == 99


def test_finalize_finalizes_hashable_fig_dict_key():
    """A hashable (eq=False) Fig used as a dict key is finalized and isolated.

    ``copy_tree`` copies the key and ``_finalize_value`` finalizes it, so a
    Fig-keyed dict is handled like its values -- and the source is untouched.
    """

    class Key(Fig, eq=False):
        v: int = 0

        @override
        def finalize(self) -> Self:
            self = super().finalize()
            self.v = 99
            return self

    class Holder(Fig):
        d: dict[Key, str] = field(default_factory=dict[Key, str])

    source_key = Key()
    holder = Holder(d={source_key: "val"})
    finalized = holder.copy_tree().finalize()
    key = next(iter(finalized.d))
    assert key.v == 99
    assert finalized.d[key] == "val"
    assert key is not source_key  # key copied, source isolated
    assert source_key.v == 0


def test_finalize_preserves_unchanged_tuple_identity():
    """A tuple whose elements finalize in place is preserved (not rebuilt).

    In-place finalize keeps each Fig's identity, so the tuple holding them is
    unchanged and is returned as-is -- matching ``copy_tree``'s preserve rule.
    """

    class Leaf(Fig):
        v: int = 0

        @override
        def finalize(self) -> Self:
            self = super().finalize()
            self.v = 99
            return self

    class Holder(Fig):
        t: tuple[Leaf, ...] = field(default_factory=lambda: (Leaf(),))

    holder = Holder()
    original_tuple = holder.t
    holder.finalize()
    assert holder.t is original_tuple  # identity preserved
    assert holder.t[0].v == 99  # element finalized in place


def test_finalize_finalizes_dict_with_leaf_keys():
    """A dict with ordinary (str) keys round-trips: values finalized, keys intact."""

    class Leaf(Fig):
        v: int = 0

        @override
        def finalize(self) -> Self:
            self = super().finalize()
            self.v = 99
            return self

    class Holder(Fig):
        d: dict[str, Leaf] = field(default_factory=lambda: {"k": Leaf()})

    finalized = Holder().copy_tree().finalize()
    assert finalized.d["k"].v == 99


def test_finalize_does_not_misunwrap_wrapped_field():
    """A config field literally named ``__wrapped__`` is not mistaken for a proxy.

    ``finalize`` sheds a CopyOnWrite proxy by reading ``__wrapped__``; that probe
    must not fire on a real config that happens to declare such a field.
    """

    class Weird(Fig, slots=False):
        __wrapped__: str = (
            "payload"  # field name collides with the proxy probe by design of this test
        )

    cfg = Weird()
    finalized = cfg.finalize()
    assert finalized is cfg
    assert finalized.__wrapped__ == "payload"
    assert getattr(cfg, "_finalized", False) is True


def test_finalize_dag_finalizes_shared_child_once():
    """A config shared by two fields is finalized exactly once."""

    class Leaf(Fig):
        n: int = 0

        @override
        def finalize(self) -> Self:
            self.n += 1
            return super().finalize()

    class Root(Fig):
        a: Leaf = field(default_factory=Leaf)
        b: Leaf = field(default_factory=Leaf)

    shared = Leaf()
    root = Root(a=shared, b=shared)
    root.finalize()
    assert root.a is root.b
    assert root.a.n == 1  # finalized once, not twice


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
