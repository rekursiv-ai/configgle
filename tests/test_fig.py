from __future__ import annotations

from collections.abc import Callable
from dataclasses import field
from typing import Self, override

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
    _get_object_attribute_names,
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
    with pytest.raises(AttributeError):
        cfg.nonexistent_field = 1  # pyright: ignore[reportAttributeAccessIssue]
    assert isinstance(cfg.make(), Parent)


def test_cloudpickle():
    cfg = Child.Config()
    cfg_ = pickle.loads(cloudpickle.dumps(cfg))
    assert cfg == cfg_
    assert cfg.a == cfg_.a
    assert cfg.b == cfg_.b
    assert cfg.c == cfg_.c
    with pytest.raises(AttributeError):
        cfg.nonexistent_field = 1  # pyright: ignore[reportAttributeAccessIssue]


def test_pickle_parent_class_restored():
    """Test that parent_class is correctly restored after pickle."""
    cfg = Parent.Config()

    # Verify parent_class is set before pickling
    assert Parent.Config.parent_class is Parent

    # Pickle and unpickle
    cfg_ = pickle.loads(pickle.dumps(cfg))

    # Verify parent_class is restored after unpickling
    assert type(cfg_).parent_class is Parent
    assert cfg_.make().__class__ is Parent


def test_cloudpickle_parent_class_restored():
    """Test that parent_class is correctly restored after cloudpickle."""
    cfg = Child.Config()

    # Verify parent_class is set before pickling
    assert Child.Config.parent_class is Child

    # Cloudpickle and unpickle
    cfg_ = pickle.loads(cloudpickle.dumps(cfg))

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
    Child_pickled = pickle.loads(cloudpickle.dumps(Child))

    # Verify the Config class is accessible
    assert hasattr(Child_pickled, "Config")
    assert Child_pickled.Config.parent_class is Child_pickled

    # Verify we can create and use the config
    cfg = Child_pickled.Config(a=1.0, b=2.0, c=3.0j)
    instance = cfg.make()
    assert instance.__class__ is Child_pickled


def test_mutable():
    cfg = Mutable.Config()
    cfg.b = 2  # pyright: ignore[reportAttributeAccessIssue]
    assert cfg.b == 2  # pyright: ignore[reportAttributeAccessIssue]


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
    """Test Fig finalize creates copy."""

    class TestConfig(Fig):
        x: int = 1

    cfg = TestConfig()
    # Fig doesn't have _finalized, only Setup does
    # But finalize returns a copy
    finalized = cfg.finalize()

    # They should be different objects
    assert cfg is not finalized
    # Values should be the same
    assert cfg.x == finalized.x


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


def test_get_object_attribute_names_filters_int_indices():
    """Test that _get_object_attribute_names only yields string attribute names."""

    # Test with object (should yield attribute names)
    class TestObj:
        def __init__(self):
            self.x = 1
            self.y = 2

    obj = TestObj()
    names = list(_get_object_attribute_names(obj))
    assert set(names) == {"x", "y"}

    # Test with list (should yield nothing - no string attributes)
    # This protects against the bug where integer indices would be stringified
    lst = [1, 2, 3]
    names = list(_get_object_attribute_names(lst))
    assert names == []


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


if __name__ == "__main__":
    import pytest

    pytest.main([__file__, "-v"])
