"""Tests for core.config.inline."""

from __future__ import annotations

from typing import Protocol, Self, cast, override

import copy
import dataclasses

import pytest

from configgle.codec import from_plain, to_plain
from configgle.custom_types import Makeable, MutableNamespace
from configgle.fig import Fig
from configgle.inline import InlineConfig, PartialConfig


class _DynamicLookup(Protocol):
    def __getattr__(self, key: str) -> object: ...


def test_inline_config_serializes_as_a_plain_object() -> None:
    config: InlineConfig[str] = InlineConfig(str, 1, base=8)

    assert to_plain(config) == {
        "py/object": "configgle.inline.InlineConfig",
        "func": {"py/type": "builtins.str"},
        "_finalized": False,
        "_args": [1],
        "_kwargs": {"base": 8},
    }


def test_a_finalized_inline_config_stays_finalized() -> None:
    config: InlineConfig[str] = InlineConfig(str, 1).finalize()

    restored = from_plain(to_plain(config), object, allow_imports=True)

    assert isinstance(restored, InlineConfig)
    assert restored._finalized is True
    assert restored == config


def test_an_inline_config_cycle_round_trips() -> None:
    config: InlineConfig[object] = InlineConfig(list)
    config._args.append(config)

    restored = from_plain(to_plain(config), object, allow_imports=True)

    assert isinstance(restored, InlineConfig)
    assert restored._args[0] is restored


def test_inline_config():
    """Test InlineConfig functionality."""

    def add(a: int, b: int) -> int:
        return a + b

    # Test basic creation.
    cfg = InlineConfig(add, 1, 2)
    assert cfg.func == add
    assert cfg._args == [1, 2]
    assert cfg._kwargs == {}

    # Test make.
    result = cfg.make()
    assert result == 3

    # Test with kwargs.
    cfg2 = InlineConfig(add, a=5, b=10)
    assert cfg2.make() == 15

    # Test with mixed args and kwargs.
    cfg3 = InlineConfig(add, 3, b=7)
    assert cfg3.make() == 10


def test_inline_config_with_nested_make():
    """Test InlineConfig with nested makes."""

    class SimpleConfig:
        """Simple config without make (not a Fig)."""

        def __init__(self, value: int):
            self.value = value

        def finalize(self) -> Self:
            return copy.copy(self)

    def multiply(cfg: SimpleConfig) -> int:
        return cfg.value * 2

    # Test with object that has finalize but not make.
    cfg = SimpleConfig(5)
    inline_cfg = InlineConfig(multiply, cfg)

    # Make should call finalize on nested objects.
    result = inline_cfg.make()
    assert result == 10


def test_inline_config_finalize():
    """Test InlineConfig.finalize."""

    class SimpleConfig:
        """Simple config with finalize and make."""

        def __init__(self, x: int):
            self.x = x
            self._finalized = False

        def make(self) -> object:
            return self.finalize()

        def finalize(self) -> Self:
            new = copy.copy(self)
            new._finalized = True
            return new

    cfg = InlineConfig(lambda c: c.x * 2, SimpleConfig(1))  # pyright: ignore[reportUnknownLambdaType, reportUnknownVariableType, reportUnknownArgumentType, reportUnknownMemberType] -- The test exercises intentionally dynamic config construction.
    assert cfg._finalized is False

    finalized = cfg.finalize()  # pyright: ignore[reportUnknownVariableType] -- InlineConfig's callable type is erased by the dynamic test fixture.

    assert finalized._finalized is True
    # Should finalize nested configs.
    assert finalized._args[0]._finalized is True  # pyright: ignore[reportAttributeAccessIssue, reportUnknownMemberType] -- The dynamic test fixture stores erased config arguments.  # ty: ignore[unresolved-attribute] -- The dynamic test fixture stores erased config arguments.


def test_make_does_not_refinalize_an_already_finalized_inline_tree() -> None:
    """An enclosing Config may finalize an InlineConfig before making it."""

    class Value:
        class Config(Fig["Value"]):
            parts: tuple[str, ...] = ()

            @override
            def finalize(self) -> Self:
                self.parts = ("derived", *self.parts)
                return super().finalize()

        def __init__(self, config: Config) -> None:
            self.parts = config.parts

    def parts(value: Value) -> tuple[str, ...]:
        return value.parts

    config = InlineConfig(parts, Value.Config())
    finalized = config.copy_tree().finalize()

    assert finalized.make() == ("derived",)


def test_finalized_copies_and_finalizes_once() -> None:
    class Value:
        class Config(Fig["Value"]):
            parts: tuple[str, ...] = ()

            @override
            def finalize(self) -> Self:
                self.parts = ("derived", *self.parts)
                return super().finalize()

        def __init__(self, config: Config) -> None:
            self.parts = config.parts

    def parts(value: Value) -> tuple[str, ...]:
        return value.parts

    source = InlineConfig(parts, Value.Config())
    once = source.finalized()
    twice = once.finalized()

    assert source._finalized is False
    assert (once._finalized, once is not source) == (True, True)
    assert twice.make() == ("derived",)


def test_inline_config_finalizes_a_shared_child_once() -> None:
    finalize_calls: list[None] = []

    class Value:
        class Config(Fig["Value"]):
            @override
            def finalize(self) -> Self:
                finalize_calls.append(None)
                return super().finalize()

        def __init__(self, config: Config) -> None:
            del config

    shared = Value.Config()
    config = InlineConfig(lambda left, right: (left, right), shared, shared)  # pyright: ignore[reportUnknownLambdaType, reportUnknownVariableType, reportUnknownArgumentType] -- The test intentionally exercises an untyped callback boundary.

    finalized = config.copy_tree().finalize()  # pyright: ignore[reportUnknownVariableType] -- InlineConfig's callable type is erased by the shared-child fixture.

    assert len(finalize_calls) == 1
    assert finalized._args[0] is finalized._args[1]


def test_inline_config_finalize_terminates_a_self_cycle() -> None:
    config = InlineConfig(lambda value: value)  # pyright: ignore[reportUnknownLambdaType, reportUnknownVariableType, reportUnknownArgumentType] -- The test intentionally exercises an untyped callback boundary.
    config.value = config

    finalized = config.copy_tree().finalize()  # pyright: ignore[reportUnknownVariableType] -- InlineConfig's callable type is erased by the self-cycle fixture.

    assert finalized._finalized is True
    assert finalized.value is finalized  # pyright: ignore[reportAny] -- Dynamic field read through PartialConfig/InlineConfig.__getattr__, which is Any by construction.


def test_inline_config_makes_nested_containers_and_shared_children_once() -> None:
    constructed: list[None] = []

    class Value:
        class Config(Fig["Value"]):
            value: int = 7
            """Value copied onto the runtime instance."""

        def __init__(self, config: Config) -> None:
            constructed.append(None)
            self.value = config.value

    shared = Value.Config()

    def collect(
        values_list: list[Value],
        values_tuple: tuple[Value, ...],
        values_mapping: dict[str, Value],
    ) -> tuple[list[Value], tuple[Value, ...], dict[str, Value]]:
        return (
            values_list,
            values_tuple,
            values_mapping,
        )

    config = InlineConfig(collect, [shared], (shared,), {"value": shared})

    values_list, values_tuple, values_mapping = config.make()

    assert values_list[0] is values_tuple[0]
    assert values_list[0] is values_mapping["value"]
    assert values_list[0].value == 7
    assert len(constructed) == 1


def test_inline_config_make_rejects_a_self_cycle() -> None:
    def identity(value: object) -> object:
        return value

    config = InlineConfig(identity)
    config.value = config

    with pytest.raises(ValueError, match="cyclic Makeable"):
        config.make()


def test_inline_config_make_rejects_a_positional_self_cycle() -> None:
    def identity(value: object) -> object:
        return value

    config = InlineConfig(identity)
    config._args.append(config)

    with pytest.raises(ValueError, match="cyclic Makeable"):
        config.make()


def test_inline_config_attr_access():
    """Test InlineConfig attribute access via kwargs."""

    def func(a: int, b: int) -> int:
        return a + b

    cfg = InlineConfig(func)
    cfg.a = 5  # Should go to kwargs.
    cfg.b = 10  # Should go to kwargs.

    assert cfg.a == 5  # pyright: ignore[reportAny] -- Dynamic field read through PartialConfig/InlineConfig.__getattr__, which is Any by construction.
    assert cfg.b == 10  # pyright: ignore[reportAny] -- Dynamic field read through PartialConfig/InlineConfig.__getattr__, which is Any by construction.
    assert cfg._kwargs == {"a": 5, "b": 10}

    result = cfg.make()
    assert result == 15


def test_inline_config_delattr():
    """Test InlineConfig.__delattr__."""

    def func(a: int = 0) -> int:
        return a

    cfg = InlineConfig(func)
    cfg.a = 5

    assert cfg._kwargs == {"a": 5}

    del cfg.a
    assert cfg._kwargs == {}

    # Should raise error if trying to delete non-existent
    # The __delattr__ tries kwargs first, then falls through to object.__delattr__.
    with pytest.raises(AttributeError):
        del cfg.nonexistent


def test_inline_config_repr():
    """Test InlineConfig.__repr__."""

    def add(a: int, b: int) -> int:
        return a + b

    cfg = InlineConfig(add, 1, 2, c=3)
    repr_str = repr(cfg)
    assert "InlineConfig" in repr_str
    assert "add" in repr_str or "function" in repr_str


def test_partial_config():
    """Test PartialConfig."""

    def multiply(a: int, b: int, c: int = 1) -> int:
        return a * b * c

    # Create partial with some args.
    cfg = PartialConfig(multiply, 2, c=10)
    partial_func = cfg.make()

    # Should create a functools.partial.
    result = partial_func(b=3)
    assert result == 60  # 2 * 3 * 10.


def test_inline_configs_compare_by_value() -> None:
    """Two configs naming the same call are equal, so a fork's diff is honest."""

    def add(a: int, b: int = 0) -> int:
        return a + b

    assert PartialConfig(add, 1, b=2) == PartialConfig(add, 1, b=2)
    assert PartialConfig(add, 1, b=2) != PartialConfig(add, 1, b=3)
    assert PartialConfig(add, 1) != PartialConfig(add, 2)
    assert PartialConfig(add, 1) != InlineConfig(add, 1)
    assert InlineConfig(add, 1) == InlineConfig(add, 1)
    assert PartialConfig(add) != "add"


def test_inline_config_update_from_dataclass():
    """Test InlineConfig.update from a dataclass source."""

    @dataclasses.dataclass  # house-ignore[dataclass] -- A plain stdlib dataclass is the subject under test.
    class Source:
        a: int = 10
        b: str = "hello"

    cfg = InlineConfig(lambda a, b: f"{a}-{b}")  # pyright: ignore[reportUnknownLambdaType, reportUnknownArgumentType] -- The test intentionally exercises an untyped callback boundary.
    cfg.update(Source())
    assert cfg.a == 10  # pyright: ignore[reportAny] -- Dynamic field read through PartialConfig/InlineConfig.__getattr__, which is Any by construction.
    assert cfg.b == "hello"  # pyright: ignore[reportAny] -- Dynamic field read through PartialConfig/InlineConfig.__getattr__, which is Any by construction.
    assert cfg.make() == "10-hello"


def test_inline_config_update_from_non_dataclass():
    """Test InlineConfig.update from a non-dataclass source (skips callables)."""

    class Source:
        def __init__(self):
            self.x = 42
            self.y = "data"

        def method(self) -> None:
            pass

    cfg = InlineConfig(lambda **kwargs: kwargs)  # pyright: ignore[reportUnknownLambdaType, reportUnknownVariableType, reportUnknownArgumentType] -- The test intentionally exercises an untyped callback boundary.
    cfg.update(Source())  # pyright: ignore[reportArgumentType] -- The test source intentionally lacks the update Protocol.  # ty: ignore[invalid-argument-type] -- The test source intentionally lacks the update Protocol.
    assert cfg.x == 42  # pyright: ignore[reportAny] -- Dynamic field read through PartialConfig/InlineConfig.__getattr__, which is Any by construction.
    assert cfg.y == "data"  # pyright: ignore[reportAny] -- Dynamic field read through PartialConfig/InlineConfig.__getattr__, which is Any by construction.
    # Methods should NOT be copied.
    assert "method" not in cfg._kwargs


def test_inline_config_update_with_kwargs():
    """Test InlineConfig.update with kwargs."""
    cfg = InlineConfig(lambda a, b: a + b)  # pyright: ignore[reportUnknownLambdaType, reportUnknownVariableType, reportUnknownArgumentType] -- The test intentionally exercises an untyped callback boundary.
    cfg.update(a=5, b=10)
    assert cfg.make() == 15


def test_inline_config_update_skip_missing_filters_source_and_kwargs() -> None:
    @dataclasses.dataclass  # house-ignore[dataclass] -- A plain stdlib dataclass is the subject under test.
    class Source:
        existing: int = 20
        source_only: int = 30

    cfg = InlineConfig(lambda **kwargs: kwargs, existing=10)  # pyright: ignore[reportUnknownLambdaType, reportUnknownVariableType, reportUnknownArgumentType] -- The test intentionally exercises an untyped callback boundary.

    cfg.update(Source(), skip_missing=True, existing=40, kwargs_only=50)

    assert cfg._kwargs == {"existing": 40}


def test_inline_config_update_skip_missing_filters_a_non_dataclass_source() -> None:
    """skip_missing also filters a plain-object source, not just a dataclass one."""

    class Source:
        def __init__(self) -> None:
            self.existing = 40
            self.source_only = 30

    def build(**kwargs: object) -> dict[str, object]:
        return kwargs

    cfg = InlineConfig(build, existing=10)

    # `Source` isn't a `Makeable` -- `update()` only cares that it's not None
    # and not a dataclass, which `cast` proves to the checker without giving
    # `Source` a real (unused) make/finalize/copy_tree/update implementation.
    cfg.update(cast("Makeable[object]", Source()), skip_missing=True)

    assert cfg._kwargs == {"existing": 40}


def test_inline_config_update_non_dataclass_with_property():
    """Test InlineConfig.update from source with property that raises."""

    class TrickySource:
        @property
        def broken(self) -> object:
            raise AttributeError("can't get this")

        @property
        def data(self) -> int:
            return 42

    cfg = InlineConfig(lambda **kwargs: kwargs)  # pyright: ignore[reportUnknownLambdaType, reportUnknownVariableType, reportUnknownArgumentType] -- The test intentionally exercises an untyped callback boundary.
    cfg.update(TrickySource())  # pyright: ignore[reportArgumentType] -- The malformed test source exercises attribute filtering.  # ty: ignore[invalid-argument-type] -- The malformed test source exercises attribute filtering.
    # Broken should be skipped (AttributeError), data should be skipped (callable check)
    # Actually properties return their values, not the property object itself.
    assert cfg.data == 42  # pyright: ignore[reportAny] -- Dynamic field read through PartialConfig/InlineConfig.__getattr__, which is Any by construction.
    assert "broken" not in cfg._kwargs


def test_inline_config_recursive_repr():
    """Test InlineConfig.__repr__ with self-referencing kwargs."""
    cfg = InlineConfig(lambda x: x)  # pyright: ignore[reportUnknownLambdaType, reportUnknownVariableType, reportUnknownArgumentType] -- The test intentionally exercises an untyped callback boundary.
    cfg.self_ref = cfg  # Create self-reference.
    # Should not infinitely recurse -- @reprlib.recursive_repr handles it.
    repr_str = repr(cfg)  # pyright: ignore[reportUnknownArgumentType] -- Recursive dynamic config state erases the argument type.
    assert "..." in repr_str or "InlineConfig" in repr_str


def test_dynamic_namespace_narrows_partial_config():
    """Test MutableNamespace isinstance narrows for dynamic attribute access.

    When a field is typed as Makeable[...], isinstance(x, PartialConfig) narrows
    to PartialConfig[Unknown] in basedpyright, causing reportUnknownMemberType
    warnings. MutableNamespace is a non-generic protocol that avoids this::

        assert isinstance(some.field, MutableNamespace)
        some.field.lr = 0.1  # no basedpyright warning

    """

    def make_thing(lr: float = 0.1, weight_decay: float = 0.01) -> str:
        return f"lr={lr}, wd={weight_decay}"

    field: Makeable[object] = PartialConfig(make_thing, lr=0.5)

    assert isinstance(field, MutableNamespace)
    field.lr = 0.2
    field.weight_decay = 0.0
    assert field.lr == 0.2
    assert field.weight_decay == 0.0


def test_dynamic_namespace_narrows_inline_config():
    """Test MutableNamespace works for InlineConfig too."""

    def add(a: int = 0, b: int = 0) -> int:
        return a + b

    field: Makeable[object] = InlineConfig(add, a=1)

    assert isinstance(field, MutableNamespace)
    field.b = 2
    assert field.make() == 3


def test_dynamic_namespace_rejects_fig():
    """Test MutableNamespace returns False for Fig configs (no __getattr__)."""

    class MyClass:
        class Config(Fig):
            x: int = 0

    config = MyClass.Config()
    assert not isinstance(config, MutableNamespace)


def test_setattr_fallback_before_kwargs_initialized():
    """Test __setattr__ fallback when _kwargs is not yet set."""

    # Subclass without slots so object.__setattr__ can succeed.
    class DictConfig(InlineConfig[object]):
        pass

    cfg = object.__new__(DictConfig)
    # _kwargs slot is unset, so setting an arbitrary attr falls through
    # the except AttributeError path to object.__setattr__.
    cfg.custom = "value"
    assert object.__getattribute__(cfg, "custom") == "value"


def test_inline_config_reads_reserved_attributes_after_missing_dynamic_key() -> None:
    """Dynamic lookup reports the missing key after checking kwargs."""
    config: InlineConfig[str] = InlineConfig(str)
    assert config.func is str
    lookup: _DynamicLookup = config
    with pytest.raises(AttributeError, match="missing"):
        lookup.__getattr__("missing")


def test_inline_config_make_reuses_shared_nested_config_across_args_and_kwargs() -> (
    None
):
    """Making preserves identity when one nested config appears twice."""

    class Value:
        class Config(Fig["Value"]):
            value: int = 1

        def __init__(self, config: Config):
            self.value = config.value

    def make_pair(left: object, *, right: object) -> tuple[object, object]:
        return left, right

    shared = Value.Config()
    config = InlineConfig(make_pair, shared, right=shared)
    left, right = config.make()
    assert left is right


def test_inline_config_reserved_slots_and_equality() -> None:
    """Reserved slots remain ordinary attributes and equality is type-sensitive."""
    config = InlineConfig(str)
    config.func = repr
    assert config.func is repr
    assert config == InlineConfig(repr)
    assert config != InlineConfig(str)
    assert config != object()


def test_inline_config_finalized_copy_skips_second_finalize() -> None:
    """A finalized copy is not finalized again."""
    calls: list[None] = []

    class Child:
        def finalize(self) -> Self:
            calls.append(None)
            return self

    def identity(value: object) -> object:
        return value

    child = Child()
    config = InlineConfig(identity, child)
    finalized = config.finalized()
    finalized.finalized()
    assert len(calls) == 1


def test_inline_config_update_dataclass_skips_then_continues() -> None:
    """Skipping an unknown dataclass field does not stop later fields."""

    @dataclasses.dataclass(kw_only=True, slots=True)
    class Source:
        source_only: int = 1
        existing: int = 2

    def collect(**kwargs: object) -> dict[str, object]:
        return kwargs

    config = InlineConfig(collect, existing=0)
    config.update(Source(), skip_missing=True)
    assert config._kwargs == {"existing": 2}


def test_inline_config_dataclass_branch_copies_callable_fields() -> None:
    """Dataclass fields are copied even when their values are callable."""

    @dataclasses.dataclass(kw_only=True, slots=True)
    class Source:
        callback: object = print

    def collect(**kwargs: object) -> dict[str, object]:
        return kwargs

    config = InlineConfig(collect)
    config.update(Source())
    assert config._kwargs == {"callback": print}


def test_inline_config_update_non_dataclass_skips_private_then_continues() -> None:
    """Private source attributes are ignored without truncating traversal."""

    class Source:
        _private = 1
        existing = 2

    def collect(**kwargs: object) -> dict[str, object]:
        return kwargs

    config = InlineConfig(collect, existing=0)
    config.update(cast("Makeable[object]", Source()))
    assert config._kwargs["existing"] == 2
    assert "_private" not in config._kwargs


def test_inline_config_update_non_dataclass_skips_unknown_then_continues() -> None:
    """Unknown public attributes do not stop later accepted attributes."""

    class Source:
        aaa_unknown = 1
        existing = 2

        @override
        def __dir__(self) -> list[str]:
            return ["aaa_unknown", "existing"]

    def collect(**kwargs: object) -> dict[str, object]:
        return kwargs

    config = InlineConfig(collect, existing=0)
    config.update(cast("Makeable[object]", Source()), skip_missing=True)
    assert config._kwargs == {"existing": 2}


def test_inline_config_update_kwargs_skips_then_continues() -> None:
    """Unknown keyword overrides are ignored without stopping later overrides."""

    def collect(**kwargs: object) -> dict[str, object]:
        return kwargs

    config = InlineConfig(collect, existing=0)
    config.update(skip_missing=True, unknown=1, existing=2)
    assert config._kwargs == {"existing": 2}


if __name__ == "__main__":
    from configgle.lib.testing.main import test_main

    test_main(__file__)
