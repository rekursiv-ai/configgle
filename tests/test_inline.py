"""Tests for core.config.inline."""

from __future__ import annotations

from typing import Self

import copy
import dataclasses

import pytest

from configgle.inline import InlineConfig, PartialConfig


def test_inline_config():
    """Test InlineConfig functionality."""

    def add(a: int, b: int) -> int:
        return a + b

    # Test basic creation
    cfg = InlineConfig(add, 1, 2)
    assert cfg.func == add
    assert cfg._args == [1, 2]
    assert cfg._kwargs == {}

    # Test make
    result = cfg.make()
    assert result == 3

    # Test with kwargs
    cfg2 = InlineConfig(add, a=5, b=10)
    assert cfg2.make() == 15

    # Test with mixed args and kwargs
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

    # Test with object that has finalize but not make
    cfg = SimpleConfig(5)
    inline_cfg = InlineConfig(multiply, cfg)

    # Make should call finalize on nested objects
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

    cfg = InlineConfig(lambda c: c.x * 2, SimpleConfig(1))  # pyright: ignore[reportUnknownLambdaType, reportUnknownVariableType, reportUnknownArgumentType, reportUnknownMemberType]
    assert cfg._finalized is False

    finalized = cfg.finalize()  # pyright: ignore[reportUnknownVariableType]  # InlineConfig[Unknown]

    assert finalized._finalized is True
    # Should finalize nested configs
    assert finalized._args[0]._finalized is True  # pyright: ignore[reportAttributeAccessIssue, reportUnknownMemberType]  # ty: ignore[unresolved-attribute]


def test_inline_config_attr_access():
    """Test InlineConfig attribute access via kwargs."""

    def func(a: int, b: int) -> int:
        return a + b

    cfg = InlineConfig(func)
    cfg.a = 5  # Should go to kwargs
    cfg.b = 10  # Should go to kwargs

    assert cfg.a == 5
    assert cfg.b == 10
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
    # The __delattr__ tries kwargs first, then falls through to object.__delattr__
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

    # Create partial with some args
    cfg = PartialConfig(multiply, 2, c=10)
    partial_func = cfg.make()

    # Should create a functools.partial
    result = partial_func(b=3)
    assert result == 60  # 2 * 3 * 10


def test_inline_config_update_from_dataclass():
    """Test InlineConfig.update from a dataclass source."""

    @dataclasses.dataclass
    class Source:
        a: int = 10
        b: str = "hello"

    cfg = InlineConfig(lambda a, b: f"{a}-{b}")  # pyright: ignore[reportUnknownLambdaType, reportUnknownArgumentType]
    cfg.update(Source())
    assert cfg.a == 10
    assert cfg.b == "hello"
    assert cfg.make() == "10-hello"


def test_inline_config_update_from_non_dataclass():
    """Test InlineConfig.update from a non-dataclass source (skips callables)."""

    class Source:
        def __init__(self):
            self.x = 42
            self.y = "data"

        def method(self) -> None:
            pass

    cfg = InlineConfig(lambda **kwargs: kwargs)  # pyright: ignore[reportUnknownLambdaType, reportUnknownVariableType, reportUnknownArgumentType]
    cfg.update(Source())  # pyright: ignore[reportArgumentType]  # type: ignore[invalid-argument-type]
    assert cfg.x == 42
    assert cfg.y == "data"
    # Methods should NOT be copied
    assert "method" not in cfg._kwargs


def test_inline_config_update_with_kwargs():
    """Test InlineConfig.update with kwargs."""
    cfg = InlineConfig(lambda a, b: a + b)  # pyright: ignore[reportUnknownLambdaType, reportUnknownVariableType, reportUnknownArgumentType]
    cfg.update(a=5, b=10)
    assert cfg.make() == 15


def test_inline_config_update_skip_missing_ignored():
    """Test InlineConfig.update ignores skip_missing (by design)."""
    cfg = InlineConfig(lambda a: a)  # pyright: ignore[reportUnknownLambdaType, reportUnknownVariableType, reportUnknownArgumentType]
    # skip_missing is silently ignored for InlineConfig
    cfg.update(skip_missing=True, a=99)
    assert cfg.a == 99


def test_inline_config_update_non_dataclass_with_property():
    """Test InlineConfig.update from source with property that raises."""

    class TrickySource:
        @property
        def broken(self) -> object:
            raise AttributeError("can't get this")

        @property
        def data(self) -> int:
            return 42

    cfg = InlineConfig(lambda **kwargs: kwargs)  # pyright: ignore[reportUnknownLambdaType, reportUnknownVariableType, reportUnknownArgumentType]
    cfg.update(TrickySource())  # pyright: ignore[reportArgumentType]  # type: ignore[invalid-argument-type]
    # broken should be skipped (AttributeError), data should be skipped (callable check)
    # Actually properties return their values, not the property object itself
    assert cfg.data == 42
    assert "broken" not in cfg._kwargs


def test_inline_config_recursive_repr():
    """Test InlineConfig.__repr__ with self-referencing kwargs."""
    cfg = InlineConfig(lambda x: x)  # pyright: ignore[reportUnknownLambdaType, reportUnknownVariableType, reportUnknownArgumentType]
    cfg.self_ref = cfg  # Create self-reference
    # Should not infinitely recurse — @reprlib.recursive_repr handles it
    repr_str = repr(cfg)  # pyright: ignore[reportUnknownArgumentType]  # InlineConfig[Unknown]
    assert "..." in repr_str or "InlineConfig" in repr_str


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
