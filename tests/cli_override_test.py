from __future__ import annotations

from dataclasses import field

import pytest

from configgle import Fig, apply_overrides


class ChildJob:
    """Nested job used to exercise depth in override traversal."""

    class Config(Fig["ChildJob"]):
        lr: float = 1e-3
        steps: int = 10

    def __init__(self, config: Config):
        self.lr = config.lr
        self.steps = config.steps


class NestedJob:
    """Job whose config nests another Fig, for depth-override tests."""

    class Config(Fig["NestedJob"]):
        name: str = ""
        enabled: bool = False
        child: ChildJob.Config = field(default_factory=ChildJob.Config)

    def __init__(self, config: Config):
        self.config = config


class FrozenInner(Fig["object"], frozen=True):
    """Frozen nested config for frozen-override testing."""

    lr: float = 1e-3


class FrozenJob:
    """Job whose config (and nested config) are frozen Figs."""

    class Config(Fig["FrozenJob"], frozen=True):
        name: str = ""
        inner: FrozenInner = field(default_factory=FrozenInner)

    def __init__(self, config: Config):
        self.config = config


def test_apply_overrides_top_level_scalar() -> None:
    """A top-level override casts the RHS to the field's declared type."""
    config = NestedJob.Config()
    apply_overrides(config, ["enabled=true", "name=run_a"])
    assert config.enabled is True
    assert config.name == "run_a"


def test_apply_overrides_nested_depth() -> None:
    """A dotted override walks into a nested Fig and casts the leaf."""
    config = NestedJob.Config()
    apply_overrides(config, ["child.lr=3e-4", "child.steps=99"])
    assert config.child.lr == 3e-4
    assert isinstance(config.child.lr, float)
    assert config.child.steps == 99
    assert isinstance(config.child.steps, int)


def test_apply_overrides_frozen_fig() -> None:
    """Overrides write through frozen Figs, top-level and nested."""
    config = FrozenJob.Config()
    apply_overrides(config, ["name=frozen_run", "inner.lr=2e-4"])
    assert config.name == "frozen_run"
    assert config.inner.lr == 2e-4
    assert isinstance(config.inner.lr, float)


def test_apply_overrides_unknown_path_raises() -> None:
    """An override naming a non-existent field is a hard error."""
    config = NestedJob.Config()
    with pytest.raises(ValueError, match="no field `nonexistent`"):
        apply_overrides(config, ["child.nonexistent=1"])


def test_apply_overrides_malformed_spec_raises() -> None:
    """An override missing ``=`` is rejected."""
    config = NestedJob.Config()
    with pytest.raises(ValueError, match="expected PATH=VALUE"):
        apply_overrides(config, ["child.lr"])


def test_apply_overrides_scalar_for_config_field_raises_valueerror() -> None:
    """Setting a nested-config field to a scalar raises ValueError.

    Not a codec TypeError. ``child`` is a Fig; ``child=5`` is a likely typo
    for ``child.lr=5`` and must fail with a path-naming ValueError.
    """
    config = NestedJob.Config()
    with pytest.raises(ValueError, match="child"):
        apply_overrides(config, ["child=5"])


def test_apply_overrides_empty_segment_paths_raise() -> None:
    """Paths with empty segments are rejected with a clear message."""
    config = NestedJob.Config()
    for spec in ["=1", "child.=1", "child..lr=1", ".child=1"]:
        with pytest.raises(ValueError, match="no empty segments"):
            apply_overrides(config, [spec])


if __name__ == "__main__":
    from configgle.lib.testing.main import test_main

    test_main(__file__)
