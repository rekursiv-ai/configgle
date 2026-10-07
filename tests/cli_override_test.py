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


@pytest.mark.parametrize("raw", ["123", "1.5", "run_a"])
def test_apply_overrides_keeps_the_text_for_a_str_field(raw: str) -> None:
    """``name=123`` means the text "123": a str field reads the raw text."""
    config = NestedJob.Config()
    apply_overrides(config, [f"name={raw}"])
    assert config.name == raw


def test_apply_overrides_reads_a_quoted_json_string_as_its_text() -> None:
    """``name="a b"`` is the JSON string, so the quotes are not kept."""
    config = NestedJob.Config()
    apply_overrides(config, ['name="a b"'])
    assert config.name == "a b"


@pytest.mark.parametrize("raw", ["abc", "[1]", "{}"])
def test_apply_overrides_rejects_text_no_reading_fits(raw: str) -> None:
    """A value neither its JSON nor its bare text can type fails, naming the path."""
    config = NestedJob.Config()
    with pytest.raises(ValueError, match=r"child\.steps"):
        apply_overrides(config, [f"child.steps={raw}"])


def test_apply_overrides_reads_a_number_for_a_numeric_field() -> None:
    """``steps=7`` is the number, not the text."""
    config = NestedJob.Config()
    apply_overrides(config, ["child.steps=7", "child.lr=0.5"])
    assert config.child.steps == 7
    assert config.child.lr == 0.5


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


def test_apply_overrides_path_through_a_scalar_field_raises_valueerror() -> None:
    """A path that walks PAST a leaf scalar into a non-existent hop fails.

    ``name`` is a plain ``str`` field; ``name.sub`` tries to traverse one hop
    further, into the string value itself, which is not a config node.
    """
    config = NestedJob.Config()
    with pytest.raises(ValueError, match="traverses non-config str"):
        apply_overrides(config, ["name.sub=1"])


def test_apply_overrides_empty_segment_paths_raise() -> None:
    """Paths with empty segments are rejected with a clear message."""
    config = NestedJob.Config()
    for spec in ["=1", "child.=1", "child..lr=1", ".child=1"]:
        with pytest.raises(
            ValueError,
            match=r"^Malformed override .*field path with no empty segments \(e\.g\. `step\.lr`\)\.$",
        ):
            apply_overrides(config, [spec])


def test_apply_overrides_preserves_equals_in_value() -> None:
    """Only the first equals separates a path from its string value."""
    config = NestedJob.Config()
    apply_overrides(config, ["name=left=right"])
    assert config.name == "left=right"


def test_apply_overrides_reports_exact_invalid_leaf_path() -> None:
    """Invalid leaf diagnostics identify the real config type and path."""
    config = NestedJob.Config()
    with pytest.raises(
        ValueError,
        match=r"^Override path `child\.missing` has no field `missing` on ChildJob\.Config\.$",
    ):
        apply_overrides(config, ["child.missing=1"])


def test_apply_overrides_reports_exact_invalid_intermediate_path() -> None:
    """Invalid intermediate diagnostics retain the complete dotted path."""
    config = NestedJob.Config()
    with pytest.raises(
        ValueError,
        match=r"^Override path `child\.missing\.lr` has no field `missing` on ChildJob\.Config\.$",
    ):
        apply_overrides(config, ["child.missing.lr=1"])


if __name__ == "__main__":
    from configgle.lib.testing.main import test_main

    test_main(__file__)
