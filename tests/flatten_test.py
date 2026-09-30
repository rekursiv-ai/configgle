"""Tests for lossless flat config encodings."""

from __future__ import annotations

from dataclasses import field

import pytest

from configgle.fig import Fig
from configgle.flatten import ABSENT, decode, dumps, encode, flatten, loads


class _Norm:
    class Config(Fig["_Norm"]):
        eps: float = 1e-6
        """Numerical floor."""

    def __init__(self, config: Config) -> None:
        del config


class _Block:
    class Config(Fig["_Block"]):
        window: int = 1024
        """Attention window."""

        norm: _Norm.Config = field(default_factory=_Norm.Config)
        """Normalization."""

    def __init__(self, config: Config) -> None:
        del config


def _tree(*, windows: tuple[int, ...] = (1024,) * 4, lr: float = 0.02) -> object:
    blocks = [
        {
            "py/object": "m.Block",
            "window": window,
            "norm": {"py/object": "m.Norm", "eps": 1e-6},
        }
        for window in windows
    ]
    return {
        "py/object": "m.Loop",
        "model": {"block": blocks, "width": 512},
        "optimizer": {"py/object": "m.Adam", "lr": lr, "betas": [0.9, 0.95]},
        "empty": {},
        "none": [],
        "odd key": {"a.b": 1, "[0]": 2},
    }


def test_flatten_emits_every_leaf_in_natural_order() -> None:
    entries = flatten({"b": [{"x": 1}] * 11, "a": {}, "c": [], "d": "s"})
    paths = [path for path, _ in entries]
    assert paths[:3] == ["a", "b[0].x", "b[1].x"]
    assert paths.index("b[2].x") < paths.index("b[10].x")
    assert dict(entries) == {
        "a": "{}",
        **{f"b[{i}].x": "1" for i in range(11)},
        "c": "[]",
        "d": "'s'",
    }


def test_flatten_quotes_keys_that_are_not_identifiers() -> None:
    assert flatten({"a.b": 1, "[0]": 2, "py/object": 3}) == [
        ("['[0]']", "2"),
        ("['a.b']", "1"),
        ("py/object", "3"),
    ]


def test_flatten_rejects_non_string_keys() -> None:
    with pytest.raises(TypeError, match="key"):
        flatten({1: 2})


def test_identical_siblings_become_one_reference() -> None:
    lines = encode(flatten(_tree()))
    assert ("model.block[1]", "@^[0]") in lines
    assert ("model.block[3]", "@^[0]") in lines
    assert not any(path.startswith("model.block[2].") for path, _ in lines)


def test_near_duplicate_sibling_carries_only_its_override() -> None:
    lines = encode(flatten(_tree(windows=(1024, 1024, 1024, 7))))
    assert ("model.block[3]", "@^[0]") in lines
    assert ("model.block[3].window", "7") in lines
    assert [path for path, _ in lines if path.startswith("model.block[3]")] == [
        "model.block[3]",
        "model.block[3].window",
    ]


def test_reference_prefers_fewest_overrides_then_earliest() -> None:
    lines = encode(flatten(_tree(windows=(1, 2, 2, 2))))
    assert ("model.block[2]", "@^[1]") in lines
    assert ("model.block[3]", "@^[1]") in lines


def test_absent_marks_leaves_the_reference_has_but_the_copy_lacks() -> None:
    tree = {
        "block": [
            {"a": 1, "b": 2, "c": 3, "d": 4, "extra": {"deep": 5}},
            {"a": 1, "b": 2, "c": 3, "d": 4},
        ],
    }
    entries = flatten(tree)
    lines = encode(entries)
    assert ("block[1]", "@^[0]") in lines
    assert ("block[1].extra", ABSENT) in lines
    assert decode(lines) == entries


def test_reference_is_skipped_when_raw_lines_are_shorter() -> None:
    lines = encode(flatten({"x": [{"a": 1}, {"a": 2}]}))
    assert lines == [("x[0].a", "1"), ("x[1].a", "2")]


def test_same_type_configs_at_one_depth_share_references() -> None:
    tree = {
        "left": {"a": 1, "b": 2, "c": 3, "norm": {"py/object": "m.N", "e": 1, "w": 2}},
        "right": {"x": 1, "y": 2, "z": 3, "norm": {"py/object": "m.N", "e": 1, "w": 2}},
    }
    lines = encode(flatten(tree))
    assert ("right.norm", "@left.norm") in lines


@pytest.mark.parametrize(
    "tree",
    [
        _tree(),
        _tree(windows=(1024, 1024, 1024, 7)),
        _tree(windows=(1, 2, 3, 2, 1)),
        {"b": [{"x": {"y": 1, "z": 2}, "k": 3}] * 3 + [{"x": {"y": 1}, "k": 3}]},
        {"n": [[{"a": 1, "b": 2}] * 3, [{"a": 1, "b": 2}] * 3]},
        {"leaf": [{"a": {"deep": 1}, "b": 2, "c": 3}, {"a": 9, "b": 2, "c": 3}]},
        {"leaf": [{"a": 9, "b": 2, "c": 3}, {"a": {"deep": 1}, "b": 2, "c": 3}]},
        {"quoted": [{"1": {"a.b": 1, "c": 2}}, {"1": {"a.b": 1, "c": 2}}]},
    ],
)
def test_decode_inverts_encode(tree: object) -> None:
    entries = flatten(tree)
    assert decode(encode(entries)) == entries
    assert decode(loads(dumps(encode(entries)))) == entries


def test_base_encoding_stores_only_the_delta() -> None:
    base = flatten(_tree())
    variant = flatten(_tree(windows=(1024, 1024, 5, 1024), lr=0.5))
    lines = encode(variant, base=base)
    assert lines == [("model.block[2].window", "5"), ("optimizer.lr", "0.5")]
    assert decode(lines, base=base) == variant
    assert encode(base, base=base) == []


def test_base_encoding_removes_and_replaces_subtrees() -> None:
    base = flatten({"a": {"x": 1, "y": 2}, "b": 3, "c": {"k": 1}, "gone": {"q": 1}})
    variant = flatten({"a": 7, "b": {"z": 1}, "c": {"k": 1}, "new": [1]})
    lines = encode(variant, base=base)
    assert decode(lines, base=base) == variant
    assert ("gone", ABSENT) in lines


def test_outline_round_trips_and_drops_repeated_prefixes() -> None:
    lines = encode(flatten(_tree(windows=(1024, 1024, 1024, 7))))
    text = dumps(lines)
    assert loads(text) == lines
    assert "model.block[1]" not in text
    assert "model.block[3]" not in text
    assert "\n  [3] = @^[0]\n   window = 7\n" in text


@pytest.mark.parametrize(
    ("text", "line"),
    [
        ("a\n", 1),
        ("= 1\n", 1),
        ("a = \n", 1),
        ("a b = 1\n", 1),
        ("a.b = 1\n    c = 2\n", 2),
        ("a.b = 1\n  c = 2\n", 2),
    ],
)
def test_loads_rejects_malformed_outlines(text: str, line: int) -> None:
    with pytest.raises(ValueError, match=f"line {line}"):
        loads(text)


def test_decode_rejects_dangling_reference() -> None:
    with pytest.raises(ValueError, match="@missing"):
        decode([("a", "@missing")])
    with pytest.raises(ValueError, match="climbs past the root"):
        decode([("a", "@^^b")])


def test_absolute_reference_used_when_shorter_than_climbing() -> None:
    shared = {"py/object": "m.N", "eps": 1, "weight": 2, "bias": 3}
    tree = {"a": {"b": {"c": {"d": shared}}}, "z": {"y": {"x": {"w": dict(shared)}}}}
    lines = encode(flatten(tree))
    assert ("z.y.x.w", "@a.b.c.d") in lines
    assert decode(lines) == flatten(tree)


def test_config_serialization_round_trips() -> None:
    config = _Block.Config()
    config.norm.eps = 0.5
    entries = flatten(config.serialize())
    assert ("norm.eps", "0.5") in entries
    assert decode(loads(dumps(encode(entries)))) == entries


if __name__ == "__main__":
    from configgle.lib.testing.main import test_main

    test_main(__file__)
