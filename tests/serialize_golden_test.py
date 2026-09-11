"""Frozen wire bytes for every tag ``serialize`` emits.

``serialize_test.py`` proves the format round-trips; a round-trip passes just as
well after both sides shift together, so it cannot see the format MOVE. This
freezes the bytes themselves, one case per tag, and fails when they change.

The wire is a durable format: a config serialized by one version is read by a
later one, and a tree whose bytes shift silently stops matching what an earlier
run recorded. So a diff here is a compatibility decision, never a test to
refresh -- regenerate only alongside that decision::

    CONFIGGLE_REGENERATE_GOLDEN=1 uv --quiet run --frozen pytest \
        serialize_golden_test.py

The goldens are inline rather than a data file because only ``*_test.py`` is
copied to the standalone package's ``tests/``, so a sibling data file would be
absent there. The dotted paths inside the wire also name THIS module, which
differs between the two checkouts -- hence ``{module}`` in every golden,
substituted at compare time.
"""

from __future__ import annotations

from collections import OrderedDict
from collections.abc import Callable, Mapping
from dataclasses import field
from decimal import Decimal
from pathlib import PurePosixPath
from typing import Final, NamedTuple, override

import enum
import json
import os

import pytest

from configgle.custom_json import (
    DecodeCapabilities,
    GraphHooks,
    decode_graph,
    encode_graph,
    resolve_import,
)
from configgle.fig import Fig
from configgle.inline import InlineConfig


MODULE: Final = __name__
"""This module's dotted path, which every ``py/object`` in a golden names."""

_ENV_REGENERATE: Final = "CONFIGGLE_REGENERATE_GOLDEN"


class Color(enum.Enum):
    """An enum, whose reduce recipe the wire has to carry."""

    RED = "red"


class Coord(NamedTuple):
    """A namedtuple: reduced by value, so it takes no reference index."""

    lat: float
    lon: float


class Weight:
    """A leaf with no useful reduce, serialized through a hook."""

    def __init__(self, data: list[float]) -> None:
        self.data = data

    @override
    def __eq__(self, other: object) -> bool:
        return isinstance(other, Weight) and other.data == self.data

    @override
    def __hash__(self) -> int:
        return hash(tuple(self.data))


class Leaf:
    """The innermost config, and the object a DAG case shares."""

    class Config(Fig["Leaf"]):
        k: int = 1
        """A leaf value."""

    def __init__(self, config: Config) -> None:
        self.k = config.k


class Branch:
    """A config holding another, so the wire nests ``py/object``."""

    class Config(Fig["Branch"]):
        leaf: Leaf.Config = field(default_factory=Leaf.Config)
        """The child leaf."""

        tag: str = "t"
        """A label."""

    def __init__(self, config: Config) -> None:
        self.leaf = config.leaf.make()


class Pair:
    """A config whose two slots may hold one shared child."""

    class Config(Fig["Pair"]):
        a: Leaf.Config = field(default_factory=Leaf.Config)
        """First slot."""

        b: Leaf.Config = field(default_factory=Leaf.Config)
        """Second slot; set to ``a`` to exercise ``py/id``."""

    def __init__(self, config: Config) -> None:
        del config


class Cyclic:
    """A config that can point at itself, so a cycle has an anchor."""

    class Config(Fig["Cyclic"], slots=False):
        peer: object = None
        """Another config, possibly this one."""

    def __init__(self, config: Config) -> None:
        del config


def _shared_leaf_pair() -> Pair.Config:
    """Return a config whose two slots hold the SAME child."""
    config = Pair.Config()
    config.b = config.a
    return config


def _self_cycle() -> Cyclic.Config:
    """Return a config that is its own peer."""
    config = Cyclic.Config()
    config.peer = config
    return config


def _nested_branch() -> Branch.Config:
    """Return a branch with a non-default leaf."""
    config = Branch.Config()
    config.leaf.k = 3
    return config


def _shared_list() -> dict[str, object]:
    """One list reachable by two paths, so the second is a back-reference."""
    shared = [1, 2]
    return {"x": shared, "y": {"z": shared}}


def _cyclic_list() -> list[object]:
    """Return a list containing itself."""
    cycle: list[object] = [1]
    cycle.append(cycle)
    return cycle


def _empty_containers() -> dict[str, object]:
    """Every empty container, which each tag spells differently."""
    return {"a": [], "b": {}, "c": (), "d": set[int]()}


CASES: Final[Mapping[str, Callable[[], object]]] = {
    "primitives": lambda: {"a": 1, "b": 2.5, "c": None, "d": True, "e": "s"},
    "nonfinite": lambda: [float("inf"), float("-inf")],
    "bytes": lambda: b"\x00\xff binary",
    "tuple": lambda: (1, (2, 3), ()),
    "set": lambda: {1},
    "frozenset": lambda: frozenset({1}),
    "path": lambda: PurePosixPath("/opt/scratch/x"),
    "decimal": lambda: Decimal("1.25"),
    "enum": lambda: Color.RED,
    "namedtuple": lambda: Coord(1.0, 2.0),
    "ordereddict": lambda: OrderedDict([("b", 1), ("a", 2)]),
    "type_reference": lambda: PurePosixPath,
    "function_reference": lambda: json.dumps,
    "reserved_keys": lambda: {"py/object": 1, "json://x": 2, "plain": 3},
    "nonstring_keys": lambda: {1: "int", (2, 3): "tuple", "s": "str"},
    "empty_containers": _empty_containers,
    "fig_leaf": lambda: Leaf.Config(k=7),
    "fig_nested": _nested_branch,
    "fig_shared_child": _shared_leaf_pair,
    "fig_self_cycle": _self_cycle,
    "shared_list": _shared_list,
    "cyclic_list": _cyclic_list,
    "hooked_leaf": lambda: Weight([1.5]),
    "inline_config": lambda: InlineConfig(PurePosixPath, "/opt/scratch/x"),
}
"""One builder per wire feature. A tag with no case here is unfrozen."""

GOLDEN: Final[Mapping[str, str]] = {
    "primitives": '{"a":1,"b":2.5,"c":null,"d":true,"e":"s"}',
    "nonfinite": '[{"py/float":"inf"},{"py/float":"-inf"}]',
    "bytes": '{"py/b64":"AP8gYmluYXJ5"}',
    "tuple": '{"py/tuple":[1,{"py/tuple":[2,3]},{"py/tuple":[]}]}',
    "set": '{"py/set":[1]}',
    "frozenset": '{"py/reduce":[{"py/type":"builtins.frozenset"},{"py/tuple":[[1]]}]}',
    "path": (
        '{"py/reduce":[{"py/type":"pathlib.PurePosixPath"},'
        '{"py/tuple":["/opt/scratch/x"]}]}'
    ),
    "decimal": ('{"py/reduce":[{"py/type":"decimal.Decimal"},{"py/tuple":["1.25"]}]}'),
    "enum": ('{"py/reduce":[{"py/type":"{module}.Color"},{"py/tuple":["red"]}]}'),
    "namedtuple": (
        '{"py/reduce":[{"py/function":"copyreg.__newobj__"},'
        '{"py/tuple":[{"py/type":"{module}.Coord"},1.0,2.0]}]}'
    ),
    "ordereddict": (
        '{"py/reduce":[{"py/type":"collections.OrderedDict"},'
        '{"py/tuple":[]},null,null,[{"py/tuple":["b",1]},'
        '{"py/tuple":["a",2]}]]}'
    ),
    "type_reference": '{"py/type":"pathlib.PurePosixPath"}',
    "function_reference": '{"py/function":"json.dumps"}',
    "reserved_keys": (
        '{"json://\\"py/object\\"":1,"json://\\"json://x\\"":2,"plain":3}'
    ),
    "nonstring_keys": (
        '{"json://1":"int","json://{\\"py/tuple\\": [2, 3]}":"tuple","s":"str"}'
    ),
    "empty_containers": '{"a":[],"b":{},"c":{"py/tuple":[]},"d":{"py/set":[]}}',
    "fig_leaf": '{"py/object":"{module}.Leaf.Config","k":7}',
    "fig_nested": (
        '{"py/object":"{module}.Branch.Config",'
        '"leaf":{"py/object":"{module}.Leaf.Config","k":3},"tag":"t"}'
    ),
    "fig_shared_child": (
        '{"py/object":"{module}.Pair.Config",'
        '"a":{"py/object":"{module}.Leaf.Config","k":1},"b":{"py/id":1}}'
    ),
    "fig_self_cycle": ('{"py/object":"{module}.Cyclic.Config","peer":{"py/id":0}}'),
    "shared_list": '{"x":[1,2],"y":{"z":{"py/id":1}}}',
    "cyclic_list": '[1,{"py/id":0}]',
    "hooked_leaf": '{"py/hook":["{module}.Weight",[1.5]]}',
    "inline_config": (
        '{"py/inline":["configgle.inline.InlineConfig",'
        '{"func":{"py/type":"pathlib.PurePosixPath"},'
        '"args":["/opt/scratch/x"],"kwargs":{}}]}'
    ),
}
"""The exact bytes each case serializes to, with ``{module}`` for this module.

Regenerate deliberately; see the module docstring. A changed entry means a
payload written by an older configgle no longer reads back the same way.
"""


@pytest.mark.parametrize("name", sorted(CASES), ids=sorted(CASES))
def test_a_case_serializes_to_its_frozen_bytes(name: str) -> None:
    wire = json.dumps(
        encode_graph(CASES[name](), hooks=_hooks()),
        separators=(",", ":"),
        sort_keys=False,
    )

    if os.environ.get(_ENV_REGENERATE) == "1":
        pytest.skip(f"regenerating: {name} -> {wire.replace(MODULE, '{module}')}")
    assert wire == GOLDEN[name].replace("{module}", MODULE)


@pytest.mark.parametrize("name", sorted(CASES), ids=sorted(CASES))
def test_the_frozen_bytes_still_decode_to_the_value(name: str) -> None:
    # Freezing bytes nobody can read again would pass while the format is
    # broken, so the golden is decoded rather than only compared. The decoded
    # value is compared by RE-SERIALIZING it: ``==`` recurses forever on the
    # cyclic cases, and a value whose class defines no ``__eq__`` compares by
    # identity, which a fresh decode never satisfies.
    wire = GOLDEN[name].replace("{module}", MODULE)

    restored = _decode_graph(json.loads(wire), hooks=_hooks())

    assert (
        json.dumps(encode_graph(restored, hooks=_hooks()), separators=(",", ":"))
        == wire
    )


def test_every_case_has_a_golden() -> None:
    # Without this, adding a case and forgetting its golden raises KeyError in
    # one parametrization rather than reporting the gap.
    assert sorted(GOLDEN) == sorted(CASES)


@pytest.mark.parametrize(
    "tag",
    [
        "py/b64",
        "py/float",
        "py/function",
        "py/hook",
        "py/id",
        "py/inline",
        "py/object",
        "py/reduce",
        "py/set",
        "py/tuple",
        "py/type",
        "json://",
    ],
)
def test_every_wire_tag_is_frozen_by_some_case(tag: str) -> None:
    # The goldens above only defend the tags they happen to contain. This is
    # what fails when a tag is added to the format and left unfrozen.
    assert any(tag in wire for wire in GOLDEN.values()), f"no golden emits {tag}"


def _decode_graph(tree: object, *, hooks: GraphHooks) -> object:
    return decode_graph(
        tree,
        hooks=hooks,
        capabilities=DecodeCapabilities(resolve=resolve_import, apply_reduce=True),
    )


def _hooks() -> GraphHooks:
    """Return the hook table the golden cases serialize under."""
    return {Weight: (_encode_weight, Weight)}


def _encode_weight(weight: Weight) -> list[float]:
    """Encode a hooked leaf as the payload its hook stores."""
    return weight.data


if __name__ == "__main__":
    from configgle.lib.testing.main import test_main

    test_main(__file__)
