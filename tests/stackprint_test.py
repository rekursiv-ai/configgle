"""Tests for ``configgle.stackprint``."""

from __future__ import annotations

from collections import (
    ChainMap,
    Counter,
    OrderedDict,
    UserDict,
    UserList,
    UserString,
    defaultdict,
    deque,
)
from dataclasses import dataclass, field
from pprint import PrettyPrinter
from types import MappingProxyType, SimpleNamespace
from typing import TypedDict

import io

import pytest

from configgle.lib.testing.resources import recursion_headroom
from configgle.stackprint import StackPrinter, plain_repr


@dataclass(kw_only=True, slots=True)
class Point:
    x: int = 0
    y: list[int] = field(default_factory=list[int])


@dataclass(kw_only=True, slots=True)
class Link:
    next: Link | None = None
    label: str = "link"


@dataclass(kw_only=True, slots=True)
class Holder:
    item: object = None


def _cyclic_list() -> list[object]:
    value: list[object] = [1]
    value.append(value)
    return value


def _cyclic_dict() -> dict[str, object]:
    value: dict[str, object] = {"n": 1}
    value["self"] = value
    return value


def _chain_without_maps() -> ChainMap[str, int]:
    chain = ChainMap[str, int]()
    chain.maps = []
    return chain


def _cyclic_holder() -> Holder:
    holder = Holder()
    holder.item = [holder, {"again": holder}]
    return holder


_VALUES: list[object] = [
    0,
    -1_234_567,
    True,
    None,
    1.5,
    "",
    "short",
    "a long line of words " * 6,
    "first line\nsecond line that is rather long to wrap\nthird",
    b"",
    b"abcdefghijklmnopqrstuvwxyz" * 4,
    bytearray(b"abcdefghijklmnopqrstuvwxyz" * 3),
    [],
    [1, 2, 3],
    list(range(40)),
    [[1, 2], [3, [4, 5, [6, 7]]], "x" * 30],
    (),
    (1,),
    tuple(range(30)),
    {},
    {"b": 1, "a": [1, 2, 3] * 8},
    {(1, 2): {"k": "v" * 40}, 3: None},
    {1, 2, 3},
    set(range(30)),
    frozenset[int](),
    frozenset({"a", "b"}),
    frozenset(range(30)),
    OrderedDict[str, int](),
    OrderedDict(a=1, b=list(range(25))),
    defaultdict[str, list[int]](list),
    defaultdict(list, {"a": list(range(30))}),
    Counter[str](),
    Counter("abracadabra" * 8),
    ChainMap[str, int](),
    _chain_without_maps(),
    ChainMap[str, object]({"a": list(range(20))}, {"b": 2}),
    deque[int](),
    deque(range(30)),
    deque(range(30), maxlen=40),
    UserDict({"a": list(range(30))}),
    UserList(range(30)),
    UserString("u" * 100),
    MappingProxyType({"a": list(range(30))}),
    SimpleNamespace(),
    SimpleNamespace(a=1, b=list(range(30))),
    Point(),
    Point(x=10**9, y=list(range(30))),
    Link(next=Link(next=Link(label="end" * 20))),
    _cyclic_list(),
    _cyclic_dict(),
    _cyclic_holder(),
    {"same": [1, 2], "other": [[1, 2]] * 20},
]


class _Options(TypedDict, total=False):
    width: int
    indent: int
    depth: int
    compact: bool
    sort_dicts: bool
    underscore_numbers: bool


def _kind(value: object) -> str:
    return type(value).__name__


_OPTIONS: list[_Options] = [
    {},
    {"width": 1},
    {"width": 20},
    {"width": 40, "indent": 4},
    {"width": 30, "compact": True},
    {"depth": 1},
    {"depth": 2},
    {"sort_dicts": False, "underscore_numbers": True, "width": 50},
]


@pytest.mark.parametrize("options", _OPTIONS, ids=repr)
@pytest.mark.parametrize("value", _VALUES, ids=_kind)
def test_it_lays_out_what_pretty_printer_lays_out(
    value: object,
    options: _Options,
) -> None:
    expected = PrettyPrinter(**options).pformat(value)

    assert StackPrinter(**options).pformat(value) == expected


@pytest.mark.parametrize("width", range(4, 24))
@pytest.mark.parametrize(
    "value",
    [b"abcd", b"abcde", b"abcdefghijklm", ("one two three four five",), ([1, 2, 3],)],
    ids=repr,
)
def test_it_wraps_as_pretty_printer_at_every_width(value: object, width: int) -> None:
    """Every width puts a last chunk or item exactly on the boundary at least once."""
    expected = PrettyPrinter(width=width).pformat(value)

    assert StackPrinter(width=width).pformat(value) == expected


@pytest.mark.parametrize("width", range(8, 24))
def test_top_level_bytes_keep_the_callers_allowance(width: int) -> None:
    """A caller's trailing allowance adds to the parentheses' own, at every width."""
    data = b"abcdefghijklm"
    ours, theirs = io.StringIO(), io.StringIO()

    StackPrinter(width=width)._format(data, ours, 0, 3, {}, 0)
    PrettyPrinter(width=width)._format(data, theirs, 0, 3, {}, 0)

    assert ours.getvalue() == theirs.getvalue()


@pytest.mark.parametrize("value", _VALUES, ids=_kind)
def test_its_one_line_text_matches_pretty_printer(value: object) -> None:
    expected = PrettyPrinter().format(value, {}, 0, 0)

    assert StackPrinter().format(value, {}, 0, 0) == expected


@pytest.mark.parametrize("value", _VALUES, ids=_kind)
def test_plain_repr_matches_repr(value: object) -> None:
    assert plain_repr(value) == repr(value)


@pytest.mark.parametrize("value", _VALUES, ids=_kind)
def test_its_private_hooks_match_pretty_printers(value: object) -> None:
    """``_format`` writes and ``_repr`` returns what the base class's would."""
    ours, theirs = StackPrinter(width=20, depth=2), PrettyPrinter(width=20, depth=2)
    ours_stream, theirs_stream = io.StringIO(), io.StringIO()

    ours._format(value, ours_stream, 2, 1, {}, 0)
    theirs._format(value, theirs_stream, 2, 1, {}, 0)

    assert ours_stream.getvalue() == theirs_stream.getvalue()
    assert ours._repr(value, {}, 1) == theirs._repr(value, {}, 1)


def test_pprint_writes_the_layout_and_a_newline_to_its_stream() -> None:
    value = {"a": list(range(30))}
    ours, theirs = io.StringIO(), io.StringIO()

    StackPrinter(width=20, stream=ours).pprint(value)
    PrettyPrinter(width=20, stream=theirs).pprint(value)

    assert ours.getvalue() == theirs.getvalue()


def test_a_tree_deeper_than_the_headroom_lays_out() -> None:
    value: object = 0
    for _ in range(200):
        value = [value, "x" * 30]

    with recursion_headroom(100):
        text = StackPrinter(width=40).pformat(value)

    assert text.count("'" + "x" * 30 + "'") == 200


def test_a_dataclass_chain_deeper_than_the_headroom_reprs() -> None:
    value = Link()
    for _ in range(200):
        value = Link(next=value)

    with recursion_headroom(100):
        text = plain_repr(value)

    assert text.count("Link(") == 201


if __name__ == "__main__":
    from configgle.lib.testing.runner import test_main

    test_main(__file__)
