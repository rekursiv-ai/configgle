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
from typing import NamedTuple, Protocol, cast, override
from typing_extensions import TypedDict

import io

import pytest

from configgle.lib.testing.resources import recursion_headroom
from configgle.stackprint import ReprMemo, StackPrinter, plain_repr


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


@dataclass(kw_only=True, slots=True)
class Named:
    """A dataclass with its own ``__repr__``, which ``pprint`` never spreads."""

    text: str = "named " * 20

    @override
    def __repr__(self) -> str:
        return self.text


@dataclass(kw_only=True, repr=False, slots=True)
class Quiet(Point):
    """Inherits ``Point``'s generated repr but declares ``repr=False`` itself."""


class Opaque:
    @override
    def __repr__(self) -> str:
        return "opaque " * 20


class Blank:
    @override
    def __repr__(self) -> str:
        return ""


class Loud(int):
    @override
    def __repr__(self) -> str:
        return "loud"


class Shout(list[int]):
    @override
    def __repr__(self) -> str:
        return "shout"


class Pair(NamedTuple):
    left: object
    right: object


class Space(SimpleNamespace):
    pass


class Borrowed:
    """No dataclass, though its ``__repr__`` is one's generated repr."""

    __repr__ = Point.__repr__

    def __init__(self) -> None:
        self.x = 10**9
        self.y = list(range(30))


def _self_holder() -> Holder:
    holder = Holder()
    holder.item = holder
    return holder


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
    Named(),
    Quiet(x=10**9, y=list(range(30))),
    Opaque(),
    Blank(),
    object(),
    Loud(10**6),
    Shout([1, 2]),
    Pair(left=list(range(20)), right="x" * 30),
    Space(a=1, b=list(range(30))),
    Borrowed(),
    Point,
    _self_holder(),
    {frozenset({1}): 2, frozenset({3}): 1},
    {1, "a", None, (2,), 3.5, b"z"},
    {1j, 2j, 3j, 4j},
    Counter({(1, 2): 3, (3, 4): 1}),
    {"a": _cyclic_list(), "b": 1},
    {"b": 1, "a": 2},
    [bytearray(b"abcdefghijklmnopqrstuvwxyz" * 3)],
    deque([[i] for i in range(15)]),
    deque([[i] for i in range(15)], maxlen=20),
]


class _Options(TypedDict, total=False, closed=True):
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


@pytest.mark.parametrize(
    "options",
    [{"indent": 2.0}, {"width": 40.0}, {"compact": 1}],
    ids=repr,
)
def test_it_coerces_options_as_pretty_printer_does(
    options: dict[str, object],
) -> None:
    value = {"alpha": list(range(30)), "beta": {"gamma": "x" * 50}}
    expected = cast(_Printer, PrettyPrinter)(**options).pformat(value)

    assert cast(_Printer, StackPrinter)(**options).pformat(value) == expected


class _Printer(Protocol):
    def __call__(self, **options: object) -> PrettyPrinter: ...


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


def _as_flags(text: tuple[str, object, bool]) -> tuple[str, bool, bool]:
    """``pprint`` reports an empty repr unreadable as ``''``; the flag is its truth."""
    return text[0], bool(text[1]), text[2]


@pytest.mark.parametrize("value", _VALUES, ids=_kind)
def test_its_one_line_text_matches_pretty_printer(value: object) -> None:
    expected = _as_flags(PrettyPrinter().format(value, {}, 0, 0))

    assert StackPrinter().format(value, {}, 0, 0) == expected


@pytest.mark.parametrize("seen", [False, True])
@pytest.mark.parametrize(("maxlevels", "level"), [(1, 1), (2, 1), (2, 0)])
@pytest.mark.parametrize("value", _VALUES, ids=_kind)
def test_its_one_line_text_matches_pretty_printer_at_a_depth(
    value: object,
    maxlevels: int,
    level: int,
    *,
    seen: bool,
) -> None:
    """A depth cut, and a value already on the path, read as in ``pprint``."""
    context = {id(value): 1} if seen else {}
    expected = _as_flags(
        PrettyPrinter().format(value, dict(context), maxlevels, level),
    )

    assert StackPrinter().format(value, dict(context), maxlevels, level) == expected


def _swept() -> list[object]:
    """Small values of every layout, so a width sweep crosses each line's end."""
    return [
        "ab cd\nef gh",
        "ab cd ef",
        b"abcdefghij",
        bytearray(b"abcdefghij"),
        [1, [2, 3]],
        (1, (2, 3)),
        {"a": [1, 2], "b": 3},
        {1, 2, 3},
        frozenset({1, 2}),
        OrderedDict(a=[1, 2]),
        defaultdict(list, {"a": [1, 2]}),
        Counter("aab"),
        ChainMap[str, list[int]]({"a": [1]}, {"b": [2]}),
        deque([[1], [2]]),
        deque([[1], [2]], maxlen=3),
        MappingProxyType({"a": [1, 2]}),
        SimpleNamespace(a=[1], b=2),
        Point(x=1, y=[2, 3]),
        UserList([1, [2]]),
        # Each singleton tuple adds 2 to its last item's allowance, so a compact
        # last item meets an allowance larger than its line.
        (((((["ab", "cd"],),),),),),
        ((((((1, 2),),),),),),
        ((((("ab cd ef gh",),),),),),
        # Deep enough that the allowance outgrows a short string's line.
        ((((((("a b",),),),),),),),
        # A first line of three words breaks before its last one.
        "aaaa bb cc\nx",
        Counter({"alpha beta": 3, "gamma": 2, "delta epsilon zeta": 1}),
        # A Counter's last count is what its allowance bounds.
        Counter({"a": [1, 2, 3], "b": [4, 5]}),
    ]


@pytest.mark.parametrize("value", _swept(), ids=_kind)
def test_it_lays_out_as_pretty_printer_at_every_width_and_place(value: object) -> None:
    """Every width, indent, and position puts some line's end on the boundary."""
    mismatches: list[tuple[object, int, int, bool]] = []
    for placed in (value, [value, 1], (1, value), {"k": value}):
        for indent in (1, 2):
            for compact in (False, True):
                for width in range(1, 40):
                    options: _Options = {
                        "width": width,
                        "indent": indent,
                        "compact": compact,
                    }
                    expected = PrettyPrinter(**options).pformat(placed)
                    if StackPrinter(**options).pformat(placed) != expected:
                        mismatches.append((placed, width, indent, compact))

    assert mismatches == []


def test_keys_that_tie_keep_pretty_printers_order() -> None:
    """Two NaN keys are distinct yet neither sorts first; insertion order stands."""
    first, second = float("nan"), float("nan")
    value = {first: [2] * 30, second: [1] * 30}

    expected = PrettyPrinter(width=20).pformat(value)

    assert StackPrinter(width=20).pformat(value) == expected


def test_it_rejects_a_depth_as_pretty_printer_does() -> None:
    with pytest.raises(ValueError, match=r"^depth must be > 0$"):
        StackPrinter(depth=0)


class _Remembering(StackPrinter):
    """Hands one-line text a memo, and writes each leaf in angle brackets."""

    def __init__(self, memo: ReprMemo) -> None:
        super().__init__()
        self._reprs = memo

    @override
    def _leaf_repr(self, node: object) -> str:
        return f"<{node!r}>"


def test_a_subclass_memo_and_leaf_shape_the_one_line_text() -> None:
    point = Point(x=1, y=[2])
    printer = _Remembering({id(point.y): (point.y, "<remembered>")})

    assert printer.pformat(point) == "Point(x=<1>, y=<remembered>)"


@pytest.mark.parametrize(
    "value",
    [[1, [2]], {"k": [1]}, Point(x=1, y=[2])],
    ids=_kind,
)
def test_plain_repr_remembers_each_container_it_builds(value: object) -> None:
    memo: ReprMemo = {}

    text = plain_repr(value, memo=memo)

    pinned, remembered = memo[id(value)]
    assert pinned is value
    assert remembered == text


def test_a_memo_answers_for_the_root_itself() -> None:
    value = [1, 2]

    assert (
        plain_repr(value, memo={id(value): (value, "<remembered>")}) == "<remembered>"
    )


def test_a_memo_entry_pinned_to_another_object_is_not_used() -> None:
    value, other = [1, 2], [3]

    assert plain_repr(value, memo={id(value): (other, "<stale>")}) == "[1, 2]"


@pytest.mark.parametrize("value", _VALUES, ids=_kind)
def test_plain_repr_matches_repr(value: object) -> None:
    assert plain_repr(value) == repr(value)


@pytest.mark.parametrize("value", _VALUES, ids=_kind)
def test_plain_repr_with_a_shared_memo_matches_repr_twice(value: object) -> None:
    memo: ReprMemo = {}

    assert plain_repr(value, memo=memo) == repr(value)
    assert plain_repr([value, value], memo=memo) == repr([value, value])


def test_a_memo_reuses_a_finished_subtree_without_rewalking_it() -> None:
    inner = [1, [2]]
    memo: ReprMemo = {id(inner): (inner, "<remembered>")}

    assert plain_repr([inner, 3], memo=memo) == "[<remembered>, 3]"


def test_a_memo_keeps_no_text_written_around_a_recursion_marker() -> None:
    cyclic: list[object] = [1]
    cyclic.append(cyclic)
    memo: ReprMemo = {}

    assert plain_repr(cyclic, memo=memo) == repr(cyclic)
    assert id(cyclic) not in memo
    assert plain_repr([cyclic], memo=memo) == repr([cyclic])


def test_a_leaf_writes_each_node_it_does_not_open() -> None:
    @dataclass(kw_only=True, slots=True)
    class Pair:
        left: object
        right: object

    text = plain_repr(Pair(left=[1.5], right={"k": 2}), leaf=lambda node: f"<{node}>")

    assert text == f"{Pair.__qualname__}(left=[<1.5>], right={{<k>: <2>}})"


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
        # A bare list literal here never converges in pyrefly's inference.
        value = list[object]([value, "x" * 30])

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
