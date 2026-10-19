"""Tests for ``configgle.tree``."""

from __future__ import annotations

from collections import defaultdict, deque
from dataclasses import dataclass, field
from types import MappingProxyType
from typing import TYPE_CHECKING, ClassVar, NamedTuple, cast

from treekle.absent import ABSENT

import pytest

from configgle.lib.testing.resources import recursion_headroom
from configgle.tree import (
    Work,
    children,
    drive,
    traverse,
    tree_map,
    walk,
    walk_into,
)


if TYPE_CHECKING:
    from collections.abc import Generator


@dataclass(kw_only=True, slots=True)
class Box:
    item: object = None


class _Leaf:
    """A plain object with no dataclass fields or own ``__slots__``: a leaf."""

    def __init__(self, tag: str) -> None:
        self.tag = tag


class Pair(NamedTuple):
    left: object
    right: object


class Bare:
    def __init__(self) -> None:
        self.item = [1]


class Unset:
    __slots__ = ("kept", "unset")

    def __init__(self) -> None:
        self.kept = 1
        self.unset = 2
        del self.unset


def _boxed(node: object) -> object:
    return "boxed" if isinstance(node, Box) else ABSENT


def _tens(node: object) -> object:
    return node * 10 if type(node) is int else ABSENT


class TestTreeMap:
    def test_the_default_visit_walks_into_every_node(self) -> None:
        assert walk_into(Box()) is ABSENT

    def test_a_list_deeper_than_the_recursion_limit_copies(self) -> None:
        depth = 200
        value: list[object] = [0]
        for _ in range(depth):
            value = [value]

        with recursion_headroom(100):
            rebuilt = tree_map(value)

        source, copied = value, rebuilt
        for _ in range(depth):
            assert copied is not source
            source = cast("list[object]", source[0])
            copied = cast("list[object]", copied[0])
        assert copied == [0]

    def test_a_data_object_chain_deeper_than_the_recursion_limit_copies(self) -> None:
        depth = 200
        value = Box()
        for _ in range(depth):
            value = Box(item=value)

        with recursion_headroom(100):
            rebuilt = tree_map(value)

        source: object = value
        for _ in range(depth):
            assert isinstance(source, Box)
            assert isinstance(rebuilt, Box)
            assert rebuilt is not source
            source, rebuilt = source.item, rebuilt.item

    def test_in_place_sets_changed_attributes_on_the_object_itself(self) -> None:
        value = Box(item=Box())

        rebuilt = tree_map(
            value,
            visit=lambda node: "x" if node is value.item else ABSENT,
            in_place=True,
        )

        assert rebuilt is value
        assert value.item == "x"

    def test_a_replaced_node_is_not_walked(self) -> None:
        seen: list[object] = []

        def visit(node: object) -> object:
            seen.append(node)
            return _boxed(node)

        rebuilt = tree_map([Box(item=[1])], visit=visit)

        assert rebuilt == ["boxed"]
        assert len(seen) == 2

    def test_a_shared_done_map_never_matches_a_freed_root(self) -> None:
        done: dict[int, object] = {}
        for index in range(50):
            root: tuple[int, dict[str, int]] = (index, {})

            mapped = tree_map(root, done=done)

            assert mapped == (index, {})

    def test_a_shared_part_stays_shared(self) -> None:
        shared = [1]

        rebuilt = tree_map([shared, shared])

        assert rebuilt[0] is rebuilt[1]
        assert rebuilt[0] is not shared

    def test_an_unchanged_tuple_is_kept_and_a_changed_one_rebuilt(self) -> None:
        unchanged = (1, "a")
        changed = Pair(left=[1], right=2)

        kept = tree_map(unchanged)
        rebuilt = tree_map(changed)

        assert kept is unchanged
        assert type(rebuilt) is Pair
        assert rebuilt == changed
        assert rebuilt.left is not changed.left

    def test_a_mapping_rebuilds_its_keys_and_values(self) -> None:
        source = {"a": [1]}

        rebuilt = tree_map(source, visit=lambda node: "b" if node == "a" else ABSENT)

        assert rebuilt == {"b": [1]}
        assert rebuilt["b"] is not source["a"]

    def test_a_set_rebuilds_its_members(self) -> None:
        source = {1, 2}

        rebuilt = tree_map(
            source,
            visit=lambda node: node * 10 if type(node) is int else ABSENT,
        )

        assert rebuilt == {10, 20}
        assert rebuilt is not source

    def test_a_cycle_through_a_data_object_closes_on_its_copy(self) -> None:
        value = Box()
        value.item = [value]

        rebuilt = tree_map(value)

        assert rebuilt.item == [rebuilt]
        assert cast("list[object]", rebuilt.item)[0] is rebuilt
        assert rebuilt is not value

    def test_a_cycle_through_a_list_closes_on_its_copy(self) -> None:
        value: list[object] = []
        value.append(value)

        rebuilt = tree_map(value)

        assert rebuilt[0] is rebuilt
        assert rebuilt is not value

    def test_a_cycle_through_a_tuple_closes_on_a_rebuilt_tuple(self) -> None:
        inner: list[object] = []
        value = (inner,)
        inner.append(value)

        rebuilt = tree_map(value)

        copied = rebuilt[0]
        assert copied is not inner
        assert cast("tuple[object, ...]", copied[0])[0] is copied

    def test_a_tuple_reached_twice_mid_build_is_rebuilt_each_time(self) -> None:
        first: list[object] = []
        second: list[object] = []
        value = (first, second)
        first.append(value)
        second.append(value)

        rebuilt = tree_map(value)

        copied = cast("tuple[list[tuple[object, ...]], ...]", rebuilt)
        assert copied[0][0][0] is copied[0]
        assert copied[0][0][1] is copied[1]

    def test_a_changed_frozenset_is_rebuilt_as_a_frozenset(self) -> None:
        rebuilt = tree_map(
            frozenset({1, 2}),
            visit=lambda node: node * 10 if type(node) is int else ABSENT,
        )

        assert type(rebuilt) is frozenset
        assert rebuilt == {10, 20}

    def test_a_read_only_mapping_is_rebuilt_as_its_own_type(self) -> None:
        source = MappingProxyType({"a": [1]})

        rebuilt = tree_map(source)

        assert type(rebuilt) is MappingProxyType
        assert dict(rebuilt) == {"a": [1]}
        assert rebuilt["a"] is not source["a"]

    def test_a_deque_is_rebuilt_as_a_deque(self) -> None:
        source = deque([[1]])

        rebuilt = tree_map(source)

        assert type(rebuilt) is deque
        assert rebuilt == source
        assert rebuilt[0] is not source[0]

    def test_a_dict_subclass_keeps_its_type_and_state(self) -> None:
        source = defaultdict(list[int], k=[1])

        rebuilt = tree_map(source)

        assert type(rebuilt) is defaultdict
        assert rebuilt.default_factory is source.default_factory
        assert rebuilt == source
        assert rebuilt["k"] is not source["k"]

    def test_an_object_without_fields_is_kept(self) -> None:
        value = Bare()

        assert tree_map([value])[0] is value

    def test_a_replaced_root_is_returned_as_is(self) -> None:
        assert tree_map(Box(), visit=_boxed) == "boxed"

    def test_a_node_mapped_during_its_own_visit_reads_as_that_result(self) -> None:
        inner = [1]
        done: dict[int, object] = {}

        def visit(node: object) -> object:
            if node is inner:
                tree_map(node, visit=_tens, done=done)
            return ABSENT

        rebuilt = tree_map([inner], visit=visit, done=done)

        assert rebuilt == [[10]]

    def test_a_deque_holding_itself_raises(self) -> None:
        value: deque[object] = deque()
        value.append(value)

        with pytest.raises(
            ValueError,
            match=r"^cannot walk a cycle through a deque$",
        ):
            tree_map(value)


class TestChildren:
    @pytest.mark.parametrize(
        ("value", "expected"),
        [
            ([5, 6], [(0, 5), (1, 6)]),
            ((5,), [(0, 5)]),
            ({"k": 5}, [("k", 5)]),
            ({5}, [(None, 5)]),
            (Box(item=5), [("item", 5)]),
            (Unset(), [("kept", 1)]),
            (Bare(), []),
            ("text", []),
            (5, []),
        ],
        ids=["list", "tuple", "dict", "set", "Box", "Unset", "Bare", "str", "int"],
    )
    def test_each_part_is_named_by_where_it_sits(
        self,
        value: object,
        expected: list[tuple[object, object]],
    ) -> None:
        assert list(children(value)) == expected


@dataclass(kw_only=True, slots=True)
class _TLeaf:
    width: int = 1


@dataclass(kw_only=True, slots=True)
class _TNorm:
    eps: float = 1e-6


@dataclass(kw_only=True, slots=True)
class _TBlock:
    proj: _TLeaf | _TNorm = field(default_factory=_TLeaf)
    norm: _TLeaf | _TNorm = field(default_factory=_TNorm)


@dataclass(kw_only=True, slots=True)
class _TStack:
    head: _TLeaf | _TNorm = field(default_factory=_TLeaf)
    blocks: list[_TBlock] = field(default_factory=list[_TBlock])
    extras: dict[str, object] = field(default_factory=dict[str, object])
    pair: tuple[object, object] = field(default_factory=lambda: (_TLeaf(), _TLeaf()))


def _stack() -> _TStack:
    stack = _TStack()
    stack.blocks = [_TBlock(), _TBlock()]
    stack.extras = {"aux": _TLeaf()}
    return stack


def test_traverse_yields_fqn_for_fields_lists_dicts_and_tuples() -> None:
    fqns = [m.fqn for m in traverse(_stack(), _TLeaf)]
    assert fqns == [
        "blocks[0].proj",
        "blocks[1].proj",
        "extras['aux']",
        "head",
        "pair[0]",
        "pair[1]",
    ]


def test_traverse_yields_the_root_when_it_matches() -> None:
    cfg = _stack()
    matches = list(traverse(cfg, _TStack))
    assert len(matches) == 1
    assert matches[0].fqn == ""
    assert matches[0].node is cfg
    assert matches[0].parent is None
    assert matches[0].attr is None


def test_traverse_stops_at_a_match_unless_recurse() -> None:
    """A matched subtree is one unit by default; ``recurse`` opens it."""
    cfg = _stack()
    shallow = [m.fqn for m in traverse(cfg, (_TBlock, _TLeaf))]
    assert shallow == [
        "blocks[0]",
        "blocks[1]",
        "extras['aux']",
        "head",
        "pair[0]",
        "pair[1]",
    ]
    deep = [m.fqn for m in traverse(cfg, (_TBlock, _TLeaf), recurse=True)]
    assert "blocks[0].proj" in deep
    assert deep.index("blocks[0]") < deep.index("blocks[0].proj")


def test_traverse_replace_rewrites_in_every_container_kind() -> None:
    cfg = _stack()
    for match in traverse(cfg, _TLeaf):
        match.replace(_TNorm(eps=0.5))
    assert isinstance(cfg.head, _TNorm)
    assert all(isinstance(b.proj, _TNorm) for b in cfg.blocks)
    assert isinstance(cfg.extras["aux"], _TNorm)
    assert all(isinstance(p, _TNorm) for p in cfg.pair)
    assert list(traverse(cfg, _TLeaf)) == []


def test_traverse_replace_rewrites_a_direct_list_element() -> None:
    """A match whose parent is a plain ``list`` is replaced by index."""
    cfg = _stack()
    for match in list(traverse(cfg, _TBlock)):
        match.replace(_TLeaf(width=9))
    assert all(isinstance(b, _TLeaf) for b in cfg.blocks)


def test_traverse_replace_rewrites_a_tuple_element_nested_in_a_list() -> None:
    """A tuple inside a list: its element's parent is a ``_TupleSlot`` whose
    OWN parent is the list.
    """

    @dataclass(kw_only=True, slots=True)
    class Holder:
        pairs: list[tuple[object, object]] = field(
            default_factory=lambda: [(_TLeaf(), _TLeaf())],
        )

    cfg = Holder()
    for match in list(traverse(cfg, _TLeaf)):
        match.replace(_TNorm(eps=0.5))
    assert isinstance(cfg.pairs[0][0], _TNorm)
    assert isinstance(cfg.pairs[0][1], _TNorm)


def test_traverse_replace_rebuilds_a_namedtuple_tuple_slot() -> None:
    @dataclass(kw_only=True, slots=True)
    class Holder:
        pair: Pair = field(default_factory=lambda: Pair(_TLeaf(), _TLeaf()))

    cfg = Holder()
    for match in list(traverse(cfg, _TLeaf)):
        match.replace(_TNorm(eps=0.5))
    assert isinstance(cfg.pair, Pair)
    assert all(isinstance(item, _TNorm) for item in cfg.pair)


def test_traverse_replace_rewrites_a_doubly_nested_tuple_element() -> None:
    """A tuple nested in another tuple: the inner slot's parent is a slot too."""

    @dataclass(kw_only=True, slots=True)
    class Holder:
        nested: tuple[tuple[object, object], ...] = field(
            default_factory=lambda: ((_TLeaf(), _TLeaf()),),
        )

    cfg = Holder()
    for match in list(traverse(cfg, _TLeaf)):
        match.replace(_TNorm(eps=0.5))
    assert isinstance(cfg.nested[0][0], _TNorm)
    assert isinstance(cfg.nested[0][1], _TNorm)


def test_traverse_replace_rewrites_a_tuple_element_held_in_a_mapping() -> None:
    holder: dict[str, tuple[object, object]] = {"pair": (_TLeaf(), _TLeaf())}

    for match in list(traverse(holder, _TLeaf)):
        match.replace(_TNorm())

    assert all(isinstance(item, _TNorm) for item in holder["pair"])


def test_traverse_replace_refuses_a_read_only_parent() -> None:
    (match,) = traverse(MappingProxyType({"k": _TLeaf()}), _TLeaf)

    with pytest.raises(
        TypeError,
        match=r"^cannot replace a part of a read-only mappingproxy$",
    ):
        match.replace(_TNorm())


def test_traverse_walks_into_a_set() -> None:
    """Set members are walked, though a matched leaf has no writable slot."""
    leaf = _Leaf("x")
    (match,) = traverse({leaf}, _Leaf)
    assert match.fqn == "{...}"
    assert match.node is leaf


def test_traverse_replaces_a_match_in_a_mutable_sequence() -> None:
    holder: deque[object] = deque([_TLeaf()])

    (match,) = traverse(holder, _TLeaf)
    match.replace(_TNorm())

    assert isinstance(holder[0], _TNorm)


def test_traverse_replaces_a_match_under_a_none_key() -> None:
    holder: dict[object, object] = {None: _TLeaf()}

    (match,) = traverse(holder, _TLeaf)
    match.replace(_TNorm())

    assert match.fqn == "[None]"
    assert isinstance(holder[None], _TNorm)


def test_traverse_skips_an_unset_slot_on_a_data_object() -> None:
    """A declared-but-unset slot on a walked node does not raise; it's skipped."""

    class Holder:
        __slots__ = ("set_attr", "unset_attr")

        def __init__(self, leaf: _Leaf) -> None:
            self.set_attr = leaf
            self.unset_attr = leaf
            del self.unset_attr

    leaf = _Leaf("x")
    holder = Holder(leaf)
    # Must not raise on the unset slot, and still finds the set one.
    (match,) = traverse(holder, _Leaf)
    assert match.node is leaf


def test_traverse_replace_on_root_raises() -> None:
    cfg = _stack()
    (root,) = traverse(cfg, _TStack)
    with pytest.raises(
        ValueError,
        match=r"^The root of a traversal has no parent to replace it in\.$",
    ):
        root.replace(_TStack())


def test_traverse_visits_a_shared_node_once() -> None:
    cfg = _stack()
    shared = _TLeaf()
    cfg.head = shared
    cfg.extras["aux"] = shared
    matched = [m.node for m in traverse(cfg, _TLeaf)]
    assert sum(m is shared for m in matched) == 1


def test_traverse_survives_a_cycle() -> None:
    a, b = Box(), Box()
    a.item, b.item = b, a
    assert [m.node for m in traverse(a, Box, recurse=True)] == [a, b]


def test_walk_yields_each_non_leaf_node_once() -> None:
    shared = Box(item=5)
    root = [shared, {"again": shared}]

    nodes = list(walk(root))

    assert nodes[0] is root
    assert sum(node is shared for node in nodes) == 1
    assert 5 not in nodes


def test_walk_follows_only_the_parts_its_caller_names() -> None:
    inner = Box()
    root = Box(item=inner)

    assert list(walk(root, parts=_no_parts)) == [root]
    assert list(walk(root)) == [root, inner]


def _no_parts(node: object) -> tuple[()]:
    del node
    return ()


def test_traverse_skips_leaves_that_are_not_data() -> None:
    cfg = _stack()
    cfg.head = _TLeaf(width=3)
    assert [m.fqn for m in traverse(cfg, int)] == []


def test_traverse_continues_after_unset_slot() -> None:
    class Holder:
        a_unset: object
        __slots__ = ("a_unset", "child")

        def __init__(self, child: _Leaf) -> None:
            self.child = child
            self.a_unset = None
            del self.a_unset

    leaf = _Leaf("target")
    holder = Holder(leaf)
    (match,) = traverse(holder, _Leaf)
    assert match.node is leaf


def test_traverse_walks_a_dataclass_without_slots() -> None:
    class Holder:
        __dataclass_fields__: ClassVar[dict[str, object]] = {}

        def __init__(self, child: _Leaf) -> None:
            self.child = child

    leaf = _Leaf("target")
    (match,) = traverse(Holder(leaf), _Leaf)
    assert match.node is leaf


def test_traverse_walks_a_dataclass_with_builtins_module_name() -> None:
    @dataclass(kw_only=True, slots=True)
    class Holder:
        __module__ = "builtins"
        child: _Leaf

    leaf = _Leaf("target")
    (match,) = traverse(Holder(child=leaf), _Leaf)
    assert match.node is leaf


def _depth(levels: int) -> Generator[Work, object, int]:
    if levels == 0:
        return 0
    below = cast(int, (yield _depth(levels - 1)))
    return below + 1


def _sum_of_depths(depths: list[int]) -> Generator[Work, object, int]:
    total = 0
    for levels in depths:
        total += cast(int, (yield _depth(levels)))
    return total


class TestDrive:
    def test_it_returns_what_the_work_returns(self) -> None:
        assert drive(_depth(0)) == 0

    def test_each_yielded_part_sends_its_result_back(self) -> None:
        assert drive(_sum_of_depths([1, 2, 3])) == 6

    def test_work_deeper_than_the_headroom_runs(self) -> None:
        with recursion_headroom(100):
            assert drive(_depth(200)) == 200


if __name__ == "__main__":
    from configgle.lib.testing.runner import test_main

    test_main(__file__)
