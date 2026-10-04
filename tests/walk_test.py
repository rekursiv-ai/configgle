from __future__ import annotations

from collections.abc import Iterator
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, ClassVar, NamedTuple, Self, cast, override

import collections
import random
import socket
import types

import pytest

from configgle.custom_types import LateBound
from configgle.fig import Fig
from configgle.walk import (
    _finalize_value,
    _get_object_attribute_names,
    _make_value,
    bind_late,
    copy_tree,
    traverse,
)


if TYPE_CHECKING:
    from collections.abc import Iterator


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
    # This protects against the bug where integer indices would be stringified.
    lst = [1, 2, 3]
    names = list(_get_object_attribute_names(lst))
    assert names == []


def test_get_object_attribute_names_with_string_slots():
    """Test _get_object_attribute_names with __slots__ as a string."""

    class StringSlots:
        __slots__ = "value"  # noqa: PLC0205 -- The string form is the branch-specific fixture under test.

        def __init__(self):
            self.value = 42

    obj = StringSlots()
    names = list(_get_object_attribute_names(obj))
    assert "value" in names


def test_get_object_attribute_names_inherited_slots_across_mro():
    """Slots declared on a base class are yielded for a subclass instance."""

    class Base:
        __slots__ = ("base_attr",)

        def __init__(self) -> None:
            self.base_attr = 1

    class Derived(Base):
        __slots__ = ("derived_attr",)

        @override
        def __init__(self) -> None:
            super().__init__()
            self.derived_attr = 2

    obj = Derived()
    assert set(_get_object_attribute_names(obj)) == {"base_attr", "derived_attr"}


def test_get_object_attribute_names_slots_and_dict_combined():
    """An object with both __slots__ and __dict__ yields the union, deduped."""

    class Both:
        __slots__ = ("__dict__", "slotted")  # __dict__ in slots enables both.

        def __init__(self):
            self.slotted = 1
            self.dynamic = 2  # Lands in __dict__.

    obj = Both()
    names = list(_get_object_attribute_names(obj))
    assert set(names) == {"slotted", "dynamic"}
    assert len(names) == len(set(names))  # No duplicates.


def test_get_object_attribute_names_yields_unset_slot():
    """A declared-but-unset slot is still yielded (the caller guards getattr).

    ``_get_object_attribute_names`` reports declared field names; whether each is
    currently set is the caller's concern (it wraps ``getattr`` in try/except).
    """

    class HasUnset:
        # ``unset_attr`` is declared but never assigned -- exactly what this test
        # exercises (the name is yielded though the slot is empty).
        __slots__ = ("set_attr", "unset_attr")  # pyright: ignore[reportUninitializedInstanceVariable] -- The unset slot is the test subject and is read only through the guarded walker.

        def __init__(self):
            self.set_attr = 1

    obj = HasUnset()
    assert set(_get_object_attribute_names(obj)) == {"set_attr", "unset_attr"}


def test_get_object_attribute_names_skips_internal_attrs():
    """``_finalized`` is never yielded, even when a real attribute.

    A ``__dict__`` entry or a slot) by that name exists.
    """

    class Internal:
        def __init__(self):
            self.real = 1
            self._finalized = True  # Bookkeeping, must not be yielded.

    obj = Internal()
    names = set(_get_object_attribute_names(obj))
    assert "real" in names
    assert "_finalized" not in names


def test_get_object_attribute_names_deduplicates_slot_and_dict_names():
    class Base:
        __slots__ = ("shared",)

        def __init__(self) -> None:
            self.shared = 1

    class Derived(Base):
        __slots__ = ("__dict__",)

    obj = Derived()
    vars(obj)["shared"] = 2
    names = list(_get_object_attribute_names(obj))
    assert names.count("shared") == 1


def test_get_object_attribute_names_empty_slots():
    """A class with empty ``__slots__`` and no ``__dict__`` yields nothing."""

    class Empty:
        __slots__ = ()

    assert list(_get_object_attribute_names(Empty())) == []


class _Leaf:
    """A non-config object (no dataclass fields, no own __slots__): a leaf."""

    def __init__(self, tag: str) -> None:
        self.tag = tag


class _Child:
    class Config(Fig["_Child"]):
        v: int = 0

    def __init__(self, config: Config) -> None:
        self.v = config.v


class _Parent:
    class Config(Fig["_Parent"]):
        child: _Child.Config = field(default_factory=_Child.Config)
        children: list[_Child.Config] = field(
            default_factory=lambda: [_Child.Config(), _Child.Config()],
        )
        mapping: dict[str, _Child.Config] = field(
            default_factory=lambda: {"a": _Child.Config()},
        )
        nums: list[int] = field(default_factory=lambda: [1, 2, 3])
        leaf: _Leaf = field(default_factory=lambda: _Leaf("shared"))
        scalar: int = 7

    def __init__(self, config: Config) -> None:
        del config


def test_copy_tree_copies_nested_fig():
    """A nested Fig is duplicated; the original is untouched by mutation."""
    cfg = _Parent.Config()
    copied = copy_tree(cfg)
    assert copied is not cfg
    assert copied.child is not cfg.child
    copied.child.v = 99
    assert cfg.child.v == 0


def test_copy_tree_copies_list_of_figs():
    """A list of Figs and its elements are all duplicated."""
    cfg = _Parent.Config()
    copied = copy_tree(cfg)
    assert copied.children is not cfg.children
    assert copied.children[0] is not cfg.children[0]
    copied.children[0].v = 5
    assert cfg.children[0].v == 0


def test_copy_tree_copies_dict_of_figs():
    """A dict of Figs and its values are duplicated."""
    cfg = _Parent.Config()
    copied = copy_tree(cfg)
    assert copied.mapping is not cfg.mapping
    assert copied.mapping["a"] is not cfg.mapping["a"]
    copied.mapping["a"].v = 5
    assert cfg.mapping["a"].v == 0


def test_copy_tree_copies_leaf_container_but_aliases_leaves():
    """A list of pure leaves is copied (mutable), its int elements shared."""
    cfg = _Parent.Config()
    copied = copy_tree(cfg)
    assert copied.nums is not cfg.nums  # The list itself is fresh.
    copied.nums.append(4)
    assert cfg.nums == [1, 2, 3]  # `original` list unaffected.


def test_copy_tree_aliases_non_config_leaves():
    """Non-config objects (no dataclass fields / own __slots__) are shared."""
    cfg = _Parent.Config()
    copied = copy_tree(cfg)
    # The leaf object is aliased by reference -- not copied.
    assert copied.leaf is cfg.leaf
    assert copied.scalar == 7


def test_copy_tree_recurses_slotted_objects():
    """A plain object with its own __slots__ holding a Fig is recursed into."""

    class Holder:
        __slots__ = ("inner",)

        def __init__(self, inner: _Child.Config) -> None:
            self.inner = inner

    inner = _Child.Config()
    holder = Holder(inner)
    copied = copy_tree(holder)
    assert copied is not holder
    assert copied.inner is not inner  # The Fig inside was copied.


def test_copy_tree_aliases_primitives():
    """Primitives and types pass through unchanged (shared)."""
    assert copy_tree(5) == 5
    assert copy_tree("x") == "x"
    assert copy_tree(None) is None
    assert copy_tree(int) is int


def test_copy_tree_copies_list_of_ints():
    """A ``list[int]`` is copied (fresh, mutable) with values preserved."""
    original = [1, 2, 3]
    copied = copy_tree(original)
    assert copied is not original
    assert copied == [1, 2, 3]
    copied.append(4)
    assert original == [1, 2, 3]


def test_copy_tree_empty_containers():
    """Empty list/dict/set copy to fresh, equal-but-distinct containers."""
    empty_list: list[int] = []
    assert copy_tree(empty_list) == []
    assert copy_tree(empty_list) is not empty_list
    empty_dict: dict[str, int] = {}
    assert copy_tree(empty_dict) is not empty_dict
    empty_set: set[int] = set()
    assert copy_tree(empty_set) is not empty_set


def test_copy_tree_empty_list_member_of_fig():
    """An empty-list member of a Fig is copied to a fresh, independent list."""

    class HasEmpty:
        class Config(Fig["HasEmpty"]):
            items: list[int] = field(default_factory=list[int])

        def __init__(self, config: Config) -> None:
            del config

    cfg = HasEmpty.Config()
    copied = copy_tree(cfg)
    assert copied.items is not cfg.items
    copied.items.append(1)
    assert cfg.items == []


def test_copy_tree_preserves_immutable_leaf_containers():
    """A tuple/frozenset of pure leaves is preserved (not needlessly rebuilt)."""
    leaf_tuple = (1, 2, 3)
    assert copy_tree(leaf_tuple) is leaf_tuple
    leaf_frozenset = frozenset({1, 2, 3})
    assert copy_tree(leaf_frozenset) is leaf_frozenset


def test_copy_tree_rebuilds_immutable_container_with_copied_fig():
    """A tuple of Figs is rebuilt to carry the copied (mutable) Fig elements."""
    block = (_Child.Config(), _Child.Config())
    copied = copy_tree(block)
    assert copied is not block  # Rebuilt because an element was copied.
    assert copied[0] is not block[0]
    copied[0].v = 9
    assert block[0].v == 0  # `original` Fig untouched.


def test_copy_tree_preserves_namedtuple_type():
    """A namedtuple member is reconstructed as the same namedtuple type."""

    class Pair(NamedTuple):
        a: int
        b: int

    pair = Pair(1, 2)
    copied = copy_tree(pair)
    assert copied == Pair(1, 2)


def test_copy_tree_rebuilds_namedtuple_when_an_element_is_copied():
    """A namedtuple holding a mutable element is reconstructed, not aliased."""

    class Pair(NamedTuple):
        a: list[int]
        b: int

    pair = Pair([1, 2], 3)
    copied = copy_tree(pair)
    assert type(copied) is Pair
    assert copied == Pair([1, 2], 3)
    assert copied.a is not pair.a  # The mutable element was actually copied.


def test_copy_tree_skips_an_unset_slot():
    """A declared-but-unset slot on a data object is left unset, not raised on."""

    class HasUnset:
        __slots__ = ("set_attr", "unset_attr")

        def __init__(self) -> None:
            self.set_attr = 1
            # Assigned then deleted, not left off entirely: the checker's
            # unset-slot analysis reads __init__'s assignments, not runtime
            # state, so a slot never assigned at all is a checker error
            # (reportUninitializedInstanceVariable) rather than the declared-
            # but-unset condition this test targets.
            self.unset_attr = 0
            del self.unset_attr

    copied = copy_tree(HasUnset())
    assert copied.set_attr == 1
    assert not hasattr(copied, "unset_attr")


def test_copy_tree_delegates_to_custom_method():
    """A config overriding ``copy_tree`` controls its own copy semantics."""
    sentinel: list[str] = []

    class Custom:
        class Config(Fig["Custom"]):
            x: int = 0

            @override
            def copy_tree(self, visited: dict[int, object] | None = None) -> Self:
                sentinel.append("called")
                return super().copy_tree(visited)

        def __init__(self, config: Config) -> None:
            del config

    cfg = Custom.Config()
    copied = copy_tree(cfg)
    assert sentinel == ["called"]  # The override ran.
    assert copied is not cfg


def test_copy_tree_preserves_dag_identity():
    """A sub-config shared by two fields stays shared in the copy (not split)."""

    class Leaf(Fig):
        v: int = 0

    class Root(Fig):
        a: Leaf = field(default_factory=Leaf)
        b: Leaf = field(default_factory=Leaf)

    shared = Leaf()
    root = Root(a=shared, b=shared)
    assert root.a is root.b
    copied = copy_tree(root)
    assert copied.a is copied.b  # Identity preserved across the copy.
    assert copied.a is not shared  # But it is a fresh copy.


def test_copy_tree_handles_cycles():
    """A cyclic reference does not cause infinite recursion."""

    class Node(Fig, slots=False):
        peer: object = None

    a = Node()
    b = Node()
    a.peer = b
    b.peer = a  # Cycle: a -> b -> a.
    copied = copy_tree(a)
    assert copied is not a
    peer = cast(Node, copied.peer)
    assert peer is not b
    assert cast(Node, peer.peer) is copied  # Cycle re-closed onto the copy.


def test_copy_tree_dag_in_list():
    """Shared identity is preserved when the same config appears twice in a list."""

    class Leaf(Fig):
        v: int = 0

    class Root(Fig):
        items: list[Leaf] = field(default_factory=list[Leaf])

    shared = Leaf()
    root = Root(items=[shared, shared])
    copied = copy_tree(root)
    assert copied.items[0] is copied.items[1]
    assert copied.items[0] is not shared


def test_bind_late_does_not_walk_into_an_imported_module():
    """A module-valued attribute is a leaf, not a subtree to search.

    ``random`` (or any module) reached from a built object exposes the whole
    imported graph; walking it hits third-party objects whose permissive
    ``__getattr__`` synthesizes a callable ``modules`` and hijacks the walk
    (``pytest.mark`` raised ``TypeError``; ``wandb``'s disabled shim raised
    ``KeyError``).
    """

    class Trap:
        """Stands in for ``pytest.mark``: any attribute read is a landmine."""

        def __getattr__(self, name: str) -> object:
            raise AssertionError(f"bind_late must not probe {name!r} here")

    module = types.ModuleType("fake_module_with_a_trap")
    # ``vars(module)[...]``, not ``module.trap = ...``: a fresh ``ModuleType``
    # declares no attributes, so the literal form is a static error.
    vars(module)["trap"] = Trap()

    class Holder:
        def __init__(self) -> None:
            self.rng = module

    bind_late(Holder())


def test_bind_late_does_not_probe_modules_through_instance_getattr():
    """``modules`` is a METHOD, so it is looked up on the class, never the instance.

    ``wandb``'s disabled shim is a ``dict`` subclass whose ``__getattr__``
    forwards to ``__getitem__``, so an instance-level probe raised ``KeyError``
    rather than reporting "no such method".
    """

    class Shim(dict[str, object]):
        def __getattr__(self, key: str) -> object:
            return self[key]  # Raises KeyError for any absent name.

    class Holder:
        def __init__(self) -> None:
            self.client = Shim()

    bind_late(Holder())


def test_bind_late_does_not_bind_a_foreign_object_that_merely_has_bind():
    """``LateBound`` is runtime-checkable, so it matches on the METHOD NAME alone.

    A live ``socket`` has ``bind(address)``; reached from a built tree, the walk
    called ``sock.bind(root)`` and raised ``TypeError: a bytes-like object is
    required``. Structural intent needs an explicit opt-in, not a name collision.
    """
    with socket.socket() as sock:

        class Holder:
            def __init__(self) -> None:
                self.transport = sock

        bind_late(Holder())  # Must not call sock.bind.


def test_bind_late_skips_an_unset_slot():
    """A declared-but-unset slot on a built object does not raise; it's skipped."""

    class Holder:
        __slots__ = ("set_attr", "unset_attr")

        def __init__(self) -> None:
            self.set_attr = 1
            self.unset_attr = 0
            del self.unset_attr

    bind_late(Holder())  # Must not raise despite the unset slot.


class _TorchModule:
    """Stands in for ``torch.nn.Module``, which ``bind_late`` knows by import path.

    configgle stays torch-free, so the walk names torch's class instead of
    importing it; to the walk, a class at that module and qualname is the real
    one.
    """


_TorchModule.__module__ = "torch.nn.modules.module"
_TorchModule.__qualname__ = "Module"


def test_bind_late_walks_a_torch_module_through_modules():
    """A torch module's subtree is what ``modules()`` yields."""
    seen: list[object] = []

    class Late(LateBound):
        @override
        def bind(self, root: object) -> None:
            seen.append(root)

    hidden = Late()  # No attribute holds it: only ``modules()`` reaches it.

    class Net(_TorchModule):
        def modules(self) -> Iterator[object]:
            yield self
            yield hidden

    root = Net()
    bind_late(root)
    assert seen == [root]


def test_bind_late_does_not_call_a_foreign_modules_method():
    """Only torch's module tree is walked through ``modules()``; others are data.

    A catalog whose ``modules(self, kind)`` was reachable from a built object
    was called with no arguments, and ``make()`` raised ``TypeError``. A
    zero-argument ``modules`` is no safer: it is still someone else's method.
    Both objects are walked as ordinary data carriers, so what they hold is
    still reached.
    """
    seen: list[object] = []

    class Late(LateBound):
        @override
        def bind(self, root: object) -> None:
            seen.append(root)

    class Catalog:
        def __init__(self) -> None:
            self.late = Late()

        def modules(self, kind: str) -> list[str]:
            return [kind]

    class Registry:
        def __init__(self) -> None:
            self.late = Late()

        def modules(self) -> list[str]:
            raise AssertionError("bind_late must not call Registry.modules")

    root = {"catalog": Catalog(), "registry": Registry()}
    bind_late(root)
    assert seen == [root, root]


def test_bind_late_reaches_late_bound_objects_in_every_container():
    """Builtin containers and other Mappings and Sequences are all searched."""
    bound: list[str] = []

    class Late(LateBound):
        def __init__(self, name: str) -> None:
            self.name = name

        @override
        def bind(self, root: object) -> None:
            bound.append(self.name)

    root = [
        (Late("tuple"),),
        {"key": [Late("dict")]},
        {Late("set")},
        frozenset({Late("frozenset")}),
        types.MappingProxyType({"key": Late("mapping")}),
        collections.deque([Late("sequence")]),
    ]
    bind_late(root)
    assert sorted(bound) == [
        "dict",
        "frozenset",
        "mapping",
        "sequence",
        "set",
        "tuple",
    ]


def test_bind_late_still_binds_through_ordinary_attributes():
    """The module skip must not cost the walk its normal reach."""
    seen: list[object] = []

    class Late(LateBound):
        @override
        def bind(self, root: object) -> None:
            seen.append(root)

    class Inner:
        def __init__(self) -> None:
            self.late = Late()
            self.rng = random  # `a` module sits beside the real subtree.

    class Root:
        def __init__(self) -> None:
            self.inner = Inner()

    root = Root()
    bind_late(root)
    assert seen == [root]


class _TLeaf:
    class Config(Fig["_TLeaf"]):
        width: int = 1

    def __init__(self, config: Config) -> None:
        del config


class _TNorm:
    class Config(Fig["_TNorm"]):
        eps: float = 1e-6

    def __init__(self, config: Config) -> None:
        del config


class _TBlock:
    class Config(Fig["_TBlock"]):
        proj: _TLeaf.Config = field(default_factory=_TLeaf.Config)
        norm: _TNorm.Config = field(default_factory=_TNorm.Config)

    def __init__(self, config: Config) -> None:
        del config


class _TStack:
    class Config(Fig["_TStack"]):
        head: _TLeaf.Config = field(default_factory=_TLeaf.Config)
        blocks: list[_TBlock.Config] = field(default_factory=list[_TBlock.Config])
        extras: dict[str, _TLeaf.Config] = field(
            default_factory=dict[str, _TLeaf.Config],
        )
        pair: tuple[_TLeaf.Config, _TLeaf.Config] = field(
            default_factory=lambda: (_TLeaf.Config(), _TLeaf.Config()),
        )

    def __init__(self, config: Config) -> None:
        del config


def _stack() -> _TStack.Config:
    cfg = _TStack.Config()
    cfg.blocks = [_TBlock.Config(), _TBlock.Config()]
    cfg.extras = {"aux": _TLeaf.Config()}
    return cfg


def test_traverse_yields_fqn_for_fields_lists_dicts_and_tuples() -> None:
    fqns = [m.fqn for m in traverse(_stack(), _TLeaf.Config)]
    assert fqns == [
        "head",
        "blocks[0].proj",
        "blocks[1].proj",
        "extras['aux']",
        "pair[0]",
        "pair[1]",
    ]


def test_traverse_yields_the_root_when_it_matches() -> None:
    cfg = _stack()
    matches = list(traverse(cfg, _TStack.Config))
    assert len(matches) == 1
    assert matches[0].fqn == ""
    assert matches[0].config is cfg
    assert matches[0].parent is None
    assert matches[0].attr is None


def test_traverse_stops_at_a_match_unless_recurse() -> None:
    """A matched subtree is one unit by default; ``recurse`` opens it."""
    cfg = _stack()
    shallow = [m.fqn for m in traverse(cfg, (_TBlock.Config, _TLeaf.Config))]
    assert shallow == [
        "head",
        "blocks[0]",
        "blocks[1]",
        "extras['aux']",
        "pair[0]",
        "pair[1]",
    ]
    deep = [m.fqn for m in traverse(cfg, (_TBlock.Config, _TLeaf.Config), recurse=True)]
    assert "blocks[0].proj" in deep
    assert deep.index("blocks[0]") < deep.index("blocks[0].proj")


def test_traverse_replace_rewrites_in_every_container_kind() -> None:
    cfg = _stack()
    for match in traverse(cfg, _TLeaf.Config):
        match.replace(_TNorm.Config(eps=0.5))
    assert isinstance(cfg.head, _TNorm.Config)
    assert all(isinstance(b.proj, _TNorm.Config) for b in cfg.blocks)
    assert isinstance(cfg.extras["aux"], _TNorm.Config)
    assert all(isinstance(p, _TNorm.Config) for p in cfg.pair)
    assert list(traverse(cfg, _TLeaf.Config)) == []


def test_traverse_replace_rewrites_a_direct_list_element() -> None:
    """A match whose parent is a plain ``list`` is replaced by index."""
    cfg = _stack()
    for match in list(traverse(cfg, _TBlock.Config)):
        match.replace(_TLeaf.Config(width=9))
    assert all(isinstance(b, _TLeaf.Config) for b in cfg.blocks)


def test_traverse_replace_rewrites_a_tuple_element_nested_in_a_list() -> None:
    """A tuple inside a list: its element's parent is a ``_TupleSlot`` whose
    OWN parent is the list.
    """

    class Holder:
        class Config(Fig["Holder"]):
            pairs: list[tuple[_TLeaf.Config, _TLeaf.Config]] = field(
                default_factory=lambda: [(_TLeaf.Config(), _TLeaf.Config())],
            )

        def __init__(self, config: Config) -> None:
            del config

    cfg = Holder.Config()
    for match in list(traverse(cfg, _TLeaf.Config)):
        match.replace(_TNorm.Config(eps=0.5))
    assert isinstance(cfg.pairs[0][0], _TNorm.Config)
    assert isinstance(cfg.pairs[0][1], _TNorm.Config)


def test_traverse_replace_rebuilds_a_namedtuple_tuple_slot() -> None:
    class Pair(NamedTuple):
        left: _TLeaf.Config
        right: _TLeaf.Config

    class Holder:
        class Config(Fig["Holder"]):
            pair: Pair = field(
                default_factory=lambda: Pair(_TLeaf.Config(), _TLeaf.Config()),
            )

        def __init__(self, config: Config) -> None:
            del config

    cfg = Holder.Config()
    for match in list(traverse(cfg, _TLeaf.Config)):
        match.replace(_TNorm.Config(eps=0.5))
    assert isinstance(cfg.pair, Pair)
    assert all(isinstance(item, _TNorm.Config) for item in cfg.pair)


def test_traverse_replace_rewrites_a_doubly_nested_tuple_element() -> None:
    """A tuple nested in another tuple: the inner slot's parent is a slot too."""

    class Holder:
        class Config(Fig["Holder"]):
            nested: tuple[tuple[_TLeaf.Config, _TLeaf.Config], ...] = field(
                default_factory=lambda: ((_TLeaf.Config(), _TLeaf.Config()),),
            )

        def __init__(self, config: Config) -> None:
            del config

    cfg = Holder.Config()
    for match in list(traverse(cfg, _TLeaf.Config)):
        match.replace(_TNorm.Config(eps=0.5))
    assert isinstance(cfg.nested[0][0], _TNorm.Config)
    assert isinstance(cfg.nested[0][1], _TNorm.Config)


def test_traverse_walks_into_a_set() -> None:
    """Set members are walked, though a matched leaf has no writable slot."""
    leaf = _Leaf("x")
    (match,) = traverse({leaf}, _Leaf)
    assert match.fqn == "{...}"
    assert match.config is leaf


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
    assert match.config is leaf


def test_traverse_replace_on_root_raises() -> None:
    cfg = _stack()
    (root,) = traverse(cfg, _TStack.Config)
    with pytest.raises(
        ValueError,
        match=r"^The root of a traversal has no parent to replace it in\.$",
    ):
        root.replace(_TStack.Config())


def test_traverse_visits_a_shared_node_once() -> None:
    cfg = _stack()
    shared = _TLeaf.Config()
    cfg.head = shared
    cfg.extras["aux"] = shared
    matched = [m.config for m in traverse(cfg, _TLeaf.Config)]
    assert sum(m is shared for m in matched) == 1


def test_traverse_survives_a_cycle() -> None:
    class _Node:
        class Config(Fig["_Node"]):
            next: object = None

        def __init__(self, config: Config) -> None:
            del config

    a, b = _Node.Config(), _Node.Config()
    a.next, b.next = b, a
    assert [m.config for m in traverse(a, _Node.Config, recurse=True)] == [a, b]


def test_traverse_skips_leaves_that_are_not_data() -> None:
    cfg = _stack()
    cfg.head.width = 3
    assert [m.fqn for m in traverse(cfg, int)] == []


def test_make_value_defaults_and_preserves_an_unchanged_tuple() -> None:
    """The top-level call supplies no ``made``/``making``; both default fresh."""
    result = _make_value((1, 2, 3))
    assert result == (1, 2, 3)
    assert type(result) is tuple


def test_make_value_rebuilds_a_namedtuple_of_makeables() -> None:
    class Pair(NamedTuple):
        a: object
        b: object

    pair = Pair(_Child.Config(v=3), _Child.Config(v=5))
    result = _make_value(pair)
    assert type(result) is Pair
    made_a = cast(_Child, result.a)
    made_b = cast(_Child, result.b)
    assert made_a.v == 3
    assert made_b.v == 5


class _Replacing:
    """A ``Finalizeable`` whose ``finalize`` returns a DIFFERENT instance."""

    def __init__(self, tag: str) -> None:
        self.tag = tag

    def finalize(self) -> _Replacing:
        return _Replacing(self.tag + "!")


def test_finalize_value_rebuilds_a_tuple_when_an_element_changes() -> None:
    original = (_Replacing("a"), 1)
    finalized = _finalize_value(original)
    assert finalized is not original
    assert finalized[0].tag == "a!"


def test_finalize_value_rebuilds_a_namedtuple_when_an_element_changes() -> None:
    class Pair(NamedTuple):
        a: object
        b: int

    pair = Pair(_Replacing("a"), 1)
    finalized = _finalize_value(pair)
    assert type(finalized) is Pair
    assert cast(_Replacing, finalized.a).tag == "a!"


def test_finalize_value_reassigns_a_changed_attribute_in_place() -> None:
    class Holder:
        __slots__ = ("child",)

        def __init__(self, child: _Replacing) -> None:
            self.child = child

    holder = Holder(_Replacing("a"))
    result = _finalize_value(holder)
    assert result is holder  # The object itself is finalized in place.
    assert holder.child.tag == "a!"


def test_copy_tree_preserves_shared_identity_in_immutable_containers() -> None:
    class Leaf(Fig, eq=False):
        value: int = 0

    shared = Leaf()
    copied = copy_tree((shared, shared))
    assert copied[0] is copied[1]
    assert copied[0] is not shared

    copied_set = copy_tree(frozenset({shared}))
    copied_member = next(iter(copied_set))
    assert copied_member is not shared


def test_copy_tree_copies_mapping_keys_and_set_members() -> None:
    class Leaf(Fig, eq=False):
        value: int = 0

    key = Leaf()
    source = {key: {key}}
    copied = copy_tree(source)
    copied_key = next(iter(copied))
    copied_member = next(iter(copied[copied_key]))
    assert copied_key is copied_member
    assert copied_key is not key
    assert copied_key.value == 0


def test_copy_tree_copies_dataclass_fields() -> None:
    @dataclass(kw_only=True, slots=True)
    class Holder:
        child: _Child.Config

    child = _Child.Config()
    copied = copy_tree(Holder(child=child))
    assert copied is not None
    assert copied.child is not child


def test_copy_tree_copies_a_dataclass_without_slots() -> None:
    class Holder:
        __dataclass_fields__: ClassVar[dict[str, object]] = {}

        def __init__(self, child: _Child.Config) -> None:
            self.child = child

    child = _Child.Config()
    copied = copy_tree(Holder(child))
    assert copied.child is not child


def test_finalize_value_skips_an_already_finalized_value() -> None:
    class Finalized:
        _finalized = True

        def finalize(self) -> Self:
            raise AssertionError("already finalized")

    value = Finalized()
    assert _finalize_value(value) is value


def test_finalize_value_preserves_an_in_place_tuple() -> None:
    class InPlace:
        def finalize(self) -> Self:
            return self

    value = (InPlace(), 1)
    assert _finalize_value(value) is value


def test_finalize_value_rebuilds_a_list_when_an_element_changes() -> None:
    class Replacing:
        def finalize(self) -> Self:
            return type(self)()

    value = [Replacing()]
    finalized = _finalize_value(value)
    assert finalized is not value
    assert isinstance(finalized[0], Replacing)
    assert finalized[0] is not value[0]


def test_finalize_value_handles_mapping_set_and_dataclass() -> None:
    class Replacing:
        def __init__(self, tag: str) -> None:
            self.tag = tag

        def finalize(self) -> Self:
            return type(self)(self.tag + "!")

    @dataclass(kw_only=True, slots=True)
    class Holder:
        child: Replacing

    source = {"item": Replacing("a")}
    finalized_mapping = _finalize_value(source)
    assert finalized_mapping["item"].tag == "a!"
    assert finalized_mapping is not source

    source_set = {Replacing("b")}
    finalized_set = _finalize_value(source_set)
    assert {item.tag for item in finalized_set} == {"b!"}
    assert finalized_set is not source_set

    holder = Holder(child=Replacing("c"))
    assert _finalize_value(holder) is holder
    assert holder.child.tag == "c!"

    class PlainHolder:
        __dataclass_fields__: ClassVar[dict[str, object]] = {}

        def __init__(self, child: Replacing) -> None:
            self.child = child

    plain_holder = PlainHolder(Replacing("d"))
    finalized_plain = _finalize_value(plain_holder)
    assert finalized_plain is plain_holder
    assert finalized_plain.child.tag == "d!"


def test_finalize_value_continues_after_unset_slot() -> None:
    class ReplacingForSlot:
        finalized: bool

        def __init__(self) -> None:
            self.finalized = False

        def finalize(self) -> Self:
            self.finalized = True
            return self

    class Holder:
        a_unset: object

        __slots__ = ("a_unset", "child")

        def __init__(self, child: ReplacingForSlot) -> None:
            self.child = child
            self.a_unset = None
            del self.a_unset

    holder = Holder(ReplacingForSlot())
    _finalize_value(holder)
    assert holder.child.finalized


def test_make_value_materializes_lists_mappings_and_shared_nodes() -> None:
    config = _Child.Config(v=8)
    result = _make_value([config, config])
    assert result[0] is result[1]
    assert result[0].v == 8

    mapped = _make_value({"first": config, "second": config})
    assert list(mapped) == ["first", "second"]
    assert mapped["first"] is mapped["second"]
    assert mapped["first"] is not config


def test_make_value_rebuilds_a_plain_tuple_of_makeables() -> None:
    config = _Child.Config(v=4)
    result = _make_value((config,))
    assert type(result) is tuple
    assert result[0].v == 4
    assert result[0] is not config


def test_make_value_reuses_an_existing_tuple_cache_entry() -> None:
    config = _Child.Config(v=5)
    cached = object()
    result = _make_value((config,), made={id(config): cached})
    assert result == (cached,)


def test_make_value_reuses_a_made_mapping_key_and_value() -> None:
    class Hashable:
        class Config(Fig["Hashable"], eq=False):
            value: int = 3

        def __init__(self, config: Config) -> None:
            self.value = config.value

    original = Hashable.Config()
    result = _make_value({original: original})
    made_key = next(iter(result))
    assert made_key is result[made_key]
    assert made_key is not original


def test_make_value_propagates_cycle_detection_through_containers() -> None:
    config = _Child.Config()

    class Hashable:
        class Config(Fig["Hashable"], eq=False):
            value: int = 3

        def __init__(self, config: Config) -> None:
            self.value = config.value

    key = Hashable.Config()
    for container in ([config], (config,), {"config": config}, {key: "value"}):
        with pytest.raises(
            ValueError,
            match=r"^cannot materialize a cyclic Makeable graph$",
        ):
            _make_value(container, making={id(config), id(key)})


def test_make_value_reports_makeable_cycles_exactly() -> None:
    config = _Child.Config()
    with pytest.raises(
        ValueError,
        match=r"^cannot materialize a cyclic Makeable graph$",
    ):
        _make_value(config, making={id(config)})


def test_bind_late_continues_after_unset_slot() -> None:
    seen: list[object] = []

    class Late(LateBound):
        @override
        def bind(self, root: object) -> None:
            seen.append(root)

    class Holder:
        a_unset: object
        __slots__ = ("a_unset", "child")

        def __init__(self) -> None:
            self.child = Late()
            self.a_unset = None
            del self.a_unset

    holder = Holder()
    bind_late(holder)
    assert seen == [holder]


def test_bind_late_walks_sequences() -> None:
    seen: list[object] = []

    class Late(LateBound):
        @override
        def bind(self, root: object) -> None:
            seen.append(root)

    root = [Late()]
    bind_late(root)
    assert seen == [root]


def test_bind_late_walks_a_dataclass_without_slots() -> None:
    seen: list[object] = []

    class Late(LateBound):
        @override
        def bind(self, root: object) -> None:
            seen.append(root)

    class Holder:
        __module__ = "builtins"
        __dataclass_fields__: ClassVar[dict[str, object]] = {}

        def __init__(self, child: Late) -> None:
            self.child = child

    root = Holder(Late())
    bind_late(root)
    assert seen == [root]


def test_bind_late_walks_dataclass_even_when_module_name_is_builtins() -> None:
    seen: list[object] = []

    class Late(LateBound):
        @override
        def bind(self, root: object) -> None:
            seen.append(root)

    @dataclass(kw_only=True, slots=True)
    class Holder:
        __module__ = "builtins"
        child: Late

    root = Holder(child=Late())
    bind_late(root)
    assert seen == [root]


def test_bind_late_walks_nonbuiltin_module_names_exactly() -> None:
    seen: list[object] = []

    class Late(LateBound):
        def __init__(self, label: str) -> None:
            self.label = label

        @override
        def bind(self, root: object) -> None:
            del root
            seen.append(self.label)

    class First:
        __module__ = "XXbuiltinsXX"

        def __init__(self) -> None:
            self.child = Late("first")

    class Second:
        __module__ = "BUILTINS"

        def __init__(self) -> None:
            self.child = Late("second")

    first, second = First(), Second()
    bind_late([first, second])
    assert seen == ["second", "first"]


def test_bind_late_walks_slots_even_when_module_name_is_builtins() -> None:
    seen: list[object] = []

    class Late(LateBound):
        @override
        def bind(self, root: object) -> None:
            seen.append(root)

    class Holder:
        __module__ = "builtins"
        __slots__ = ("child",)

        def __init__(self) -> None:
            self.child = Late()

    root = Holder()
    bind_late(root)
    assert seen == [root]


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
    assert match.config is leaf


def test_copy_tree_continues_after_unset_slot() -> None:
    class Holder:
        a_unset: object
        __slots__ = ("a_unset", "child")

        def __init__(self, child: _Child.Config) -> None:
            self.child = child
            self.a_unset = None
            del self.a_unset

    child = _Child.Config()
    holder = Holder(child)
    copied = copy_tree(holder)
    assert copied.child is not child


def test_traverse_walks_a_dataclass_without_slots() -> None:
    class Holder:
        __dataclass_fields__: ClassVar[dict[str, object]] = {}

        def __init__(self, child: _Leaf) -> None:
            self.child = child

    leaf = _Leaf("target")
    (match,) = traverse(Holder(leaf), _Leaf)
    assert match.config is leaf


def test_traverse_walks_a_dataclass_with_builtins_module_name() -> None:
    @dataclass(kw_only=True, slots=True)
    class Holder:
        __module__ = "builtins"
        child: _Leaf

    leaf = _Leaf("target")
    (match,) = traverse(Holder(child=leaf), _Leaf)
    assert match.config is leaf


if __name__ == "__main__":
    from configgle.lib.testing.main import test_main

    test_main(__file__)
