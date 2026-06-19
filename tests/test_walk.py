from __future__ import annotations

from dataclasses import field
from typing import NamedTuple, Self, cast, override

import pytest

from configgle.fig import Fig
from configgle.walk import (
    _get_object_attribute_names,
    copy_tree,
)


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
    # This protects against the bug where integer indices would be stringified
    lst = [1, 2, 3]
    names = list(_get_object_attribute_names(lst))
    assert names == []


def test_get_object_attribute_names_with_string_slots():
    """Test _get_object_attribute_names with __slots__ as a string."""

    class StringSlots:
        __slots__ = "value"  # noqa: PLC0205  # intentionally string for branch test

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
        __slots__ = ("__dict__", "slotted")  # __dict__ in slots enables both

        def __init__(self):
            self.slotted = 1
            self.dynamic = 2  # lands in __dict__

    obj = Both()
    names = list(_get_object_attribute_names(obj))
    assert set(names) == {"slotted", "dynamic"}
    assert len(names) == len(set(names))  # no duplicates


def test_get_object_attribute_names_yields_unset_slot():
    """A declared-but-unset slot is still yielded (the caller guards getattr).

    ``_get_object_attribute_names`` reports declared field names; whether each is
    currently set is the caller's concern (it wraps ``getattr`` in try/except).
    """

    class HasUnset:
        # ``unset_attr`` is declared but never assigned -- exactly what this test
        # exercises (the name is yielded though the slot is empty).
        __slots__ = ("set_attr", "unset_attr")  # pyright: ignore[reportUninitializedInstanceVariable] -- unset slot is the test subject

        def __init__(self):
            self.set_attr = 1

    obj = HasUnset()
    assert set(_get_object_attribute_names(obj)) == {"set_attr", "unset_attr"}


def test_get_object_attribute_names_skips_internal_attrs():
    """``_finalized`` is never yielded, even when a real attribute (a ``__dict__``
    entry or a slot) by that name exists.
    """

    class Internal:
        def __init__(self):
            self.real = 1
            self._finalized = True  # bookkeeping, must not be yielded

    obj = Internal()
    names = set(_get_object_attribute_names(obj))
    assert "real" in names
    assert "_finalized" not in names


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
            default_factory=lambda: [_Child.Config(), _Child.Config()]
        )
        mapping: dict[str, _Child.Config] = field(
            default_factory=lambda: {"a": _Child.Config()}
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
    assert copied.nums is not cfg.nums  # the list itself is fresh
    copied.nums.append(4)
    assert cfg.nums == [1, 2, 3]  # original list unaffected


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
    assert copied.inner is not inner  # the Fig inside was copied


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
    assert copied is not block  # rebuilt because an element was copied
    assert copied[0] is not block[0]
    copied[0].v = 9
    assert block[0].v == 0  # original Fig untouched


def test_copy_tree_preserves_namedtuple_type():
    """A namedtuple member is reconstructed as the same namedtuple type."""

    class Pair(NamedTuple):
        a: int
        b: int

    pair = Pair(1, 2)
    copied = copy_tree(pair)
    assert isinstance(copied, Pair)
    assert copied == Pair(1, 2)


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
    assert sentinel == ["called"]  # the override ran
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
    assert copied.a is copied.b  # identity preserved across the copy
    assert copied.a is not shared  # but it is a fresh copy


def test_copy_tree_handles_cycles():
    """A cyclic reference does not cause infinite recursion."""

    class Node(Fig, slots=False):
        peer: object = None

    a = Node()
    b = Node()
    a.peer = b
    b.peer = a  # cycle: a -> b -> a
    copied = copy_tree(a)
    assert copied is not a
    peer = cast(Node, copied.peer)
    assert peer is not b
    assert cast(Node, peer.peer) is copied  # cycle re-closed onto the copy


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


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
