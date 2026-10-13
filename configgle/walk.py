"""Structural tree walks for the config lifecycle: copy, finalize, make, find.

Each walk is ``tree``'s ``tree_map``, ``traverse``, or ``walk`` with a
config-specific visit, so every one walks the same shapes (Figs, slotted
objects, lists, dicts, sets, tuples) with no recursion:

- ``copy_tree`` -- a "semi-deep" copy: every nested config and mutable container
  is duplicated while leaf values (primitives, tensors, loggers) are aliased.
- ``finalize_value`` -- finalizes every nested ``Finalizeable``, in place.
- ``make_value`` -- builds every nested ``Makeable``.
- ``traverse`` -- finds every node of a type, with the handle to replace it.

These functions reference no concrete config class; they dispatch on the
``Finalizeable`` and ``Makeable`` protocols, so ``fig`` imports this module and
this module never imports back up.
"""

from __future__ import annotations

from collections.abc import (
    Iterable,
    Iterator,
    Mapping,
    Sequence,
    Set as AbstractSet,
)
from dataclasses import dataclass
from functools import partial
from types import FunctionType, ModuleType
from typing import Protocol, TypeGuard, cast

from treekle.absent import ABSENT

from configgle.custom_types import Finalizeable, LateBound, Makeable
from configgle.tree import (
    Match as TreeMatch,
    attributes,
    traverse as tree_traverse,
    tree_map,
    walk,
)


__all__ = [
    "Match",
    "bind_late",
    "copy_tree",
    "finalize_value",
    "make_value",
    "traverse",
]


def copy_tree[ValueT](
    value: ValueT,
    visited: dict[int, object] | None = None,
) -> ValueT:
    """Copy a config tree down through Figs/containers, aliasing leaf values.

    Every nested config (Fig, dataclass, or any object with its own
    ``__slots__``) and every mutable container is duplicated, while leaf values
    (primitives, tensors, loggers -- anything that is not a data container) are
    shared by reference. This is the copy a config needs before in-place
    mutation, without deep-copying heavy leaves like tensors. A tuple or
    frozenset is kept unless a part of it was copied.

    An object whose class overrides ``copy_tree`` controls how it is copied:
    the walk calls ``value.copy_tree(visited)`` for it. This is the dual of how
    a ``Fig`` overrides ``finalize``.

    Args:
      value: The config (or container/value) to copy.
      visited: Maps ``id(obj)`` to its copy, so a node reached twice -- shared
        by two fields, or through a cycle -- is copied once. Callers may omit
        it; an override threads it through ``super().copy_tree(visited)``.

    Returns:
      copied: A structural copy; nested configs/containers fresh, leaves aliased.

    """
    visited = {} if visited is None else visited
    own = _own_copy(value, root=None, visited=visited)
    return _copy_walk(value, visited) if own is ABSENT else cast(ValueT, own)


@dataclass(frozen=True, slots=True, kw_only=True)
class Match[ConfigT]:
    """One node ``traverse`` found, with the handle needed to swap it out.

    Attributes:
      fqn: Dotted path from the root, with ``[i]`` for sequence positions and
        ``[key]`` for mapping keys: ``blocks[0].attn``, ``extras['aux']``.
        Empty for the root itself.
      config: The matched node.
      parent: The container or object holding it; ``None`` at the root.
      attr: Where in ``parent`` it lives -- an attribute name, a sequence
        index, or a mapping key; ``None`` at the root.

    """

    fqn: str
    config: ConfigT
    parent: object | None
    attr: str | int | object | None

    def replace(self, new: object) -> None:
        """Write ``new`` where ``config`` sits in ``parent``.

        A tuple parent is rebuilt in ITS parent, since a tuple cannot be
        written into; the rebuilt tuple keeps the parent's own type.

        Args:
          new: The node to write in place of ``config``.

        """
        TreeMatch(
            fqn=self.fqn,
            node=self.config,
            parent=self.parent,
            attr=self.attr,
        ).replace(new)


def traverse[ConfigT](
    root: object,
    cls: type[ConfigT] | tuple[type[ConfigT], ...],
    *,
    recurse: bool = False,
) -> Iterator[Match[ConfigT]]:
    """Yield every node of type ``cls`` reachable from ``root``, with its slot.

    Walks the same shapes ``copy_tree`` walks -- Figs and other data objects,
    lists, tuples, dicts, sets -- and yields a :class:`Match` for each node that
    is an instance of ``cls``. A node is visited once even when two fields
    share it. Primitives and non-data objects (tensors, loggers) are leaves.

    Args:
      root: The config tree to search. Yielded first if it matches.
      cls: A type or tuple of types to match with ``isinstance``.
      recurse: Whether to keep descending into a matched node. Off, a matched
        subtree is one unit -- the shape a caller replacing whole nodes wants.
        On, nested matches are yielded too, parents before children.

    Yields:
      match: The node, its dotted path, and the handle to replace it.

    """
    for found in tree_traverse(root, cls, recurse=recurse):
        yield Match[ConfigT](
            fqn=found.fqn,
            config=found.node,
            parent=found.parent,
            attr=found.attr,
        )


def bind_late(root: object) -> None:
    """Call ``bind(root)`` on every ``LateBound`` reachable from ``root``, once.

    Walks the BUILT tree, not a config tree, so the shapes differ from
    ``copy_tree``: a torch module exposes its subtree through ``modules()``,
    and otherwise attributes and containers are followed as ``finalize_value``
    follows them. Every object the built tree holds is visited, since a
    ``LateBound`` may sit anywhere in it.

    Args:
      root: The outermost built object; what every ``bind`` receives.

    """
    parts = partial(_built_parts, torch_module_types={})
    for node in walk(root, parts=parts):
        if isinstance(node, LateBound):
            node.bind(root)


def make_value[ValueT](
    value: ValueT,
    made: dict[int, object] | None = None,
    making: set[int] | None = None,
) -> ValueT:
    """Materialize nested Makeables through lists, tuples, and mappings.

    Args:
      value: The tree whose Makeables to build.
      made: Maps ``id(config)`` to what it built, so a shared config builds once.
      making: Ids of configs being built, so a cycle raises instead of looping.

    Returns:
      materialized: ``value`` with every reachable Makeable replaced by its build.

    Raises:
      ValueError: A Makeable reaches itself through its arguments.

    """
    return tree_map(
        value,
        visit=partial(
            _made,
            made={} if made is None else made,
            making=set[int]() if making is None else making,
        ),
    )


# The whole tree was already duplicated by ``copy_tree`` before ``finalize`` ran, so
# finalizing in place is isolated from the caller's original. A child ``finalize`` may
# return a different object than it received; that result takes its place.
def finalize_value[ValueT](value: ValueT) -> ValueT:
    """Finalize every pending ``Finalizeable`` reachable through containers."""
    return tree_map(value, visit=_finalized, in_place=True)


class _ModuleTree(Protocol):
    """The part of ``torch.nn.Module`` that ``bind_late`` calls."""

    def modules(self) -> Iterator[object]: ...


# ``ModuleType`` has no parts: a module-valued attribute (e.g. the shared ``random`` a
# seedless RNG falls back to) otherwise opens the entire imported graph, where a
# third-party object's permissive ``__getattr__`` details the walk far from anything
# the caller built. Builtin containers come before the torch and ABC probes, which
# they can never satisfy; on a tree of millions of small lists those probes cost a
# quarter of the walk.
def _built_parts(
    node: object,
    *,
    torch_module_types: dict[type, bool],
) -> Iterable[object]:
    """Return the parts of a built object that ``bind_late`` walks into."""
    if isinstance(node, (list, tuple, set, frozenset)):
        return cast(Iterable[object], node)
    if isinstance(node, dict):
        return cast(dict[object, object], node).values()
    if isinstance(node, (ModuleType, FunctionType)):
        return ()
    if _is_torch_module(node, torch_module_types):
        return node.modules()
    if isinstance(node, Mapping):
        return cast(Mapping[object, object], node).values()
    if isinstance(node, (Sequence, AbstractSet)):
        return cast(Iterable[object], node)
    if (
        hasattr(type(node), "__dataclass_fields__")
        or "__slots__" in type(node).__dict__
        or type(node).__module__ != "builtins"
    ):
        # A user-defined object is walked as a data carrier; a builtin
        # (a random.Random, a logger's lock) is a leaf.
        return [part for _, part in attributes(node)]
    return ()


# A Finalizeable is its own finalization unit: its own ``finalize`` owns its fields, so
# the walk stops there, which also ends a cycle back to an already-finalized node.
def _finalized(node: object) -> object:
    """Return a ``Finalizeable`` finalized once, else ``ABSENT`` to walk on."""
    if not isinstance(node, Finalizeable):
        return ABSENT
    return node if getattr(node, "_finalized", None) else node.finalize()


# Recognized by import path, never by a ``modules`` method: duck typing called
# whatever ``modules`` a built object's class happened to define, and a catalog's
# ``modules(self, kind)`` made ``make()`` raise ``TypeError``. Naming the class also
# keeps torch out of configgle's imports; an instance cannot exist until torch is
# imported anyway.
def _is_torch_module(
    node: object,
    known: dict[type, bool],
) -> TypeGuard[_ModuleTree]:
    """Whether ``node`` is a ``torch.nn.Module``; ``known`` caches the answer by type."""
    cls = type(node)
    found = known.get(cls)
    if found is None:
        found = known[cls] = any(
            base.__module__ == "torch.nn.modules.module"
            and base.__qualname__ == "Module"
            for base in cls.__mro__
        )
    return found


def _copy_walk[ValueT](value: ValueT, visited: dict[int, object]) -> ValueT:
    """Copy ``value``'s tree, handing each overriding node to its own ``copy_tree``."""
    return tree_map(
        value,
        visit=partial(_own_copy, root=value, visited=visited),
        done=visited,
    )


# The root is walked, never handed back to its own ``copy_tree``: ``Maker.copy_tree``
# and every ``super().copy_tree(visited)`` land in ``_copy_walk`` on it. The default
# is walked inline too, so only an override costs a stack frame; ``Maker`` is named
# rather than imported because ``fig`` imports this module.
def _own_copy(node: object, *, root: object, visited: dict[int, object]) -> object:
    """Return ``node`` copied by its class's own ``copy_tree``, else ``ABSENT``."""
    own: object = getattr(type(node), "copy_tree", None)
    if (
        node is root
        or not callable(own)
        or (
            isinstance(own, FunctionType)
            and own.__module__ == "configgle.fig"
            and own.__qualname__ == "Maker.copy_tree"
        )
    ):
        return ABSENT
    return own(node, visited)


def _made(node: object, *, made: dict[int, object], making: set[int]) -> object:
    """Return a Makeable built once, else walk a list, tuple, or mapping."""
    if isinstance(node, Makeable):
        key = id(node)
        if key in made:
            return made[key]
        if key in making:
            raise ValueError("cannot materialize a cyclic Makeable graph")
        making.add(key)
        try:
            result = node.make()
        finally:
            making.remove(key)
        made[key] = result
        return result
    return ABSENT if isinstance(node, (list, tuple, Mapping)) else node
