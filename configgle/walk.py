"""Structural tree walks for the config lifecycle: copy and finalize.

The pure structural recursion underlying ``Fig``'s lifecycle, kept apart from
the class hierarchy in ``fig.py``. Two walks live here:

- ``copy_tree`` -- a "semi-deep" copy: every nested config and mutable container
  is duplicated while leaf values (primitives, tensors, loggers) are aliased.
- ``_finalize_value`` -- finalizes every nested ``Finalizeable`` reachable
  through containers and data objects, in place.

The two mirror each other shape-for-shape (the same Figs/slotted objects/lists/
dicts/sets/tuples are walked); ``finalize`` is "``copy_tree`` then mutate".

These functions reference no concrete config class. They dispatch to user types
structurally -- ``getattr(type(value), "copy_tree")`` and
``isinstance(value, Finalizeable)`` -- so this module imports only
``custom_types`` and the stdlib. That keeps the dependency edge one-way
(``fig`` -> ``walk`` -> ``custom_types``): ``fig.py``'s methods call down into
these walks; this module never imports back up.
"""

from __future__ import annotations

from collections.abc import (
    Callable,
    Iterator,
    Mapping,
    Sequence,
    Set as AbstractSet,
)
from typing import cast

import copy

from configgle.custom_types import Finalizeable, Makeable


__all__ = [
    "copy_tree",
]


_SKIP_ATTRS = frozenset(("__weakref__", "__dict__", "_finalized"))


def _get_object_attribute_names(obj: object) -> Iterator[str]:
    """Yield attribute names, excluding __weakref__, __dict__, and _finalized."""
    seen = set[str]()
    if hasattr(type(obj), "__slots__"):
        for cls in type(obj).__mro__:
            # CPython accepts ANY iterable of identifiers for ``__slots__`` --
            # a list and a set are as legal as a tuple -- so narrowing to
            # ``str | tuple`` would silently drop every field of such a class.
            raw_slots = getattr(cls, "__slots__", ())
            slots = (
                (raw_slots,)
                if isinstance(raw_slots, str)
                else (str(s) for s in raw_slots)
            )
            for slot in slots:
                if slot not in seen and slot not in _SKIP_ATTRS:
                    seen.add(slot)
                    yield slot
    if hasattr(obj, "__dict__"):
        for key in sorted(vars(obj)):
            if key not in seen and key not in _SKIP_ATTRS:
                seen.add(key)
                yield key


def _copy_immutable_container(
    value: tuple[object, ...] | frozenset[object],
    visited: dict[int, object],
) -> object:
    """Copy a tuple/frozenset, preserving it when no element changed.

    An immutable container cannot be mutated in place, so it is rebuilt only to
    carry a freshly copied (mutable) element; otherwise the original is returned.
    The caller restores the precise type (the element type is erased at runtime).

    Args:
      value: The tuple or frozenset to copy.
      visited: Shared ``id(obj) -> copy`` map threaded through the recursion.

    Returns:
      copied: The original when every element is unchanged, else a rebuilt
        container holding the copied elements.

    """
    items: list[object] = list(value)
    copied: list[object] = [copy_tree(item, visited) for item in items]
    if all(c is o for c, o in zip(copied, items, strict=True)):
        return value  # nothing inside changed -- keep the immutable original
    if isinstance(value, frozenset):
        return frozenset(copied)
    if type(value) is tuple:
        return tuple(copied)
    # Namedtuple subclass: reconstructed by positional unpacking. Its field types
    # are erased at runtime, so neither checker can model ``type(value)(*copied)``.
    return type(value)(*copied)  # ty: ignore[invalid-argument-type]  # pyright: ignore[reportArgumentType] -- namedtuple field types are erased


def _copy_slots(value: object, visited: dict[int, object]) -> object:
    """Structurally copy one data object (dataclass / ``__slots__`` carrier).

    The terminal copy walk shared by both ``copy_tree`` entry points: the free
    function reaches it for plain data objects (the leg after leaves, custom
    ``copy_tree`` delegation, and containers), and ``Fig.copy_tree`` calls it as
    its base implementation. Keeping it here -- not inside ``Fig.copy_tree`` --
    is what lets the method be a thin dispatch target: the free function
    delegates to ``value.copy_tree(visited)`` for any object that defines one, so
    if ``Fig.copy_tree`` delegated back to the free function the two would
    recurse without bound. This helper is that base case.

    ``visited`` (required here -- the caller has already created it) maps
    ``id(obj)`` to its copy, so a node reached twice via a DAG or cycle is copied
    once and the shared/back reference is preserved.

    Args:
      value: The data object to shallow-copy and recurse into.
      visited: Shared ``id(obj) -> copy`` map threaded through the recursion.

    Returns:
      copied: A fresh ``copy.copy`` of ``value`` with nested configs/containers
        replaced by their copies and leaf values aliased.

    """
    cached = visited.get(id(value))
    if cached is not None:
        return cached
    r = copy.copy(value)
    visited[id(value)] = r
    for name in _get_object_attribute_names(r):
        try:
            attr_value = getattr(r, name)
        except AttributeError:
            # Declared-but-unset slot: nothing to copy.
            continue
        copied_attr = copy_tree(attr_value, visited)
        if copied_attr is not attr_value:
            object.__setattr__(r, name, copied_attr)
    return r


def copy_tree[ValueT](
    value: ValueT, visited: dict[int, object] | None = None
) -> ValueT:
    """Copy a config tree down through Figs/containers, aliasing leaf values.

    Returns a copy in which every nested config (Fig, dataclass, or any object
    with its own ``__slots__``) and every container holding such configs is
    duplicated, while leaf values (primitives, tensors, loggers -- anything that
    is not a data container) are shared by reference. This is the copy a config
    needs before in-place mutation: it isolates the structure that ``finalize``
    (or any edit) may touch, without the cost/incorrectness of deep-copying heavy
    leaves like tensors.

    An object that defines its own ``copy_tree`` (every ``Maker``, plus any user
    type) controls how it is copied: the free function delegates to
    ``value.copy_tree(visited)``. This is the dual of how a ``Fig`` overrides
    ``finalize`` to control how it is finalized. The default ``Maker.copy_tree``
    performs the same structural walk this function does for plain data objects.

    ``visited`` maps ``id(obj)`` to its copy, so a sub-config reached more than
    once -- shared by two fields (a DAG) or via a cycle -- is copied once and the
    shared/back reference is preserved.

    The traversal mirrors ``_finalize_value``'s -- the same shapes (Figs,
    slotted objects, lists, dicts including keys, sets, tuples) are walked, minus
    the ``finalize`` call -- so ``finalize`` is "copy_tree then mutate". Both
    preserve an immutable container (tuple/frozenset) whose elements are all
    unchanged.

    Args:
      value: The config (or container/value) to copy.
      visited: Maps ``id(obj)`` to its copy; supplied internally during
        recursion. Callers may omit it (a fresh map is created).

    Returns:
      copied: A structural copy; nested configs/containers fresh, leaves aliased.

    """
    if visited is None:
        visited = {}

    # Primitives and types: immutable leaves, shared as-is.
    if isinstance(value, (type, int, float, str, bytes, bool, type(None))):
        return value

    # Delegate to a custom/Maker ``copy_tree`` so the object controls its own
    # copy semantics (the dual of overriding ``finalize``). Checked after leaves
    # so primitives never hit attribute lookup.
    own = getattr(type(value), "copy_tree", None)
    if callable(own):
        return cast(ValueT, own(value, visited))

    # Immutable containers (tuple, frozenset): preserve the original unless a
    # mutable element inside it was copied. They cannot be mutated in place, so
    # the only reason to rebuild one is to carry a freshly copied element.
    if isinstance(value, (tuple, frozenset)):
        container = cast("tuple[object, ...] | frozenset[object]", value)
        return cast(ValueT, _copy_immutable_container(container, visited))

    # Mutable containers (list, dict, set): always copy so an in-place mutation
    # never reaches the original. Narrowed through ``object`` because
    # subtracting the classes excluded above from an unsolved ``ValueT``
    # empties the surviving ABC intersections.
    node = cast(object, value)
    if isinstance(node, Sequence) and not isinstance(node, (str, bytes)):
        copied = [copy_tree(v, visited) for v in node]
    elif isinstance(node, Mapping):
        # Bound here rather than off the narrowed union below: that union's
        # mapping arm carries an unsolved key type into ``type(...)``.
        mapping = cast(Mapping[object, object], node)
        rebuild_container = cast(Callable[[object], ValueT], type(mapping))
        return rebuild_container(
            {copy_tree(k, visited): copy_tree(v, visited) for k, v in mapping.items()}
        )
    elif isinstance(node, AbstractSet):
        copied = {copy_tree(v, visited) for v in node}
    else:
        # Only recurse into data containers (dataclasses or classes with their
        # own ``__slots__``). Plain objects without config data (loggers, file
        # handles, tensors) are leaves: aliased.
        obj_type = type(value)
        if not (
            hasattr(obj_type, "__dataclass_fields__")
            or "__slots__" in obj_type.__dict__
        ):
            return value

        return cast(ValueT, _copy_slots(value, visited))

    # ``value`` is narrowed to a Sequence/Mapping/AbstractSet union here, so
    # ``type(value)`` statically resolves to ``type[object]`` and the call reads
    # as ``object.__init__``. Name the real contract: a one-argument iterable
    # constructor whose element type is erased at runtime.
    ctor = cast(Callable[[object], ValueT], type(node))
    return ctor(copied)


def _finalize_value[ValueT](value: ValueT) -> ValueT:
    """Finalize nested Fig instances, recursing through containers and objects.

    Visits every nested ``Finalizeable`` reachable through sequences, mappings,
    and data objects (dataclasses or classes with their own ``__slots__``) and
    finalizes it. The whole tree was already duplicated by ``copy_tree`` before
    ``finalize`` ran, so the work here is isolated from the caller's original.

    A child ``finalize`` may return a *different* object than it received, so the
    returned value is threaded back: containers are rebuilt and object attrs are
    reassigned when an element changed identity.

    Args:
      value: Value to finalize recursively.

    Returns:
      finalized_value: The finalized value (a new object when a nested finalize
        produced one, else the same value mutated in place).

    """
    # A Finalizeable is its own finalization unit: finalize it if pending, else
    # it is already done. Either way return WITHOUT recursing into its fields --
    # its own ``finalize`` owns that, and stopping here terminates cyclic trees
    # (a back-edge to an already-finalized node returns instead of looping).
    if isinstance(value, Finalizeable):
        if not getattr(value, "_finalized", False):
            return value.finalize()
        return value

    # Classes, types, and primitives never need finalization.
    if isinstance(value, (type, int, float, str, bytes, bool, type(None))):
        return value

    node = cast(object, value)  # See ``copy_tree``.
    if isinstance(node, Sequence) and not isinstance(node, (str, bytes)):
        finalized_items: list[object] = [_finalize_value(v) for v in node]
        if isinstance(node, tuple):
            # Preserve the original tuple/namedtuple when no element changed
            # identity (an in-place finalize); rebuild only to carry a replaced
            # element. Matches ``_copy_immutable_container``.
            if all(f is o for f, o in zip(finalized_items, node, strict=True)):
                return value
            if type(node) is tuple:
                # One iterable argument for a plain tuple; a namedtuple takes
                # its fields positionally instead.
                return cast(Callable[[object], ValueT], tuple)(finalized_items)
            rebuild = cast(Callable[..., ValueT], type(node))
            return rebuild(*finalized_items)
        finalized = finalized_items
    elif isinstance(node, Mapping):
        # Bound here rather than off the narrowed union below: that union's
        # mapping arm carries an unsolved key type into ``type(...)``.
        mapping = cast(Mapping[object, object], node)
        rebuild_container = cast(Callable[[object], ValueT], type(mapping))
        finalized = {_finalize_value(k): _finalize_value(v) for k, v in mapping.items()}
        return rebuild_container(finalized)
    elif isinstance(node, AbstractSet):
        # An ``eq=False`` Fig is hashable and can be a set member, so its
        # elements are finalized and the set is rebuilt. (A default ``eq=True``
        # Fig is unhashable and cannot appear here.)
        finalized = {_finalize_value(v) for v in node}
    else:
        # Only recurse into data containers (dataclasses or classes with their
        # own __slots__). Skip plain objects like loggers, file handles, etc.
        obj_type = type(value)
        if not (
            hasattr(obj_type, "__dataclass_fields__")
            or "__slots__" in obj_type.__dict__
        ):
            return value
        for name in _get_object_attribute_names(value):
            try:
                attr_value = getattr(value, name)
            except AttributeError:
                continue
            finalized_attr = _finalize_value(attr_value)
            if finalized_attr is not attr_value:
                object.__setattr__(value, name, finalized_attr)
        return value

    # Reconstruct the container with the finalized items.
    # See ``copy_tree``: the narrowed union erases the concrete container type.
    ctor = cast(Callable[[object], ValueT], type(node))
    return ctor(finalized)


def _make_value[ValueT](
    value: ValueT,
    made: dict[int, object] | None = None,
    making: set[int] | None = None,
) -> ValueT:
    """Materialize nested Makeables through sequences and mappings."""
    if made is None:
        made = {}
    if making is None:
        making = set()

    if isinstance(value, Makeable):
        value_id = id(value)
        if value_id in made:
            return cast(ValueT, made[value_id])
        if value_id in making:
            raise ValueError("cannot materialize a cyclic Makeable graph")
        making.add(value_id)
        try:
            result = value.make()
        finally:
            making.remove(value_id)
        made[value_id] = result
        return cast(ValueT, result)

    if isinstance(value, (type, int, float, str, bytes, bool, type(None))):
        return value

    if isinstance(value, list):
        list_value = cast(list[object], value)
        materialized_items: list[object] = [
            _make_value(item, made, making) for item in list_value
        ]
        return cast(ValueT, materialized_items)
    if isinstance(value, tuple):
        tuple_value = cast(tuple[object, ...], value)
        materialized_items = [_make_value(item, made, making) for item in tuple_value]
        if all(
            materialized is original
            for materialized, original in zip(
                materialized_items,
                tuple_value,
                strict=True,
            )
        ):
            return cast(ValueT, tuple_value)
        if type(tuple_value) is tuple:
            return cast(ValueT, tuple(materialized_items))
        return cast(
            ValueT,
            type(tuple_value)(*materialized_items),  # ty: ignore[invalid-argument-type]  # pyright: ignore[reportArgumentType] -- namedtuple field types are erased at runtime
        )
    if isinstance(value, Mapping):
        materialized = {
            _make_value(key, made, making): _make_value(item, made, making)
            for key, item in cast(Mapping[object, object], value).items()
        }
        # See ``copy_tree``: the narrowed union erases the concrete mapping type.
        ctor = cast(Callable[[object], ValueT], type(cast(object, value)))
        return ctor(materialized)
    return value
