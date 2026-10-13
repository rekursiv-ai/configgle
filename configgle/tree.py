"""Walk, search, and map object graphs without recursion.

``children`` names a value's parts; ``walk`` yields every node once;
``traverse`` finds the nodes of a type, each with the handle to replace it;
``tree_map`` copies a graph part by part, letting a ``visit`` callback replace
any node; and ``drive`` runs work that yields the work of each part it needs,
for a walk whose result needs its parts' results. All treat the same shapes as parts: sequences, mappings, sets, and
data objects (dataclasses or classes with their own ``__slots__``). Anything
else is a leaf.
"""

from __future__ import annotations

from collections.abc import (
    Callable,
    Generator,
    Iterable,
    Iterator,
    Mapping,
    MutableMapping,
    MutableSequence,
    Sequence,
    Set as AbstractSet,
)
from dataclasses import dataclass
from typing import Final, cast

import copy
import operator

from treekle.absent import ABSENT
from treekle.reflect import attribute_names


__all__ = [
    "LEAVES",
    "Match",
    "Work",
    "attributes",
    "children",
    "drive",
    "traverse",
    "tree_map",
    "walk",
    "walk_into",
]


type Work = Generator[Work, object, object]
"""Work that yields the work of each part it needs, and gets back that result."""


LEAVES: Final = (type, int, float, complex, str, bytes, type(None))
"""Values with no parts, kept as they are."""


def children(value: object) -> Iterator[tuple[object, object]]:
    """Yield each part of ``value`` with where it sits.

    Args:
      value: The node whose parts to name.

    Yields:
      where: An index, a mapping key, an attribute name, or ``None`` for a set
        member.
      part: The part itself; an unset slot is skipped.

    """
    if isinstance(value, LEAVES):
        return
    if isinstance(value, Mapping):
        yield from cast("Mapping[object, object]", value).items()
    elif isinstance(value, Sequence):
        yield from enumerate(value)
    elif isinstance(value, AbstractSet):
        for member in value:
            yield None, member
    elif _holds_data(value):
        yield from attributes(value)


def attributes(value: object) -> Iterator[tuple[str, object]]:
    """Yield each set attribute of a data object with its name.

    Args:
      value: The object whose state to name.

    Yields:
      name: The attribute's name.
      part: Its value; an unset slot is skipped.

    """
    for name in attribute_names(value):
        part = getattr(value, name, _UNSET)
        if part is not _UNSET:
            yield name, part


def drive[T](work: Generator[Work, object, T]) -> T:
    """Run ``work`` and every part it waits on; return what ``work`` returns.

    ``work`` yields a part's work wherever plain recursion would call it, and
    receives that part's result back. Every pending piece waits on one explicit
    stack, so work as deep as its input never reaches the recursion limit.

    Args:
      work: The work to finish.

    Returns:
      result: ``work``'s return value.

    """
    stack: list[Work] = [work]
    sent: object = None
    while True:
        try:
            part = stack[-1].send(sent)
        except StopIteration as done:
            stack.pop()
            sent = cast(object, done.value)
            if not stack:
                return cast("T", sent)
            continue
        stack.append(part)
        sent = None


def walk_into(node: object) -> object:
    """Replace no node: ``tree_map``'s default visit, so it copies the graph.

    Args:
      node: The node reached.

    Returns:
      replacement: Always ``ABSENT``, so every node is walked into.

    """
    del node
    return ABSENT


def walk(
    root: object,
    *,
    parts: Callable[[object], Iterable[object]] | None = None,
) -> Iterator[object]:
    """Yield every node reachable from ``root`` once, leaves excluded.

    Args:
      root: Where the walk starts; yielded first unless it is a leaf.
      parts: Returns the parts of a node to walk into; ``None`` walks the parts
        ``children`` names. A caller with its own notion of a part, such as a
        module tree, passes it here.

    Yields:
      node: Each non-leaf node, the first time it is reached.

    """
    seen = set[int]()
    stack: list[object] = [root]
    while stack:
        node = stack.pop()
        if isinstance(node, LEAVES) or id(node) in seen:
            continue
        seen.add(id(node))
        yield node
        stack.extend(
            (part for _, part in children(node)) if parts is None else parts(node),
        )


@dataclass(frozen=True, slots=True, kw_only=True)
class Match[NodeT]:
    """One node ``traverse`` found, with the handle needed to swap it out.

    Attributes:
      fqn: Dotted path from the root, with ``[i]`` for sequence positions and
        ``[key]`` for mapping keys: ``blocks[0].attn``, ``extras['aux']``.
        Empty for the root itself.
      node: The matched node.
      parent: The container or object holding it; ``None`` at the root.
      attr: Where in ``parent`` it lives -- an attribute name, a sequence
        index, or a mapping key; ``None`` at the root.

    """

    fqn: str
    node: NodeT
    parent: object | None
    attr: str | int | object | None

    def replace(self, new: object) -> None:
        """Write ``new`` where ``node`` sits in ``parent``.

        A tuple parent is rebuilt in ITS parent, since a tuple cannot be
        written into; the rebuilt tuple keeps the parent's own type.

        Args:
          new: The node to write in its place.

        Raises:
          ValueError: The match is the root, which has no parent to write to.
          TypeError: The parent is a read-only container.

        """
        if self.parent is None:
            raise ValueError("The root of a traversal has no parent to replace it in.")
        _replace_at(self.parent, self.attr, new)


def traverse[NodeT](
    root: object,
    cls: type[NodeT] | tuple[type[NodeT], ...],
    *,
    recurse: bool = False,
) -> Iterator[Match[NodeT]]:
    """Yield every node of type ``cls`` reachable from ``root``, with its slot.

    Walks the parts ``children`` names and yields a :class:`Match` for each
    node that is an instance of ``cls``. A node is visited once even when two
    parents share it.

    Args:
      root: The graph to search. Yielded first if it matches.
      cls: A type or tuple of types to match with ``isinstance``.
      recurse: Whether to keep descending into a matched node. Off, a matched
        subtree is one unit -- the shape a caller replacing whole nodes wants.
        On, nested matches are yielded too, parents before children.

    Yields:
      match: The node, its dotted path, and the handle to replace it.

    """
    seen = set[int]()
    stack: list[Iterator[_Slot]] = [iter([("", None, None, root)])]
    while stack:
        slot = next(stack[-1], None)
        if slot is None:
            stack.pop()
            continue
        fqn, parent, attr, node = slot
        if isinstance(node, LEAVES):
            continue
        if id(node) in seen:
            continue
        seen.add(id(node))
        if isinstance(node, cls):
            yield Match(fqn=fqn, node=node, parent=parent, attr=attr)
            if not recurse:
                continue
        stack.append(_slots(node, fqn=fqn, parent=parent, attr=attr))


def tree_map[T](
    value: T,
    *,
    visit: Callable[[object], object] = walk_into,
    in_place: bool = False,
    done: dict[int, object] | None = None,
) -> T:
    """Map ``value``'s graph, replacing each node ``visit`` returns a value for.

    ``visit`` sees every node once, parents before their parts, and returns
    its replacement, or ``ABSENT`` to walk into it. A replaced node's parts are
    not walked. Mutable containers are always rebuilt; a tuple or frozenset is
    kept unless a part changed; a leaf is kept. A node reached twice reads as
    one result, so shared parts stay shared and a cycle closes on its copy. A
    cycle back to a tuple still being built rebuilds that tuple once more from
    the parts already copied. With the default ``visit`` this is a copy: every
    container and data object fresh, every leaf shared.

    Args:
      value: The root of the graph.
      visit: Returns a node's replacement, or ``ABSENT`` to walk into it.
      in_place: Set changed attributes on each data object itself rather than
        on a shallow copy of it.
      done: Maps ``id(node)`` to its result. Pass one map to several calls so
        they share each node's one result. The map also holds each node mapped,
        under ``~id(node)``, so its ``id`` cannot pass to a later object.

    Returns:
      mapped: The mapped root.

    Raises:
      ValueError: A cycle returns to a container built only after its parts with
        no other node copied in between, such as a deque holding itself.

    """
    state = _TreeMap(
        visit=visit,
        in_place=in_place,
        done={} if done is None else done,
    )
    root = state.enter(value)
    if not isinstance(root, _Frame):
        return cast("T", root)
    stack = [root]
    while True:
        frame = stack[-1]
        if frame.index < len(frame.parts):
            part = state.enter(frame.parts[frame.index])
            frame.index += 1
            if isinstance(part, _Frame):
                stack.append(part)
            else:
                frame.built.append(part)
            continue
        stack.pop()
        result = state.finish(frame)
        if not stack:
            return cast("T", result)
        stack[-1].built.append(result)


class _Frame:
    """A node whose parts are being rebuilt."""

    __slots__ = ("built", "first", "index", "names", "node", "parts", "target")

    def __init__(
        self,
        *,
        node: object,
        parts: list[object],
        target: object,
        names: tuple[str, ...] = (),
    ) -> None:
        self.node = node
        self.parts = parts
        self.target: object = target
        self.names = names
        self.built: list[object] = []
        self.index = 0
        self.first = False


class _TreeMap:
    """The state of one ``tree_map``: each node's result by ``id``.

    Attributes:
      visit: Returns a node's replacement, or ``ABSENT``.
      in_place: Whether data objects are changed rather than copied.
      done: Each finished or registered node's result, by ``id``.
      pending: ``len(done)`` when each tuple-like node still being built last
        opened, by ``id``.

    """

    __slots__ = ("done", "in_place", "pending", "visit")

    def __init__(
        self,
        *,
        visit: Callable[[object], object],
        in_place: bool,
        done: dict[int, object],
    ) -> None:
        self.visit = visit
        self.in_place = in_place
        self.done = done
        self.pending: dict[int, int] = {}

    def enter(self, node: object) -> object:
        """Return ``node``'s result, or a frame that rebuilds its parts.

        Args:
          node: The node reached.

        Returns:
          entered: Its result when already known or replaced, else its frame.

        """
        if isinstance(node, LEAVES):
            replaced = self.visit(node)
            return node if replaced is ABSENT else replaced
        key = id(node)
        if key in self.done:
            return self.done[key]
        if key in self.pending:
            return self._reopen(node)
        replaced = self.visit(node)
        if replaced is not ABSENT:
            self._record(node, replaced)
            return replaced
        if key in self.done:
            return self.done[key]
        frame = self._frame(node)
        if frame is None:
            self._record(node, node)
            return node
        if frame.target is _UNSET:
            self.pending[key] = len(self.done)
            frame.first = True
        else:
            self._record(node, frame.target)
        return frame

    def finish(self, frame: _Frame) -> object:
        """Build ``frame``'s node from its rebuilt parts and record it.

        Args:
          frame: A frame whose parts have all been rebuilt.

        Returns:
          built: The rebuilt node.

        """
        node, parts, built, target = frame.node, frame.parts, frame.built, frame.target
        if target is _UNSET:
            assembled = _assembled(node, parts, built)
            if frame.first:
                del self.pending[id(node)]
            self._record(node, assembled)
            return assembled
        if isinstance(target, list):
            cast("list[object]", target).extend(built)
        elif isinstance(target, set):
            cast("set[object]", target).update(built)
        elif isinstance(target, dict):
            cast("dict[object, object]", target).update(_pairs(built))
        else:
            for name, part, new in zip(frame.names, parts, built, strict=True):
                if new is not part:
                    object.__setattr__(target, name, new)
        return frame.target

    def _record(self, node: object, result: object) -> None:
        """Record ``result`` as ``node``'s, holding ``node`` alive in ``done``."""
        self.done[id(node)] = result
        self.done[~id(node)] = node

    def _reopen(self, node: object) -> _Frame:
        """Open a tuple-like ``node`` again for a cycle that reached it mid-build."""
        key = id(node)
        if self.pending[key] == len(self.done):
            raise ValueError(f"cannot walk a cycle through a {type(node).__name__}")
        self.pending[key] = len(self.done)
        return _built_after(node)

    def _frame(self, value: object) -> _Frame | None:
        """Return a frame for ``value``'s parts, else ``None`` for a leaf object."""
        node: object = value
        kind: type = type(node)
        if kind is list or kind is set:
            parts = list(cast("Iterable[object]", node))
            empty: object = list[object]() if kind is list else set[object]()
            return _Frame(node=node, parts=parts, target=empty)
        if isinstance(node, Mapping):
            mapping = cast("Mapping[object, object]", node)
            pairs: list[object] = [item for pair in mapping.items() for item in pair]
            return _Frame(node=value, parts=pairs, target=_emptied(mapping))
        if isinstance(node, (Sequence, AbstractSet)):
            return _built_after(node)
        if not _holds_data(node):
            return None
        names: list[str] = []
        parts: list[object] = []
        for name, part in attributes(node):
            names.append(name)
            parts.append(part)
        target = node if self.in_place else copy.copy(node)
        return _Frame(node=node, parts=parts, target=target, names=tuple(names))


def _replace_at(parent: object, attr: str | int | object | None, new: object) -> None:
    """Write ``new`` at ``attr`` in ``parent``: an index, a key, or an attribute."""
    if isinstance(parent, _TupleSlot):
        parent.write(cast(int, attr), new)
    elif isinstance(parent, MutableSequence):
        cast(MutableSequence[object], parent)[cast(int, attr)] = new
    elif isinstance(parent, MutableMapping):
        cast(MutableMapping[object, object], parent)[attr] = new
    elif isinstance(parent, (Sequence, Mapping)):
        kind: type = type(cast(object, parent))
        raise TypeError(f"cannot replace a part of a read-only {kind.__name__}")
    else:
        object.__setattr__(parent, cast(str, attr), new)


@dataclass(frozen=True, slots=True, kw_only=True)
class _TupleSlot:
    """A tuple standing in as ``parent`` so ``replace`` can rebuild it.

    Tuples are immutable, so a match inside one records the tuple's OWN
    position and rebuilds it there on write.
    """

    tuple_parent: object
    tuple_attr: str | int | object

    def current(self) -> tuple[object, ...]:
        """Read the tuple as its parent holds it now, after any earlier rebuild.

        Returns:
          held: The tuple currently stored at this slot.

        """
        parent, attr = self.tuple_parent, self.tuple_attr
        if isinstance(parent, Mapping):
            held = cast(Mapping[object, object], parent)[attr]
        elif isinstance(parent, Sequence):
            held = parent[cast(int, attr)]
        elif isinstance(parent, _TupleSlot):
            held = parent.current()[cast(int, attr)]
        else:
            held = cast(object, getattr(parent, cast(str, attr)))
        return cast(tuple[object, ...], held)

    def write(self, index: int, new: object) -> None:
        items = self.current()
        rebuilt = list(items)
        rebuilt[index] = new
        as_tuple = (
            tuple(rebuilt) if type(items) is tuple else type(items)(*rebuilt)  # ty: ignore[invalid-argument-type] -- Namedtuple field types are erased at runtime.  # pyright: ignore[reportArgumentType] -- Namedtuple field types are erased at runtime.
        )
        _replace_at(self.tuple_parent, self.tuple_attr, as_tuple)


type _Slot = tuple[str, object | None, object | None, object]
"""A node's dotted path, its parent, where it sits in the parent, and the node."""


def _slots(
    node: object,
    *,
    fqn: str,
    parent: object | None,
    attr: object | None,
) -> Iterator[_Slot]:
    """Yield the slot of each of ``node``'s parts, named as ``Match.fqn`` names it."""
    if isinstance(node, (Mapping, Sequence)):
        # A tuple cannot be written into, so its parts name the tuple's own slot.
        holder: object = (
            _TupleSlot(tuple_parent=parent, tuple_attr=attr)
            if isinstance(node, tuple)
            else cast(object, node)
        )
        for where, part in children(cast(object, node)):
            yield f"{fqn}[{where!r}]", holder, where, part
    elif isinstance(node, AbstractSet):
        for _, part in children(node):
            yield f"{fqn}{{...}}", None, None, part
    else:
        for name, part in attributes(node):
            yield f"{fqn}.{name}" if fqn else name, node, name, part


def _built_after(node: object) -> _Frame:
    """Return a frame for a container built only once its parts are."""
    return _Frame(node=node, parts=list(cast("Iterable[object]", node)), target=_UNSET)


# A dict subclass keeps its own state (a ``defaultdict``'s factory) through
# ``copy.copy``; its constructor need not take a mapping.
def _emptied(mapping: Mapping[object, object]) -> object:
    """Return an empty dict of ``mapping``'s class to fill, else ``_UNSET``."""
    if type(mapping) is dict:
        return dict[object, object]()
    if not isinstance(mapping, dict):
        return _UNSET
    emptied = copy.copy(mapping)
    emptied.clear()
    return emptied


def _assembled(node: object, parts: list[object], built: list[object]) -> object:
    """Return ``node`` rebuilt from ``built``; an unchanged immutable one as is."""
    kind: type = type(node)
    if issubclass(kind, (tuple, frozenset)) and all(map(operator.is_, built, parts)):
        return node
    if kind is tuple:
        return tuple(built)
    if isinstance(node, tuple):
        # A namedtuple takes its fields positionally.
        return cast("Callable[..., object]", kind)(*built)
    if isinstance(node, Mapping):
        return cast("Callable[[object], object]", kind)(dict(_pairs(built)))
    return cast("Callable[[object], object]", kind)(built)


def _pairs(flat: list[object]) -> Iterator[tuple[object, object]]:
    """Return a flat ``[key, value, ...]`` list's pairs."""
    return zip(flat[::2], flat[1::2], strict=True)


def _holds_data(value: object) -> bool:
    """Return whether ``value`` is a dataclass or has its own ``__slots__``."""
    kind = type(value)
    return hasattr(kind, "__dataclass_fields__") or "__slots__" in kind.__dict__


_UNSET: Final = object()
"""Marks an unset slot, or a node built only once its parts are."""
