"""Lossless line encodings of nested configuration trees.

``flatten`` turns a JSON-shaped tree, such as ``Fig.serialize()`` output, into
sorted ``(path, repr)`` leaves. ``encode`` shortens those leaves without losing
any: a subtree that nearly repeats an earlier sibling, or an earlier
same-typed node, becomes ``path = @other`` followed by only the leaves that
differ. ``@^^x`` climbs two levels from ``path`` before descending to ``x``.
Given a ``base``, ``encode`` keeps just the delta. ``decode`` inverts
``encode`` exactly. ``dumps`` and ``loads`` write the lines
as an indented outline that prints each shared path prefix once.

Paths use ``.name`` for identifier keys, ``[i]`` for list positions, and
``['key']`` for any other key. ``<absent>`` deletes a subtree.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from typing import Final, cast

import ast
import re


__all__ = [
    "ABSENT",
    "Entry",
    "decode",
    "dumps",
    "encode",
    "flatten",
    "loads",
]

ABSENT: Final = "<absent>"

type Entry = tuple[str, str]
type _Path = tuple[str | int, ...]
type _Tree = str | dict[str | int, _Tree]

_IDENTIFIER: Final = re.compile(r"[A-Za-z_][A-Za-z0-9_/]*")
_SEGMENT: Final = re.compile(
    r"(\.?)([A-Za-z_][A-Za-z0-9_/]*)"
    r"|\[(\d+)\]"
    r"|\[('(?:[^'\\]|\\.)*'|\"(?:[^\"\\]|\\.)*\")\]",
)


def flatten(tree: object) -> list[Entry]:
    """Return every leaf of a JSON-shaped tree as sorted ``(path, repr)`` pairs.

    Mappings need string keys; lists and tuples are indexed. An empty container
    is a leaf rendered ``{}`` or ``[]``, so no structure is lost.

    Args:
      tree: Nested mappings, sequences, and scalars.

    Returns:
      entries: Dotted paths with each leaf's ``repr``, in structural order.

    Raises:
      TypeError: A mapping key is not a string.

    """
    leaves: list[tuple[_Path, str]] = []
    _visit(tree, (), leaves)
    return [(_format(path), value) for path, value in leaves]


def encode(entries: Iterable[Entry], *, base: Iterable[Entry] = ()) -> list[Entry]:
    """Shorten flat entries into reference and override lines.

    Decoding starts from ``base`` (empty by default), and every subtree that
    differs from that state is written the cheapest way. It is either its own
    lines, or ``path = @other`` plus override lines, where ``other`` is an
    earlier sibling or an earlier node of the same ``py/object`` type. Overrides are chosen the same way, so references nest. Ties go
    to the earliest candidate, and a reference is used only when it is shorter.

    Args:
      entries: Flat entries, as returned by ``flatten``.
      base: Flat entries the result is stored against; empty for none.

    Returns:
      lines: Encoded lines that ``decode`` expands back into ``entries``.

    """
    target = _tree((_parse(path), value) for path, value in entries)
    start = _tree((_parse(path), value) for path, value in base)
    lines = _Encoder(target).lines((), start)
    return [(_format(path), value) for path, value in lines]


def decode(lines: Iterable[Entry], *, base: Iterable[Entry] = ()) -> list[Entry]:
    """Expand encoded lines back into the flat entries they encode.

    Args:
      lines: Lines produced by ``encode``.
      base: The ``base`` given to ``encode``; empty for none.

    Returns:
      entries: Flat entries in structural order.

    Raises:
      ValueError: A reference names a subtree that does not exist.

    """
    result = {_parse(path): value for path, value in base}
    for text, value in lines:
        path = _parse(text)
        if value.startswith("@"):
            source = _resolve(path, value)
            size = len(source)
            copied = {
                path + key[size:]: leaf
                for key, leaf in result.items()
                if len(key) > size and key[:size] == source
            }
            if not copied:
                raise ValueError(f"Reference {value} at {text} names no subtree.")
            _clear(result, path)
            result.update(copied)
        else:
            _clear(result, path)
            if value != ABSENT:
                result[path] = value
    return [(_format(path), result[path]) for path in sorted(result, key=_order)]


def dumps(lines: Iterable[Entry]) -> str:
    """Render lines as an outline that states each shared path prefix once.

    Args:
      lines: Encoded or flat lines in structural order.

    Returns:
      text: One-space-per-level outline; ``loads`` reverses it.

    """
    root = _Node()
    for text, value in lines:
        node = root
        for segment in _parse(text):
            node = node.children.setdefault(segment, _Node())
        node.value = value
    out: list[str] = []
    _render(root, indent=0, out=out)
    return "".join(f"{line}\n" for line in out)


def loads(text: str) -> list[Entry]:
    """Parse an outline written by ``dumps`` back into lines.

    Args:
      text: Outline text.

    Returns:
      lines: Full-path lines in outline order.

    Raises:
      ValueError: The outline is malformed; the message names the line.

    """
    lines: list[Entry] = []
    stack: list[_Open] = []
    for number, line in enumerate(text.splitlines(), start=1):
        body = line.lstrip(" ")
        indent = len(line) - len(body)
        while stack and stack[-1].indent >= indent:
            _close(stack.pop())
        if indent != len(stack):
            raise ValueError(f"Outline line {number}: unexpected indentation.")
        relative, rest = _split(body, number=number)
        if stack:
            stack[-1].children += 1
        path = (stack[-1].path if stack else ()) + relative
        if rest:
            lines.append((_format(path), rest))
        stack.append(_Open(indent=indent, path=path, line=number, header=not rest))
    for opened in stack:
        _close(opened)
    return lines


@dataclass(slots=True, kw_only=True)
class _Node:
    value: str | None = None
    children: dict[str | int, _Node] = field(default_factory=dict[str | int, "_Node"])


@dataclass(slots=True, kw_only=True)
class _Open:
    indent: int
    path: _Path
    line: int
    header: bool
    children: int = 0


def _visit(node: object, path: _Path, leaves: list[tuple[_Path, str]]) -> None:
    if isinstance(node, Mapping):
        mapping = cast(Mapping[object, object], node)
        items: dict[str, object] = {}
        for key, value in mapping.items():
            if not isinstance(key, str):
                raise TypeError(f"Mapping key {key!r} at {_format(path)!r} is not str.")
            items[key] = value
        if not items:
            leaves.append((path, "{}"))
        for key in sorted(items):
            _visit(items[key], (*path, key), leaves)
    elif isinstance(node, (list, tuple)):
        sequence = cast(Sequence[object], node)
        if not sequence:
            leaves.append((path, "[]"))
        for index, item in enumerate(sequence):
            _visit(item, (*path, index), leaves)
    else:
        leaves.append((path, repr(node)))


class _Encoder:
    """Choose, per subtree, the cheapest of its own lines or a reference."""

    def __init__(self, target: _Tree) -> None:
        self._fingerprints: dict[int, int] = {}
        self._interned: dict[object, int] = {}
        self._nodes: dict[_Path, _Tree] = {}
        self._candidates: dict[_Path, list[_Path]] = {}
        self._memo: dict[tuple[_Path, int], tuple[int, str | None]] = {}
        self._index(target, (), groups={})

    def lines(self, path: _Path, start: _Tree | None) -> list[tuple[_Path, str]]:
        """Return the lines turning decode state ``start`` into the target at ``path``.

        Args:
          path: Absolute path of the target subtree.
          start: What ``decode`` holds at ``path`` before these lines; None if empty.

        Returns:
          lines: Encoded lines, in structural order.

        """
        target = self._nodes[path]
        if start is not None and self._fingerprint(start) == self._fingerprint(target):
            return []
        if isinstance(target, str):
            return [(path, target)]
        _, choice = self._cost(path, start)
        if choice is None:
            return self._children(path, start)
        begin = None if choice == ABSENT else self._nodes[_resolve(path, choice)]
        return [(path, choice), *self._children(path, begin)]

    def _index(
        self,
        node: _Tree,
        path: _Path,
        *,
        groups: dict[tuple[object, ...], list[_Path]],
    ) -> None:
        self._nodes[path] = node
        if isinstance(node, str):
            return
        if path and node:
            keys: list[tuple[object, ...]] = [("sibling", path[:-1])]
            kind = node.get("py/object")
            if isinstance(kind, str):
                keys.append(("type", kind))
            self._candidates[path] = list(
                dict.fromkeys(
                    other
                    for key in keys
                    for other in groups.get(key, ())
                    if path[: len(other)] != other
                ),
            )
            for key in keys:
                groups.setdefault(key, []).append(path)
        for key, child in node.items():
            self._index(child, (*path, key), groups=groups)

    # The head is None to edit ``start`` in place, ``<absent>`` to clear it and rebuild,
    # or ``@other`` to copy an earlier subtree and override it.
    def _cost(self, path: _Path, start: _Tree | None) -> tuple[int, str | None]:
        """Return the cheapest line cost at ``path`` and the head line achieving it."""
        target = self._nodes[path]
        if start is not None and self._fingerprint(start) == self._fingerprint(target):
            return 0, None
        if isinstance(target, str):
            return _size(path, target), None
        memo_key = (path, -1 if start is None else self._fingerprint(start))
        if (cached := self._memo.get(memo_key)) is not None:
            return cached
        best: tuple[int, str | None] = (self._children_cost(path, start), None)
        heads = [_reference(path, other) for other in self._candidates.get(path, ())]
        if isinstance(start, dict):
            heads.insert(0, ABSENT)
        for head in heads:
            reference = _size(path, head)
            if reference >= best[0]:
                continue
            begin = None if head == ABSENT else self._nodes[_resolve(path, head)]
            cost = reference + self._children_cost(path, begin)
            if cost < best[0]:
                best = (cost, head)
        self._memo[memo_key] = best
        return best

    def _children_cost(self, path: _Path, start: _Tree | None) -> int:
        target = cast(dict[str | int, _Tree], self._nodes[path])
        state = start if isinstance(start, dict) else {}
        cost = sum(self._cost((*path, key), state.get(key))[0] for key in target)
        return cost + sum(
            _size((*path, key), ABSENT) for key in state if key not in target
        )

    def _children(self, path: _Path, start: _Tree | None) -> list[tuple[_Path, str]]:
        target = cast(dict[str | int, _Tree], self._nodes[path])
        state = start if isinstance(start, dict) else {}
        lines = [
            line for key in target for line in self.lines((*path, key), state.get(key))
        ]
        lines.extend(((*path, key), ABSENT) for key in state if key not in target)
        return lines

    def _fingerprint(self, node: _Tree) -> int:
        """Return a small integer equal for equal subtrees."""
        if isinstance(node, str):
            shape: object = node
        elif (cached := self._fingerprints.get(id(node))) is not None:
            return cached
        else:
            shape = tuple(
                (key, self._fingerprint(child)) for key, child in node.items()
            )
        number = self._interned.setdefault(shape, len(self._interned))
        if not isinstance(node, str):
            self._fingerprints[id(node)] = number
        return number


def _reference(path: _Path, other: _Path) -> str:
    """Spell ``other`` as seen from ``path``; each ``^`` climbs one level."""
    shared = 0
    while path[shared] == other[shared]:
        shared += 1
    relative = "@" + "^" * (len(path) - shared) + _format(other[shared:])
    absolute = f"@{_format(other)}"
    return relative if len(relative) < len(absolute) else absolute


# Without a leading ``^`` the reference is already absolute.
def _resolve(path: _Path, reference: str) -> _Path:
    """Return the absolute path a ``@`` reference written at ``path`` names."""
    text = reference[1:]
    body = text.lstrip("^")
    up = len(text) - len(body)
    if not up:
        return _parse(body)
    if up > len(path):
        raise ValueError(
            f"Reference {reference} at {_format(path)} climbs past the root.",
        )
    return path[: len(path) - up] + (_parse(body) if body else ())


def _tree(leaves: Iterable[tuple[_Path, str]]) -> _Tree:
    """Nest flat leaves back into a tree keyed by path segment."""
    root: dict[str | int, _Tree] = {}
    for path, value in leaves:
        node = root
        for segment in path[:-1]:
            child = node.setdefault(segment, {})
            assert isinstance(child, dict), f"{_format(path)} extends a leaf."
            node = child
        node[path[-1]] = value
    return root


def _clear(result: dict[_Path, str], path: _Path) -> None:
    """Drop every leaf at, above, or below ``path``."""
    size = len(path)
    for key in [key for key in result if key[:size] == path or path[: len(key)] == key]:
        del result[key]


def _size(path: _Path, value: str) -> int:
    return len(_format(path)) + len(value) + 4


def _order(path: _Path) -> tuple[tuple[int, str, int], ...]:
    return tuple(
        (1, "", segment) if isinstance(segment, int) else (0, segment, 0)
        for segment in path
    )


def _format(path: _Path) -> str:
    parts: list[str] = []
    for segment in path:
        if isinstance(segment, int):
            parts.append(f"[{segment}]")
        elif _IDENTIFIER.fullmatch(segment):
            parts.append(f".{segment}" if parts else segment)
        else:
            parts.append(f"[{segment!r}]")
    return "".join(parts)


def _parse(text: str) -> _Path:
    path, end = _segments(text)
    if end != len(text) or not path:
        raise ValueError(f"Malformed path {text!r}.")
    return path


def _segments(text: str) -> tuple[_Path, int]:
    """Parse the longest path prefix of ``text``; return it and where it ends."""
    path: list[str | int] = []
    position = 0
    while match := _SEGMENT.match(text, position):
        dot, name, index, quoted = match.groups()
        if name is not None:
            if bool(dot) != bool(path):
                break
            path.append(name)
        elif index is not None:
            path.append(int(index))
        else:
            path.append(cast(str, ast.literal_eval(quoted)))
        position = match.end()
    return tuple(path), position


def _split(body: str, *, number: int) -> tuple[_Path, str]:
    """Split one outline line into its relative path and value."""
    relative, end = _segments(body)
    rest = body[end:]
    if not relative or (rest and (not rest.startswith(" = ") or len(rest) == 3)):
        raise ValueError(f"Outline line {number}: expected 'path' or 'path = value'.")
    return relative, rest[3:]


def _close(opened: _Open) -> None:
    if opened.header and not opened.children:
        raise ValueError(f"Outline line {opened.line}: header has no children.")


def _render(node: _Node, *, indent: int, out: list[str]) -> None:
    for head, first in node.children.items():
        path: list[str | int] = [head]
        child = first
        while child.value is None and len(child.children) == 1:
            ((segment, child),) = child.children.items()
            path.append(segment)
        text = " " * indent + _format(tuple(path))
        out.append(text if child.value is None else f"{text} = {child.value}")
        _render(child, indent=indent + 1, out=out)
