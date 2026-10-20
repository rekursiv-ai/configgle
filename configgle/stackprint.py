"""A ``PrettyPrinter`` that lays out a tree of any depth without recursing.

``StackPrinter`` renders exactly what ``pprint.PrettyPrinter`` renders. Each
method that recursed into a part instead yields the part's own work, and
``tree.drive`` runs every pending piece of work on one explicit stack, so a
tree deeper than the recursion limit prints. Subclasses hook the generator methods:
``_text`` renders a value on one line, ``_block`` lays it out across lines.
"""

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
from pprint import PrettyPrinter
from types import MappingProxyType, SimpleNamespace
from typing import IO, TYPE_CHECKING, ClassVar, Final, Protocol, cast, override

import dataclasses
import functools
import re
import sys

from configgle.tree import Work, drive


if TYPE_CHECKING:
    from collections.abc import Callable, Generator, Iterator, Sequence

    from _typeshed import SupportsWrite


__all__ = [
    "StackPrinter",
    "plain_repr",
]


type _Text = tuple[str, bool, bool]
"""One-line text, whether it reads back with ``eval``, and whether it recursed."""


def plain_repr(value: object) -> str:
    """Return ``repr(value)``, walking builtin containers and dataclasses on a stack.

    Builtin ``repr`` recurses once per level, as a dataclass's generated
    ``__repr__`` does; every other type keeps its own ``__repr__``.

    Args:
      value: The value to represent.

    Returns:
      text: Exactly ``repr(value)``.

    """
    active: set[int] = set()
    root = _opened(value, active)
    if isinstance(root, str):
        return root
    stack = [root]
    while True:
        frame = stack[-1]
        if len(frame.parts) < len(frame.children):
            part = _opened(frame.children[len(frame.parts)], active)
            if isinstance(part, str):
                frame.parts.append(part)
            else:
                stack.append(part)
            continue
        text = frame.joined()
        active.discard(frame.ident)
        stack.pop()
        if not stack:
            return text
        stack[-1].parts.append(text)


class StackPrinter(PrettyPrinter):
    """``PrettyPrinter`` whose layout runs on an explicit stack, not the call stack."""

    def __init__(
        self,
        indent: int = 1,
        width: int = 80,
        depth: int | None = None,
        stream: IO[str] | None = None,
        *,
        compact: bool = False,
        sort_dicts: bool = True,
        underscore_numbers: bool = False,
    ) -> None:
        super().__init__(
            indent=indent,
            width=width,
            depth=depth,
            stream=stream,
            compact=compact,
            sort_dicts=sort_dicts,
            underscore_numbers=underscore_numbers,
        )
        # The base class sets these as well; typeshed does not declare them. Read
        # them back off the base rather than restating its ``int``/``bool``
        # coercion, so ``indent=2.0`` lays out exactly as ``PrettyPrinter``.
        self._indent_per_level: int = vars(self)["_indent_per_level"]
        self._width: int = vars(self)["_width"]
        self._depth = depth
        self._stream: IO[str] = sys.stdout if stream is None else stream
        self._compact: bool = vars(self)["_compact"]
        self._sort_dicts = sort_dicts
        self._underscore_numbers = underscore_numbers

    @override
    def pprint(self, object: object) -> None:
        self._stream.write(drive(self._block(object, 0, 0, {}, 0)))
        self._stream.write("\n")

    @override
    def pformat(self, object: object) -> str:
        return drive(self._block(object, 0, 0, {}, 0))

    @override
    def format(
        self,
        object: object,
        context: dict[int, int],
        maxlevels: int,
        level: int,
    ) -> tuple[str, bool, bool]:
        return drive(self._text(object, context, maxlevels, level))

    @override
    def _format(
        self,
        object: object,
        stream: SupportsWrite[str],
        indent: int,
        allowance: int,
        context: dict[int, int],
        level: int,
    ) -> None:
        stream.write(drive(self._block(object, indent, allowance, context, level)))

    @override
    def _repr(self, object: object, context: dict[int, int], level: int) -> str:
        return drive(self._repr_text(object, context, level))

    def _text(
        self,
        obj: object,
        context: dict[int, int],
        maxlevels: int,
        level: int,
    ) -> Generator[Work, object, _Text]:
        """Render ``obj`` on one line, as ``PrettyPrinter._safe_repr`` does."""
        kind = type(obj)
        if kind in _BUILTIN_SCALARS:
            return repr(obj), True, False
        own_repr = cast(object, kind.__repr__)
        if issubclass(kind, int) and own_repr is cast(object, int.__repr__):
            text = f"{obj:_d}" if self._underscore_numbers else repr(obj)
            return text, True, False
        if issubclass(kind, dict) and own_repr is cast(object, dict.__repr__):
            return (yield from self._dict_text(obj, context, maxlevels, level))
        if (issubclass(kind, list) and own_repr is cast(object, list.__repr__)) or (
            issubclass(kind, tuple) and own_repr is cast(object, tuple.__repr__)
        ):
            return (yield from self._sequence_text(obj, context, maxlevels, level))
        rep = plain_repr(obj)
        return rep, bool(rep) and not rep.startswith("<"), False

    def _dict_text(
        self,
        obj: object,
        context: dict[int, int],
        maxlevels: int,
        level: int,
    ) -> Generator[Work, object, _Text]:
        """Render a plain dict on one line, each key and value as its own part."""
        mapping = cast("dict[object, object]", obj)
        if not mapping:
            return "{}", True, False
        objid = id(obj)
        if maxlevels and level >= maxlevels:
            return "{...}", False, objid in context
        if objid in context:
            return _recursion(obj), False, True
        context[objid] = 1
        readable, recursive = True, False
        components: list[str] = []
        items = (
            sorted(mapping.items(), key=_safe_tuple)
            if self._sort_dicts
            else list(mapping.items())
        )
        for key, value in items:
            key_text = cast(
                _Text,
                (yield self._text(key, context, maxlevels, level + 1)),
            )
            value_text = cast(
                _Text,
                (yield self._text(value, context, maxlevels, level + 1)),
            )
            components.append(f"{key_text[0]}: {value_text[0]}")
            readable = readable and key_text[1] and value_text[1]
            recursive = recursive or key_text[2] or value_text[2]
        del context[objid]
        return "{" + ", ".join(components) + "}", readable, recursive

    def _sequence_text(
        self,
        obj: object,
        context: dict[int, int],
        maxlevels: int,
        level: int,
    ) -> Generator[Work, object, _Text]:
        """Render a plain list or tuple on one line, each item as its own part."""
        items = cast("Sequence[object]", obj)
        if issubclass(type(obj), list):
            if not items:
                return "[]", True, False
            opener, closer = "[", "]"
        elif len(items) == 1:
            opener, closer = "(", ",)"
        else:
            if not items:
                return "()", True, False
            opener, closer = "(", ")"
        objid = id(obj)
        if maxlevels and level >= maxlevels:
            return f"{opener}...{closer}", False, objid in context
        if objid in context:
            return _recursion(obj), False, True
        context[objid] = 1
        readable, recursive = True, False
        components: list[str] = []
        for item in items:
            text, item_readable, item_recursive = cast(
                _Text,
                (yield self._text(item, context, maxlevels, level + 1)),
            )
            components.append(text)
            readable = readable and item_readable
            recursive = recursive or item_recursive
        del context[objid]
        return opener + ", ".join(components) + closer, readable, recursive

    def _repr_text(
        self,
        obj: object,
        context: dict[int, int],
        level: int,
    ) -> Generator[Work, object, str]:
        """Render ``obj`` on one line within this printer's depth."""
        text, _, _ = cast(
            _Text,
            (yield self._text(obj, context.copy(), self._depth or 0, level)),
        )
        return text

    def _block(
        self,
        obj: object,
        indent: int,
        allowance: int,
        context: dict[int, int],
        level: int,
    ) -> Generator[Work, object, str]:
        """Lay ``obj`` out as ``PrettyPrinter._format`` does: one line if it fits."""
        objid = id(obj)
        if objid in context:
            return _recursion(obj)
        rep = yield from self._repr_text(obj, context, level)
        if len(rep) <= self._width - indent - allowance:
            return rep
        layout = self._layout_for(type(obj))
        if layout is None and not _has_generated_repr(obj):
            return rep
        context[objid] = 1
        laid = (
            self._dataclass_block(obj, indent, allowance, context, level + 1)
            if layout is None
            else layout(obj, indent, allowance, context, level + 1)
        )
        text = laid if isinstance(laid, str) else cast(str, (yield from laid))
        del context[objid]
        return text

    def _layout_for(self, kind: type) -> Callable[..., str | Work] | None:
        """Return the layout ``PrettyPrinter`` dispatches ``kind`` to, by its repr."""
        return self._layouts_by_repr.get(cast(object, kind.__repr__))

    @functools.cached_property
    def _layouts_by_repr(self) -> dict[object, Callable[..., str | Work]]:
        """Each type ``PrettyPrinter`` spreads over lines, keyed by its repr."""
        return {
            cast(object, dict.__repr__): self._dict_block,
            cast(object, OrderedDict.__repr__): self._ordered_dict_block,
            cast(object, list.__repr__): self._list_block,
            cast(object, tuple.__repr__): self._tuple_block,
            cast(object, set.__repr__): self._set_block,
            cast(object, frozenset.__repr__): self._set_block,
            cast(object, str.__repr__): self._str_block,
            cast(object, bytes.__repr__): self._bytes_block,
            cast(object, bytearray.__repr__): self._bytearray_block,
            cast(object, MappingProxyType.__repr__): self._mappingproxy_block,
            cast(object, SimpleNamespace.__repr__): self._simplenamespace_block,
            cast(object, defaultdict.__repr__): self._default_dict_block,
            cast(object, Counter.__repr__): self._counter_block,
            cast(object, ChainMap.__repr__): self._chain_map_block,
            cast(object, deque.__repr__): self._deque_block,
            cast(object, UserDict.__repr__): self._wrapped_data_block,
            cast(object, UserList.__repr__): self._wrapped_data_block,
            cast(object, UserString.__repr__): self._wrapped_data_block,
        }

    def _dataclass_block(
        self,
        obj: object,
        indent: int,
        allowance: int,
        context: dict[int, int],
        level: int,
    ) -> Generator[Work, object, str]:
        """Lay a dataclass out as its class name and its ``repr`` fields."""
        cls_name = type(obj).__name__
        items = [
            (item.name, cast(object, getattr(obj, item.name)))
            for item in dataclasses.fields(cast("_Dataclass", obj))
            if item.repr
        ]
        body = yield from self._namespace_items_block(
            items,
            indent + len(cls_name) + 1,
            allowance,
            context,
            level,
        )
        return f"{cls_name}({body})"

    def _dict_block(
        self,
        obj: object,
        indent: int,
        allowance: int,
        context: dict[int, int],
        level: int,
    ) -> Generator[Work, object, str]:
        mapping = cast("dict[object, object]", obj)
        parts = ["{"]
        if self._indent_per_level > 1:
            parts.append((self._indent_per_level - 1) * " ")
        if mapping:
            items = (
                sorted(mapping.items(), key=_safe_tuple)
                if self._sort_dicts
                else list(mapping.items())
            )
            parts.append(
                (
                    yield from self._dict_items_block(
                        items,
                        indent,
                        allowance + 1,
                        context,
                        level,
                    )
                ),
            )
        parts.append("}")
        return "".join(parts)

    def _ordered_dict_block(
        self,
        obj: object,
        indent: int,
        allowance: int,
        context: dict[int, int],
        level: int,
    ) -> Generator[Work, object, str]:
        mapping = cast("OrderedDict[object, object]", obj)
        if not mapping:
            return repr(mapping)
        name = type(mapping).__name__
        body = cast(
            str,
            (
                yield self._block(
                    list(mapping.items()),
                    indent + len(name) + 1,
                    allowance + 1,
                    context,
                    level,
                )
            ),
        )
        return f"{name}({body})"

    def _list_block(
        self,
        obj: object,
        indent: int,
        allowance: int,
        context: dict[int, int],
        level: int,
    ) -> Generator[Work, object, str]:
        items = cast("list[object]", obj)
        body = yield from self._items_block(
            items,
            indent,
            allowance + 1,
            context,
            level,
        )
        return f"[{body}]"

    def _tuple_block(
        self,
        obj: object,
        indent: int,
        allowance: int,
        context: dict[int, int],
        level: int,
    ) -> Generator[Work, object, str]:
        items = cast("tuple[object, ...]", obj)
        closer = ",)" if len(items) == 1 else ")"
        body = yield from self._items_block(
            items,
            indent,
            allowance + len(closer),
            context,
            level,
        )
        return f"({body}{closer}"

    def _set_block(
        self,
        obj: object,
        indent: int,
        allowance: int,
        context: dict[int, int],
        level: int,
    ) -> Generator[Work, object, str]:
        members = cast("set[object] | frozenset[object]", obj)
        if not members:
            return repr(members)
        kind = type(members)
        if kind is set:
            opener, closer = "{", "}"
        else:
            opener, closer = kind.__name__ + "({", "})"
            indent += len(kind.__name__) + 1
        body = yield from self._items_block(
            sorted(members, key=_SafeKey),
            indent,
            allowance + len(closer),
            context,
            level,
        )
        return opener + body + closer

    def _str_block(
        self,
        obj: object,
        indent: int,
        allowance: int,
        context: dict[int, int],
        level: int,
    ) -> str:
        """Wrap a long string into adjacent literals, as ``PrettyPrinter`` does."""
        del context
        text = cast(str, obj)
        if not text:
            return repr(text)
        chunks: list[str] = []
        lines = text.splitlines(True)
        if level == 1:
            indent += 1
            allowance += 1
        max_width1 = max_width = self._width - indent
        rep = ""
        for i, line in enumerate(lines):
            rep = repr(line)
            if i == len(lines) - 1:
                max_width1 -= allowance
            if len(rep) <= max_width1:
                chunks.append(rep)
            else:
                chunks.extend(
                    _wrapped_line(
                        line,
                        max_width=max_width,
                        allowance=allowance if i == len(lines) - 1 else 0,
                    ),
                )
        if len(chunks) == 1:
            return rep
        body = ("\n" + " " * indent).join(chunks)
        return f"({body})" if level == 1 else body

    def _bytes_block(
        self,
        obj: object,
        indent: int,
        allowance: int,
        context: dict[int, int],
        level: int,
    ) -> str:
        """Wrap long bytes into adjacent literals, as ``PrettyPrinter`` does."""
        del context
        data = cast(bytes, obj)
        if len(data) <= 4:
            return repr(data)
        parens = level == 1
        if parens:
            indent += 1
            allowance += 1
        body = ("\n" + " " * indent).join(
            _wrap_bytes_repr(data, self._width - indent, allowance),
        )
        return f"({body})" if parens else body

    def _bytearray_block(
        self,
        obj: object,
        indent: int,
        allowance: int,
        context: dict[int, int],
        level: int,
    ) -> str:
        data = bytes(cast(bytearray, obj))
        body = self._bytes_block(data, indent + 10, allowance + 1, context, level + 1)
        return f"bytearray({body})"

    def _mappingproxy_block(
        self,
        obj: object,
        indent: int,
        allowance: int,
        context: dict[int, int],
        level: int,
    ) -> Generator[Work, object, str]:
        proxy = cast("MappingProxyType[object, object]", obj)
        body = cast(
            str,
            (
                yield self._block(
                    proxy.copy(),
                    indent + 13,
                    allowance + 1,
                    context,
                    level,
                )
            ),
        )
        return f"mappingproxy({body})"

    def _simplenamespace_block(
        self,
        obj: object,
        indent: int,
        allowance: int,
        context: dict[int, int],
        level: int,
    ) -> Generator[Work, object, str]:
        namespace = cast(SimpleNamespace, obj)
        # ``SimpleNamespace``'s own repr says "namespace"; a subclass uses its name.
        cls_name = (
            "namespace"
            if type(namespace) is SimpleNamespace
            else type(namespace).__name__
        )
        items = list(cast("dict[str, object]", vars(namespace)).items())
        body = yield from self._namespace_items_block(
            items,
            indent + len(cls_name) + 1,
            allowance,
            context,
            level,
        )
        return f"{cls_name}({body})"

    def _default_dict_block(
        self,
        obj: object,
        indent: int,
        allowance: int,
        context: dict[int, int],
        level: int,
    ) -> Generator[Work, object, str]:
        mapping = cast("defaultdict[object, object]", obj)
        if not mapping:
            return repr(mapping)
        factory = yield from self._repr_text(mapping.default_factory, context, level)
        name = type(mapping).__name__
        indent += len(name) + 1
        body = yield from self._dict_block(
            mapping,
            indent,
            allowance + 1,
            context,
            level,
        )
        return f"{name}({factory},\n{' ' * indent}{body})"

    def _counter_block(
        self,
        obj: object,
        indent: int,
        allowance: int,
        context: dict[int, int],
        level: int,
    ) -> Generator[Work, object, str]:
        counter = cast("Counter[object]", obj)
        if not counter:
            return repr(counter)
        name = type(counter).__name__
        pad = (self._indent_per_level - 1) * " " if self._indent_per_level > 1 else ""
        body = yield from self._dict_items_block(
            counter.most_common(),
            indent + len(name) + 1,
            allowance + 2,
            context,
            level,
        )
        return f"{name}({{{pad}{body}}})"

    def _chain_map_block(
        self,
        obj: object,
        indent: int,
        allowance: int,
        context: dict[int, int],
        level: int,
    ) -> Generator[Work, object, str]:
        chain = cast("ChainMap[object, object]", obj)
        if not chain.maps:
            return repr(chain)
        name = type(chain).__name__
        indent += len(name) + 1
        parts = [name + "("]
        for i, mapping in enumerate(chain.maps):
            last = i == len(chain.maps) - 1
            parts.append(
                cast(
                    str,
                    (
                        yield self._block(
                            mapping,
                            indent,
                            allowance + 1 if last else 1,
                            context,
                            level,
                        )
                    ),
                ),
            )
            parts.append(")" if last else ",\n" + " " * indent)
        return "".join(parts)

    def _deque_block(
        self,
        obj: object,
        indent: int,
        allowance: int,
        context: dict[int, int],
        level: int,
    ) -> Generator[Work, object, str]:
        items = cast("deque[object]", obj)
        if not items:
            return repr(items)
        name = type(items).__name__
        indent += len(name) + 1
        if items.maxlen is None:
            body = yield from self._items_block(
                items,
                indent,
                allowance + 2,
                context,
                level,
            )
            return f"{name}([{body}])"
        body = yield from self._items_block(items, indent, 2, context, level)
        maxlen = yield from self._repr_text(items.maxlen, context, level)
        return f"{name}([{body}],\n{' ' * indent}maxlen={maxlen})"

    def _wrapped_data_block(
        self,
        obj: object,
        indent: int,
        allowance: int,
        context: dict[int, int],
        level: int,
    ) -> Generator[Work, object, str]:
        """Lay out a ``UserDict``, ``UserList``, or ``UserString`` as its data."""
        data = cast(
            "UserDict[object, object] | UserList[object] | UserString",
            obj,
        ).data
        return cast(
            str,
            (yield self._block(data, indent, allowance, context, level - 1)),
        )

    def _dict_items_block(
        self,
        items: Sequence[tuple[object, object]],
        indent: int,
        allowance: int,
        context: dict[int, int],
        level: int,
    ) -> Generator[Work, object, str]:
        """Lay out ``key: value`` pairs, one per line."""
        indent += self._indent_per_level
        parts: list[str] = []
        for i, (key, value) in enumerate(items):
            last = i == len(items) - 1
            key_text = yield from self._repr_text(key, context, level)
            value_text = cast(
                str,
                (
                    yield self._block(
                        value,
                        indent + len(key_text) + 2,
                        allowance if last else 1,
                        context,
                        level,
                    )
                ),
            )
            parts.append(f"{key_text}: {value_text}")
        return (",\n" + " " * indent).join(parts)

    def _namespace_items_block(
        self,
        items: list[tuple[str, object]],
        indent: int,
        allowance: int,
        context: dict[int, int],
        level: int,
    ) -> Generator[Work, object, str]:
        """Lay out ``name=value`` pairs, one per line."""
        parts: list[str] = []
        for i, (key, value) in enumerate(items):
            last = i == len(items) - 1
            # A value already being laid out reads as a dataclass's own repr does.
            text = (
                "..."
                if id(value) in context
                else cast(
                    str,
                    (
                        yield self._block(
                            value,
                            indent + len(key) + 1,
                            allowance if last else 1,
                            context,
                            level,
                        )
                    ),
                )
            )
            parts.append(f"{key}={text}")
        return (",\n" + " " * indent).join(parts)

    def _items_block(
        self,
        items: Sequence[object],
        indent: int,
        allowance: int,
        context: dict[int, int],
        level: int,
    ) -> Generator[Work, object, str]:
        """Lay out items one per line, or several per line when compact."""
        indent += self._indent_per_level
        parts: list[str] = (
            [(self._indent_per_level - 1) * " "] if self._indent_per_level > 1 else []
        )
        newline = ",\n" + " " * indent
        delim = ""
        width = max_width = self._width - indent + 1
        for i, item in enumerate(items):
            last = i == len(items) - 1
            if last:
                max_width -= allowance
                width -= allowance
            if self._compact:
                rep = yield from self._repr_text(item, context, level)
                needed = len(rep) + 2
                if width < needed:
                    width = max_width
                    if delim:
                        delim = newline
                if width >= needed:
                    width -= needed
                    parts.extend((delim, rep))
                    delim = ", "
                    continue
            parts.append(delim)
            delim = newline
            parts.append(
                cast(
                    str,
                    (
                        yield self._block(
                            item,
                            indent,
                            allowance if last else 1,
                            context,
                            level,
                        )
                    ),
                ),
            )
        return "".join(parts)


class _Dataclass(Protocol):
    __dataclass_fields__: ClassVar[dict[str, dataclasses.Field[object]]]


class _Comparable(Protocol):
    def __lt__(self, other: object, /) -> bool: ...


class _SafeKey:
    """Order values that refuse ``<`` by type name, then id, as ``pprint`` does."""

    __slots__ = ("obj",)

    def __init__(self, obj: object) -> None:
        self.obj = obj

    def __lt__(self, other: _SafeKey) -> bool:
        try:
            return cast(_Comparable, self.obj) < other.obj
        except TypeError:
            return (str(type(self.obj)), id(self.obj)) < (
                str(type(other.obj)),
                id(other.obj),
            )


def _safe_tuple(pair: tuple[object, object]) -> tuple[_SafeKey, _SafeKey]:
    return _SafeKey(pair[0]), _SafeKey(pair[1])


def _recursion(obj: object) -> str:
    return f"<Recursion on {type(obj).__name__} with id={id(obj)}>"


def _has_generated_repr(obj: object) -> bool:
    """Return whether ``PrettyPrinter`` lays ``obj`` out as a dataclass."""
    if not dataclasses.is_dataclass(obj) or isinstance(obj, type):
        return False
    params = cast(object, getattr(type(obj), "__dataclass_params__", None))
    wrapped = cast(object, getattr(type(obj).__repr__, "__wrapped__", None))
    return bool(getattr(params, "repr", False)) and "__create_fn__" in cast(
        str,
        getattr(wrapped, "__qualname__", ""),
    )


def _wrapped_line(line: str, *, max_width: int, allowance: int) -> Iterator[str]:
    """Yield the literals one string line splits into at word boundaries."""
    # ``pprint`` splits on ``\S*\s*`` and pops the empty match that always ends it;
    # requiring a character yields the same words without that trailing match.
    words = [match.group() for match in re.finditer(r"\S+\s*|\s+", line)]
    current = ""
    for j, word in enumerate(words):
        candidate = current + word
        limit = max_width - allowance if j == len(words) - 1 else max_width
        if len(repr(candidate)) > limit:
            if current:
                yield repr(current)
            current = word
        else:
            current = candidate
    if current:
        yield repr(current)


def _wrap_bytes_repr(data: bytes, width: int, allowance: int) -> Iterator[str]:
    """Yield the literals long bytes split into, four bytes at a time."""
    current = b""
    last = len(data) // 4 * 4
    for i in range(0, len(data), 4):
        part = data[i : i + 4]
        candidate = current + part
        if i == last:
            width -= allowance
        if len(repr(candidate)) > width:
            if current:
                yield repr(current)
            current = part
        else:
            current = candidate
    if current:
        yield repr(current)


@dataclasses.dataclass(kw_only=True, slots=True)
class _ReprFrame:
    """A container ``plain_repr`` is part way through."""

    ident: int
    opener: str
    closer: str
    children: list[object]
    labels: list[str]
    pairs: bool = False
    parts: list[str] = dataclasses.field(default_factory=list[str])

    def joined(self) -> str:
        """Return the container's text from its parts' text.

        Returns:
          text: The opener, each part's text joined by commas, then the closer.

        """
        if self.pairs:
            items = [
                f"{key}: {value}"
                for key, value in zip(self.parts[::2], self.parts[1::2], strict=True)
            ]
        else:
            items = [
                label + part
                for label, part in zip(self.labels, self.parts, strict=True)
            ]
        return self.opener + ", ".join(items) + self.closer


def _opened(node: object, active: set[int]) -> str | _ReprFrame:
    """Return ``node``'s repr when it has no parts, else a frame for its parts."""
    kind = type(node)
    if kind in _CONTAINER_TEXT:
        empty, marker, opener, closer = _CONTAINER_TEXT[kind]
        container = cast("Sequence[object] | set[object] | frozenset[object]", node)
        if not container:
            return empty
        if id(node) in active:
            return marker
        if kind is dict:
            mapping = cast("dict[object, object]", node)
            children = [item for pair in mapping.items() for item in pair]
            active.add(id(node))
            return _ReprFrame(
                ident=id(node),
                opener=opener,
                closer=closer,
                children=children,
                labels=[],
                pairs=True,
            )
        if kind is tuple and len(container) == 1:
            closer = ",)"
        active.add(id(node))
        return _ReprFrame(
            ident=id(node),
            opener=opener,
            closer=closer,
            children=list(container),
            labels=[""] * len(container),
        )
    generated = cast(object, getattr(vars(kind).get("__repr__"), "__wrapped__", None))
    if "__create_fn__" not in cast(str, getattr(generated, "__qualname__", "")):
        return repr(node)
    if id(node) in active:
        return "..."
    names = [
        item.name for item in dataclasses.fields(cast("_Dataclass", node)) if item.repr
    ]
    active.add(id(node))
    return _ReprFrame(
        ident=id(node),
        opener=f"{kind.__qualname__}(",
        closer=")",
        children=[cast(object, getattr(node, name)) for name in names],
        labels=[f"{name}=" for name in names],
    )


_BUILTIN_SCALARS: Final = frozenset(
    {str, bytes, bytearray, float, complex, bool, type(None)},
)
"""Types whose one-line text is their ``repr``, as in ``pprint``."""


_CONTAINER_TEXT: Final[dict[type, tuple[str, str, str, str]]] = {
    list: ("[]", "[...]", "[", "]"),
    tuple: ("()", "(...)", "(", ")"),
    dict: ("{}", "{...}", "{", "}"),
    set: ("set()", "set(...)", "{", "}"),
    frozenset: ("frozenset()", "frozenset(...)", "frozenset({", "})"),
}
"""Each builtin container's repr when empty, when recursive, and around its items."""
