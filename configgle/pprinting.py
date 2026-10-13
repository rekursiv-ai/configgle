"""Pretty printing utilities for Fig config objects."""

from __future__ import annotations

from difflib import SequenceMatcher, unified_diff
from itertools import zip_longest
from typing import (
    IO,
    TYPE_CHECKING,
    Final,
    Literal,
    TypeVar,
    assert_never,
    cast,
    override,
)

import ast
import contextlib
import dataclasses
import functools
import inspect
import io
import re
import shutil
import sys
import textwrap
import tokenize
import types
import warnings

from configgle.coloring import color_change, color_config, color_diff
from configgle.custom_types import (
    DataclassLike,
    DefersCall,
    Finalizeable,
    PformatOptions,
)
from configgle.stackprint import StackPrinter, plain_repr
from configgle.walk import copy_tree


if TYPE_CHECKING:
    from collections.abc import Generator, Sequence

    from configgle.tree import Work


__all__ = [
    "format_diff",
    "pformat",
    "pprint",
]

_T = TypeVar("_T")


_DEFAULT_CONTINUATION_PIPE_THRESHOLD: Final = 50

_SHORT_SEQUENCE_MAX_WIDTH: Final = 40


def pformat(
    obj: object,
    indent: int = 4,
    width: int = 80,
    depth: int | None = None,
    *,
    compact: bool = False,
    # sort_dicts=False preserves insertion order (usually meaningful for configs).
    # underscore_numbers=True improves readability of large numbers (1_000_000).
    sort_dicts: bool = False,
    underscore_numbers: bool = True,
    finalize: bool = True,
    mask_memory_addresses: bool = True,
    extra_compact: bool = True,
    continuation_pipe: int = _DEFAULT_CONTINUATION_PIPE_THRESHOLD,
    hide_default_values: bool = True,
    short_sequence_max_width: int = _SHORT_SEQUENCE_MAX_WIDTH,
) -> str:
    """Format object as a string with Fig-aware pretty printing.

    Args:
      obj: Object to format.
      indent: Spaces per indent level.
      width: Maximum line width.
      depth: Maximum nesting depth (None for unlimited).
      compact: Use compact format for sequences.
      sort_dicts: Sort dictionary keys.
      underscore_numbers: Use underscores in large numbers.
      finalize: Auto-finalize unfinalized configs before printing.
      mask_memory_addresses: Replace memory addresses with placeholder.
      extra_compact: Use extra compact formatting.
      continuation_pipe: Lines threshold for continuation pipes (0=always, -1=never).
      hide_default_values: Omit fields matching literal defaults. Factory-backed
        fields remain visible without executing their factories.
      short_sequence_max_width: Max width for single-line sequences.

    Returns:
      formatted: Pretty-printed string representation.

    """
    printer = FigPrinter(
        indent=indent,
        width=width,
        depth=depth,
        compact=compact,
        sort_dicts=sort_dicts,
        underscore_numbers=underscore_numbers,
        finalize=finalize,
        mask_memory_addresses=mask_memory_addresses,
        extra_compact=extra_compact,
        continuation_pipe=continuation_pipe,
        hide_default_values=hide_default_values,
        short_sequence_max_width=short_sequence_max_width,
    )
    return printer.pformat(obj)


def pprint(
    obj: object,
    stream: IO[str] | None = None,
    indent: int = 4,
    width: int = 80,
    depth: int | None = None,
    *,
    color: bool | None = None,
    compact: bool = False,
    # The following differ from the Python standard lib.
    sort_dicts: bool = False,
    underscore_numbers: bool = True,
    finalize: bool = True,
    mask_memory_addresses: bool = True,
    extra_compact: bool = True,
    continuation_pipe: int = _DEFAULT_CONTINUATION_PIPE_THRESHOLD,
    hide_default_values: bool = True,
    short_sequence_max_width: int = _SHORT_SEQUENCE_MAX_WIDTH,
) -> None:
    """Pretty-print object with Fig-aware formatting.

    Args:
      obj: Object to print.
      stream: Output stream (defaults to sys.stdout).
      color: Highlight config syntax with ANSI terminal colors; ``None`` colors
        only a terminal stream.
      indent: Spaces per indent level.
      width: Maximum line width.
      depth: Maximum nesting depth (None for unlimited).
      compact: Use compact format for sequences.
      sort_dicts: Sort dictionary keys.
      underscore_numbers: Use underscores in large numbers.
      finalize: Auto-finalize unfinalized configs before printing.
      mask_memory_addresses: Replace memory addresses with placeholder.
      extra_compact: Use extra compact formatting.
      continuation_pipe: Lines threshold for continuation pipes (0=always, -1=never).
      hide_default_values: Omit fields matching literal defaults. Factory-backed
        fields remain visible without executing their factories.
      short_sequence_max_width: Max width for single-line sequences.

    """
    printer = FigPrinter(
        stream=stream,
        indent=indent,
        width=width,
        depth=depth,
        compact=compact,
        sort_dicts=sort_dicts,
        underscore_numbers=underscore_numbers,
        finalize=finalize,
        mask_memory_addresses=mask_memory_addresses,
        extra_compact=extra_compact,
        continuation_pipe=continuation_pipe,
        hide_default_values=hide_default_values,
        short_sequence_max_width=short_sequence_max_width,
    )
    sink = sys.stdout if stream is None else stream
    if sink.isatty() if color is None else color:
        sink.write(color_config(printer.pformat(obj)) + "\n")
    else:
        printer.pprint(obj)


class FigPrinter(StackPrinter):
    """``StackPrinter`` with Fig-specific formatting enhancements."""

    def __init__(
        self,
        stream: IO[str] | None = None,
        indent: int = 4,
        width: int = 80,
        depth: int | None = None,
        *,
        compact: bool = False,
        # The following differ from the Python standard lib.
        sort_dicts: bool = False,
        underscore_numbers: bool = True,
        finalize: bool = True,
        mask_memory_addresses: bool = True,
        extra_compact: bool = True,
        continuation_pipe: int = _DEFAULT_CONTINUATION_PIPE_THRESHOLD,
        hide_default_values: bool = True,
        short_sequence_max_width: int = _SHORT_SEQUENCE_MAX_WIDTH,
    ):
        super().__init__(
            indent=indent,
            width=width,
            depth=depth,
            stream=stream,
            compact=compact,
            sort_dicts=sort_dicts,
            underscore_numbers=underscore_numbers,
        )
        self._finalize = finalize
        self._mask_memory_addresses = (
            _mask_memory_addresses if mask_memory_addresses else None
        )
        self._extra_compact = extra_compact
        self._continuation_pipe = continuation_pipe
        self._hide_default_values = hide_default_values
        self._short_sequence_max_width = short_sequence_max_width
        self._finalized_copies: dict[int, tuple[object, object]] | None = None
        self._formats: (
            dict[
                tuple[int, int, int, tuple[int, ...]],
                tuple[object, tuple[str, bool, bool]],
            ]
            | None
        ) = None
        self._layouts: (
            dict[tuple[int, int, int, int, tuple[int, ...]], tuple[object, str]] | None
        ) = None

    def field_values(self, obj: object, *, depth: int | None) -> dict[str, str]:
        """Map each field path of ``obj`` to its rendered leaf or class name.

        Each node is finalized for display exactly as ``pformat`` finalizes it.

        Args:
          obj: Root of the tree to flatten.
          depth: Levels to descend before rendering a subtree whole (None for
            unlimited).

        Returns:
          values: Paths such as ``a.b[0]['k']`` in field order; ``""`` is the root.

        """
        values: dict[str, str] = {}
        with self._render_scope():
            self._collect_field_values(obj, values=values, levels=depth)
        return values

    @override
    def pprint(self, object: object) -> None:
        with self._render_scope():
            super().pprint(self._try_to_finalize(object))

    @override
    def pformat(self, object: object) -> str:
        with self._render_scope():
            return super().pformat(self._try_to_finalize(object))

    # CPython's ``_format`` renders a node's whole subtree to measure it, then lays
    # a too-wide one out item by item; the extra-compact layout renders each item
    # once to try one line and again to lay it out. Each node's text is therefore
    # requested once per enclosing level, which made renders exponential in depth.
    # Within one render a node's text depends only on the arguments in these keys,
    # so each is rendered once. The stored node pins its ``id`` against reuse.
    @override
    def _text(
        self,
        obj: object,
        context: dict[int, int],
        maxlevels: int,
        level: int,
    ) -> Generator[Work, object, tuple[str, bool, bool]]:
        if self._formats is None:
            return (yield from self._format_text(obj, context, maxlevels, level))
        key = (id(obj), maxlevels, level, tuple(context))
        cached = self._formats.get(key)
        if cached is None or cached[0] is not obj:
            rendered = yield from self._format_text(obj, context, maxlevels, level)
            cached = (obj, rendered)
            self._formats[key] = cached
        return cached[1]

    @override
    def _block(
        self,
        obj: object,
        indent: int,
        allowance: int,
        context: dict[int, int],
        level: int,
    ) -> Generator[Work, object, str]:
        if self._layouts is None:
            return (yield from self._layout(obj, indent, allowance, context, level))
        key = (id(obj), indent, allowance, level, tuple(context))
        cached = self._layouts.get(key)
        if cached is None or cached[0] is not obj:
            laid = yield from self._layout(obj, indent, allowance, context, level)
            cached = (obj, laid)
            self._layouts[key] = cached
        return cached[1]

    @contextlib.contextmanager
    def _render_scope(self) -> Generator[None]:
        """Share finalized copies and rendered text across one top-level render."""
        if self._finalized_copies is not None:
            yield
            return
        self._finalized_copies, self._formats, self._layouts = {}, {}, {}
        try:
            yield
        finally:
            self._finalized_copies = self._formats = self._layouts = None

    def _format_text(
        self,
        obj: object,
        context: dict[int, int],
        maxlevels: int,
        level: int,
    ) -> Generator[Work, object, tuple[str, bool, bool]]:
        """Render ``obj`` on one line, qualifying functions and masking addresses."""
        obj = self._try_to_finalize(obj)
        if isinstance(obj, DefersCall):
            return (
                yield from self._inline_repr(
                    obj,
                    context=context,
                    maxlevels=maxlevels,
                    level=level,
                )
            )
        text, readable, recursive = yield from super()._text(
            obj,
            context,
            maxlevels,
            level,
        )
        text = _qualify_function_reprs(obj, text)
        if self._mask_memory_addresses is not None:
            text = self._mask_memory_addresses(text)
        return text, readable, recursive

    def _layout(
        self,
        obj: object,
        indent: int,
        allowance: int,
        context: dict[int, int],
        level: int,
    ) -> Generator[Work, object, str]:
        """Lay ``obj`` out across lines as ``_block`` would, without the cache."""
        obj = self._try_to_finalize(obj)
        if not isinstance(obj, DefersCall):
            return (yield from super()._block(obj, indent, allowance, context, level))
        inline = obj
        if id(inline) in context:
            return "..."
        if self._depth is None or level < self._depth:
            rep = yield from self._repr_text(inline, context, level)
            if len(rep) > self._width - indent - allowance:
                return (
                    yield from self._pprint_inline(
                        inline,
                        indent=indent,
                        allowance=allowance,
                        context=context,
                        level=level + 1,
                    )
                )
        return (yield from super()._block(inline, indent, allowance, context, level))

    def _finalize_pending(self, obj: object) -> bool:
        """Return whether ``_try_to_finalize`` would copy and finalize ``obj``."""
        # An exact builtin type has no ``finalize``, and the runtime protocol check
        # costs microseconds on each of a render's thousands of leaves.
        if type(obj) in (int, float, str, bool, bytes, type(None), list, dict, tuple):
            return False
        finalized_flag = getattr(obj, "_finalized", False)
        return self._finalize and not finalized_flag and isinstance(obj, Finalizeable)

    def _inline_repr(
        self,
        obj: DefersCall,
        *,
        context: dict[int, int],
        maxlevels: int,
        level: int,
    ) -> Generator[Work, object, tuple[str, bool, bool]]:
        name = type(obj).__qualname__
        if id(obj) in context:
            return "...", False, True
        if maxlevels and level >= maxlevels:
            return f"{name}(...)", False, False
        context[id(obj)] = 1
        parts: list[tuple[str, tuple[str, bool, bool]]] = []
        try:
            for key, value in _rendered_inline_items(obj):
                part = cast(
                    "tuple[str, bool, bool]",
                    (yield self._text(value, context, maxlevels, level + 1)),
                )
                parts.append((key, part))
        finally:
            del context[id(obj)]
        text = ", ".join(f"{key}={part[0]}" if key else part[0] for key, part in parts)
        return (
            f"{name}({text})",
            all(part[1] for _, part in parts),
            any(part[2] for _, part in parts),
        )

    def _pprint_inline(
        self,
        obj: DefersCall,
        *,
        indent: int,
        allowance: int,
        context: dict[int, int],
        level: int,
    ) -> Generator[Work, object, str]:
        name = type(obj).__qualname__
        items = _rendered_inline_items(obj)
        item_indent, closing_indent = _get_level_indents(level, self._indent_per_level)
        if not self._extra_compact:
            item_indent, closing_indent = indent + len(name) + 1, indent
        parts = [name + "(\n"]
        context[id(obj)] = 1
        try:
            for index, (key, value) in enumerate(items):
                prefix = f"{key}=" if key else ""
                parts.append(" " * item_indent + prefix)
                parts.append(
                    cast(
                        str,
                        (
                            yield self._block(
                                value,
                                item_indent + len(prefix),
                                allowance + 1 if index == len(items) - 1 else 1,
                                context,
                                level,
                            )
                        ),
                    ),
                )
                parts.append(",\n" if index < len(items) - 1 else "\n")
            parts.append(" " * closing_indent + ")")
        finally:
            del context[id(obj)]
        return "".join(parts)

    def _collect_field_values(
        self,
        root: object,
        *,
        values: dict[str, str],
        levels: int | None,
    ) -> None:
        """Record each field path's rendered leaf, walking the tree on a stack."""
        ancestors: set[int] = set()
        stack: list[tuple[str, object, int | None] | int] = [("", root, levels)]
        while stack:
            entry = stack.pop()
            if isinstance(entry, int):
                ancestors.discard(entry)
                continue
            path, value, left = entry
            value = self._try_to_finalize(value)
            children = _field_children(value)
            if id(value) in ancestors or not children or left == 0:
                values[path] = self.pformat(value)
                continue
            if dataclasses.is_dataclass(value) and not isinstance(value, type):
                values[path] = type(value).__qualname__
            ancestors.add(id(value))
            stack.append(id(value))
            stack.extend(
                (
                    f"{path}.{name}"
                    if path and not name.startswith("[")
                    else path + name,
                    child,
                    None if left is None else left - 1,
                )
                for name, child in reversed(children)
            )

    # ``finalize`` mutates in place, so the tree is copied first (via ``copy_tree``,
    # which duplicates the config spine but aliases heavy leaves like tensors) to keep
    # printing side-effect-free.
    def _try_to_finalize(self, obj: _T) -> _T:
        """Copy the config tree then finalize it for display purposes."""
        if not self._finalize_pending(obj):
            return obj
        assert isinstance(obj, Finalizeable)
        cached = (
            None
            if self._finalized_copies is None
            else self._finalized_copies.get(id(obj))
        )
        # ``is obj`` guards address reuse: CPython hands a reclaimed id() to
        # the next allocation, so a bare id() hit can belong to a config that
        # has since been collected.
        if cached is not None and cached[0] is obj:
            return cast(_T, cached[1])
        try:
            finalized = copy_tree(obj).finalize()
            if self._finalized_copies is not None:
                self._finalized_copies[id(obj)] = (obj, finalized)
            obj = finalized
        except Exception as e:  # noqa: BLE001 -- any finalize failure degrades to printing the unfinalized tree.
            warnings.warn(f"{type(e).__name__}: {e}", stacklevel=2)
        return obj

    # ``StackPrinter`` lays a dataclass out through ``_dataclass_block``. This hides
    # default-valued fields and uses the extra-compact layout.
    @override
    def _dataclass_block(
        self,
        obj: object,
        indent: int,
        allowance: int,
        context: dict[int, int],
        level: int,
    ) -> Generator[Work, object, str]:
        cls_name = obj.__class__.__qualname__
        indent += len(cls_name) + 1
        items = [
            (f.name, cast(object, getattr(obj, f.name)))
            for f in dataclasses.fields(cast(DataclassLike, obj))
            if f.repr
        ]
        if self._hide_default_values:
            items = _filter_non_default_items(obj, items)
        body = yield from self._namespace_items_block(
            items,
            indent,
            allowance,
            context,
            level,
        )
        return f"{cls_name}({body})"

    @override
    def _namespace_items_block(
        self,
        items: list[tuple[str, object]],
        indent: int,
        allowance: int,
        context: dict[int, int],
        level: int,
    ) -> Generator[Work, object, str]:
        """Put each parameter on its own line at a fixed, level-based indent."""
        if not self._extra_compact:
            return (
                yield from super()._namespace_items_block(
                    items,
                    indent,
                    allowance,
                    context,
                    level,
                )
            )
        if not items:
            return ""
        item_indent, base_indent = _get_level_indents(level, self._indent_per_level)
        parts = ["\n"]
        for i, (key, value) in enumerate(items):
            last = i == len(items) - 1
            parts.append(f"{' ' * item_indent}{key}=")
            if id(value) in context:
                parts.append("...")
            else:
                parts.append(
                    (
                        yield from self._format_namespace_value(
                            value,
                            context=context,
                            level=level,
                            item_indent=item_indent,
                            allowance=allowance if last else 1,
                            num_items=len(items),
                        )
                    ),
                )
            if not last:
                parts.append(",\n")
        parts.append("\n" + " " * base_indent)
        return "".join(parts)

    def _format_namespace_value(
        self,
        value: object,
        *,
        context: dict[int, int],
        level: int,
        item_indent: int,
        allowance: int,
        num_items: int,
    ) -> Generator[Work, object, str]:
        """Format a namespace value with collapsing and continuation pipes."""
        formatted_value = cast(
            str,
            (yield self._block(value, item_indent, allowance, context, level)),
        )
        formatted_value = _collapse_multiline_value(
            formatted_value,
            self._short_sequence_max_width,
        )
        if _should_add_continuation_pipes(
            formatted_value,
            num_items,
            self._continuation_pipe,
        ):
            formatted_value = "\n".join(
                _add_pipes_to_lines(formatted_value.split("\n"), item_indent),
            )
        return formatted_value

    @override
    def _items_block(
        self,
        items: Sequence[object],
        indent: int,
        allowance: int,
        context: dict[int, int],
        level: int,
    ) -> Generator[Work, object, str]:
        """Use a level-based indent instead of an accumulated one."""
        if not self._extra_compact:
            return (
                yield from super()._items_block(
                    items,
                    indent,
                    allowance,
                    context,
                    level,
                )
            )
        one_line = yield from self._one_line_items(
            items,
            context,
            level,
            indent=indent,
            allowance=allowance,
        )
        if self._should_format_on_one_line(len(one_line) + 2, indent, allowance):
            return one_line
        return (yield from self._multiline_items(items, context, level))

    def _one_line_items(
        self,
        items: Sequence[object],
        context: dict[int, int],
        level: int,
        *,
        indent: int,
        allowance: int,
    ) -> Generator[Work, object, str]:
        """Format items on a single line, stopping once the line cannot fit."""
        pieces: list[str] = []
        content_width = 0
        for item in items:
            pieces.append(cast(str, (yield self._block(item, 0, 0, context, level))))
            content_width += len(pieces[-1]) + 2
            # The width only grows, so a prefix that cannot fit rules the line
            # out; rendering the rest, as a list of large configs did, is waste.
            if not self._should_format_on_one_line(content_width, indent, allowance):
                break
        return ", ".join(pieces)

    def _should_format_on_one_line(
        self,
        content_width: int,
        indent: int,
        allowance: int,
    ) -> bool:
        """Determine if items should be formatted on one line."""
        # Short seqs stay one-line at any depth (content_width excludes indent)
        # For longer sequences, check if they fit within the available width.
        return (
            content_width < self._short_sequence_max_width
            or indent + content_width + allowance <= self._width
        )

    def _multiline_items(
        self,
        items: Sequence[object],
        context: dict[int, int],
        level: int,
    ) -> Generator[Work, object, str]:
        """Format items across multiple lines with level-based indent."""
        item_indent, base_indent = _get_level_indents(level, self._indent_per_level)
        parts = ["\n"]
        for i, item in enumerate(items):
            parts.append(" " * item_indent)
            if id(item) in context:
                parts.append("...")
            else:
                laid = cast(
                    str,
                    (yield self._block(item, item_indent, 1, context, level)),
                )
                parts.append(
                    _collapse_multiline_value(laid, self._short_sequence_max_width),
                )
            if i < len(items) - 1:
                parts.append(",\n")
        parts.append("\n" + " " * base_indent)
        return "".join(parts)


def format_diff(
    configs: list[object],
    *,
    mode: Literal["diff", "sxs", "sxs-compact", "sxs-full", "fields"],
    n: int,
    color: bool | None,
    options: PformatOptions,
) -> str:
    """Compare config trees without modifying inputs.

    Args:
      configs: Baseline followed by comparison configs.
      mode: Unified lines, complete aligned trees, or changed field paths.
      n: Context lines for ``diff`` and ``sxs-compact``.
      color: Highlight changes with ANSI terminal colors; ``None`` colors only
        when stdout is a terminal.
      options: Pretty formatting options; side-by-side and fields modes divide
        ``width`` among their columns.

    Returns:
      diff: Formatted differences, or an empty string for equal configs.

    """
    color = bool(sys.stdout.isatty()) if color is None else color
    match mode:
        case "fields":
            return _fields_diff(configs, options=options, color=color)
        case "sxs" | "sxs-compact" | "sxs-full":
            return _sxs_diff(configs, mode=mode, n=n, color=color, options=options)
        case "diff":
            return _unified_diff(configs, n=n, color=color, options=options)
        case _:
            assert_never(mode)


def _unified_diff(
    configs: list[object],
    *,
    n: int,
    color: bool,
    options: PformatOptions,
) -> str:
    """Render one unified diff per comparison config against the baseline."""
    columns = [pformat(cfg, **options).splitlines() for cfg in configs]
    sections: list[str] = []
    for index, after in enumerate(columns[1:], 1):
        lines = list(unified_diff(columns[0], after, n=n, lineterm=""))
        if not lines:
            continue
        text = "\n".join(color_diff(line) if color else line for line in lines)
        if len(columns) > 2:
            text = f"config[0] → config[{index}]\n{text}"
        sections.append(text)
    return "\n\n".join(sections)


def _sxs_diff(
    configs: list[object],
    *,
    mode: Literal["sxs", "sxs-compact", "sxs-full"],
    n: int,
    color: bool,
    options: PformatOptions,
) -> str:
    """Render configs as equal columns, widened to keep each header whole."""
    width = options.get("width", shutil.get_terminal_size().columns)
    column_width = max(
        len(f"config[{len(configs) - 1}]"),
        (width - 3 * (len(configs) - 1)) // len(configs),
    )
    column_options = options.copy()
    column_options["width"] = column_width
    columns = [pformat(cfg, **column_options).splitlines() for cfg in configs]
    return _side_by_side_diff(
        columns,
        n=n,
        width=column_width,
        color=color,
        compact=mode != "sxs-full",
    )


def _fields_diff(
    configs: list[object],
    *,
    options: PformatOptions,
    color: bool,
) -> str:
    """Render only changed paths, with blank cells for absent fields."""
    leaf_options = options.copy()
    leaf_options["depth"] = None
    leaf_options["width"] = max(1, options.get("width", 80) // (len(configs) + 1))
    leaf_options["continuation_pipe"] = -1
    printer = FigPrinter(**leaf_options)
    trees = [
        printer.field_values(source, depth=options.get("depth")) for source in configs
    ]
    paths = dict.fromkeys(path for tree in trees for path in tree)
    rows = [
        (path or "(class)", *(tree.get(path) for tree in trees))
        for path in paths
        if len({tree.get(path) for tree in trees}) > 1
    ]
    if not rows:
        return ""
    widths = [
        max(len(line) for row in rows for line in (row[i] or "").splitlines() or [""])
        for i in range(len(configs) + 1)
    ]
    header = ("field", *(f"config[{i}]" for i in range(len(configs))))
    widths = [
        max(width, len(label)) for width, label in zip(widths, header, strict=True)
    ]
    output = _render_field_row(header, widths=widths, color=False)
    for row in rows:
        output.extend(_render_field_row(row, widths=widths, color=color))
    return "\n".join(output)


def _rendered_inline_items(value: DefersCall) -> list[tuple[str, object]]:
    """Return a call's arguments in rendered order; positional keys are empty."""
    func, args, kwargs = value.deferred_call()
    return [("", func), *(("", arg) for arg in args), *kwargs.items()]


def _field_children(value: object) -> list[tuple[str, object]]:
    """Expose config fields, including inline callable arguments."""
    if isinstance(value, DefersCall):
        func, args, kwargs = value.deferred_call()
        return [
            ("func", func),
            *((f"[{i}]", arg) for i, arg in enumerate(args)),
            *kwargs.items(),
        ]
    if dataclasses.is_dataclass(value) and not isinstance(value, type):
        return [
            (field.name, getattr(value, field.name))
            for field in dataclasses.fields(value)
            if field.repr
        ]
    if isinstance(value, dict):
        return [
            (f"[{key!r}]", child)
            for key, child in cast(dict[object, object], value).items()
        ]
    if isinstance(value, (list, tuple)):
        return [
            (f"[{i}]", child)
            for i, child in enumerate(cast(list[object] | tuple[object, ...], value))
        ]
    return []


def _render_field_row(
    row: tuple[str | None, ...],
    *,
    widths: list[int],
    color: bool,
) -> list[str]:
    """Align multiline values without coloring the shared path column."""
    output: list[str] = []
    for chunks in zip_longest(
        *[(cell or "").splitlines() or [""] for cell in row],
        fillvalue="",
    ):
        cells = [chunks[0].ljust(widths[0])]
        for index, chunk in enumerate(chunks[1:], 1):
            cell = chunk.ljust(widths[index])
            if color and chunk and (index == 1 or row[index] != row[1]):
                cell = color_change(cell, baseline=index == 1)
            cells.append(cell)
        output.append(" │ ".join(cells).rstrip())
    return output


def _align_diff_column(
    baseline: list[str],
    variant: list[str],
) -> tuple[list[str | None], list[list[str]]]:
    """Align a variant to baseline lines and insertion gaps."""
    lines: list[str | None] = [None] * len(baseline)
    gaps: list[list[str]] = [[] for _ in range(len(baseline) + 1)]
    matcher = SequenceMatcher(a=baseline, b=variant, autojunk=False)
    for _, start, stop, other_start, other_stop in matcher.get_opcodes():
        incoming = variant[other_start:other_stop]
        count = min(stop - start, len(incoming))
        lines[start : start + count] = incoming[:count]
        gaps[stop].extend(incoming[count:])
    return lines, gaps


def _side_by_side_diff(
    columns: list[list[str]],
    *,
    n: int,
    width: int,
    color: bool,
    compact: bool = False,
) -> str:
    """Render baseline-aligned rows with equally sized config columns."""
    aligned = [_align_diff_column(columns[0], column) for column in columns[1:]]
    rows: list[tuple[str | None, ...]] = []
    for index in range(len(columns[0]) + 1):
        rows.extend(
            (None, *inserted)
            for inserted in zip_longest(*(gaps[index] for _, gaps in aligned))
        )
        if index < len(columns[0]):
            rows.append((columns[0][index], *(lines[index] for lines, _ in aligned)))
    changed = [index for index, row in enumerate(rows) if len(set(row)) > 1]
    if not changed:
        return ""
    visible = (
        sorted(
            {
                j
                for i in changed
                for j in range(max(0, i - n), min(len(rows), i + n + 1))
            },
        )
        if compact
        else range(len(rows))
    )
    header = tuple(f"config[{i}]" for i in range(len(columns)))
    output = _render_diff_row(header, width=width, color=False)
    for index in visible:
        output.extend(_render_diff_row(rows[index], width=width, color=color))
    return "\n".join(output)


def _render_diff_row(
    row: tuple[str | None, ...],
    *,
    width: int,
    color: bool,
) -> list[str]:
    """Wrap physical lines into equally sized cells without dropping characters."""
    wrapped = [
        [
            line[start : start + width]
            for line in (value or "").splitlines()
            for start in range(0, max(1, len(line)), width)
        ]
        or [""]
        for value in row
    ]
    output: list[str] = []
    for chunks in zip_longest(*wrapped, fillvalue=""):
        cells: list[str] = []
        for index, chunk in enumerate(chunks):
            cell = chunk.ljust(width)
            differs = (
                any(value != row[0] for value in row[1:])
                if index == 0
                else row[index] != row[0]
            )
            if color and differs and chunk:
                cell = color_change(cell, baseline=index == 0)
            cells.append(cell)
        output.append(" │ ".join(cells).rstrip())
    return output


def _qualify_function_reprs(
    value: object,
    rendered: str,
    *,
    active: frozenset[int] = frozenset(),
) -> str:
    """Replace function reprs with qualified names or lambda source, plus bindings."""
    # Every function repr starts so; skipping the walk without one saves a pass
    # over each value's subtree.
    if "<function " not in rendered:
        return rendered
    functions: list[types.FunctionType] = []
    _collect_functions(value, functions, set())
    if not any(repr(function) in rendered for function in functions):
        return rendered
    spans = _string_token_spans(rendered)
    for function in functions:
        qualified = _function_text(function, active=active)
        rendered, spans = _replace_unquoted(
            rendered,
            repr(function),
            qualified,
            spans=spans,
        )
        if "'" in qualified or '"' in qualified:
            spans = _string_token_spans(rendered)
    return rendered


def _function_text(function: types.FunctionType, *, active: frozenset[int]) -> str:
    """Name a function and append the values it captured, e.g. ``f {k=1}``."""
    name = (
        _lambda_source(function.__code__)
        if function.__name__ == "<lambda>"
        else f"{function.__module__}.{function.__qualname__}"
    )
    # A module-level function's name fixes its defaults, so they add only noise;
    # lambdas and locals are made per call, and their bindings tell them apart.
    # One capturing itself (a recursive local) renders by name inside.
    is_made_per_call = (
        function.__name__ == "<lambda>" or "<locals>" in function.__qualname__
    )
    if not is_made_per_call or id(function) in active:
        return name
    active |= {id(function)}
    bindings = [
        f"{key}={_qualify_function_reprs(bound, plain_repr(bound), active=active)}"
        for key, bound in _function_bindings(function)
    ]
    return f"{name} {{{', '.join(bindings)}}}" if bindings else name


def _function_bindings(function: types.FunctionType) -> list[tuple[str, object]]:
    """Return captured closure values and parameter defaults, by name."""
    code = function.__code__
    cells = zip(code.co_freevars, function.__closure__ or (), strict=True)
    # Zero-arg ``super()`` captures ``__class__``, which says nothing about behavior.
    bindings: list[tuple[str, object]] = [
        (key, cast(object, cell.cell_contents))
        for key, cell in cells
        if key != "__class__" and _cell_is_bound(cell)
    ]
    defaults = function.__defaults__ or ()
    positional = code.co_varnames[: code.co_argcount]
    bindings.extend(
        zip(positional[len(positional) - len(defaults) :], defaults, strict=True),
    )
    bindings.extend((function.__kwdefaults__ or {}).items())
    return bindings


def _cell_is_bound(cell: types.CellType) -> bool:
    try:
        # Reading an empty cell is the only emptiness test.
        _ = cast(object, cell.cell_contents)
    except ValueError:
        return False
    return True


# Rendering a deep tree requalifies each level's subtree, so every lambda is
# looked up once per enclosing level; re-reading and re-parsing its source file
# each time made a 16-level render take 44 seconds.
@functools.cache
def _lambda_source(code: types.CodeType) -> str:
    """Find the lambda whose body matches this code object's source positions."""
    try:
        lines, first_line = inspect.getsourcelines(code)
        original = "".join(lines)
        source = textwrap.dedent(original)
        tree = ast.parse(source)
    except (OSError, TypeError, SyntaxError):
        return "<lambda>"
    positions = set(code.co_positions())
    original_lines = original.splitlines()
    source_lines = source.splitlines()
    for node in ast.walk(tree):
        if not isinstance(node, ast.Lambda):
            continue
        body = node.body
        end_lineno = cast(int, body.end_lineno)
        end_col_offset = cast(int, body.end_col_offset)
        start_indent = len(original_lines[body.lineno - 1]) - len(
            source_lines[body.lineno - 1],
        )
        end_indent = len(original_lines[end_lineno - 1]) - len(
            source_lines[end_lineno - 1],
        )
        position = (
            first_line + body.lineno - 1,
            first_line + end_lineno - 1,
            body.col_offset + start_indent,
            end_col_offset + end_indent,
        )
        if position in positions:
            return ast.unparse(node)
    return "<lambda>"


def _collect_functions(
    root: object,
    out: list[types.FunctionType],
    ancestors: set[int],
) -> None:
    """Append each function in one object tree, in traversal order."""
    stack: list[object | _Leave] = [root]
    while stack:
        current = stack.pop()
        if isinstance(current, _Leave):
            ancestors.remove(current.identity)
            continue
        if isinstance(current, types.FunctionType):
            out.append(current)
            continue
        identity = id(current)
        if identity in ancestors:
            continue
        ancestors.add(identity)
        stack.append(_Leave(identity=identity))
        stack.extend(reversed(_function_repr_children(current)))


@dataclasses.dataclass(frozen=True, slots=True, kw_only=True)
class _Leave:
    """Marks where ``_collect_functions`` leaves a node's subtree."""

    identity: int


def _replace_unquoted(
    text: str,
    bare: str,
    qualified: str,
    *,
    spans: list[tuple[int, int]],
) -> tuple[str, list[tuple[int, int]]]:
    """Replace ``bare``'s first unquoted occurrence; return text and shifted spans."""
    for match in re.finditer(re.escape(bare), text):
        start, end = match.span()
        if any(start < stop and end > begin for begin, stop in spans):
            continue
        shift = len(qualified) - len(bare)
        shifted = [
            (begin + shift, stop + shift) if begin >= end else (begin, stop)
            for begin, stop in spans
        ]
        return text[:start] + qualified + text[end:], shifted
    return text, spans


def _string_token_spans(text: str) -> list[tuple[int, int]]:
    """Return absolute spans occupied by Python string tokens."""
    # Every string token holds a quote; most reprs hold none.
    if "'" not in text and '"' not in text:
        return []
    line_offsets = [0]
    for line in text.splitlines(keepends=True):
        line_offsets.append(line_offsets[-1] + len(line))
    spans = list[tuple[int, int]]()
    try:
        for token in tokenize.generate_tokens(io.StringIO(text).readline):
            if token.type == tokenize.STRING:
                spans.append(
                    (
                        line_offsets[token.start[0] - 1] + token.start[1],
                        line_offsets[token.end[0] - 1] + token.end[1],
                    ),
                )
    except tokenize.TokenError:
        pass
    return spans


def _function_repr_children(value: object) -> list[object]:
    """Return children whose rendered function reprs need qualification."""
    if isinstance(value, functools.partial):
        children: list[object] = [value.func]
        children.extend(cast(tuple[object, ...], value.args))
        children.extend(cast(dict[str, object], value.keywords).values())
        return children
    if isinstance(value, dict):
        return [*value.keys(), *value.values()]
    if isinstance(value, (list, tuple, set, frozenset)):
        return list(
            cast(
                list[object] | tuple[object, ...] | set[object] | frozenset[object],
                value,
            ),
        )
    if dataclasses.is_dataclass(value) and not isinstance(value, type):
        return [
            getattr(value, field.name)
            for field in dataclasses.fields(value)
            if hasattr(value, field.name)
        ]
    return []


def _get_level_indents(level: int, indent_per_level: int) -> tuple[int, int]:
    """Return (item_indent, base_indent) for a given nesting level."""
    item_indent = indent_per_level * (level + 1)
    base_indent = item_indent - indent_per_level
    return item_indent, base_indent


def _collapse_multiline_value(formatted_value: str, max_width: int) -> str:
    """Collapse multiline value to a single line if short enough."""
    if "\n" not in formatted_value:
        return formatted_value

    # Collapsing keeps every non-space character, so more than ``max_width`` of
    # them keeps the value multiline. Counting stops there: tokenizing and
    # collapsing each level's whole subtree text dominated the render.
    visible = 0
    for word in re.finditer(r"\S+", formatted_value):
        visible += word.end() - word.start()
        if visible > max_width:
            return formatted_value

    if _contains_repeated_string_whitespace(formatted_value):
        return formatted_value

    oneline = re.sub(r"\s+", " ", formatted_value).strip()
    oneline = oneline.replace("( ", "(").replace(" )", ")")

    if len(oneline) <= max_width:
        return oneline
    return formatted_value


def _contains_repeated_string_whitespace(value: str) -> bool:
    """Return whether collapsing whitespace would alter a string token."""
    try:
        tokens = tokenize.generate_tokens(io.StringIO(value).readline)
        return any(
            token.type == tokenize.STRING and re.search(r"\s{2,}", token.string)
            for token in tokens
        )
    except tokenize.TokenError:
        return True


def _replace_char_at_column(line: str, column: int, char: str) -> str:
    """Replace character at column position if it's whitespace."""
    if len(line) > column and line[column].isspace():
        return line[:column] + char + line[column + 1 :]
    return line


def _add_pipes_to_lines(lines: list[str], pipe_column: int) -> list[str]:
    """Add continuation pipes to lines at the given column."""
    if not lines:
        return lines

    result = [lines[0]]
    for i, line in enumerate(lines[1:], 1):
        is_last = i == len(lines) - 1
        pipe_char = " " if is_last else "│"
        result.append(_replace_char_at_column(line, pipe_column, pipe_char))

    return result


def _should_add_continuation_pipes(
    formatted_value: str,
    num_items: int,
    continuation_pipe_threshold: int,
) -> bool:
    """Determine if continuation pipes should be added to formatted value."""
    if continuation_pipe_threshold < 0:
        return False
    if num_items <= 1:
        return False
    if "\n" not in formatted_value:
        return False

    num_lines = formatted_value.count("\n") + 1
    return num_lines >= continuation_pipe_threshold


def _filter_non_default_items(
    obj: object,
    items: list[tuple[str, object]],
) -> list[tuple[str, object]]:
    """Filter fields equal to side-effect-free, scalar-comparable defaults."""
    fields = {
        field.name: field for field in dataclasses.fields(cast(DataclassLike, obj))
    }
    filtered = list[tuple[str, object]]()
    for name, value in items:
        field = fields[name]
        if field.default is dataclasses.MISSING:
            filtered.append((name, value))
            continue
        default_value = cast(object, field.default)
        try:
            if value != default_value:
                filtered.append((name, value))
        except Exception:  # noqa: BLE001 -- a field whose __eq__ raises is shown, not hidden.
            filtered.append((name, value))
    return filtered


def _mask_memory_addresses(text: str) -> str:
    """Replace object-repr memory addresses outside string tokens."""
    matches = list(re.finditer(r"(?<= at )0x[0-9a-fA-F]+", text))
    if not matches:
        return text
    string_spans = _string_token_spans(text)
    for match in reversed(matches):
        open_spans = [
            (start, end) for start, end in string_spans if match.start() < end
        ]
        if any(match.end() > start for start, _ in open_spans):
            continue
        # One literal for every masked address, on every platform. Goldens are
        # shared across machines, so the width must not come from the recording
        # host: sizing it from ``id(object())`` made a golden recorded on a 64-bit
        # interpreter unreproducible on a 32-bit one, and sizing it per match
        # leaked the original address's length. (0xdefaced is prime.)
        text = text[: match.start()] + "0xdefacedeface" + text[match.end() :]
    return text
