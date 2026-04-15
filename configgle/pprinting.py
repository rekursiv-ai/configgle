"""Pretty printing utilities for Fig config objects."""

from __future__ import annotations

from collections.abc import Callable
from pprint import PrettyPrinter as _PrettyPrinter
from typing import IO, Protocol, TypeVar, override

import copy
import dataclasses
import io
import re
import warnings

from configgle.custom_types import Finalizeable


__all__ = [
    "FigPrinter",
    "pformat",
    "pprint",
]

_T = TypeVar("_T")
_T_contra = TypeVar("_T_contra", contravariant=True)


class SupportsWrite(Protocol[_T_contra]):
    """Protocol for objects that support write method."""

    def write(self, s: _T_contra, /) -> object: ...


# Default threshold for continuation pipes (lines)
_DEFAULT_CONTINUATION_PIPE_THRESHOLD = 50

# Maximum width for sequences to always stay on one line
_SHORT_SEQUENCE_MAX_WIDTH = 40


def pformat(
    obj: object,
    indent: int = 8,
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
      hide_default_values: Omit fields with default values.
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
    indent: int = 8,
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
) -> None:
    """Pretty-print object with Fig-aware formatting.

    Args:
      obj: Object to print.
      stream: Output stream (defaults to sys.stdout).
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
      hide_default_values: Omit fields with default values.
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
    return printer.pprint(obj)


class FigPrinter(_PrettyPrinter):
    """PrettyPrinter subclass with Fig-specific formatting enhancements."""

    def __init__(
        self,
        stream: IO[str] | None = None,
        indent: int = 8,
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
        # Explicitly set inherited private attrs for type checking
        # (parent sets these same values, but type checkers don't see it)
        self._indent_per_level: int = indent
        self._width: int = width
        self._finalize = finalize
        self._mask_memory_addresses = (
            _MASK_MEMORY_ADDRESSES_FN if mask_memory_addresses else None
        )
        self._extra_compact = extra_compact
        self._continuation_pipe = continuation_pipe
        self._hide_default_values = hide_default_values
        self._short_sequence_max_width = short_sequence_max_width

    @override
    def pprint(self, object: object) -> None:
        return super().pprint(self._try_to_finalize(object))

    @override
    def pformat(self, object: object) -> str:
        return super().pformat(self._try_to_finalize(object))

    @override
    def format(
        self,
        object: object,
        context: dict[int, int],
        maxlevels: int,
        level: int,
    ) -> tuple[str, bool, bool]:
        if (
            self._finalize
            and callable(getattr(object, "make", None))
            and callable(getattr(object, "finalize", None))
            and not getattr(object, "_finalized", False)
        ):
            warnings.warn(
                f"Found potentially unfinalized dataclass: {object}.",
                stacklevel=2,
            )
        repr_, readable, recursive = super().format(
            object,
            context,
            maxlevels,
            level,
        )
        if self._mask_memory_addresses is not None:
            repr_ = self._mask_memory_addresses(repr_)
        return repr_, readable, recursive

    def _try_to_finalize(self, obj: _T) -> _T:
        """Deep-copy then finalize for display purposes.

        Deep-copies first so that finalization (which may mutate) doesn't
        alter the caller's config. This keeps printing side-effect-free.
        """
        if (
            self._finalize
            and isinstance(obj, Finalizeable)
            and not getattr(obj, "_finalized", False)
        ):
            try:
                obj = copy.deepcopy(obj)
                obj = obj.finalize()  # ty: ignore[invalid-assignment]
            except Exception as e:  # noqa: BLE001
                warnings.warn(str(e), stacklevel=2)
        return obj

    def _pprint_dataclass(
        self,
        obj: object,
        stream: SupportsWrite[str],
        indent: int,
        allowance: int,
        context: dict[int, int],
        level: int,
    ) -> None:
        """Format a dataclass instance.

        CPython's PrettyPrinter dispatches to ``_pprint_dataclass`` for
        dataclass instances. We override it to hide default-valued fields
        and use our extra-compact layout.
        """
        cls_name = obj.__class__.__qualname__
        indent += len(cls_name) + 1
        items = [
            (f.name, getattr(obj, f.name))
            for f in dataclasses.fields(obj)  # pyright: ignore[reportArgumentType]  # ty: ignore[invalid-argument-type]
            if f.repr
        ]

        # Filter out default values if requested
        if self._hide_default_values:
            items = _filter_non_default_items(obj, items)

        stream.write(cls_name + "(")
        self._format_namespace_items(items, stream, indent, allowance, context, level)
        stream.write(")")

    def _format_namespace_items(
        self,
        items: list[tuple[str, object]],
        stream: SupportsWrite[str],
        indent: int,
        allowance: int,
        context: dict[int, int],
        level: int,
    ) -> None:
        """Override to use fixed indent and put each parameter on its own line."""
        if not self._extra_compact:
            # PrettyPrinter private method
            super()._format_namespace_items(  # pyright: ignore[reportAttributeAccessIssue,reportUnknownMemberType]  # ty: ignore[unresolved-attribute]
                items,
                stream,
                indent,
                allowance,
                context,
                level,
            )
            return

        if not items:
            return

        write = stream.write
        write("\n")

        item_indent, base_indent_val = _get_level_indents(
            level,
            self._indent_per_level,
        )
        base_indent = " " * base_indent_val

        for i, (key, ent) in enumerate(items):
            last = i == len(items) - 1

            write(" " * item_indent)
            write(key)
            write("=")

            if id(ent) in context:
                write("...")
            else:
                formatted_value = self._format_namespace_value(
                    ent,
                    context,
                    level,
                    item_indent,
                    allowance if last else 1,
                    len(items),
                )
                write(formatted_value)

            if not last:
                write(",\n")

        write("\n")
        write(base_indent)

    def _format_namespace_value(
        self,
        value: object,
        context: dict[int, int],
        level: int,
        item_indent: int,
        allowance: int,
        num_items: int,
    ) -> str:
        """Format a namespace value with collapsing and continuation pipes."""
        # Format value to string
        temp_stream = io.StringIO()
        self._format(value, temp_stream, item_indent, allowance, context, level)
        formatted_value = temp_stream.getvalue()

        # Try to collapse short multiline values onto one line
        formatted_value = _collapse_multiline_value(
            formatted_value,
            self._short_sequence_max_width,
        )

        # Add continuation pipes if needed
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
    def _format_items(
        self,
        items: list[object],
        stream: SupportsWrite[str],
        indent: int,
        allowance: int,
        context: dict[int, int],
        level: int,
    ) -> None:
        """Override to use level-based indent instead of accumulated indent."""
        if not self._extra_compact:
            super()._format_items(items, stream, indent, allowance, context, level)
            return

        one_line_str = self._try_format_items_on_one_line(items, context, level)
        content_width = len(one_line_str) + 2  # Add 2 for surrounding brackets/parens

        if self._should_format_on_one_line(content_width, indent, allowance):
            stream.write(one_line_str)
        else:
            self._format_items_multiline(items, stream, context, level)

    def _try_format_items_on_one_line(
        self,
        items: list[object],
        context: dict[int, int],
        level: int,
    ) -> str:
        """Try to format items on a single line."""
        one_line = io.StringIO()
        delim = ""
        for item in items:
            one_line.write(delim)
            self._format(item, one_line, 0, 0, context, level)
            delim = ", "
        return one_line.getvalue()

    def _should_format_on_one_line(
        self,
        content_width: int,
        indent: int,
        allowance: int,
    ) -> bool:
        """Determine if items should be formatted on one line."""
        # Keep short sequences on one line regardless of nesting depth
        # (content_width doesn't include indent, so short tuples stay compact even when deeply nested)
        # For longer sequences, check if they fit within the available width
        return (
            content_width < self._short_sequence_max_width
            or indent + content_width + allowance <= self._width
        )

    def _format_items_multiline(
        self,
        items: list[object],
        stream: SupportsWrite[str],
        context: dict[int, int],
        level: int,
    ) -> None:
        """Format items across multiple lines with level-based indent."""
        write = stream.write
        write("\n")

        item_indent, base_indent_val = _get_level_indents(
            level,
            self._indent_per_level,
        )
        indent_str = " " * item_indent

        for i, ent in enumerate(items):
            last = i == len(items) - 1
            write(indent_str)

            if id(ent) in context:
                write("...")
            else:
                formatted_value = self._format_and_collapse_item(
                    ent,
                    context,
                    level,
                    item_indent,
                )
                stream.write(formatted_value)

            if not last:
                write(",\n")

        write("\n")
        write(" " * base_indent_val)

    def _format_and_collapse_item(
        self,
        item: object,
        context: dict[int, int],
        level: int,
        item_indent: int,
    ) -> str:
        """Format an item to a string and collapse if short enough."""
        temp_stream = io.StringIO()
        self._format(item, temp_stream, item_indent, 1, context, level)
        formatted_value = temp_stream.getvalue()
        return _collapse_multiline_value(
            formatted_value,
            self._short_sequence_max_width,
        )


def _get_level_indents(level: int, indent_per_level: int) -> tuple[int, int]:
    """Return (item_indent, base_indent) for a given nesting level."""
    item_indent = indent_per_level * (level + 1)
    base_indent = item_indent - indent_per_level
    return item_indent, base_indent


def _collapse_multiline_value(formatted_value: str, max_width: int) -> str:
    """Collapse multiline value to a single line if short enough."""
    if "\n" not in formatted_value:
        return formatted_value

    # Remove newlines and collapse whitespace
    oneline = re.sub(r"\s+", " ", formatted_value.replace("\n", ""))
    # Clean up spaces around parentheses
    oneline = oneline.replace("( ", "(").replace(" )", ")")

    # Use collapsed version if short enough
    if len(oneline) <= max_width:
        return oneline
    return formatted_value


def _replace_char_at_column(line: str, column: int, char: str) -> str:
    """Replace character at column position if it's whitespace."""
    if len(line) > column and line[column].isspace():
        return line[:column] + char + line[column + 1 :]
    return line


def _add_pipes_to_lines(lines: list[str], pipe_column: int) -> list[str]:
    """Add continuation pipes to lines at the given column."""
    if not lines:
        return lines

    result = [lines[0]]  # First line unchanged
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
    return continuation_pipe_threshold == 0 or num_lines >= continuation_pipe_threshold


def _filter_non_default_items(
    obj: object,
    items: list[tuple[str, object]],
) -> list[tuple[str, object]]:
    """Filter out items whose values match the default-constructed instance."""
    try:
        # Get the class and instantiate a default instance
        cls = type(obj)
        default_obj = cls()

        # Filter items - keep only non-default values
        filtered = list[tuple[str, object]]()
        for name, value in items:
            default_value = getattr(default_obj, name)
            if value != default_value:
                filtered.append((name, value))

        return filtered
    except Exception:  # noqa: BLE001
        # If we can't create defaults (e.g., required args), return all items
        return items


def _make_memory_address_masker() -> Callable[[str], str]:
    """Replace memory addresses (e.g., ``0x7f...``) with a fixed placeholder.

    Object reprs include addresses that change every run, making string
    comparisons and snapshot tests brittle. A fixed placeholder gives
    stable, reproducible output.
    """
    n = len(str(lambda: None)[:-1].split(" at 0x")[-1])
    pattern = re.compile(rf"0x[a-f0-9]{{{n}}}")
    # Fun fact: 0x0defaced is a prime number.
    replace = "0x0defaced0defaced"
    replace = replace[: min(len(replace), 2 + n)]

    def mask_memory_addresses(x: str) -> str:
        return pattern.sub(replace, x)

    return mask_memory_addresses


_MASK_MEMORY_ADDRESSES_FN = _make_memory_address_masker()
