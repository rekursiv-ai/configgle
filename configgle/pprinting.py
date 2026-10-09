"""Pretty printing utilities for Fig config objects."""

from __future__ import annotations

from pprint import PrettyPrinter
from typing import IO, Final, Protocol, TypeVar, cast, override

import dataclasses
import functools
import io
import re
import tokenize
import types
import warnings

from configgle.custom_types import DataclassLike, Finalizeable
from configgle.walk import copy_tree


__all__ = [
    "pformat",
    "pprint",
]

_T = TypeVar("_T")
_T_contra = TypeVar("_T_contra", contravariant=True)


class SupportsWrite(Protocol[_T_contra]):
    """Protocol for objects that support write method."""

    def write(self, s: _T_contra, /) -> object: ...


_DEFAULT_CONTINUATION_PIPE_THRESHOLD: Final = 50

_SHORT_SEQUENCE_MAX_WIDTH: Final = 40

_ADDRESS_PATTERN: Final = r"(?<= at )0x[0-9a-fA-F]+"

# One literal for every masked address, on every platform. Goldens are shared
# across machines, so the width must not come from the recording host: sizing it
# from ``id(object())`` made a golden recorded on a 64-bit interpreter
# unreproducible on a 32-bit one, and sizing it per match leaked the original
# address's length. (0xdefaced is prime.)
_ADDRESS_MASK: Final = "0xdefacedeface"


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
    return printer.pprint(obj)


class FigPrinter(PrettyPrinter):
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
        # re-set inherited private attrs; type checkers don't see parent's writes.
        self._indent_per_level: int = indent
        self._width: int = width
        self._depth: int | None = depth
        self._finalize = finalize
        self._mask_memory_addresses = (
            _mask_memory_addresses if mask_memory_addresses else None
        )
        self._extra_compact = extra_compact
        self._continuation_pipe = continuation_pipe
        self._hide_default_values = hide_default_values
        self._short_sequence_max_width = short_sequence_max_width
        self._finalized_copies: dict[int, tuple[object, object]] | None = None

    @override
    def pprint(self, object: object) -> None:
        owns_cache = self._finalized_copies is None
        if owns_cache:
            self._finalized_copies = {}
        try:
            return super().pprint(self._try_to_finalize(object))
        finally:
            if owns_cache:
                self._finalized_copies = None

    @override
    def pformat(self, object: object) -> str:
        owns_cache = self._finalized_copies is None
        if owns_cache:
            self._finalized_copies = {}
        try:
            return super().pformat(self._try_to_finalize(object))
        finally:
            if owns_cache:
                self._finalized_copies = None

    @override
    def format(
        self,
        object: object,
        context: dict[int, int],
        maxlevels: int,
        level: int,
    ) -> tuple[str, bool, bool]:
        object = self._try_to_finalize(object)
        repr_, readable, recursive = super().format(
            object,
            context,
            maxlevels,
            level,
        )
        repr_ = _qualify_function_reprs(object, repr_)
        if self._mask_memory_addresses is not None:
            repr_ = self._mask_memory_addresses(repr_)
        return repr_, readable, recursive

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
        object = self._try_to_finalize(object)
        # CPython renders a container's or dataclass's whole subtree only to measure
        # it, then lays a too-wide one out item by item and discards the text.
        # Qualifying and masking that text made a render quadratic in depth: every
        # level re-tokenized its descendants. A lower bound on the processed width
        # decides CPython's branch without the text.
        if (
            (type(object) in (list, tuple, dict) or _is_laid_out_by_field(object))
            and id(object) not in context
            and (floor := self._repr_floor(object, context.copy())) is not None
            and floor > self._width - indent - allowance
        ):
            context[id(object)] = 1
            self._expand(object, stream, indent, allowance, context, level + 1)
            del context[id(object)]
            return
        super()._format(object, stream, indent, allowance, context, level)

    def _expand(  # noqa: PLR0917 -- mirrors CPython's positional pprint dispatch signature.
        self,
        obj: object,
        stream: SupportsWrite[str],
        indent: int,
        allowance: int,
        context: dict[int, int],
        level: int,
    ) -> None:
        """Lay ``obj`` out with the method CPython's ``_format`` dispatches it to."""
        if isinstance(obj, list):
            items = cast(list[object], obj)
            self._pprint_list(items, stream, indent, allowance, context, level)
        elif isinstance(obj, tuple):
            values = cast(tuple[object, ...], obj)
            self._pprint_tuple(values, stream, indent, allowance, context, level)
        elif isinstance(obj, dict):
            mapping = cast(dict[object, object], obj)
            self._pprint_dict(mapping, stream, indent, allowance, context, level)
        else:
            self._pprint_dataclass(obj, stream, indent, allowance, context, level)

    # Qualifying only inserts module paths, and masking shortens a repr by at most
    # each address's excess over the mask, so the raw text bounds the processed one;
    # a container's bound sums its items'. ``None`` marks a node whose render
    # finalizes something, recurses, or is cut by ``depth``: CPython's own path
    # renders it, with the same side effects.
    def _repr_floor(self, obj: object, context: dict[int, int]) -> int | None:
        """Return a lower bound on the length of ``format(obj)``, or ``None``."""
        if self._finalize_pending(obj):
            return None
        if type(obj) not in (list, tuple, dict):
            # ``_safe_repr`` renders a builtin container's subclass item by item.
            if isinstance(obj, (list, tuple, dict)):
                return None
            rendered = repr(obj)
            return len(rendered) - sum(
                max(0, address.end() - address.start() - len(_ADDRESS_MASK))
                for address in re.finditer(_ADDRESS_PATTERN, rendered)
            )
        if not obj:
            return 2
        identity = id(obj)
        if self._depth is not None or identity in context:
            return None
        if isinstance(obj, dict):
            mapping = cast(dict[object, object], obj)
            items = [*mapping.keys(), *mapping.values()]
        else:
            items = list(cast(list[object] | tuple[object, ...], obj))
        # Brackets and ", " between items; a dict adds ": " per pair.
        total = 2 * len(items)
        context[identity] = 1
        for item in items:
            floor = self._repr_floor(item, context)
            # The caller discards this context copy, so it needs no cleanup here.
            if floor is None:
                return None
            total += floor
        del context[identity]
        return total

    def _finalize_pending(self, obj: object) -> bool:
        """Return whether ``_try_to_finalize`` would copy and finalize ``obj``."""
        # An exact builtin type has no ``finalize``, and the runtime protocol check
        # costs microseconds on each of a render's thousands of leaves.
        if type(obj) in (int, float, str, bool, bytes, type(None), list, dict, tuple):
            return False
        finalized_flag = getattr(obj, "_finalized", False)
        return self._finalize and not finalized_flag and isinstance(obj, Finalizeable)

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

    # CPython's PrettyPrinter dispatches to ``_pprint_dataclass`` for dataclass
    # instances. We override it to hide default-valued fields and use our extra-compact
    # layout.
    def _pprint_dataclass(  # noqa: PLR0917 -- pprint override; CPython dispatches positionally, keyword-only params break it.
        self,
        obj: object,
        stream: SupportsWrite[str],
        indent: int,
        allowance: int,
        context: dict[int, int],
        level: int,
    ) -> None:
        """Format a dataclass instance."""
        cls_name = obj.__class__.__qualname__
        indent += len(cls_name) + 1
        items = [
            (f.name, getattr(obj, f.name))
            for f in dataclasses.fields(cast(DataclassLike, obj))
            if f.repr
        ]

        if self._hide_default_values:
            items = _filter_non_default_items(obj, items)

        stream.write(cls_name + "(")
        self._format_namespace_items(items, stream, indent, allowance, context, level)
        stream.write(")")

    def _format_namespace_items(  # noqa: PLR0917 -- pprint override; CPython dispatches positionally, keyword-only params break it.
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
            super()._format_namespace_items(  # pyright: ignore[reportAttributeAccessIssue,reportUnknownMemberType] -- Private PrettyPrinter method is absent from typeshed.  # ty: ignore[unresolved-attribute] -- Private PrettyPrinter method is absent from typeshed.
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
                    context=context,
                    level=level,
                    item_indent=item_indent,
                    allowance=allowance if last else 1,
                    num_items=len(items),
                )
                write(formatted_value)

            if not last:
                write(",\n")

        write("\n")
        write(base_indent)

    def _format_namespace_value(
        self,
        value: object,
        *,
        context: dict[int, int],
        level: int,
        item_indent: int,
        allowance: int,
        num_items: int,
    ) -> str:
        """Format a namespace value with collapsing and continuation pipes."""
        temp_stream = io.StringIO()
        self._format(value, temp_stream, item_indent, allowance, context, level)
        formatted_value = temp_stream.getvalue()

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

        one_line_str = self._try_format_items_on_one_line(
            items,
            context,
            level,
            indent=indent,
            allowance=allowance,
        )
        content_width = len(one_line_str) + 2

        if self._should_format_on_one_line(content_width, indent, allowance):
            stream.write(one_line_str)
        else:
            self._format_items_multiline(items, stream, context, level)

    def _try_format_items_on_one_line(
        self,
        items: list[object],
        context: dict[int, int],
        level: int,
        *,
        indent: int,
        allowance: int,
    ) -> str:
        """Format items on a single line, stopping once the line cannot fit."""
        pieces: list[str] = []
        content_width = 0
        for item in items:
            piece = io.StringIO()
            self._format(item, piece, 0, 0, context, level)
            pieces.append(piece.getvalue())
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


def _qualify_function_reprs(value: object, rendered: str) -> str:
    """Add module paths to function reprs nested in supported containers."""
    # Every function repr starts so; skipping the walk without one saves a pass
    # over each value's subtree.
    if "<function " not in rendered:
        return rendered
    functions: list[types.FunctionType] = []
    _collect_functions(value, functions, set())
    if not any(repr(function) in rendered for function in functions):
        return rendered
    # Tokenized once: re-tokenizing per function cost 2.5s on one config. A
    # function repr holds no quote, so a replacement only shifts later spans.
    spans = _string_token_spans(rendered)
    for function in functions:
        bare = repr(function)
        qualified = bare.replace("<function ", f"<function {function.__module__}.")
        rendered, spans = _replace_unquoted(rendered, bare, qualified, spans=spans)
    return rendered


def _collect_functions(
    current: object,
    out: list[types.FunctionType],
    ancestors: set[int],
) -> None:
    """Append each function in one object tree, in traversal order."""
    if isinstance(current, types.FunctionType):
        out.append(current)
        return
    identity = id(current)
    if identity in ancestors:
        return
    ancestors.add(identity)
    for child in _function_repr_children(current):
        _collect_functions(child, out, ancestors)
    ancestors.remove(identity)


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


class _ReprParam(Protocol):
    """The one ``__dataclass_params__`` member CPython's ``_format`` reads."""

    repr: bool


class _DataclassLayout(Protocol):
    """A dataclass as CPython's ``_format`` inspects it."""

    __dataclass_params__: _ReprParam


# CPython's own test, narrowed to a function ``__wrapped__``, so every object this
# accepts CPython lays out field by field too. A generated ``__repr__`` is never a
# ``_dispatch`` key, which CPython checks first. Cast, not a runtime protocol: this
# runs on every dataclass a render meets.
def _is_laid_out_by_field(obj: object) -> bool:
    """Return whether CPython's ``_format`` expands ``obj`` with ``_pprint_dataclass``."""
    if not dataclasses.is_dataclass(obj) or isinstance(obj, type):
        return False
    wrapped = cast(object, getattr(obj.__repr__, "__wrapped__", None))
    return (
        isinstance(wrapped, types.FunctionType)
        and "__create_fn__" in wrapped.__qualname__
        and cast(_DataclassLayout, obj).__dataclass_params__.repr
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
    matches = list(re.finditer(_ADDRESS_PATTERN, text))
    if not matches:
        return text
    string_spans = _string_token_spans(text)
    for match in reversed(matches):
        open_spans = [
            (start, end) for start, end in string_spans if match.start() < end
        ]
        if any(match.end() > start for start, _ in open_spans):
            continue
        text = text[: match.start()] + _ADDRESS_MASK + text[match.end() :]
    return text
