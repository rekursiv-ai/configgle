"""Tests for pprinting module."""

from __future__ import annotations

from collections.abc import Callable
from io import StringIO
from pathlib import Path
from typing import Self, override

import ast
import copy
import dataclasses
import functools
import glob
import inspect
import re
import types
import warnings

import pytest

from configgle import Fig, PartialConfig, pprinting
from configgle.pprinting import (
    _MASKED_MEMORY_ADDRESS,
    FigPrinter,
    _add_pipes_to_lines,
    _collapse_multiline_value,
    _filter_non_default_items,
    _get_level_indents,
    _mask_memory_addresses,
    _qualify_function_reprs,
    _replace_char_at_column,
    _should_add_continuation_pipes,
    pformat,
    pprint,
)


class MockConfigurable:
    """Mock configurable object for testing."""

    def __init__(self, value: int, finalized: bool = False):
        self.value = value
        self._finalized = finalized

    def make(self) -> Self:
        return self

    def finalize(self) -> Self:
        new = copy.copy(self)
        new._finalized = True
        return new


def test_pformat_basic():
    """Test pformat function with basic object."""
    result = pformat({"a": 1, "b": 2})
    assert "'a': 1" in result
    assert "'b': 2" in result


def test_pformat_with_options():
    """Test pformat with various options."""
    obj = {"a": 1_000_000, "b": 2_000_000}

    # Test with underscore_numbers
    result = pformat(obj, underscore_numbers=True)
    assert "1_000_000" in result

    # Test without underscore_numbers
    result = pformat(obj, underscore_numbers=False)
    assert "1000000" in result
    assert "1_000_000" not in result


def test_pformat_mask_memory_addresses():
    """Test pformat with mask_memory_addresses option."""

    class Obj:
        pass

    obj = Obj()
    result = pformat(obj, mask_memory_addresses=True)
    assert _MASKED_MEMORY_ADDRESS in result
    assert repr(obj) not in result


def _identity_function(
    function: Callable[[str], str],
) -> Callable[[str], str]:
    return function


def _partial_function(function: Callable[[str], str]) -> PartialConfig[str]:
    return PartialConfig(function)


@pytest.mark.parametrize("wrap", [_identity_function, _partial_function])
@pytest.mark.parametrize("mask_memory_addresses", [False, True])
def test_function_repr_preserves_callable_identity(
    wrap: Callable[[Callable[[str], str]], object],
    mask_memory_addresses: bool,
) -> None:
    re_escape = pformat(
        wrap(re.escape),
        mask_memory_addresses=mask_memory_addresses,
    )
    glob_escape = pformat(
        wrap(glob.escape),
        mask_memory_addresses=mask_memory_addresses,
    )

    assert re_escape != glob_escape
    assert "re.escape" in re_escape
    assert "glob.escape" in glob_escape


def test_function_repr_preserves_callable_identity_in_sets() -> None:
    rendered = pformat({re.escape, glob.escape})

    assert "re.escape" in rendered
    assert "glob.escape" in rendered


def test_function_qualification_binds_modules_to_exact_addresses() -> None:
    first = types.FunctionType((lambda: None).__code__, {"__name__": "first_module"})
    second = types.FunctionType((lambda: None).__code__, {"__name__": "second_module"})
    first.__qualname__ = second.__qualname__ = "shared"
    rendered = f"[{second!r}, {first!r}]"

    qualified = _qualify_function_reprs([first, second], rendered)

    assert qualified == (
        f"[<function second_module.shared at {hex(id(second))}>, "
        f"<function first_module.shared at {hex(id(first))}>]"
    )


def test_function_qualification_ignores_repr_inside_string_value() -> None:
    function = types.FunctionType(
        (lambda: None).__code__,
        {"__name__": "string_collision_module"},
    )
    function.__qualname__ = "shared"
    rendered = repr([repr(function), function])

    qualified = _qualify_function_reprs([repr(function), function], rendered)

    assert qualified == (
        f"[{repr(function)!r}, "
        f"<function string_collision_module.shared at {hex(id(function))}>]"
    )


def test_function_qualification_handles_aliased_containers() -> None:
    function = types.FunctionType(
        (lambda: None).__code__,
        {"__name__": "aliased_module"},
    )
    function.__qualname__ = "shared"
    child = [function]
    rendered = repr([child, child])

    qualified = _qualify_function_reprs([child, child], rendered)

    expected = f"<function aliased_module.shared at {hex(id(function))}>"
    assert qualified == f"[[{expected}], [{expected}]]"


def test_owned_files_have_no_long_nested_functions() -> None:
    # Located through the imported module, not as a sibling file: the export
    # moves tests to ``tests/`` and sources into the package, so the two stop
    # sharing a directory.
    paths = [Path(__file__), Path(inspect.getfile(pprinting))]

    assert {path.name: _long_nested_functions(path) for path in paths} == {
        "pprinting.py": [],
        "pprinting_test.py": [],
    }


def test_pformat_finalize():
    """Test pformat with finalize option."""
    cfg = MockConfigurable(42, finalized=False)

    # With finalize=True (default) - MockConfigurable satisfies Makeable,
    # so it gets finalized automatically (no warning).
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        result = pformat(cfg, finalize=True)
        assert "MockConfigurable" in result
        assert len(w) == 0

    # With finalize=False - should not warn
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        result = pformat(cfg, finalize=False)
        assert "MockConfigurable" in result
        assert len(w) == 0


def test_pprint_basic():
    """Test pprint function."""
    stream = StringIO()
    pprint({"a": 1, "b": 2}, stream=stream)
    output = stream.getvalue()
    assert "'a': 1" in output
    assert "'b': 2" in output


def test_pprint_with_options():
    """Test pprint with various options."""
    stream = StringIO()
    obj = {"a": 1_000_000}

    pprint(
        obj,
        stream=stream,
        indent=2,
        width=120,
        underscore_numbers=True,
        finalize=False,
    )
    output = stream.getvalue()
    assert "1_000_000" in output


def test_pretty_printer_init():
    """Test FigPrinter initialization."""
    pp = FigPrinter(
        indent=2,
        width=120,
        depth=3,
        compact=True,
        sort_dicts=True,
        underscore_numbers=True,
        finalize=True,
        mask_memory_addresses=True,
    )
    assert pp._finalize is True
    assert pp._mask_memory_addresses is not None


def test_pretty_printer_pprint():
    """Test FigPrinter.pprint method."""
    stream = StringIO()
    pp = FigPrinter(stream=stream)
    pp.pprint({"a": 1, "b": 2})
    output = stream.getvalue()
    assert "'a': 1" in output


def test_pretty_printer_pformat():
    """Test FigPrinter.pformat method."""
    pp = FigPrinter()
    result = pp.pformat({"a": 1, "b": 2})
    assert "'a': 1" in result


def test_pretty_printer_format_finalizes_nested_configs_without_warning() -> None:
    """Test FigPrinter.format finalizes configs reached below the root."""
    finalizations = 0

    class UnfinalizedConfig:
        def __init__(self) -> None:
            self._finalized = False

        def make(self) -> Self:
            return self

        def finalize(self) -> Self:
            nonlocal finalizations
            finalizations += 1
            new = copy.copy(self)
            new._finalized = True
            return new

    pp = FigPrinter(finalize=True)
    cfg = UnfinalizedConfig()

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        pp.format(cfg, {}, 0, 0)

    assert not caught
    assert finalizations == 1


def test_nested_config_finalization_is_independent_of_render_width() -> None:
    class DerivedConfig(Fig):
        value: int = -1

        @override
        def finalize(self) -> Self:
            if self.value == -1:
                self.value = 7
            return super().finalize()

    wide = pformat({"cfg": DerivedConfig()}, width=200, hide_default_values=False)
    narrow = pformat({"cfg": DerivedConfig()}, width=20, hide_default_values=False)

    assert "value=7" in wide
    assert "value=7" in narrow
    assert "value=-1" not in narrow


def test_multiline_speculative_format_finalizes_nested_config_once() -> None:
    finalizations = 0

    class Config(Fig):
        value: int = 0

        @override
        def finalize(self) -> Self:
            nonlocal finalizations
            finalizations += 1
            return super().finalize()

    pformat(
        [Config()],
        width=10,
        hide_default_values=False,
        short_sequence_max_width=5,
    )

    assert finalizations == 1


def test_finalized_copy_cache_survives_address_reuse() -> None:
    """A reclaimed config's address must not serve another config's copy.

    The cache is keyed by ``id()``, which CPython reuses as soon as the original
    is collected. Without holding the key object, a config allocated at that
    address during the same render silently receives the previous config's
    finalized copy.
    """

    class Config(Fig):
        n: int = 0

        @override
        def finalize(self) -> Self:
            self.n += 1
            return super().finalize()

    printer = FigPrinter()
    printer._finalized_copies = {}
    first = Config()
    first_copy = printer._try_to_finalize(first)
    # Reuse is a race against the allocator, so the collision is staged rather
    # than raced: a live config whose address the cache already maps must still
    # finalize as itself.
    second = Config()
    printer._finalized_copies[id(second)] = (first, first_copy)

    assert printer._try_to_finalize(second) is not first_copy


def test_pretty_printer_format_with_memory_masking():
    """Test FigPrinter.format with memory address masking."""

    class Obj:
        pass

    pp = FigPrinter(mask_memory_addresses=True)
    obj = Obj()
    result, _, _ = pp.format(obj, {}, 0, 0)
    assert _MASKED_MEMORY_ADDRESS in result


@pytest.mark.parametrize("address", ["0x7f8b9c0a", "0x7f8b9c0a1b20"])
def test_mask_memory_addresses_function(address: str) -> None:
    """Test the memory address masking function."""
    result = _mask_memory_addresses(f"Object at {address}; configured=0xdeadbeef")
    assert address not in result
    assert _MASKED_MEMORY_ADDRESS in result
    assert "configured=0xdeadbeef" in result


@pytest.mark.parametrize("address", ["0x7f8b9c0a", "0x7f8b9c0a1b20", "0xabc"])
def test_masked_address_is_one_platform_independent_literal(address: str) -> None:
    """Every masked address renders as the same literal everywhere.

    Goldens are shared across machines, so the placeholder cannot depend on the
    recording host: sizing it from the local pointer width makes a 64-bit
    golden unreproducible on a 32-bit interpreter, and sizing it per match
    leaks the original address's length into the output.
    """
    masked = _mask_memory_addresses(f"<Obj at {address}>")

    assert masked == f"<Obj at {_MASKED_MEMORY_ADDRESS}>"
    assert _MASKED_MEMORY_ADDRESS == "0xdefacedeface"


def test_memory_address_masking_preserves_quoted_string_data() -> None:
    address_data = "object at 0x7f8b9c0a"

    rendered = pformat({"message": address_data}, mask_memory_addresses=True)

    assert address_data in rendered


def test_pretty_printer_try_to_finalize():
    """Test FigPrinter._try_to_finalize method."""

    class FinalizableConfig(Fig):
        value: int = 42

    pp = FigPrinter(finalize=True)
    cfg = FinalizableConfig()

    finalized = pp._try_to_finalize(cfg)
    assert finalized is not cfg
    assert finalized.value == 42


def test_pretty_printer_try_to_finalize_with_error():
    """Test FigPrinter._try_to_finalize handles errors."""

    class BadConfig(Fig):
        """Config that raises error during finalize."""

        value: int = 42

        @override
        def finalize(self) -> Self:
            raise ValueError("Cannot finalize")

    pp = FigPrinter(finalize=True)
    cfg = BadConfig()

    # Should catch the error and warn
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        _result = pp._try_to_finalize(cfg)
        # Should warn about the error
        assert len(w) >= 1
        assert "Cannot finalize" in str(w[0].message)


def test_pretty_printer_no_finalize():
    """Test FigPrinter with finalize=False."""

    class Config(Fig):
        value: int = 42

    pp = FigPrinter(finalize=False)
    cfg = Config()

    # Should not finalize
    result = pp._try_to_finalize(cfg)
    assert result is cfg


# ---------------------------------------------------------------------------
# Dataclass formatting (extra_compact path)
# ---------------------------------------------------------------------------


@dataclasses.dataclass(kw_only=True, slots=True)
class _SimpleData:
    x: int = 1
    y: str = "hello"
    description: str = "a somewhat long default description value"


@dataclasses.dataclass(kw_only=True, slots=True)
class _NestedData:
    inner: _SimpleData = dataclasses.field(default_factory=_SimpleData)
    values: list[int] = dataclasses.field(default_factory=lambda: [1, 2, 3])


class TestPprintDataclass:
    """Test extra-compact dataclass formatting."""

    def test_pformat_dataclass_with_defaults_hidden(self):
        """Dataclass fields matching defaults should be hidden."""
        obj = _SimpleData()
        # Use narrow width to force PrettyPrinter to use _pprint_dataclass dispatch
        result = pformat(obj, extra_compact=True, hide_default_values=True, width=40)
        # All fields are defaults, so should get compact empty parens
        assert "_SimpleData()" in result

    def test_pformat_dataclass_with_non_defaults(self):
        """Non-default fields should be shown."""
        obj = _SimpleData(x=99, y="world")
        result = pformat(obj, extra_compact=True, hide_default_values=True, width=40)
        assert "x=99" in result
        assert "y='world'" in result

    def test_pformat_dataclass_show_all_values(self):
        """With hide_default_values=False, all fields should be shown."""
        obj = _SimpleData()
        result = pformat(obj, extra_compact=True, hide_default_values=False, width=40)
        assert "x=1" in result
        assert "y='hello'" in result

    def test_pformat_nested_dataclass(self):
        """Nested dataclass should be formatted recursively."""
        obj = _NestedData(inner=_SimpleData(x=42))
        result = pformat(obj, extra_compact=True, hide_default_values=True)
        assert "_NestedData" in result
        assert "x=42" in result

    def test_pformat_dataclass_no_extra_compact(self):
        """With extra_compact=False, standard formatting is used."""
        obj = _SimpleData(x=99)
        result = pformat(obj, extra_compact=False, hide_default_values=False)
        assert "x=99" in result


class TestFormatNamespaceItems:
    """Test _format_namespace_items extra-compact path."""

    def test_empty_items(self):
        """Empty items should produce no output."""
        obj = _SimpleData()
        result = pformat(obj, extra_compact=True, hide_default_values=True)
        # All defaults → empty items → "()"
        assert "()" in result

    def test_multiple_items(self):
        """Multiple items should each appear on their own line."""
        obj = _SimpleData(x=99, y="world")
        result = pformat(
            obj,
            extra_compact=True,
            hide_default_values=True,
            width=20,
        )
        assert "x=99" in result
        assert "y='world'" in result


class TestFormatList:
    """Test _pprint_list and _format_items extra-compact paths."""

    def test_short_list_on_one_line(self):
        """Short lists should be formatted on one line."""
        result = pformat([1, 2, 3], extra_compact=True)
        assert "[1, 2, 3]" in result

    def test_long_list_multiline(self):
        """Long lists should be formatted across multiple lines."""
        obj = list(range(20))
        result = pformat(obj, extra_compact=True, width=30)
        assert "\n" in result

    def test_empty_list(self):
        """Empty list should be '[]'."""
        result = pformat([], extra_compact=True)
        assert "[]" in result

    def test_list_no_extra_compact(self):
        """With extra_compact=False, standard list formatting is used."""
        result = pformat([1, 2, 3], extra_compact=False)
        assert "1" in result


class TestContinuationPipes:
    """Test continuation pipe logic."""

    def test_should_add_pipes_always(self):
        """With threshold 0, pipes are always added for multiline."""
        assert _should_add_continuation_pipes("a\nb", 2, 0) is True

    def test_should_not_add_pipes_disabled(self):
        """With threshold -1, pipes are never added."""
        assert _should_add_continuation_pipes("a\nb", 2, -1) is False

    def test_should_not_add_pipes_single_item(self):
        """Single items never get pipes."""
        assert _should_add_continuation_pipes("a\nb", 1, 0) is False

    def test_should_not_add_pipes_single_line(self):
        """Single-line values never get pipes."""
        assert _should_add_continuation_pipes("no newline", 3, 0) is False

    def test_should_add_pipes_threshold(self):
        """Pipes should be added when lines >= threshold."""
        multiline = "\n".join(["line"] * 10)
        assert _should_add_continuation_pipes(multiline, 2, 5) is True
        assert _should_add_continuation_pipes("a\nb", 2, 5) is False

    def test_add_pipes_to_lines(self):
        """Pipes should be placed at correct column."""
        lines = ["first", "  second", "  third", "  last"]
        result = _add_pipes_to_lines(lines, 0)
        assert result[0] == "first"  # First line unchanged
        assert result[1].startswith("│")
        assert result[2].startswith("│")
        assert result[3].startswith(" ")  # Last line gets space

    def test_add_pipes_empty(self):
        """Empty lines list returns empty."""
        assert _add_pipes_to_lines([], 0) == []


class TestCollapseMultiline:
    """Test _collapse_multiline_value."""

    def test_no_newline_passthrough(self):
        """Single-line values pass through unchanged."""
        assert _collapse_multiline_value("hello", 40) == "hello"

    def test_short_multiline_collapses(self):
        """Short multiline values collapse to one line."""
        result = _collapse_multiline_value("(\n  1,\n  2\n)", 40)
        assert "\n" not in result

    def test_collapse_preserves_adjacent_token_boundary(self) -> None:
        assert _collapse_multiline_value("foo\nbar", 40) == "foo bar"

    def test_collapse_removes_formatting_boundary_whitespace(self) -> None:
        assert _collapse_multiline_value("\n  1,\n  2\n", 40) == "1, 2"

    def test_long_multiline_stays(self):
        """Long multiline values stay multiline."""
        long_val = "(\n" + "  very_long_name=very_long_value,\n" * 5 + ")"
        result = _collapse_multiline_value(long_val, 10)
        assert "\n" in result

    def test_collapse_preserves_string_literal_whitespace(self) -> None:
        multiline = "['a  ',\n'b']"

        collapsed = _collapse_multiline_value(multiline, 40)

        assert ast.literal_eval(collapsed) == ast.literal_eval(multiline)


class _AmbiguousValue:
    __hash__ = object.__hash__

    @override
    def __eq__(self, other: object) -> bool:
        del other
        raise ValueError("ambiguous comparison")


@dataclasses.dataclass(kw_only=True, slots=True)
class _MixedDefaults:
    ordinary: int = 1
    ambiguous: _AmbiguousValue = dataclasses.field(default_factory=_AmbiguousValue)


def test_one_ambiguous_default_does_not_unhide_unrelated_defaults() -> None:
    result = pformat(
        _MixedDefaults(),
        hide_default_values=True,
        width=20,
    )

    assert "ordinary=" not in result


class TestFilterNonDefaultItems:
    """Test _filter_non_default_items."""

    def test_filtering_does_not_execute_construction_side_effects(self) -> None:
        constructions = 0
        factory_calls = [0]

        @dataclasses.dataclass(kw_only=True, slots=True)
        class SideEffectData:
            value: int = 1
            items: list[int] = dataclasses.field(
                default_factory=functools.partial(_default_items, factory_calls),
            )

            def __post_init__(self) -> None:
                nonlocal constructions
                constructions += 1

        obj = SideEffectData()
        assert (constructions, factory_calls) == (1, [1])

        filtered = _filter_non_default_items(
            obj,
            [("value", 1), ("items", obj.items)],
        )

        assert filtered == [("items", [])]
        assert (constructions, factory_calls) == (1, [1])

    def test_filters_defaults(self):
        """Default values should be filtered out."""
        obj = _SimpleData()
        items: list[tuple[str, object]] = [("x", 1), ("y", "hello")]
        result = _filter_non_default_items(obj, items)
        assert result == []

    def test_keeps_non_defaults(self):
        """Non-default values should be kept."""
        obj = _SimpleData(x=99, y="world")
        items: list[tuple[str, object]] = [("x", 99), ("y", "world")]
        result = _filter_non_default_items(obj, items)
        assert ("x", 99) in result
        assert ("y", "world") in result

    def test_keeps_fields_without_declared_defaults(self):
        """Fields without declared defaults remain visible."""

        @dataclasses.dataclass(kw_only=True, slots=True)
        class RequiredFields:
            x: int

        obj = RequiredFields(x=42)
        items: list[tuple[str, object]] = [("x", 42)]
        result = _filter_non_default_items(obj, items)
        assert result == items


class TestUtilityFunctions:
    """Test standalone utility functions."""

    def test_get_level_indents(self):
        """Level indents should be calculated correctly."""
        item_indent, base_indent = _get_level_indents(0, 8)
        assert item_indent == 8
        assert base_indent == 0

        item_indent, base_indent = _get_level_indents(2, 4)
        assert item_indent == 12
        assert base_indent == 8

    def test_replace_char_at_column(self):
        """Character should be replaced at column if whitespace."""
        assert _replace_char_at_column("  hello", 0, "│") == "│ hello"
        assert _replace_char_at_column("hello", 0, "│") == "hello"  # Not whitespace
        assert _replace_char_at_column("x", 5, "│") == "x"  # Out of bounds


class TestPprintListDispatch:
    """Test _pprint_list dispatch for long lists."""

    def test_long_list_triggers_pprint_list(self):
        """A long list inside a dataclass should trigger _pprint_list dispatch."""

        @dataclasses.dataclass(kw_only=True, slots=True)
        class WithList:
            items: list[int] = dataclasses.field(
                default_factory=lambda: list(range(20)),
            )

        obj = WithList()
        # width=40 forces multiline formatting
        result = pformat(obj, extra_compact=True, hide_default_values=False, width=40)
        assert "items=" in result
        # The list should be formatted with brackets
        assert "[" in result
        assert "]" in result

    def test_list_no_extra_compact_dispatch(self):
        """With extra_compact=False, parent _pprint_list is used."""

        @dataclasses.dataclass(kw_only=True, slots=True)
        class WithList:
            items: list[int] = dataclasses.field(
                default_factory=lambda: list(range(20)),
            )

        obj = WithList()
        result = pformat(
            obj,
            extra_compact=False,
            hide_default_values=False,
            width=40,
        )
        assert "items=" in result

    def test_format_items_no_extra_compact(self):
        """With extra_compact=False, parent _format_items is used."""
        result = pformat(
            list(range(20)),
            extra_compact=False,
            width=40,
        )
        assert "0" in result


class TestPprintDataclassWithPipes:
    """Test full dataclass formatting with continuation pipes."""

    def test_dataclass_with_long_nested_values_gets_pipes(self):
        """Dataclass with long nested values should get continuation pipes."""
        obj = _NestedData(
            inner=_SimpleData(x=99, y="a very long string for testing"),
            values=list(range(50)),
        )
        result = pformat(
            obj,
            extra_compact=True,
            continuation_pipe=0,  # Always add pipes
            hide_default_values=False,
            width=40,
        )
        assert "│" in result

    def test_dataclass_with_pipes_disabled(self):
        """Pipes disabled should produce no pipe characters."""
        obj = _NestedData(
            inner=_SimpleData(x=99),
            values=list(range(50)),
        )
        result = pformat(
            obj,
            extra_compact=True,
            continuation_pipe=-1,
            hide_default_values=False,
            width=40,
        )
        assert "│" not in result


def test_format_items_one_line():
    """Test _format_items one-line path when list triggers dispatch but content is short."""
    # repr ~47 chars exceeds width=30, triggering _pprint_list dispatch.
    # But content_width < short_sequence_max_width=100, so _format_items writes one line.
    items = [100, 200, 300, 400, 500, 600, 700, 800, 900]
    result = pformat(
        items,
        extra_compact=True,
        hide_default_values=False,
        width=30,
        short_sequence_max_width=100,
    )
    assert "100" in result
    assert "900" in result


def test_format_namespace_items_context_cycle():
    """Test _format_namespace_items handles context cycles in multiline mode."""

    class MyClass:
        class Config(Fig):
            a: str = "a" * 40
            b: str = "b" * 40
            cyclic: object = None

        def __init__(self, config: Config):
            pass

    cfg = MyClass.Config()
    finalized = cfg.finalize()
    # Create a cycle — the multiline formatter must detect it
    object.__setattr__(finalized, "cyclic", finalized)
    printer = FigPrinter(
        extra_compact=True,
        hide_default_values=False,
        finalize=False,
        width=40,  # Force multiline so _format_namespace_items is entered
    )
    result = printer.pformat(finalized)
    assert "..." in result


def test_format_items_multiline_context_cycle():
    """Test _format_items_multiline handles context cycles in lists."""

    class MyClass:
        class Config(Fig):
            items: list[object] = dataclasses.field(default_factory=list[object])

        def __init__(self, config: Config):
            pass

    cfg = MyClass.Config()
    finalized = cfg.finalize()
    # Create list with self-reference to trigger cycle detection
    items: list[object] = [1, 2]
    items.append(items)  # self-referential list
    object.__setattr__(finalized, "items", items)
    printer = FigPrinter(
        extra_compact=True,
        hide_default_values=False,
        finalize=False,
        width=10,  # Force multiline
        short_sequence_max_width=5,
    )
    result = printer.pformat(finalized)
    assert "..." in result


def _default_items(factory_calls: list[int]) -> list[int]:
    factory_calls[0] += 1
    return []


def _long_nested_functions(path: Path) -> list[str]:
    tree = ast.parse(path.read_text())
    parents = {
        child: parent
        for parent in ast.walk(tree)
        for child in ast.iter_child_nodes(parent)
    }
    return [
        f"{node.name}:{node.lineno}"
        for node in ast.walk(tree)
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
        and _is_nested_function(node, parents)
        and node.end_lineno is not None
        and node.end_lineno - node.lineno + 1 > 3
    ]


def _is_nested_function(node: ast.AST, parents: dict[ast.AST, ast.AST]) -> bool:
    parent = parents.get(node)
    while parent is not None:
        if isinstance(parent, (ast.FunctionDef, ast.AsyncFunctionDef)):
            return True
        if isinstance(parent, (ast.ClassDef, ast.Module)):
            return False
        parent = parents.get(parent)
    return False


if __name__ == "__main__":
    from configgle.lib.testing.main import test_main

    test_main(__file__)
