"""Tests for pprinting module."""

from __future__ import annotations

from io import StringIO
from pathlib import Path
from typing import TYPE_CHECKING, Final, Literal, Self, cast, override

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

from configgle import Fig, InlineConfig, PartialConfig, pprinting
from configgle.coloring import color_config
from configgle.pprinting import (
    FigPrinter,
    _add_pipes_to_lines,
    _collapse_multiline_value,
    _contains_repeated_string_whitespace,
    _filter_non_default_items,
    _function_repr_children,
    _get_level_indents,
    _mask_memory_addresses,
    _qualify_function_reprs,
    _render_field_row,
    _replace_char_at_column,
    _replace_unquoted,
    _should_add_continuation_pipes,
    _side_by_side_diff,
    _string_token_spans,
    pformat,
    pprint,
)


if TYPE_CHECKING:
    from collections.abc import Callable

    from configgle.custom_types import PformatOptions


_THIS: Final = Path(__file__).resolve()
_CWD: Final = _THIS.parent


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

    # Test with underscore_numbers.
    result = pformat(obj, underscore_numbers=True)
    assert "1_000_000" in result

    # Test without underscore_numbers.
    result = pformat(obj, underscore_numbers=False)
    assert "1000000" in result
    assert "1_000_000" not in result


def test_pformat_mask_memory_addresses():
    """Test pformat with mask_memory_addresses option."""

    class Obj:
        pass

    obj = Obj()
    result = pformat(obj, mask_memory_addresses=True)
    assert "0xdefacedeface" in result
    assert repr(obj) not in result


def test_inline_calls_wrap_at_argument_boundaries() -> None:
    partial = PartialConfig(int, "123", base=10)
    rendered = pformat(partial, width=32, indent=2, finalize=False)
    assert rendered == (
        "PartialConfig(\n    <class 'int'>,\n    '123',\n    base=10\n  )"
    )
    assert "functools.partial" not in rendered
    assert "functools.partial" in repr(partial)
    assert pformat(PartialConfig(int), width=80) == "PartialConfig(<class 'int'>)"
    inline = InlineConfig(int, "123", base=10)
    assert "\n    base=10\n" in pformat(inline, width=32, indent=2)
    assert "'123', base=10)" in repr(inline)
    assert not inline._finalized


def test_inline_nested_values_cycles_and_depth() -> None:
    child = PartialConfig(int, base=10)
    parent = PartialConfig(list[object], [child], nested=child)
    rendered = pformat(parent, width=32, indent=2, finalize=False)
    assert rendered.count("base=10") == 2
    assert "nested=PartialConfig(\n" in rendered
    assert "functools.partial" not in rendered
    cyclic = PartialConfig(list[object])
    cyclic.loop = cyclic
    assert "..." in pformat(cyclic, width=32, finalize=False)
    assert "PartialConfig(...)" in pformat(parent, depth=1, width=32, finalize=False)
    assert "base=10" in pformat(parent, width=32, extra_compact=False)


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


def test_named_functions_render_without_repr_wrappers_or_addresses() -> None:
    assert pformat(re.escape) == "re.escape"
    assert pformat(PartialConfig(re.escape)) == "PartialConfig(re.escape)"
    assert (
        pformat(InlineConfig(re.escape, "hello")) == "InlineConfig(re.escape, 'hello')"
    )


def test_lambda_expression_source_and_same_line_identity() -> None:
    functions: tuple[Callable[[int], int], Callable[[int], int]] = (
        (lambda x: x * 2),
        (lambda x: x + 3),
    )
    first, second = functions
    assert pformat(first) == "lambda x: x * 2"
    assert pformat(second) == "lambda x: x + 3"
    assert pformat(PartialConfig(first)) == "PartialConfig(lambda x: x * 2)"
    (quoted,) = (lambda: "│ <function fake at 0x123>",)
    assert pformat(quoted) == "lambda: '│ <function fake at 0x123>'"


def test_lambda_multiline_source_is_an_expression() -> None:
    function: Callable[[int], int] = lambda x: x * 2 + 3  # noqa: E731 -- Testing lambda specifically.
    assert ast.dump(ast.parse(pformat(function), mode="eval")) == ast.dump(
        ast.parse("lambda x: x * 2 + 3", mode="eval"),
    )


def test_lambda_spanning_lines_renders_on_one_line() -> None:
    function: Callable[[int], int] = lambda x: (  # noqa: E731 -- Testing lambda specifically.
        x  # Kept on its own line so the source spans lines.
        + 1
    )
    assert pformat([function, 1]) == "[lambda x: x + 1, 1]"


def _defaulted(x: int, k: int = 3) -> int:
    return x * k


def _scaler(k: int) -> Callable[[int], int]:
    return lambda x: x * k


def _adder(k: int) -> Callable[[int], int]:
    def add(x: int) -> int:
        return x + k

    return add


class _SuperUser:
    def method(self) -> str:
        return super().__repr__()


def test_functions_render_their_captured_values() -> None:
    assert pformat(_scaler(1)) == "lambda x: x * k {k=1}"
    assert pformat(_scaler(1)) != pformat(_scaler(2))
    assert pformat(_adder(1)) == (f"{__name__}._adder.<locals>.add {{k=1}}")
    scalers: list[Callable[[int], int]] = [lambda x, k=k: x * k for k in (1, 2)]
    first, second = scalers
    assert pformat(first) == "lambda x, k=k: x * k {k=1}"
    assert pformat(second) == "lambda x, k=k: x * k {k=2}"
    keyword = (lambda *, k=3: k,)[0]
    assert pformat(keyword) == "lambda *, k=3: k {k=3}"
    assert pformat(_SuperUser.method) == f"{__name__}._SuperUser.method"
    assert pformat(_scaler) == f"{__name__}._scaler"
    assert pformat(functools.partial) == "<class 'functools.partial'>"
    assert pformat(re.escape) == "re.escape"
    assert pformat(_defaulted) == f"{__name__}._defaulted"
    scaler = _scaler(1)
    assert isinstance(scaler, types.FunctionType)
    unbound = types.FunctionType(
        scaler.__code__,
        {},
        closure=(types.CellType(),),
    )
    assert pformat(unbound) == "lambda x: x * k"


def _nested(levels: int, leaf: object) -> object:
    node: object = [leaf]
    for _ in range(levels):
        node = [leaf, node, "x" * 30]
    return node


@dataclasses.dataclass(slots=True, kw_only=True)
class _Node:
    leaf: object
    child: object
    pad: str = "x" * 30


def _nested_dataclass(levels: int, leaf: object) -> object:
    node: object = leaf
    for _ in range(levels):
        node = _Node(leaf=leaf, child=node)
    return node


class _CountingLeaf:
    def __init__(self, width: int) -> None:
        self.calls = 0
        self._width = width

    @override
    def __repr__(self) -> str:
        self.calls += 1
        return "y" * self._width


def _leaf_reprs(nest: Callable[[int, object], object], levels: int, width: int) -> int:
    leaf = _CountingLeaf(width)
    pformat(nest(levels, leaf), width=80)
    return leaf.calls


@pytest.mark.parametrize("width", [1, 13, 45])
def test_container_leaves_render_once_per_level(width: int) -> None:
    # Rendering each subtree once per enclosing level measured 393_195 here.
    assert _leaf_reprs(_nested, 16, width) == 16 + 1


@pytest.mark.parametrize("width", [1, 13, 45])
def test_dataclass_leaves_render_at_most_quadratically(width: int) -> None:
    # A generated dataclass ``__repr__`` reprs its children directly, so each
    # level re-renders its subtree once; ``origin/main`` measured 169 here.
    assert _leaf_reprs(_nested_dataclass, 16, width) <= 169


def test_lambda_without_available_source_has_no_address() -> None:
    function = types.FunctionType(
        (lambda: None).__code__.replace(co_filename="<unavailable-lambda>"),
        {},
    )
    assert pformat(function) == "<lambda>"


def test_function_qualification_binds_modules_to_exact_addresses() -> None:
    first = types.FunctionType((lambda: None).__code__, {"__name__": "first_module"})
    second = types.FunctionType((lambda: None).__code__, {"__name__": "second_module"})
    first.__name__ = first.__qualname__ = "shared"
    second.__name__ = second.__qualname__ = "shared"
    rendered = f"[{second!r}, {first!r}]"

    qualified = _qualify_function_reprs([first, second], rendered)

    assert qualified == "[second_module.shared, first_module.shared]"


def test_function_qualification_ignores_repr_inside_string_value() -> None:
    function = types.FunctionType(
        (lambda: None).__code__,
        {"__name__": "string_collision_module"},
    )
    function.__name__ = function.__qualname__ = "shared"
    rendered = repr([repr(function), function])

    qualified = _qualify_function_reprs([repr(function), function], rendered)

    assert qualified == f"[{repr(function)!r}, string_collision_module.shared]"


def test_function_qualification_handles_aliased_containers() -> None:
    function = types.FunctionType(
        (lambda: None).__code__,
        {"__name__": "aliased_module"},
    )
    function.__name__ = function.__qualname__ = "shared"
    child = [function]
    rendered = repr([child, child])

    qualified = _qualify_function_reprs([child, child], rendered)

    expected = "aliased_module.shared"
    assert qualified == f"[[{expected}], [{expected}]]"


def test_unquoted_function_repr_replace_is_a_noop_when_bare_repr_absent() -> None:
    text = "no functions here"
    spans = [(0, 2)]
    result, shifted = _replace_unquoted(
        text,
        "<function foo at 0x1>",
        "<function mod.foo at 0x1>",
        spans=spans,
    )
    assert result is text
    assert shifted is spans


def test_unquoted_function_repr_replace_skips_a_quoted_only_occurrence() -> None:
    bare = "<function foo at 0x1>"
    text = repr(bare)  # The only occurrence sits inside a string token.

    result, _ = _replace_unquoted(
        text,
        bare,
        "<function mod.foo at 0x1>",
        spans=_string_token_spans(text),
    )

    assert result == text


def test_unquoted_replace_shifts_only_the_spans_after_the_match() -> None:
    bare = "<f>"
    text = "'a' <f> 'b'"
    spans = _string_token_spans(text)
    assert spans == [(0, 3), (8, 11)]

    result, shifted = _replace_unquoted(text, bare, "<m.f>", spans=spans)

    assert result == "'a' <m.f> 'b'"
    assert shifted == [(0, 3), (10, 13)]
    assert [result[begin:stop] for begin, stop in shifted] == ["'a'", "'b'"]


def test_unquoted_replace_treats_a_span_touching_the_match_as_outside() -> None:
    # The quoted string ends where the bare repr begins: adjacent, not inside.
    text = "'s'<f>"
    result, shifted = _replace_unquoted(text, "<f>", "<m.f>", spans=[(0, 3)])
    assert result == "'s'<m.f>"
    assert shifted == [(0, 3)]
    # A span starting exactly at the match's end follows it, so it shifts.
    text = "<f>'s'"
    result, shifted = _replace_unquoted(text, "<f>", "<m.f>", spans=[(3, 6)])
    assert result == "<m.f>'s'"
    assert shifted == [(5, 8)]


@dataclasses.dataclass(kw_only=True, slots=True)
class _HangingInner:
    alpha: list[int]
    beta: str


@dataclasses.dataclass(kw_only=True, slots=True)
class _HangingOuter:
    name: _HangingInner


def test_non_compact_dataclass_fields_hang_under_the_class_name() -> None:
    rendered = pformat(
        _HangingOuter(name=_HangingInner(alpha=[2, 3], beta="x" * 30)),
        extra_compact=False,
        width=40,
    ).splitlines()
    # Continuation lines align one past ``_HangingInner(`` at its column.
    column = len("_HangingOuter(name=_HangingInner(")
    assert rendered[-1] == " " * column + "beta='" + "x" * 30 + "'))"
    assert rendered[0].startswith("_HangingOuter(name=_HangingInner(alpha=[")


def test_qualification_skips_a_quoted_copy_after_an_earlier_replacement() -> None:
    first = types.FunctionType((lambda: None).__code__, {"__name__": "alpha"})
    second = types.FunctionType((lambda: None).__code__, {"__name__": "beta"})
    first.__name__ = first.__qualname__ = "first"
    second.__name__ = second.__qualname__ = "second"
    # The first replacement lengthens the text, so the quoted copy of the second
    # repr only stays skipped if every later span shifts with it.
    rendered = repr([first, repr(second), second])

    qualified = _qualify_function_reprs([first, repr(second), second], rendered)

    assert qualified == f"[alpha.first, {repr(second)!r}, beta.second]"


def test_string_token_spans_returns_empty_on_an_unterminated_string() -> None:
    assert _string_token_spans("x = '''unterminated") == []


def test_function_repr_children_of_a_partial_includes_func_args_and_keywords() -> None:
    def add(a: int, b: int, *, c: int = 0) -> int:
        return a + b + c

    bound = functools.partial(add, 1, c=2)

    assert _function_repr_children(bound) == [add, 1, 2]


def test_repeated_string_whitespace_check_treats_unparsable_text_as_true() -> None:
    assert _contains_repeated_string_whitespace("'''unterminated") is True


def test_owned_files_have_no_long_nested_functions() -> None:
    # Located through the imported module, not as a sibling file: the export
    # moves tests to ``tests/`` and sources into the package, so the two stop
    # sharing a directory.
    paths = [_THIS, Path(inspect.getfile(pprinting))]

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

    # With finalize=False - should not warn.
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        result = pformat(cfg, finalize=False)
        assert "MockConfigurable" in result
        assert len(w) == 0


class _Terminal(StringIO):
    @override
    def isatty(self) -> bool:
        return True


def test_pprint_colors_only_terminals_by_default(
    capsys: pytest.CaptureFixture[str],
) -> None:
    log = StringIO()
    pprint({"value": 2}, stream=log)
    assert log.getvalue() == "{'value': 2}\n"
    pprint({"value": 2})
    assert capsys.readouterr().out == "{'value': 2}\n"
    terminal = _Terminal()
    pprint({"value": 2}, stream=terminal)
    assert "\x1b[95m'value'\x1b[0m" in terminal.getvalue()
    assert "\x1b[95m2\x1b[0m" in terminal.getvalue()
    forced = StringIO()
    pprint({"value": 2}, stream=forced, color=True)
    assert "\x1b[95m2\x1b[0m" in forced.getvalue()
    disabled = _Terminal()
    pprint({"value": 2}, stream=disabled, color=False)
    assert disabled.getvalue() == "{'value': 2}\n"


def test_format_diff_colors_only_a_terminal_stdout_by_default() -> None:
    plain = pprinting.format_diff([1, 2], mode="diff", n=0, color=None, options={})
    assert plain == "--- \n+++ \n@@ -1 +1 @@\n-1\n+2"


@pytest.mark.parametrize("mode", ["bogus"])
def test_format_diff_rejects_an_unknown_mode(mode: str) -> None:
    with pytest.raises(AssertionError, match="bogus"):
        pprinting.format_diff(
            [1, 2],
            mode=cast(Literal["diff"], mode),
            n=0,
            color=False,
            options={},
        )


def test_pprint_plain_format_stays_plain() -> None:
    colored = color_config(
        "Outer.Config(child=Inner(value=2), text='hi', yes=True, no=False, missing=None)",
    )
    assert "\x1b[96mOuter.Config\x1b[0m(child=\x1b[96mInner\x1b[0m(value=" in colored
    assert "\x1b[95mTrue\x1b[0m" in colored
    assert "\x1b[95mFalse\x1b[0m" in colored
    assert "\x1b[95mNone\x1b[0m" in colored
    assert pformat({"value": 2}) == "{'value': 2}"


@pytest.mark.parametrize("text", ["Config(\n  x=2\n)", "Config(\n  x=2", "a\n  b\n c"])
def test_syntax_color_preserves_multiline_and_incomplete_reprs(text: str) -> None:
    colored = color_config(text)
    assert re.sub(r"\x1b\[[0-9;]*m", "", colored) == text


def test_side_by_side_aligns_insertions_deletions_and_unequal_columns() -> None:
    result = _side_by_side_diff(
        [["a", "b", "c"], ["start", "a", "c", "end"], ["a", "B", "extra", "c"]],
        n=3,
        width=12,
        color=False,
    )
    rows = [line.split("│") for line in result.splitlines()[1:]]
    assert [[row[i].strip() for row in rows if row[i].strip()] for i in range(3)] == [
        ["a", "b", "c"],
        ["start", "a", "c", "end"],
        ["a", "B", "extra", "c"],
    ]
    common = next(row for row in rows if row[0].strip() == "c")
    assert [cell.strip() for cell in common] == ["c", "c", "c"]


def test_side_by_side_wraps_and_colors_without_losing_text() -> None:
    result = _side_by_side_diff([["abcdefgh"], ["ijklmnop"]], n=0, width=4, color=True)
    assert "\x1b[31mabcd\x1b[0m" in result
    assert "\x1b[31mefgh\x1b[0m" in result
    assert "\x1b[32mijkl\x1b[0m" in result
    assert "\x1b[32mmnop\x1b[0m" in result
    plain = re.sub(r"\x1b\[[0-9;]*m", "", result)
    assert all(len(line) <= 11 for line in plain.splitlines())


def test_sxs_modes_and_live_terminal_width(monkeypatch: pytest.MonkeyPatch) -> None:
    before = {"changed": "a" * 90, "same": 7}
    after = {"changed": "b" * 90, "same": 7}
    monkeypatch.setenv("COLUMNS", "60")
    narrow = pprinting.format_diff(
        [before, after],
        mode="sxs",
        n=0,
        color=False,
        options={},
    )
    assert max(map(len, narrow.splitlines())) <= 60
    assert "'same'" not in narrow
    monkeypatch.setenv("COLUMNS", "100")
    wide = pprinting.format_diff(
        [before, after],
        mode="sxs-compact",
        n=0,
        color=False,
        options={},
    )
    assert max(map(len, wide.splitlines())) > 60
    full = pprinting.format_diff(
        [before, after],
        mode="sxs-full",
        n=0,
        color=False,
        options={"width": 120},
    )
    assert "'same'" in full
    assert max(map(len, full.splitlines())) <= 120
    assert (
        pprinting.format_diff(
            [before, after],
            mode="sxs",
            n=0,
            color=False,
            options={"width": 100},
        )
        == wide
    )


def test_side_by_side_context_omits_unchanged_rows() -> None:
    baseline = ["a", "b", "c", "d", "e"]
    variant = ["A", "b", "c", "d", "E"]
    result = _side_by_side_diff([baseline, variant], n=0, width=12, color=False)
    assert "…" not in result
    rows = [list(map(str.strip, line.split("│"))) for line in result.splitlines()]
    assert ["b", "b"] in rows
    assert ["c", "c"] in rows
    assert ["d", "d"] in rows
    assert _side_by_side_diff([baseline, baseline], n=0, width=12, color=False) == ""


def test_fields_nested_containers_cycles_depth_and_color() -> None:
    before: dict[str, object] = {"nested": [1, None], "same": 7}
    after: dict[str, object] = {"nested": [2, None], "same": 7}
    before["cycle"] = before
    after["cycle"] = after
    result = pprinting.format_diff(
        [before, after],
        mode="fields",
        n=0,
        color=True,
        options={"finalize": False, "width": 80},
    )
    assert "['nested'][0]" in result
    assert "['same']" not in result
    assert "\x1b[31m" in result
    assert "\x1b[32m" in result
    shallow = pprinting.format_diff(
        [before, after],
        mode="fields",
        n=0,
        color=False,
        options={"depth": 1, "finalize": False},
    )
    assert "\x1b[" not in shallow


def test_field_rows_exact_alignment_color_and_multiline_blanks() -> None:
    assert _render_field_row(
        ("x", "a\nb", "z", None),
        widths=[3, 4, 2, 2],
        color=True,
    ) == [
        "x   │ \x1b[31ma   \x1b[0m │ \x1b[32mz \x1b[0m │",
        "    │ \x1b[31mb   \x1b[0m │    │",
    ]
    assert _render_field_row(
        ("x", "a", "a", "b"),
        widths=[3, 4, 2, 2],
        color=True,
    ) == ["x   │ \x1b[31ma   \x1b[0m │ a  │ \x1b[32mb \x1b[0m"]
    assert _render_field_row(
        ("x", "", None),
        widths=[3, 4, 2],
        color=False,
    ) == ["x   │      │"]


def test_side_by_side_keeps_headers_whole_when_narrow() -> None:
    result = pprinting.format_diff(
        [1, 2, 3, 4, 5],
        mode="sxs",
        n=0,
        color=False,
        options={"width": 20},
    )
    assert [cell.strip() for cell in result.splitlines()[0].split("│")] == [
        f"config[{i}]" for i in range(5)
    ]


class _DerivedLeaf(Fig):
    value: int = 2
    """Input value."""

    derived: int = 0
    """Derived value."""

    @override
    def finalize(self) -> Self:
        self.derived = self.value * 2
        return super().finalize()


def _field_rows(configs: list[object], options: PformatOptions) -> list[list[str]]:
    result = pprinting.format_diff(
        configs,
        mode="fields",
        n=0,
        color=False,
        options=options,
    )
    return [list(map(str.strip, line.split("│"))) for line in result.splitlines()]


def test_fields_finalizes_configs_nested_in_containers() -> None:
    before = [_DerivedLeaf()]
    after = [_DerivedLeaf()]
    after[0].value = 3
    assert ["[0].derived", "4", "6"] in _field_rows([before, after], {})
    assert before[0].derived == after[0].derived == 0


def test_fields_depth_counts_levels_and_shows_whole_leaves() -> None:
    rows = _field_rows(
        [{"x": {"y": {"z": 1}}}, {"x": {"y": {"z": 2}}}],
        {"depth": 1, "finalize": False},
    )
    assert rows[1:] == [["['x']", "{'y': {'z': 1}}", "{'y': {'z': 2}}"]]
    rows = _field_rows(
        [{"a.b.c": {"d": 1}}, {"a.b.c": {"d": 2}}],
        {"depth": 2, "finalize": False},
    )
    assert rows[1:] == [["['a.b.c']['d']", "1", "2"]]


def test_fields_inline_rows_follow_the_rendered_call() -> None:
    rows = _field_rows(
        [PartialConfig(int, "1", base=10), PartialConfig(float, "1", base=10)],
        {"finalize": False},
    )
    assert rows[1:] == [["func", "<class 'int'>", "<class 'float'>"]]
    rows = _field_rows(
        [InlineConfig(int, "1"), InlineConfig(int, "2")],
        {"finalize": False},
    )
    assert rows[1:] == [["[0]", "'1'", "'2'"]]


def test_fields_finalization_fallback_and_source_isolation() -> None:
    class Config(Fig):
        value: int = 1
        """Value to derive."""

        @override
        def finalize(self) -> Self:
            self.value *= 2
            return super().finalize()

    before = Config()
    after = Config()
    after.value = 3
    result = before.udiff(after, mode="fields", color=False)
    assert ["value", "2", "6"] in [
        list(map(str.strip, line.split("│"))) for line in result.splitlines()
    ]
    assert before.value == 1
    assert after.value == 3

    class Broken(Config):
        @override
        def finalize(self) -> Self:
            raise ValueError("cannot finalize")

    with pytest.warns(UserWarning, match="cannot finalize"):
        assert Broken().udiff(after, mode="fields", color=False)


def test_pprint_basic():
    """Test pprint function."""
    stream = StringIO()
    pprint({"a": 1, "b": 2}, stream=stream, color=False)
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
    assert "0xdefacedeface" in result


@pytest.mark.parametrize("address", ["0x7f8b9c0a", "0x7f8b9c0a1b20"])
def test_mask_memory_addresses_function(address: str) -> None:
    """Test the memory address masking function."""
    result = _mask_memory_addresses(f"Object at {address}; configured=0xdeadbeef")
    assert address not in result
    assert "0xdefacedeface" in result
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

    assert masked == "<Obj at 0xdefacedeface>"


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

    # Should catch the error and warn.
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        _result = pp._try_to_finalize(cfg)
        # Should warn about the error.
        assert len(w) == 1
        assert str(w[0].message) == "ValueError: Cannot finalize"


def test_pretty_printer_no_finalize():
    """Test FigPrinter with finalize=False."""

    class Config(Fig):
        value: int = 42

    pp = FigPrinter(finalize=False)
    cfg = Config()

    # Should not finalize.
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
        # Use narrow width to force PrettyPrinter to use _pprint_dataclass dispatch.
        result = pformat(obj, extra_compact=True, hide_default_values=True, width=40)
        # All fields are defaults, so should get compact empty parens.
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
        assert result[0] == "first"  # First line unchanged.
        assert result[1].startswith("│")
        assert result[2].startswith("│")
        assert result[3].startswith(" ")  # Last line gets space.

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

    def test_keeps_a_field_whose_eq_raises_against_its_default(self):
        """A field whose comparison raises is shown, not hidden."""

        class Explosive:
            @override
            def __eq__(self, other: object) -> bool:
                raise RuntimeError("no comparison")

            @override
            def __hash__(self) -> int:
                return 0

        @dataclasses.dataclass(kw_only=True, slots=True)
        class WithExplosiveDefault:
            # A plain (non-factory) default: field.default must hold the
            # actual value, not dataclasses.MISSING, to reach the `!=`
            # comparison this test targets.
            value: Explosive = dataclasses.field(default=Explosive())

        obj = WithExplosiveDefault()
        items: list[tuple[str, object]] = [("value", obj.value)]
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
        assert _replace_char_at_column("hello", 0, "│") == "hello"  # Not whitespace.
        assert _replace_char_at_column("x", 5, "│") == "x"  # Out of bounds.


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
        # width=40 forces multiline formatting.
        result = pformat(obj, extra_compact=True, hide_default_values=False, width=40)
        assert "items=" in result
        # The list should be formatted with brackets.
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
            continuation_pipe=0,  # Always add pipes.
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
    # ``repr`` ~47 chars exceeds width=30, triggering _pprint_list dispatch.
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
    # Create a cycle -- the multiline formatter must detect it.
    object.__setattr__(finalized, "cyclic", finalized)
    printer = FigPrinter(
        extra_compact=True,
        hide_default_values=False,
        finalize=False,
        width=40,  # Force multiline so _format_namespace_items is entered.
    )
    result = printer.pformat(finalized)
    assert "..." in result
    assert "XX...XX" not in result


def test_format_items_multiline_context_cycle():
    """Test _format_items_multiline handles context cycles in lists."""

    class MyClass:
        class Config(Fig):
            items: list[object] = dataclasses.field(default_factory=list[object])

        def __init__(self, config: Config):
            pass

    cfg = MyClass.Config()
    finalized = cfg.finalize()
    # Create list with self-reference to trigger cycle detection.
    items: list[object] = [1, 2]
    items.append(items)  # self-referential list.
    object.__setattr__(finalized, "items", items)
    printer = FigPrinter(
        extra_compact=True,
        hide_default_values=False,
        finalize=False,
        width=10,  # Force multiline.
        short_sequence_max_width=5,
    )
    result = printer.pformat(finalized)
    assert "..." in result
    assert "XX...XX" not in result


def test_collapse_multiline_value_exact_boundaries() -> None:
    assert _collapse_multiline_value("(\n  1,\n  2\n)", 5) == "(\n  1,\n  2\n)"
    assert _collapse_multiline_value("(\n  1,\n  2\n)", 6) == "(1, 2)"
    assert _collapse_multiline_value("(\n  1,\n  2\n)", 7) == "(1, 2)"


def test_mask_memory_addresses_exactly_preserves_string_spans() -> None:
    text = "['at 0xabc', <Obj at 0xdef>, <Obj at 0x123>]"
    assert _mask_memory_addresses(text) == (
        "['at 0xabc', <Obj at 0xdefacedeface>, <Obj at 0xdefacedeface>]"
    )


def test_mask_memory_addresses_does_not_skip_adjacent_match() -> None:
    text = "'x at 0xabc'<Obj at 0xdef>"
    assert _mask_memory_addresses(text) == "'x at 0xabc'<Obj at 0xdefacedeface>"


def test_mask_memory_addresses_adjacent_token_boundaries() -> None:
    assert _mask_memory_addresses("'abc'<Obj at 0x1234>") == (
        "'abc'<Obj at 0xdefacedeface>"
    )
    assert _mask_memory_addresses("<Obj at 0x1234>'abc'") == (
        "<Obj at 0xdefacedeface>'abc'"
    )


def test_mask_memory_addresses_handles_uppercase_digits() -> None:
    assert _mask_memory_addresses("<Obj at 0xABCDEF>") == ("<Obj at 0xdefacedeface>")


def test_mask_memory_addresses_continues_after_quoted_match() -> None:
    # The quoted address must itself match (`` at `` before it), so the reversed
    # scan skips it first and must still reach the earlier, unquoted one.
    text = "<Obj at 0xabc> 'x at 0xdef'"
    assert _mask_memory_addresses(text) == "<Obj at 0xdefacedeface> 'x at 0xdef'"


def test_mask_memory_addresses_masks_address_ending_where_string_starts() -> None:
    assert _mask_memory_addresses("x at 0x1234'abc'") == "x at 0xdefacedeface'abc'"


def test_string_token_spans_preserves_multiline_offsets() -> None:
    text = "x = 1\n'y' + <Obj at 0xabc>"
    assert _string_token_spans(text) == [(6, 9)]


def test_replace_char_at_column_rejects_end_column() -> None:
    assert _replace_char_at_column(" ", 1, "│") == " "
    assert _replace_char_at_column(" ", 0, "│") == "│"


def test_continuation_pipes_exact_thresholds() -> None:
    assert _should_add_continuation_pipes("a\nb", 2, 2) is True
    assert _should_add_continuation_pipes("a\nb", 2, 3) is False
    assert _should_add_continuation_pipes("a\nb\nc", 2, 2) is True
    assert _should_add_continuation_pipes("a\nb\nc", 2, 3) is True


def test_should_format_on_one_line_exact_boundaries() -> None:
    printer = FigPrinter(width=10, short_sequence_max_width=5)
    assert printer._should_format_on_one_line(4, 100, 100) is True
    assert printer._should_format_on_one_line(5, 100, 100) is False
    assert printer._should_format_on_one_line(6, 3, 1) is True
    assert printer._should_format_on_one_line(7, 3, 1) is False


def test_format_and_collapse_item_exact_allowance_boundary() -> None:
    printer = FigPrinter(width=31)
    item = list(range(10))
    assert printer._format_and_collapse_item(item, {}, 0, 0) == (
        "[0, 1, 2, 3, 4, 5, 6, 7, 8, 9]"
    )


def test_try_format_items_on_one_line_exact_width_boundary() -> None:
    printer = FigPrinter(width=30)
    item = list(range(10))
    assert printer._try_format_items_on_one_line(
        [item],
        {},
        0,
        indent=0,
        allowance=0,
    ) == ("[0, 1, 2, 3, 4, 5, 6, 7, 8, 9]")


def test_try_format_items_on_one_line_preserves_format_depth() -> None:
    printer = FigPrinter()
    assert (
        printer._try_format_items_on_one_line([[1, 2]], {}, 0, indent=0, allowance=0)
        == "[1, 2]"
    )


def test_try_format_items_on_one_line_exact_delimiter() -> None:
    printer = FigPrinter()
    assert (
        printer._try_format_items_on_one_line([1, 2, 3], {}, 0, indent=0, allowance=0)
        == "1, 2, 3"
    )
    assert printer._try_format_items_on_one_line([], {}, 0, indent=0, allowance=0) == ""


def test_format_items_multiline_exact_layout() -> None:
    printer = FigPrinter(indent=2, width=5, short_sequence_max_width=1)
    assert printer.pformat([1, 2]) == "[\n    1,\n    2\n  ]"


def test_pformat_options_have_exact_rendering_effects() -> None:
    assert pformat({"z": 1, "a": 2}, sort_dicts=False) == "{'z': 1, 'a': 2}"
    assert pformat({"z": 1, "a": 2}, sort_dicts=True) == "{'a': 2, 'z': 1}"
    assert pformat([1_000_000], underscore_numbers=True) == "[1_000_000]"
    assert pformat([1_000_000], underscore_numbers=False) == "[1000000]"


def test_pprint_options_have_exact_rendering_effects() -> None:
    stream = StringIO()
    pprint([1, 2], stream=stream, extra_compact=False, color=False)
    assert stream.getvalue() == "[1, 2]\n"
    stream = StringIO()
    pprint([1, 2], stream=stream, extra_compact=True, color=False)
    assert stream.getvalue() == "[1, 2]\n"


def test_exact_namespace_layout() -> None:
    assert pformat(_SimpleData(x=99, y="world"), width=20) == (
        "_SimpleData(\n        x=99,\n        y='world'\n    )"
    )


def test_fig_printer_init_records_every_option_exactly() -> None:
    printer = FigPrinter(
        indent=3,
        width=17,
        depth=2,
        compact=True,
        sort_dicts=True,
        underscore_numbers=False,
        finalize=False,
        mask_memory_addresses=False,
        extra_compact=False,
        continuation_pipe=4,
        hide_default_values=False,
        short_sequence_max_width=9,
    )
    assert printer._indent_per_level == 3
    assert printer._width == 17
    assert printer._finalize is False
    assert printer._mask_memory_addresses is None
    assert printer._extra_compact is False
    assert printer._continuation_pipe == 4
    assert printer._hide_default_values is False
    assert printer._short_sequence_max_width == 9
    assert object.__getattribute__(printer, "_compact") is True

    defaults = FigPrinter()
    assert defaults._indent_per_level == 4
    assert defaults._width == 80
    assert defaults._finalize is True
    assert defaults._mask_memory_addresses is not None
    assert defaults._extra_compact is True
    assert defaults._continuation_pipe == 50
    assert defaults._hide_default_values is True
    assert defaults._short_sequence_max_width == 40
    assert object.__getattribute__(defaults, "_compact") is False


def test_fig_printer_passes_base_validation_arguments() -> None:
    with pytest.raises(ValueError, match=r"^indent must be >= 0$"):
        FigPrinter(indent=-1)
    with pytest.raises(ValueError, match=r"^width must be != 0$"):
        FigPrinter(width=0)


def test_fig_printer_default_options_render_exactly() -> None:
    assert FigPrinter().pformat({"z": 1, "a": 2}) == "{'z': 1, 'a': 2}"
    assert FigPrinter().pformat([1_000_000]) == "[1_000_000]"


def test_fig_printer_forwards_stdlib_options_exactly() -> None:
    assert FigPrinter(sort_dicts=False).pformat({"z": 1, "a": 2}) == (
        "{'z': 1, 'a': 2}"
    )
    assert FigPrinter(sort_dicts=True).pformat({"z": 1, "a": 2}) == ("{'a': 2, 'z': 1}")
    assert FigPrinter(underscore_numbers=True).pformat([1_000_000]) == "[1_000_000]"
    assert FigPrinter(underscore_numbers=False).pformat([1_000_000]) == "[1000000]"
    assert FigPrinter(depth=1).pformat([[1, 2]]) == "[[...]]"


def test_public_default_rendering_is_exact() -> None:
    assert pformat({"z": 1, "a": 2}) == "{'z': 1, 'a': 2}"
    assert pformat([1_000_000]) == "[1_000_000]"
    assert pformat(_SimpleData()) == "_SimpleData()"

    class Obj:
        pass

    obj = Obj()
    assert "0xdefacedeface" in pformat(obj)
    stream = StringIO()
    pprint({"z": 1, "a": 2}, stream=stream, color=False)
    assert stream.getvalue() == "{'z': 1, 'a': 2}\n"


def test_public_wrappers_forward_every_option(monkeypatch: pytest.MonkeyPatch) -> None:
    calls: list[dict[str, object]] = []

    class RecordingPrinter:
        def __init__(self, **kwargs: object) -> None:
            calls.append(kwargs)

        def pformat(self, obj: object) -> str:
            del obj
            return "formatted"

        def pprint(self, obj: object) -> None:
            del obj

    monkeypatch.setattr(pprinting, "FigPrinter", RecordingPrinter)
    assert (
        pformat(
            object(),
            indent=3,
            width=17,
            depth=2,
            compact=True,
            sort_dicts=True,
            underscore_numbers=False,
            finalize=False,
            mask_memory_addresses=False,
            extra_compact=False,
            continuation_pipe=4,
            hide_default_values=False,
            short_sequence_max_width=9,
        )
        == "formatted"
    )
    stream = StringIO()
    pprint(
        object(),
        stream=stream,
        indent=3,
        width=17,
        depth=2,
        compact=True,
        sort_dicts=True,
        underscore_numbers=False,
        finalize=False,
        mask_memory_addresses=False,
        extra_compact=False,
        continuation_pipe=4,
        hide_default_values=False,
        short_sequence_max_width=9,
    )
    expected_pformat = {
        "indent": 3,
        "width": 17,
        "depth": 2,
        "compact": True,
        "sort_dicts": True,
        "underscore_numbers": False,
        "finalize": False,
        "mask_memory_addresses": False,
        "extra_compact": False,
        "continuation_pipe": 4,
        "hide_default_values": False,
        "short_sequence_max_width": 9,
    }
    pformat(object())
    default_stream = StringIO()
    pprint(object(), stream=default_stream)
    expected_pprint = {"stream": stream, **expected_pformat}
    expected_defaults = {
        "indent": 4,
        "width": 80,
        "depth": None,
        "compact": False,
        "sort_dicts": False,
        "underscore_numbers": True,
        "finalize": True,
        "mask_memory_addresses": True,
        "extra_compact": True,
        "continuation_pipe": 50,
        "hide_default_values": True,
        "short_sequence_max_width": 40,
    }
    assert calls == [
        expected_pformat,
        expected_pprint,
        expected_defaults,
        {
            "stream": default_stream,
            **expected_defaults,
            "indent": 4,
        },
    ]


def test_pformat_forwards_indent_and_width_exactly() -> None:
    assert pformat([1, 2], indent=2, width=5, short_sequence_max_width=1) == (
        "[\n    1,\n    2\n  ]"
    )


def test_pformat_forwards_depth_and_compact_exactly() -> None:
    value = [[1, 2]]
    assert pformat(value, depth=1) == "[[...]]"
    assert pformat(value, depth=None) == "[[1, 2]]"
    assert pformat([1, 2], compact=False) == "[1, 2]"
    assert pformat([1, 2], compact=True) == "[1, 2]"


def test_finalizeable_without_finalized_attribute_is_finalized() -> None:
    finalized = False

    class Config:
        def make(self) -> Self:
            return self

        def finalize(self) -> Self:
            nonlocal finalized
            finalized = True
            return self

    rendered = pformat(Config())
    assert "Config object at 0xdefacedeface" in rendered
    assert finalized


def test_pformat_forwards_mask_and_finalize_exactly() -> None:
    class Obj:
        pass

    obj = Obj()
    assert pformat(obj, mask_memory_addresses=False) == repr(obj)
    assert "0xdefacedeface" in pformat(obj, mask_memory_addresses=True)

    class Config(Fig):
        value: int = 1

        @override
        def finalize(self) -> Self:
            self.value = 2
            return super().finalize()

    assert pformat(Config(), finalize=False) == (f"{Config.__qualname__}(value=1)")
    assert pformat(Config(), finalize=True) == f"{Config.__qualname__}(value=2)"


def test_pformat_forwards_hide_default_and_short_sequence_width() -> None:
    assert pformat(_SimpleData(), hide_default_values=True) == "_SimpleData()"
    assert "x=1" in pformat(_SimpleData(), hide_default_values=False)
    assert pformat([1, 2], width=3, short_sequence_max_width=1) == (
        "[\n        1,\n        2\n    ]"
    )


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


@dataclasses.dataclass(kw_only=True, slots=True)
class _Pair:
    a: object = None
    b: object = None


@dataclasses.dataclass(kw_only=True, slots=True)
class _Trio:
    a: object = None
    b: object = None
    c: object = None


def test_one_line_items_format_with_zero_indent_and_allowance() -> None:
    # At width 20 the item fits exactly; any indent or allowance wraps it.
    assert (
        FigPrinter(width=20, indent=2).pformat([_Pair(a=[0])])
        == "[_Pair(a=[0], b=None)]"
    )


def test_multiline_items_reserve_exactly_one_column_allowance() -> None:
    value = [_Pair(a=list(range(13)))]
    assert FigPrinter(width=49, indent=2).pformat(value) == (
        "[\n    _Pair(\n      a=[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]\n    )\n  ]"
    )


def test_continuation_pipes_join_lines_with_bare_newlines() -> None:
    value = _Pair(a=_Pair(a=list(range(5)), b=list(range(5))), b=1)
    assert FigPrinter(width=8, indent=2, continuation_pipe=1).pformat(value) == (
        "_Pair(\n"
        "    a=_Pair(\n"
        "    │ a=[0, 1, 2, 3, 4],\n"
        "    │ b=[0, 1, 2, 3, 4]\n"
        "    ),\n"
        "    b=1\n"
        "  )"
    )


def test_last_namespace_field_uses_caller_allowance() -> None:
    assert FigPrinter(width=6, indent=2).pformat(_Pair(a=1, b={})) == (
        "_Pair(\n    a=1,\n    b={}\n  )"
    )


@pytest.mark.parametrize(
    ("width", "expected"),
    [
        (6, "_Trio(\n    a={ },\n    b=1\n  )"),
        (7, "_Trio(\n    a={},\n    b=1\n  )"),
    ],
)
def test_non_last_namespace_field_reserves_one_column(
    width: int,
    expected: str,
) -> None:
    assert FigPrinter(width=width, indent=2).pformat(_Trio(a={}, b=1)) == expected


class _Repr:
    """An object whose repr is fixed text."""

    def __init__(self, text: str) -> None:
        self.text = text

    @override
    def __repr__(self) -> str:
        return self.text


class _ListOf(list[object]):
    """A list subclass that keeps the builtin repr."""


class _Shrinks(Fig):
    text: str = "x" * 60
    """Long until finalized."""

    @override
    def finalize(self) -> Self:
        self.text = ""
        return super().finalize()


@dataclasses.dataclass(repr=False, kw_only=True, slots=True)
class _InheritsRepr(_Pair):
    c: object = None


# Masking turns this 32-character address into the 14-character mask.
_LONG_ADDRESS: Final = "<x at 0x" + "f" * 30 + ">"


@pytest.mark.parametrize(
    ("value", "expected"),
    [
        (_Pair(a=_Repr(_LONG_ADDRESS)), "_Pair(a=<x at 0xdefacedeface>, b=None)"),
        (
            [_Repr(_LONG_ADDRESS), _Repr(_LONG_ADDRESS)],
            "[<x at 0xdefacedeface>, <x at 0xdefacedeface>]",
        ),
        ({"k": _Repr(_LONG_ADDRESS)}, "{'k': <x at 0xdefacedeface>}"),
    ],
)
def test_a_repr_masking_shortens_to_the_width_prints_whole(
    value: object,
    expected: str,
) -> None:
    assert pformat(value, width=len(expected)) == expected


@pytest.mark.parametrize(
    ("value", "expected"),
    [
        ([_Shrinks()], "[_Shrinks(text='')]"),
        ([_ListOf([_Shrinks()])], "[[_Shrinks(text='')]]"),
    ],
)
def test_width_is_measured_on_the_finalized_items(
    value: object,
    expected: str,
) -> None:
    # CPython's item layout, unlike ours, never reproduces the one-line repr.
    rendered = pformat(
        value,
        width=len(expected),
        hide_default_values=False,
        extra_compact=False,
    )
    assert rendered == expected


def test_a_dataclass_without_its_own_repr_prints_by_repr() -> None:
    value = _InheritsRepr(a=list(range(20)))
    assert pformat(value, width=10) == repr(value)


def test_depth_cut_items_are_measured_as_cut() -> None:
    rendered = pformat([[1, 2, 3] * 5], depth=1, width=10, extra_compact=False)
    assert rendered == "[[...]]"


def test_a_self_containing_list_prints_its_recursion() -> None:
    loop: list[object] = [1]
    loop.append(loop)
    assert pformat(loop, width=5) == "[\n        1,\n        ...\n    ]"


def test_a_too_wide_tuple_lays_out_by_item() -> None:
    assert FigPrinter(width=10, indent=2).pformat(tuple(range(12))) == (
        "(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11)"
    )


def test_qualification_keeps_text_without_a_bare_function_repr() -> None:
    text = "<function elsewhere at 0x1>"
    assert _qualify_function_reprs([_identity_function], text) == text


def test_qualification_walks_a_cyclic_container_once() -> None:
    cyclic: list[object] = [_identity_function]
    cyclic.append(cyclic)
    module = _identity_function.__module__
    assert _qualify_function_reprs(cyclic, repr(cyclic)) == (
        f"[{module}.{_identity_function.__qualname__}, [...]]"
    )


def test_one_line_attempt_stops_at_the_first_item_too_wide() -> None:
    printer = FigPrinter(width=20)
    assert printer._try_format_items_on_one_line(
        ["x" * 50, "y"],
        {},
        0,
        indent=0,
        allowance=0,
    ) == repr("x" * 50)


def test_collapse_keeps_a_value_exactly_max_width_long() -> None:
    assert _collapse_multiline_value("(\n1\n)", 3) == "(1)"


def test_quote_free_text_has_no_string_spans_even_when_untokenizable() -> None:
    assert _string_token_spans("a\n    b\n  c") == []


if __name__ == "__main__":
    from configgle.lib.testing.main import test_main

    test_main(__file__)
