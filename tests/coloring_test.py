"""Tests for config representation and diff colors."""

import re

import pytest

from configgle.coloring import color_change, color_config, color_diff


@pytest.mark.parametrize(
    "text",
    [
        "",
        "Config(\n value=2\n)",
        "Config(\n value=2",
        "a\n  b\n c",
        "name",
        "module.name",
        "[1, -2.5, 3j]",
        "'True name=Config(42)'",
        '"a\\"b"',
        "<function pkg.name at 0xdefacedeface>",
    ],
)
def test_color_preserves_all_original_characters(text: str) -> None:
    assert re.sub(r"\x1b\[[0-9;]*m", "", color_config(text)) == text


def test_continuation_guides_are_dim_but_quoted_pipes_are_not() -> None:
    text = "Config(\n │   │ value='│',\n │   count=2\n)"
    rendered = color_config(text)
    assert rendered == (
        "\x1b[96mConfig\x1b[0m(\n \x1b[2;90m│\x1b[0m   "
        "\x1b[2;90m│\x1b[0m value=\x1b[95m'│'\x1b[0m,\n "
        "\x1b[2;90m│\x1b[0m   count=\x1b[95m2\x1b[0m\n)"
    )
    assert re.sub(r"\x1b\[[0-9;]*m", "", rendered) == text


def test_multiline_literal_offsets() -> None:
    assert color_config("Config(\n  value=2,\n  text='hi'\n)") == (
        "\x1b[96mConfig\x1b[0m(\n  value=\x1b[95m2\x1b[0m,\n  text=\x1b[95m'hi'\x1b[0m\n)"
    )


def test_repr_roles_and_palette() -> None:
    text = "Outer.Config(child=Inner(value=2), s='hi', t=True, f=False, n=None, ordinary=name)"
    rendered = color_config(text)
    assert rendered == (
        "\x1b[96mOuter.Config\x1b[0m(child=\x1b[96mInner\x1b[0m(value=\x1b[95m2\x1b[0m), "
        "s=\x1b[95m'hi'\x1b[0m, t=\x1b[95mTrue\x1b[0m, "
        "f=\x1b[95mFalse\x1b[0m, n=\x1b[95mNone\x1b[0m, ordinary=name)"
    )


@pytest.mark.parametrize(
    ("line", "expected"),
    [
        ("--- ", "--- "),
        ("+++ ", "+++ "),
        (" unchanged", " unchanged"),
        ("@@ hunk", "\x1b[36m@@ hunk\x1b[0m"),
        ("-removed", "\x1b[31m-removed\x1b[0m"),
        ("+added", "\x1b[32m+added\x1b[0m"),
    ],
)
def test_diff_palette(line: str, expected: str) -> None:
    assert color_diff(line) == expected


def test_changed_cell_palette() -> None:
    assert color_change("before", baseline=True) == "\x1b[31mbefore\x1b[0m"
    assert color_change("after", baseline=False) == "\x1b[32mafter\x1b[0m"


if __name__ == "__main__":
    from configgle.lib.testing.main import test_main

    test_main(__file__)
