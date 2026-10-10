"""ANSI config highlighting with Vim's dark 256-color palette."""

from __future__ import annotations

import contextlib
import io
import itertools
import tokenize


__all__ = ["color_change", "color_config", "color_diff"]


def color_config(text: str) -> str:
    """Highlight constructors and literals, leaving fields and punctuation plain.

    Args:
      text: Plain config representation, including incomplete representations.

    Returns:
      colored: Original text with ANSI styles inserted, preserving its layout.

    """
    offsets = [0]
    for line in text.splitlines(keepends=True):
        offsets.append(offsets[-1] + len(line))
    tokens: list[tokenize.TokenInfo] = []
    with contextlib.suppress(tokenize.TokenError, IndentationError):
        tokens.extend(tokenize.generate_tokens(io.StringIO(text).readline))
    # Text between consecutive edges is plain, then styled, alternately; cutting
    # there builds the result in one pass, where splicing each span rescanned it.
    edges = [0]
    styles: list[str] = []
    index = 0
    while index < len(tokens):
        token = tokens[index]
        stop_index, style = _token_style(tokens, index=index)
        if style:
            last = tokens[stop_index]
            edges.append(offsets[token.start[0] - 1] + token.start[1])
            edges.append(offsets[last.end[0] - 1] + last.end[1])
            styles.append(style)
        index = stop_index + 1
    pieces = [text[start:stop] for start, stop in itertools.pairwise(edges)]
    for span, style in enumerate(styles):
        pieces[2 * span + 1] = _styled(pieces[2 * span + 1], style=style)
    return "".join(pieces) + text[edges[-1] :]


def color_diff(line: str) -> str:
    """Color a unified diff line, leaving filename headers unchanged.

    Args:
      line: Plain unified diff line.

    Returns:
      colored: Styled change or hunk line; unchanged text otherwise.

    """
    if line.startswith("@@"):
        return _styled(line, style="36")
    if line.startswith("-") and not line.startswith("---"):
        return color_change(line, baseline=True)
    if line.startswith("+") and not line.startswith("+++"):
        return color_change(line, baseline=False)
    return line


def color_change(text: str, *, baseline: bool) -> str:
    """Color a changed baseline red or a changed variant green."""
    return _styled(text, style="31" if baseline else "32")


def _styled(text: str, *, style: str) -> str:
    return f"\x1b[{style}m{text}\x1b[0m"


def _token_style(tokens: list[tokenize.TokenInfo], *, index: int) -> tuple[int, str]:
    token = tokens[index]
    if token.type in (tokenize.STRING, tokenize.NUMBER):
        return index, "95"
    if token.string == "│" and not token.line[: token.start[1]].strip(" \t│"):
        return index, "2;90"
    if token.type != tokenize.NAME:
        return index, ""
    if token.string in ("True", "False", "None"):
        return index, "95"
    stop = index
    while (
        stop + 2 < len(tokens)
        and tokens[stop + 1].string == "."
        and tokens[stop + 2].type == tokenize.NAME
    ):
        stop += 2
    if stop + 1 < len(tokens) and tokens[stop + 1].string == "(":
        return stop, "96"
    return stop, ""
