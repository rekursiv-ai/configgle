"""Testing utilities for Configgle config goldens."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from difflib import unified_diff
from pathlib import Path
from typing import TYPE_CHECKING, Final

import os

from configgle.fig import Maker
from configgle.flatten import decode, dumps, encode, flatten, loads


if TYPE_CHECKING:
    from configgle.flatten import Entry


_ENV_REGENERATE: Final = "CONFIGGLE_REGENERATE_GOLDEN"


def assert_pprint_golden(
    *,
    test_file: str,
    name: str,
    config: Maker[object],
    normalize: Callable[[str], str] = str,
) -> None:
    """Assert a config's full finalized pprint matches its test-local golden.

    Missing goldens are written, then raise ``AssertionError`` for inspection.
    ``CONFIGGLE_REGENERATE_GOLDEN=1`` rewrites an existing golden before
    comparison.

    Args:
      test_file: ``__file__`` of the owning test module.
      name: Golden filename without its extension.
      config: Config to finalize and render with every field visible.
      normalize: Maps rendered text to its host-independent form.

    """
    golden = Path(test_file).resolve().parent / "testdata" / f"{name}.txt"
    rendered = normalize(
        config.pformat(
            finalize=True,
            mask_memory_addresses=True,
            hide_default_values=False,
        )
        + "\n",
    )
    _write_or_compare(golden, rendered, name=name)


@dataclass(frozen=True, slots=True, kw_only=True)
class FlatParent:
    """A parent config and the flat golden that records it.

    Attributes:
      golden: The parent's ``testdata`` golden written by ``assert_flat_golden``.
      factory: Builds the live parent config the child is stored against.
      parent: The parent's own parent when its golden is itself a delta.

    """

    golden: Path
    factory: Callable[[], Maker[object]]
    parent: FlatParent | None = None


def assert_flat_golden(
    *,
    test_file: str,
    name: str,
    config: Maker[object],
    parent: FlatParent | None = None,
    normalize: Callable[[str], str] = str,
) -> None:
    """Assert a config's full finalized leaves match its test-local flat golden.

    The golden is ``configgle.flatten`` output: every leaf of the finalized
    ``serialize()`` tree, with repeated subtrees as references, written as an
    outline. Nothing is truncated, so a change at any depth fails. With
    ``parent``, the golden holds only the delta from the parent, and the parent's
    own golden must still match the live parent, so a changed parent default
    fails the child too. Missing goldens are written, then raise
    ``AssertionError``; ``CONFIGGLE_REGENERATE_GOLDEN=1`` rewrites them.

    Args:
      test_file: ``__file__`` of the owning test module.
      name: Golden filename without its extension.
      config: Config to finalize and record.
      parent: Config and golden the child is stored against; None for a root.
      normalize: Maps each leaf ``repr`` to its host-independent form.

    Raises:
      AssertionError: The golden is missing, differs, or its parent is stale.

    """
    golden = Path(test_file).resolve().parent / "testdata" / f"{name}.txt"
    entries = _finalized_entries(config, normalize=normalize)
    if parent is None:
        lines = encode(entries)
    else:
        base = _finalized_entries(parent.factory(), normalize=normalize)
        if _golden_entries(parent) != base:
            raise AssertionError(
                f"{name}: parent golden {parent.golden} no longer matches its "
                "config; update the parent first.",
            )
        lines = encode(entries, base=base)
    rendered = dumps(lines)
    base = [] if parent is None else _golden_entries(parent)
    if decode(loads(rendered), base=base) != entries:
        raise AssertionError(f"{name}: the flat encoding does not round-trip.")
    _write_or_compare(golden, rendered, name=name)


def _golden_entries(parent: FlatParent) -> list[Entry]:
    """Decode a parent golden through its own chain of parent goldens."""
    base = [] if parent.parent is None else _golden_entries(parent.parent)
    return decode(loads(parent.golden.read_text(encoding="utf-8")), base=base)


def _finalized_entries(
    config: Maker[object],
    *,
    normalize: Callable[[str], str],
) -> list[Entry]:
    tree = config.copy_tree().finalize().serialize()
    return [(path, normalize(value)) for path, value in flatten(tree)]


def _write_or_compare(golden: Path, rendered: str, *, name: str) -> None:
    missing = not golden.exists()
    if missing or os.environ.get(_ENV_REGENERATE) == "1":
        golden.parent.mkdir(parents=True, exist_ok=True)
        golden.write_text(rendered, encoding="utf-8")
    if missing:
        raise AssertionError(
            f"Missing golden regenerated at {golden}; inspect it, then rerun the test.",
        )
    expected = golden.read_text(encoding="utf-8")
    if expected == rendered:
        return
    diff = "".join(
        unified_diff(
            expected.splitlines(keepends=True),
            rendered.splitlines(keepends=True),
            fromfile=str(golden),
            tofile=f"{name} (rendered)",
        ),
    )
    raise AssertionError(
        f"{name} changed; rerun with {_ENV_REGENERATE}=1 if intended.\n{diff}",
    )
