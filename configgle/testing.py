"""Testing utilities for Configgle pprint goldens."""

from __future__ import annotations

from difflib import unified_diff
from pathlib import Path
from typing import Final

import os

from configgle.fig import Maker


_ENV_REGENERATE: Final = "CONFIGGLE_REGENERATE_GOLDEN"


def assert_pprint_golden(
    *,
    test_file: str,
    name: str,
    config: Maker[object],
) -> None:
    """Assert a config's full finalized pprint matches its test-local golden.

    Missing goldens are written, then raise ``AssertionError`` for inspection.
    ``CONFIGGLE_REGENERATE_GOLDEN=1`` rewrites an existing golden before
    comparison.

    Args:
      test_file: ``__file__`` of the owning test module.
      name: Golden filename without its extension.
      config: Config to finalize and render with every field visible.

    """
    golden = Path(test_file).resolve().parent / "testdata" / f"{name}.txt"
    rendered = (
        config.pformat(
            finalize=True,
            mask_memory_addresses=True,
            hide_default_values=False,
        )
        + "\n"
    )
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
