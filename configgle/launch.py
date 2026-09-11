"""Launch a config by dotted path, with ``--override`` field edits.

Point the launcher at a factory function that returns a config; it imports the
function, applies any ``--override PATH=VALUE`` edits, and calls ``make()`` on
the result::

    # myproject/experiments.py
    from configgle import Fig, Makeable


    class Trainer:
        class Config(Fig["Trainer"]):
            lr: float = 1e-3
            steps: int = 100

        def __init__(self, config: Config):
            self.config = config

        def run(self) -> None:
            print(f"training {self.config.steps} steps at lr={self.config.lr}")


    def baseline() -> Makeable[Trainer]:
        return Trainer.Config()

Run it, overriding fields from the command line (dotted paths reach into
nested configs; values are cast to each field's declared type)::

    python -m configgle myproject.experiments.baseline
    python -m configgle myproject.experiments.baseline --override lr=3e-4

This launcher is deliberately minimal -- it exists to show the pattern and to
make ``--override`` usable out of the box. A real project usually wants its own
entry point (hardware logging, distributed setup, run naming); build it on
:func:`resolve_config` and :func:`configgle.apply_overrides` rather than
copying this file.

Examples:
  python -m configgle myproject.experiments.baseline
  python -m configgle myproject.experiments.baseline --override lr=3e-4

"""

from __future__ import annotations

from typing import Protocol, runtime_checkable

import argparse
import importlib

from configgle.cli_override import apply_overrides
from configgle.custom_types import Makeable


__all__ = ["main", "resolve_config"]


def resolve_config(path: str) -> Makeable[object]:
    """Import a dotted ``module.function`` path and call it to build a config.

    The factory indirection is what makes an experiment a function rather than a
    module-level constant: each call returns a fresh config, so one process can
    build several without them sharing mutable state.

    Args:
      path: A dotted path to a zero-argument callable returning a config, e.g.
        ``myproject.experiments.baseline``.

    Returns:
      config: The config the factory returned.

    Raises:
      ValueError: ``path`` is not dotted (no module component).
      ImportError: The module cannot be imported.
      AttributeError: The module has no such attribute.
      TypeError: The attribute is not callable, or did not return a config.

    """
    if "." not in path:
        raise ValueError(
            f"'{path}' is not a dotted path; expected module.function "
            "(e.g. myproject.experiments.baseline).",
        )
    module_name, function_name = path.rsplit(".", 1)

    try:
        module = importlib.import_module(module_name)
    except ImportError as e:
        raise ImportError(f"Cannot import module '{module_name}': {e}") from e

    try:
        function = getattr(module, function_name)
    except AttributeError as e:
        raise AttributeError(
            f"Module '{module_name}' has no attribute '{function_name}'",
        ) from e

    if not callable(function):
        raise TypeError(f"'{path}' ({function}) is not callable.")

    config = function()
    if not isinstance(config, Makeable):
        raise TypeError(
            f"'{path}' returned {type(config).__name__}, not a config.",
        )
    return config


def main() -> int:
    """Run the program; return the process exit code.

    Returns:
      result: The int.

    """
    parser = argparse.ArgumentParser(
        description=(__doc__ or "").strip(),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    _add_arguments(parser)
    args = parser.parse_args()

    config = resolve_config(args.config)
    apply_overrides(config, args.override)
    obj = config.make()
    if isinstance(obj, _Runnable):
        obj.run()
    return 0


def _add_arguments(parser: argparse.ArgumentParser) -> None:
    """Register flags on ``parser``."""
    parser.add_argument(
        "config",
        help="Dotted path to a callable returning a config, "
        "e.g. myproject.experiments.baseline.",
    )
    parser.add_argument(
        "--override",
        action="append",
        default=[],
        metavar="PATH=VALUE",
        help="Override a (possibly nested) config field, e.g. --override "
        "step.lr=3e-4. Repeatable. The value is cast to the field's type.",
    )


# This module is imported, never run directly, so it carries no run guard and
# no shebang. `__main__.py` is the sole executable; run it via
# `python -m configgle`.


@runtime_checkable
class _Runnable(Protocol):
    """An object with a no-argument ``run()``, invoked after ``make()``."""

    def run(self) -> None: ...
