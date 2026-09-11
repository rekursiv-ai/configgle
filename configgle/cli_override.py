"""Apply ``PATH=VALUE`` config overrides from a command line.

A launcher parses ``--override step.lr=3e-4`` off the command line and hands the
strings here; this module owns the config-side mechanics -- path traversal,
field validation, and type coercion -- so the launcher keeps only its argparse
wiring.

An override names a (possibly nested) field by a dotted path. Every hop is
validated against the node's *declared* fields, never an arbitrary attribute or
method, so a typo fails loudly instead of silently creating state. The leaf
value is parsed as a JSON literal (``true`` / ``3e-4`` / ``[1, 2]`` decode
naturally) and then coerced to the field's declared type.
"""

from __future__ import annotations

from typing import get_type_hints

import dataclasses
import json

from configgle.custom_json import decode
from configgle.custom_types import Makeable


__all__ = ["apply_overrides"]


def apply_overrides(config: Makeable[object], overrides: list[str]) -> None:
    """Apply ``PATH=VALUE`` config overrides in place.

    Each override names a (possibly nested) field by a dotted path and a
    raw value. Traversal validates every hop against the node's declared
    fields (never an arbitrary attribute or method); the leaf is set to the
    raw value coerced to its declared type. ``VALUE`` is first parsed as a
    JSON literal (so ``true`` / ``3e-4`` / ``[1, 2]`` decode naturally),
    falling back to the bare string, then cast against the field annotation.

    Example:
      >>> apply_overrides(cfg, ["step.lr=3e-4", "name=run_a"])

    Args:
      config: The configgle config (a dataclass) to mutate.
      overrides: ``PATH=VALUE`` strings, e.g. ``["step.lr=3e-4", "name=run"]``.

    Raises:
      ValueError: An override lacks ``=``, traverses a non-config node, names
        a field absent from the node's declared fields, or carries a value
        that cannot coerce to the field's declared type.

    """
    for override in overrides:
        if "=" not in override:
            raise ValueError(
                f"Malformed override `{override}`; expected PATH=VALUE.",
            )
        path, _, raw = override.partition("=")
        keys = path.split(".")
        if not path or "" in keys:
            raise ValueError(
                f"Malformed override `{override}`; PATH must be a dotted "
                "field path with no empty segments (e.g. `step.lr`).",
            )
        node = config
        for key in keys[:-1]:
            _, node = _override_field(node, key, path)
        leaf = keys[-1]
        annotation, _ = _override_field(node, leaf, path)
        try:
            value = decode(annotation, _parse_override_value(raw))
        except (TypeError, ValueError) as e:
            # ``decode`` raises ``TypeError`` when ``raw`` cannot coerce to the
            # field type -- e.g. a scalar for a nested-config field
            # (``child=5`` instead of ``child.lr=5``). Re-raise as a
            # path-naming ``ValueError`` so the caller sees one error contract.
            raise ValueError(
                f"Override path `{path}` cannot set `{raw}` "
                f"on a field of type {annotation}: {e}",
            ) from e
        # We own ``config`` and mutate it in place, writing the field directly on
        # the real (per-instance, ``default_factory``-built, never shared) node.
        # ``object.__setattr__`` is the primitive configgle itself uses to write
        # frozen Figs (see ``fig.py`` ``finalize``): it writes through
        # ``frozen=True`` while still honoring ``slots`` (an unknown leaf raises
        # ``AttributeError``). ``_override_field`` already rejected unknown
        # fields, so only declared, type-checked fields reach here.
        object.__setattr__(node, leaf, value)


# Validates ``key`` against ``node``'s resolved type hints -- a declared field, never an
# arbitrary attribute or method -- so each hop of an override path is checked uniformly.
# The annotation drives leaf casting; the value continues the traversal.
def _override_field(node: object, key: str, path: str) -> tuple[object, object]:
    """Return ``(annotation, value)`` of ``node``'s declared field ``key``."""
    if not dataclasses.is_dataclass(node):
        raise ValueError(
            f"Override path `{path}` traverses non-config "
            f"{type(node).__name__}, which has no field `{key}`.",
        )
    hints = get_type_hints(type(node))
    if key not in hints:
        raise ValueError(
            f"Override path `{path}` has no field `{key}` on {type(node).__name__}.",
        )
    return hints[key], getattr(node, key)


def _parse_override_value(raw: str) -> object:
    """Parse a raw override value as a JSON literal, else the bare string."""
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        return raw
