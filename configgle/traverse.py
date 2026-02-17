"""DEPRECATED Utilities for recursively traversing nested object structures.

This module provides tools for walking through complex nested data structures
including sequences, mappings, sets, and objects with __slots__ or __dict__.
"""

from __future__ import annotations

from collections.abc import (
    Callable,
    Iterator,
    Mapping,
    Sequence,
    Set as AbstractSet,
)

import re


__all__ = [
    "could_path_lead_to_pattern",
    "path_matches_pattern",
    "recursively_iterate_over_object_descendants",
    "should_recurse_for_patterns",
]


def recursively_iterate_over_object_descendants(
    value: object,
    *,
    recurse: Callable[[tuple[int | str, ...], object], bool] = lambda _path, _obj: True,
    seen: set[int] | None = None,
    path: tuple[int | str, ...] = (),
) -> Iterator[tuple[tuple[int | str, ...], object]]:
    """Recursively iterate over nested object descendants.

    Similar to TensorFlow's _flatten_module, this recursively traverses sequences,
    mappings, sets, and object attributes while tracking paths, avoiding cycles,
    and yielding (path, value) tuples.

    Args:
      value: The root value to traverse.
      recurse: Predicate (path, obj) -> bool. If True, yields obj and recurses
        into children. If False, skips. Defaults to always recursing.
      seen: Object IDs already visited (for cycle detection). Created if None.
      path: Current path to this value (tuple of indices/keys from root).

    Yields:
      path: Path tuple of int indices (sequences/sets) or str keys (mappings/attrs).
      value: Value at that path.

    """
    if seen is None:
        seen = set()

    value_id = id(value)
    if value_id in seen:
        return
    seen.add(value_id)

    if not recurse(path, value):
        return

    yield path, value

    if isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
        for i, item in enumerate(value):
            yield from recursively_iterate_over_object_descendants(
                item,
                recurse=recurse,
                seen=seen,
                path=(*path, i),
            )
    elif isinstance(value, Mapping):
        # Mapping key type is unknown at runtime
        for k, v in value.items():  # pyright: ignore[reportUnknownVariableType]
            yield from recursively_iterate_over_object_descendants(
                v,
                recurse=recurse,
                seen=seen,
                path=(*path, k),  # pyright: ignore[reportUnknownArgumentType]  # ty: ignore[invalid-argument-type]
            )
    elif isinstance(value, AbstractSet):
        for i, item in enumerate(value):
            yield from recursively_iterate_over_object_descendants(
                item,
                recurse=recurse,
                seen=seen,
                path=(*path, i),
            )
    else:
        # Handle objects with __slots__ and/or __dict__
        if hasattr(type(value), "__slots__"):
            seen_slots = set[str]()
            for cls in type(value).__mro__:
                slots = getattr(cls, "__slots__", ())
                if isinstance(slots, str):
                    slots = (slots,)
                for slot in slots:
                    if slot in seen_slots or slot == "__dict__":
                        continue
                    seen_slots.add(slot)
                    try:
                        slot_value = getattr(value, slot)
                    except AttributeError:
                        continue
                    yield from recursively_iterate_over_object_descendants(
                        slot_value,
                        recurse=recurse,
                        seen=seen,
                        path=(*path, slot),
                    )
        if hasattr(value, "__dict__"):
            for key in sorted(vars(value)):
                try:
                    attr_value = getattr(value, key)
                except AttributeError:
                    continue
                yield from recursively_iterate_over_object_descendants(
                    attr_value,
                    recurse=recurse,
                    seen=seen,
                    path=(*path, key),
                )


def path_matches_pattern(
    path: tuple[int | str, ...] | str,
    pattern: str,
) -> bool:
    """Check if a path matches a glob pattern.

    Supports glob patterns: "key" (prefix match), "key.*", "key.*.bar", "*", "**".

    Args:
      path: Path as tuple or dot-separated string.
      pattern: Glob pattern to match against.

    Returns:
      match: True if path matches pattern.

    """
    path_str = ".".join(str(p) for p in path) if isinstance(path, tuple) else path

    if pattern == "**":
        return True

    parts = pattern.split(".")
    regex_parts: list[str] = []

    for part in parts:
        if part == "*":
            regex_parts.append(r"[^.]+")
        else:
            regex_parts.append(re.escape(part))

    regex_pattern = r"\.".join(regex_parts)

    if pattern.endswith(".*") or "*" in pattern:
        regex_pattern = f"^{regex_pattern}$"
    else:
        regex_pattern = f"^{regex_pattern}(?:\\.|$)"

    return bool(re.match(regex_pattern, path_str))


def could_path_lead_to_pattern(path_str: str, pattern: str) -> bool:
    """Check if a path could potentially lead to matching the pattern.

    Returns True if:
    1. The path matches the pattern (at the target)
    2. The path is a prefix that could lead to the pattern (on the way)
    3. The path is a child of a matching path (past the target, exploring contents)

    Args:
      path_str: Dot-separated path string.
      pattern: Glob pattern to match against.

    Returns:
      could_match: True if path matches, could lead to, or is child of match.

    """
    if path_matches_pattern(path_str, pattern):
        return True

    path_parts = path_str.split(".")

    if pattern.endswith(".*"):
        prefix = pattern[:-2]
        return path_str == prefix or path_str.startswith(f"{prefix}.")

    if pattern.count("*") > 0:
        parts = pattern.split(".")
        if len(path_parts) <= len(parts):
            return all(
                part == "*" or (i < len(path_parts) and part == path_parts[i])
                for i, part in enumerate(parts[: len(path_parts)])
            )
        for i in range(len(path_parts) - len(parts) + 1):
            ancestor = ".".join(path_parts[: i + len(parts)])
            if path_matches_pattern(ancestor, pattern):
                return True
        return False

    return pattern.startswith(f"{path_str}.")


def should_recurse_for_patterns(
    path: tuple[int | str, ...],
    include: set[str] | None,
    exclude: set[str],
) -> bool:
    """Determine if we should recurse into a path based on include/exclude patterns.

    Args:
      path: Current path tuple.
      include: Include patterns (None means include all).
      exclude: Exclude patterns.

    Returns:
      recurse: True if we should traverse into this path.

    """
    if not path:
        return True

    path_str = ".".join(str(p) for p in path)

    if any(path_matches_pattern(path_str, p) for p in exclude):
        return False

    if include is None:
        return True

    return any(could_path_lead_to_pattern(path_str, p) for p in include)
