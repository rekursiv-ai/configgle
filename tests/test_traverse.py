"""Tests for core.config.traverse."""

from __future__ import annotations

from typing import TYPE_CHECKING, NamedTuple

from configgle.traverse import (
    path_matches_pattern,
    recursively_iterate_over_object_descendants,
    should_recurse_for_patterns,
)


if TYPE_CHECKING:
    from typing import Any


def test_path_matches_pattern_exact():
    """Test exact path matching."""
    assert path_matches_pattern("key", "key")
    assert path_matches_pattern(("key",), "key")
    assert not path_matches_pattern("key", "other")


def test_path_matches_pattern_prefix():
    """Test prefix matching for patterns without wildcards."""
    assert path_matches_pattern("key", "key")
    assert path_matches_pattern("key.child", "key")
    assert path_matches_pattern("key.0", "key")
    assert path_matches_pattern("key.0.grandchild", "key")
    assert not path_matches_pattern("key", "key.child")
    assert not path_matches_pattern("other", "key")


def test_path_matches_pattern_direct_wildcard():
    """Test direct child wildcard matching with .*"""
    assert path_matches_pattern("key.0", "key.*")
    assert path_matches_pattern("key.child", "key.*")
    assert not path_matches_pattern("key", "key.*")
    assert not path_matches_pattern("key.0.grandchild", "key.*")


def test_path_matches_pattern_nested_wildcard():
    """Test nested wildcard patterns like foo.*.bar."""
    assert path_matches_pattern("embeddings.clip.features", "embeddings.*.features")
    assert path_matches_pattern("embeddings.dino.features", "embeddings.*.features")
    assert not path_matches_pattern("embeddings.features", "embeddings.*.features")
    assert not path_matches_pattern(
        "embeddings.clip.features.0",
        "embeddings.*.features",
    )
    assert not path_matches_pattern("embeddings.clip", "embeddings.*.features")


def test_path_matches_pattern_multiple_wildcards():
    """Test patterns with multiple wildcards."""
    assert path_matches_pattern("a.b.c.d", "a.*.*.d")
    assert path_matches_pattern("a.x.y.d", "a.*.*.d")
    assert not path_matches_pattern("a.b.d", "a.*.*.d")
    assert not path_matches_pattern("a.b.c.d.e", "a.*.*.d")


def test_path_matches_pattern_single_wildcard():
    """Test single * matching any segment."""
    assert path_matches_pattern("x", "*")
    assert path_matches_pattern("foo", "*")
    assert not path_matches_pattern("foo.bar", "*")


def test_path_matches_pattern_double_wildcard():
    """Test ** matching everything."""
    assert path_matches_pattern("anything", "**")
    assert path_matches_pattern("foo.bar.baz", "**")
    assert path_matches_pattern("", "**")


def test_path_matches_pattern_tuple_input():
    """Test that tuple paths work correctly."""
    assert path_matches_pattern(("media_tensor", 0), "media_tensor.*")
    assert path_matches_pattern(("embeddings", "clip", "features"), "embeddings.*.*")


def test_should_recurse_for_patterns_no_filters():
    """Test recursion with no include/exclude filters."""
    assert should_recurse_for_patterns((), None, set())
    assert should_recurse_for_patterns(("a",), None, set())
    assert should_recurse_for_patterns(("a", "b"), None, set())


def test_should_recurse_for_patterns_exclude():
    """Test recursion with exclude patterns.

    Excluding "raw" automatically excludes all descendants via prefix matching.
    """
    exclude = {"raw"}
    assert should_recurse_for_patterns((), None, exclude)
    assert not should_recurse_for_patterns(("raw",), None, exclude)
    assert not should_recurse_for_patterns(("raw", "child"), None, exclude)
    assert not should_recurse_for_patterns(("raw", "0", "grandchild"), None, exclude)
    assert should_recurse_for_patterns(("other",), None, exclude)


def test_should_recurse_for_patterns_exclude_wildcard():
    """Test recursion with exclude wildcard patterns."""
    exclude = {"embeddings.*"}
    assert should_recurse_for_patterns((), None, exclude)
    assert should_recurse_for_patterns(("embeddings",), None, exclude)
    assert not should_recurse_for_patterns(("embeddings", "clip"), None, exclude)


def test_should_recurse_for_patterns_include():
    """Test recursion with include patterns."""
    include = {"media_tensor"}
    assert should_recurse_for_patterns((), include, set())
    assert should_recurse_for_patterns(("media_tensor",), include, set())
    assert should_recurse_for_patterns(("media_tensor", "0"), include, set())
    assert not should_recurse_for_patterns(("other",), include, set())


def test_should_recurse_for_patterns_include_nested():
    """Test recursion with nested include patterns."""
    include = {"embeddings.clip.features"}
    assert should_recurse_for_patterns((), include, set())
    assert should_recurse_for_patterns(("embeddings",), include, set())
    assert should_recurse_for_patterns(("embeddings", "clip"), include, set())
    assert should_recurse_for_patterns(
        ("embeddings", "clip", "features"),
        include,
        set(),
    )
    assert not should_recurse_for_patterns(("embeddings", "dino"), include, set())
    assert not should_recurse_for_patterns(("other",), include, set())


def test_should_recurse_for_patterns_include_wildcard():
    """Test recursion with include wildcard patterns.

    Note: should_recurse determines if we explore a path, not if we transfer it.
    With include="embeddings.*", we recurse into "embeddings.clip.features" to find
    tensors, but tensors there won't be transferred (path_matches_pattern filters).
    """
    include = {"embeddings.*"}
    assert should_recurse_for_patterns((), include, set())
    assert should_recurse_for_patterns(("embeddings",), include, set())
    assert should_recurse_for_patterns(("embeddings", "clip"), include, set())
    assert should_recurse_for_patterns(
        ("embeddings", "clip", "features"),
        include,
        set(),
    )
    assert not should_recurse_for_patterns(("other",), include, set())


def test_should_recurse_for_patterns_include_nested_wildcard():
    """Test recursion with nested wildcard include patterns.

    Note: We recurse into children of matching paths to find tensors.
    Filtering happens at transfer time via path_matches_pattern.
    """
    include = {"embeddings.*.features"}
    assert should_recurse_for_patterns((), include, set())
    assert should_recurse_for_patterns(("embeddings",), include, set())
    assert should_recurse_for_patterns(("embeddings", "clip"), include, set())
    assert should_recurse_for_patterns(
        ("embeddings", "clip", "features"),
        include,
        set(),
    )
    assert should_recurse_for_patterns(
        ("embeddings", "clip", "features", "0"),
        include,
        set(),
    )
    assert not should_recurse_for_patterns(("other",), include, set())


def test_should_recurse_for_patterns_include_exclude():
    """Test recursion with both include and exclude patterns."""
    include = {"embeddings"}
    exclude = {"embeddings.raw"}
    assert should_recurse_for_patterns((), include, exclude)
    assert should_recurse_for_patterns(("embeddings",), include, exclude)
    assert should_recurse_for_patterns(("embeddings", "clip"), include, exclude)
    assert not should_recurse_for_patterns(("embeddings", "raw"), include, exclude)
    assert not should_recurse_for_patterns(("other",), include, exclude)


def test_should_recurse_for_patterns_multiple_includes():
    """Test recursion with multiple include patterns."""
    include = {"media_tensor", "embeddings.clip"}
    assert should_recurse_for_patterns((), include, set())
    assert should_recurse_for_patterns(("media_tensor",), include, set())
    assert should_recurse_for_patterns(("embeddings",), include, set())
    assert should_recurse_for_patterns(("embeddings", "clip"), include, set())
    assert not should_recurse_for_patterns(("embeddings", "dino"), include, set())
    assert not should_recurse_for_patterns(("other",), include, set())


def test_path_matches_with_traversal():
    """Test using path matching with actual traversal."""
    data = {
        "media_tensor": [1, 2, 3],
        "embeddings": {"clip": {"features": [4, 5]}, "dino": {"features": [6, 7]}},
        "raw": {"data": [8, 9]},
    }

    include = {"embeddings.*.features"}
    exclude = set[str]()

    results = [
        (p, v)
        for p, v in recursively_iterate_over_object_descendants(
            data,
            recurse=lambda p, _: should_recurse_for_patterns(p, include, exclude),
        )
        if isinstance(v, int)
    ]

    values = {v for _, v in results}
    assert values == {4, 5, 6, 7}


def test_exclude_with_traversal():
    """Test using exclude patterns with actual traversal."""
    data = {
        "media_tensor": [1, 2],
        "raw": [3, 4],
        "embeddings": [5, 6],
    }

    include = None
    exclude = {"raw"}

    results = [
        (p, v)
        for p, v in recursively_iterate_over_object_descendants(
            data,
            recurse=lambda p, _: should_recurse_for_patterns(p, include, exclude),
        )
        if isinstance(v, int)
    ]

    values = {v for _, v in results}
    assert values == {1, 2, 5, 6}


def test_basic_sequence():
    """Test traversal of simple sequences."""
    data = [1, 2, 3]
    results = [
        (p, v)
        for p, v in recursively_iterate_over_object_descendants(data)
        if isinstance(v, int)
    ]
    assert results == [((0,), 1), ((1,), 2), ((2,), 3)]


def test_nested_sequence():
    """Test traversal of nested sequences."""
    data = [[1, 2], [3, 4]]
    results = [
        (p, v)
        for p, v in recursively_iterate_over_object_descendants(data)
        if isinstance(v, int)
    ]
    assert results == [((0, 0), 1), ((0, 1), 2), ((1, 0), 3), ((1, 1), 4)]


def test_basic_mapping():
    """Test traversal of simple mappings."""
    data = {"a": 1, "b": 2}
    results = [
        (p, v)
        for p, v in recursively_iterate_over_object_descendants(data)
        if isinstance(v, int)
    ]
    assert set(results) == {(("a",), 1), (("b",), 2)}


def test_nested_mapping():
    """Test traversal of nested mappings."""
    data = {"a": {"x": 1}, "b": {"y": 2}}
    results = [
        (p, v)
        for p, v in recursively_iterate_over_object_descendants(data)
        if isinstance(v, int)
    ]
    assert set(results) == {(("a", "x"), 1), (("b", "y"), 2)}


def test_mixed_sequence_mapping():
    """Test traversal of mixed sequences and mappings."""
    data = {"a": [1, 2], "b": {"c": 3}}
    results = [
        (p, v)
        for p, v in recursively_iterate_over_object_descendants(data)
        if isinstance(v, int)
    ]
    assert set(results) == {(("a", 0), 1), (("a", 1), 2), (("b", "c"), 3)}


def test_set_traversal():
    """Test traversal of sets."""
    data = {1, 2, 3}
    results = [
        (p, v)
        for p, v in recursively_iterate_over_object_descendants(data)
        if isinstance(v, int)
    ]
    # Sets are unordered, but each element should appear once with index
    assert len(results) == 3
    assert {v for _, v in results} == {1, 2, 3}


def test_cycle_detection():
    """Test that circular references are handled correctly."""
    data: list[int | list[Any]] = [1, 2]
    data.append(data)  # Create cycle

    results = [
        (p, v)
        for p, v in recursively_iterate_over_object_descendants(data)
        if isinstance(v, int)
    ]
    # Should find 1 and 2, but not loop infinitely
    assert results == [((0,), 1), ((1,), 2)]


def test_recurse_matches_root():
    """Test that root value is yielded."""
    data = 42
    results = list(recursively_iterate_over_object_descendants(data))
    assert results == [((), 42)]


def test_recurse_matches_containers():
    """Test filtering for container types."""
    data = {"a": [1, 2], "b": [3, 4]}
    results: list[tuple[tuple[int | str, ...], list[int]]] = [  # ty: ignore[invalid-assignment]
        (p, v)
        for p, v in recursively_iterate_over_object_descendants(data)
        if isinstance(v, list)
    ]
    assert len(results) == 2
    # Lists are found; convert to tuples for comparison
    assert {tuple(v) for _, v in results} == {(1, 2), (3, 4)}


def test_string_not_traversed():
    """Test that strings are not traversed character-by-character."""
    data = ["hello", "world"]
    results = [
        (p, v)
        for p, v in recursively_iterate_over_object_descendants(data)
        if isinstance(v, str)
    ]
    assert results == [((0,), "hello"), ((1,), "world")]


def test_bytes_not_traversed():
    """Test that bytes are not traversed byte-by-byte."""
    data = [b"hello", b"world"]
    results = [
        (p, v)
        for p, v in recursively_iterate_over_object_descendants(data)
        if isinstance(v, bytes)
    ]
    assert results == [((0,), b"hello"), ((1,), b"world")]


def test_object_with_slots():
    """Test traversal of objects with __slots__."""

    class SlotObject:
        __slots__ = ("x", "y")

        def __init__(self, x: int, y: int):
            self.x = x
            self.y = y

    obj = SlotObject(1, 2)
    results = [
        (p, v)
        for p, v in recursively_iterate_over_object_descendants(obj)
        if isinstance(v, int)
    ]
    # Results should include both x and y
    assert set(results) == {(("x",), 1), (("y",), 2)}


def test_object_with_dict():
    """Test traversal of objects with __dict__."""

    class DictObject:
        def __init__(self, x: int, y: int):
            self.x = x
            self.y = y

    obj = DictObject(1, 2)
    results = [
        (p, v)
        for p, v in recursively_iterate_over_object_descendants(obj)
        if isinstance(v, int)
    ]
    # Results should include both x and y (sorted by key)
    assert results == [(("x",), 1), (("y",), 2)]


def test_object_with_slots_and_dict():
    """Test traversal of objects with both __slots__ and __dict__."""

    class BaseWithSlots:
        __slots__ = ("x",)

        def __init__(self, x: int):
            self.x = x

    class DerivedWithDict(BaseWithSlots):
        __slots__ = ("__dict__",)

        def __init__(self, x: int, y: int):
            super().__init__(x)
            self.y = y

    obj = DerivedWithDict(1, 2)
    results = [
        (p, v)
        for p, v in recursively_iterate_over_object_descendants(obj)
        if isinstance(v, int)
    ]
    # Results should include both x (from slots) and y (from __dict__)
    assert set(results) == {(("x",), 1), (("y",), 2)}


def test_slots_with_mro():
    """Test traversal respects MRO for __slots__."""

    class Base:
        __slots__ = ("x",)

        def __init__(self, x: int):
            self.x = x

    class Derived(Base):
        __slots__ = ("y",)

        def __init__(self, x: int, y: int):
            super().__init__(x)
            self.y = y

    obj = Derived(1, 2)
    results = [
        (p, v)
        for p, v in recursively_iterate_over_object_descendants(obj)
        if isinstance(v, int)
    ]
    # Should find both x and y
    assert set(results) == {(("x",), 1), (("y",), 2)}


def test_slots_string_format():
    """Test handling of __slots__ as a string instead of tuple."""

    class StringSlotObject:
        __slots__ = ("x",)  # Intentionally using single-element tuple

        def __init__(self, x: int):
            self.x = x

    obj = StringSlotObject(42)
    results = [
        (p, v)
        for p, v in recursively_iterate_over_object_descendants(obj)
        if isinstance(v, int)
    ]
    assert results == [(("x",), 42)]


def test_skip_dict_slot():
    """Test that __dict__ slot is not traversed as a regular slot."""

    class WithDictSlot:
        __slots__ = ("__dict__", "x")

        def __init__(self, x: int, y: int):
            self.x = x
            self.y = y  # Goes into __dict__

    obj = WithDictSlot(1, 2)
    results = [
        (p, v)
        for p, v in recursively_iterate_over_object_descendants(obj)
        if isinstance(v, int)
    ]
    # Should find both x and y, but not try to access __dict__ as a slot
    assert set(results) == {(("x",), 1), (("y",), 2)}


def test_attribute_error_handling():
    """Test handling of AttributeError when accessing slots."""

    class LazySlotObject:
        __slots__ = ("x", "y")  # pyright: ignore[reportUninitializedInstanceVariable]

        def __init__(self):
            self.x = 1
            # y is not set, will raise AttributeError

    obj = LazySlotObject()
    results = [
        (p, v)
        for p, v in recursively_iterate_over_object_descendants(obj)
        if isinstance(v, int)
    ]
    # Should find x but gracefully skip y
    assert results == [(("x",), 1)]


def test_property_attribute_error():
    """Test handling of properties that raise AttributeError."""

    class PropertyObject:
        @property
        def bad_property(self) -> int:
            raise AttributeError("This property always fails")

        def __init__(self, x: int):
            self.x = x

    obj = PropertyObject(42)
    results = [
        (p, v)
        for p, v in recursively_iterate_over_object_descendants(obj)
        if isinstance(v, int)
    ]
    # Should find x but skip the bad property
    assert results == [(("x",), 42)]


def test_empty_sequence():
    """Test traversal of empty sequence."""
    data = list[int]()
    results = [
        (p, v)
        for p, v in recursively_iterate_over_object_descendants(data)
        if isinstance(v, int)
    ]
    assert results == []


def test_empty_mapping():
    """Test traversal of empty mapping."""
    data: dict[str, int] = {}
    results = [
        (p, v)
        for p, v in recursively_iterate_over_object_descendants(data)
        if isinstance(v, int)
    ]
    assert results == []


def test_empty_set():
    """Test traversal of empty set."""
    data: set[int] = set()
    results = [
        (p, v)
        for p, v in recursively_iterate_over_object_descendants(data)
        if isinstance(v, int)
    ]
    assert results == []


def test_namedtuple():
    """Test traversal of namedtuples."""

    class Point(NamedTuple):
        x: int
        y: int

    data = Point(1, 2)
    results = [
        (p, v)
        for p, v in recursively_iterate_over_object_descendants(data)
        if isinstance(v, int)
    ]
    assert results == [((0,), 1), ((1,), 2)]


def test_custom_filtering():
    """Test with custom filtering in user code."""
    data = {"a": 1, "b": "hello", "c": 3.14, "d": [1, 2]}

    # Find only strings
    results = [
        (p, v)
        for p, v in recursively_iterate_over_object_descendants(data)
        if isinstance(v, str)
    ]
    assert results == [(("b",), "hello")]

    # Find only floats
    results = [
        (p, v)
        for p, v in recursively_iterate_over_object_descendants(data)
        if isinstance(v, float)
    ]
    assert results == [(("c",), 3.14)]


def test_path_accuracy():
    """Test that paths accurately reflect nested structure."""
    data = {"a": {"b": {"c": [1, 2, 3]}}}
    results = [
        (p, v)
        for p, v in recursively_iterate_over_object_descendants(data)
        if isinstance(v, int)
    ]
    assert results == [
        (("a", "b", "c", 0), 1),
        (("a", "b", "c", 1), 2),
        (("a", "b", "c", 2), 3),
    ]


def test_seen_parameter():
    """Test that seen parameter can be passed in."""
    data1 = [1, 2]
    data2 = [3, 4]

    seen: set[int] = set()

    # First traversal
    results1 = [
        (p, v)
        for p, v in recursively_iterate_over_object_descendants(data1, seen=seen)
        if isinstance(v, int)
    ]
    assert len(results1) == 2

    # Second traversal with same seen set should skip data1 if it was a child
    # But since data1 is not a child of data2, both should be found
    results2 = [
        (p, v)
        for p, v in recursively_iterate_over_object_descendants(data2, seen=seen)
        if isinstance(v, int)
    ]
    assert len(results2) == 2


def test_complex_nested_structure():
    """Test with a complex nested structure combining all types."""

    class Config:
        __slots__ = ("name",)

        def __init__(self, name: str):
            self.name = name

    data = {
        "list": [1, 2, [3, 4]],
        "dict": {"a": 5, "b": {"c": 6}},
        "set": {7, 8},
        "tuple": (9, 10),
        "object": Config("test"),
    }

    results = [
        (p, v)
        for p, v in recursively_iterate_over_object_descendants(data)
        if isinstance(v, int)
    ]

    # Should find all integers 1-10
    values = {v for _, v in results}
    assert values == {1, 2, 3, 4, 5, 6, 7, 8, 9, 10}


def test_all_descendants():
    """Test yielding all descendants."""
    data = [1, 2, 3]
    results = list(recursively_iterate_over_object_descendants(data))
    # Should yield the list itself and all integers
    assert len(results) == 4  # [1, 2, 3] + 1 + 2 + 3


def test_recurse_depth_limit():
    """Test recurse parameter to limit depth."""
    data = {"a": [1, 2], "b": {"c": 3}}

    # Only traverse root and direct children
    results = list(
        recursively_iterate_over_object_descendants(
            data,
            recurse=lambda path, _: len(path) <= 1,
        ),
    )
    # Should yield dict itself, list [1, 2], and nested dict {"c": 3}
    assert len(results) == 3
    paths = {path for path, _ in results}
    assert paths == {(), ("a",), ("b",)}


def test_recurse_depth_limit_with_filtering():
    """Test recurse parameter with additional filtering."""
    data = {"a": [1, 2], "b": {"c": 3}}

    # Traverse up to depth 2 and filter for ints
    results = [
        (p, v)
        for p, v in recursively_iterate_over_object_descendants(
            data,
            recurse=lambda path, _: len(path) <= 2,
        )
        if isinstance(v, int)
    ]
    # Should yield integers at depth 2
    assert set(results) == {(("a", 0), 1), (("a", 1), 2), (("b", "c"), 3)}


def test_recurse_root_only():
    """Test recurse=lambda path, _: len(path) == 0 yields only root."""
    data = {"a": [1, 2], "b": {"c": 3}}
    results = list(
        recursively_iterate_over_object_descendants(
            data,
            recurse=lambda path, _: len(path) == 0,
        ),
    )
    # Should only yield the root dict
    assert results == [((), data)]


def test_recurse_with_object():
    """Test recurse with object attributes."""

    class Config:
        __slots__ = ("x", "y")

        def __init__(self):
            self.x = {"nested": 1}
            self.y = 2

    obj = Config()
    results = list(
        recursively_iterate_over_object_descendants(
            obj,
            recurse=lambda path, _: len(path) <= 1,
        ),
    )
    # Should yield obj itself, x dict, and y int (direct attributes only)
    assert len(results) == 3
    paths = {path for path, _ in results}
    assert paths == {(), ("x",), ("y",)}


if __name__ == "__main__":
    import pytest

    pytest.main([__file__, "-v"])
