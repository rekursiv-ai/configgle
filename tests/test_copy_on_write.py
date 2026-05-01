"""Tests for configgle.copy_on_write."""

from __future__ import annotations

from typing import Self

import copy
import dataclasses

import pytest

from configgle.copy_on_write import CopyOnWrite


@dataclasses.dataclass  # check-dataclass: ignore[kw_only,slots]
class SimpleConfig:
    """Simple config for testing."""

    value: int = 0
    name: str = "default"

    def finalize(self) -> Self:
        return copy.copy(self)


@dataclasses.dataclass  # check-dataclass: ignore[kw_only,slots]
class NestedConfig:
    """Config with nested structure."""

    inner: SimpleConfig = dataclasses.field(default_factory=SimpleConfig)
    items: list[int] = dataclasses.field(default_factory=list)

    def finalize(self) -> Self:
        return copy.copy(self)


@dataclasses.dataclass  # check-dataclass: ignore[kw_only,slots]
class DeeplyNestedConfig:
    """Config with deeply nested structure."""

    level1: NestedConfig = dataclasses.field(default_factory=NestedConfig)

    def finalize(self) -> Self:
        return copy.copy(self)


class TestCopyOnWriteBasic:
    """Test basic COW operations."""

    def test_read_without_copy(self):
        """Reading attributes should not trigger a copy."""
        original = SimpleConfig(value=42, name="test")
        with CopyOnWrite(original) as cow:
            _ = cow.value
            _ = cow.name
            assert cow._self_is_copy is False

        assert original.value == 42
        assert original.name == "test"

    def test_write_triggers_copy(self):
        """Writing an attribute should trigger a copy."""
        original = SimpleConfig(value=42, name="test")
        with CopyOnWrite(original) as cow:
            cow.value = 100
            assert cow._self_is_copy is True
            assert cow.unwrap.value == 100

        # Original unchanged
        assert original.value == 42

    def test_multiple_writes_single_copy(self):
        """Multiple writes should only copy once."""
        original = SimpleConfig(value=42, name="test")
        with CopyOnWrite(original) as cow:
            cow.value = 100
            cow.name = "modified"
            # Still the same copy
            assert cow._self_is_copy is True
            assert cow.unwrap.value == 100
            assert cow.unwrap.name == "modified"

        assert original.value == 42
        assert original.name == "test"


class TestCopyOnWriteNested:
    """Test COW with nested objects."""

    def test_nested_read_no_copy(self):
        """Reading nested attributes should not trigger copies."""
        original = NestedConfig(inner=SimpleConfig(value=42))
        with CopyOnWrite(original) as cow:
            _ = cow.inner.value
            assert cow._self_is_copy is False

    def test_nested_write_copies_chain(self):
        """Writing nested attribute should copy parent chain."""
        original = NestedConfig(inner=SimpleConfig(value=42))
        original_inner = original.inner

        with CopyOnWrite(original) as cow:
            cow.inner.value = 100

            # Both parent and child should be copied
            assert cow._self_is_copy is True
            inner_cow = cow._self_children.get("inner")
            assert inner_cow is not None
            assert inner_cow._self_is_copy is True

            # Values are updated
            assert cow.unwrap.inner.value == 100

        # Originals unchanged
        assert original.inner.value == 42
        assert original.inner is original_inner

    def test_deeply_nested_write(self):
        """Writing deeply nested attribute should copy entire chain."""
        original = DeeplyNestedConfig(level1=NestedConfig(inner=SimpleConfig(value=42)))

        with CopyOnWrite(original) as cow:
            cow.level1.inner.value = 100

            # Verify modification
            assert cow.unwrap.level1.inner.value == 100

        # Original unchanged
        assert original.level1.inner.value == 42


class TestCopyOnWriteSequences:
    """Test COW with sequences (lists, etc.)."""

    def test_list_read_no_copy(self):
        """Reading list items should not trigger copy."""
        original = NestedConfig(items=[1, 2, 3])
        with CopyOnWrite(original) as cow:
            _ = cow.items[0]
            assert cow._self_is_copy is False

    def test_list_setitem_triggers_copy(self):
        """Setting list item should trigger copy."""
        original = NestedConfig(items=[1, 2, 3])

        with CopyOnWrite(original) as cow:
            items_cow = cow.items
            items_cow[0] = 100

            assert items_cow._self_is_copy is True
            assert cow._self_is_copy is True
            assert cow.unwrap.items[0] == 100

        assert original.items[0] == 1

    def test_list_delitem_triggers_copy(self):
        """Deleting list item should trigger copy."""
        original = NestedConfig(items=[1, 2, 3])

        with CopyOnWrite(original) as cow:
            items_cow = cow.items
            del items_cow[0]

            assert items_cow._self_is_copy is True
            assert cow.unwrap.items == [2, 3]

        assert original.items == [1, 2, 3]


class TestCopyOnWriteMappings:
    """Test COW with mappings (dicts, etc.)."""

    def test_dict_read_no_copy(self):
        """Reading dict items should not trigger copy."""
        original = {"a": 1, "b": 2}
        with CopyOnWrite(original) as cow:
            _ = cow["a"]
            assert cow._self_is_copy is False

    def test_dict_setitem_triggers_copy(self):
        """Setting dict item should trigger copy."""
        original = {"a": 1, "b": 2}

        with CopyOnWrite(original) as cow:
            cow["a"] = 100
            assert cow._self_is_copy is True
            assert cow.unwrap["a"] == 100

        assert original["a"] == 1

    def test_dict_delitem_triggers_copy(self):
        """Deleting dict item should trigger copy."""
        original = {"a": 1, "b": 2}

        with CopyOnWrite(original) as cow:
            del cow["a"]
            assert cow._self_is_copy is True
            assert "a" not in cow.unwrap

        assert "a" in original


class TestCopyOnWriteDelattr:
    """Test COW with attribute deletion."""

    def test_delattr_triggers_copy(self):
        """Deleting an attribute should trigger copy."""

        class Deletable:
            def __init__(self):
                self.x = 1
                self.y = 2

        original = Deletable()

        with CopyOnWrite(original) as cow:
            del cow.x
            assert cow._self_is_copy is True
            assert not hasattr(cow.unwrap, "x")

        assert hasattr(original, "x")
        assert original.x == 1


class TestCopyOnWriteFinalize:
    """Test COW finalize integration."""

    def test_finalize_called_on_exit(self):
        """Finalize should be called on context exit if object has it."""
        finalize_called = list[int]()

        class FinalizeTracker:
            def __init__(self, value: int):
                self.value = value

            def finalize(self) -> Self:
                finalize_called.append(self.value)
                result = copy.copy(self)
                result.value *= 2
                return result

        original = FinalizeTracker(42)

        with CopyOnWrite(original):
            # No modification yet
            pass

        # Finalize should have been called
        assert 42 in finalize_called

    def test_finalize_not_called_twice(self):
        """Finalize should not be called if already finalized via method call."""
        finalize_count = [0]

        class FinalizeCounter:
            def __init__(self):
                self.value = 1

            def finalize(self) -> Self:
                finalize_count[0] += 1
                return copy.copy(self)

        original = FinalizeCounter()

        with CopyOnWrite(original) as cow:
            # Explicitly call finalize
            cow.finalize()

        # Should only be called once
        assert finalize_count[0] == 1

    def test_no_re_finalize_if_already_finalized(self):
        """CopyOnWrite should not re-finalize objects that are already finalized."""
        finalize_count = [0]

        @dataclasses.dataclass  # check-dataclass: ignore[kw_only,slots]
        class Inner:
            value: int = 0
            _finalized: bool = dataclasses.field(default=False, repr=False)

            def finalize(self) -> Self:
                finalize_count[0] += 1
                r = copy.copy(self)
                r._finalized = True
                return r

        @dataclasses.dataclass  # check-dataclass: ignore[kw_only,slots]
        class Outer:
            inner: Inner = dataclasses.field(default_factory=Inner)
            scale: int = 1
            _finalized: bool = dataclasses.field(default=False, repr=False)

            def finalize(self) -> Self:
                r = copy.copy(self)
                r.inner = r.inner.finalize()
                r._finalized = True
                # Use CopyOnWrite inside finalize (the README pattern)
                with CopyOnWrite(r) as cow:
                    cow.inner.value = r.scale * 10
                return cow.unwrap

        original = Outer(scale=5)
        result = original.finalize()

        assert result.inner.value == 50
        assert result._finalized is True
        # Inner.finalize should only be called once (by Outer.finalize),
        # not again by CopyOnWrite.__exit__
        assert finalize_count[0] == 1


class TestCopyOnWriteMethodCalls:
    """Test COW with method calls."""

    def test_method_call_copies_first(self):
        """Method calls should copy before invoking."""

        class Counter:
            def __init__(self):
                self.count = 0

            def increment(self) -> int:
                self.count += 1
                return self.count

        original = Counter()

        with CopyOnWrite(original) as cow:
            result = cow.increment()
            assert result.unwrap == 1
            assert cow.unwrap.count == 1

        assert original.count == 0


class TestCopyOnWriteRepr:
    """Test COW representation."""

    def test_repr_delegates_to_wrapped(self):
        """Repr should delegate to wrapped object."""
        original = SimpleConfig(value=42)
        cow = CopyOnWrite(original)
        assert "SimpleConfig" in repr(cow)
        assert "42" in repr(cow)


class TestCopyOnWriteMultipleParents:
    """Test COW with objects having multiple parents."""

    def test_shared_child_multiple_parents(self):
        """A shared child should update all parent references on copy."""
        shared = SimpleConfig(value=42)

        @dataclasses.dataclass  # check-dataclass: ignore[kw_only,slots]
        class Container:
            child: SimpleConfig = dataclasses.field(default_factory=SimpleConfig)

        container1 = Container(child=shared)
        container2 = Container(child=shared)

        # Create COW for both containers
        with CopyOnWrite(container1) as cow1, CopyOnWrite(container2) as cow2:
            # Get shared child through cow1
            child_cow = cow1.child

            # Register cow2 as another parent
            child_cow._self_parents.add((cow2, "child"))

            # Modify child
            child_cow.value = 100

            # Both parents should be copied
            assert cow1._self_is_copy is True
            assert cow2._self_is_copy is True

            # Both should point to the new child
            assert cow1.unwrap.child.value == 100
            assert cow2.unwrap.child.value == 100

        # Original shared child unchanged
        assert shared.value == 42


class TestCopyOnWriteDebugMode:
    """Test COW debug mode."""

    def test_debug_mode_prints(self, capsys: pytest.CaptureFixture[str]) -> None:
        """Debug mode should print operations."""
        original = SimpleConfig(value=42)

        with CopyOnWrite(original, debug=True) as cow:
            _ = cow.value
            cow.value = 100

        captured = capsys.readouterr()
        assert "get" in captured.out.lower() or "value" in captured.out
        assert "set" in captured.out.lower() or "copy" in captured.out.lower()


class TestCopyOnWriteContextManager:
    """Test COW as context manager."""

    def test_returns_self_on_enter(self):
        """Context manager should return self on enter."""
        original = SimpleConfig()
        cow = CopyOnWrite(original)
        assert cow.__enter__() is cow

    def test_read_only_access_does_not_finalize_children(self):
        """Read-only child access should not finalize (would mutate original)."""
        finalize_called = list[str]()

        class Trackable:
            def __init__(self, name: str):
                self.name = name
                self.child: Trackable | None = None

            def finalize(self) -> Self:
                finalize_called.append(self.name)
                return copy.copy(self)

        parent = Trackable("parent")
        parent.child = Trackable("child")
        original_child = parent.child

        with CopyOnWrite(parent) as cow:
            # Access child but don't mutate
            _ = cow.child

        # Root finalizes (harmless, no parent to mutate), child does not
        assert "parent" in finalize_called
        assert "child" not in finalize_called
        # Original child reference is unchanged
        assert parent.child is original_child

    def test_exits_children_first_on_mutation(self):
        """Children should be finalized before parent when mutations occur."""
        exit_order = list[str]()

        class Trackable:
            def __init__(self, name: str):
                self.name = name
                self.child: Trackable | None = None
                self.value: int = 0

            def finalize(self) -> Self:
                exit_order.append(self.name)
                return copy.copy(self)

        parent = Trackable("parent")
        parent.child = Trackable("child")

        with CopyOnWrite(parent) as cow:
            # Mutate the child — triggers copy of child and parent
            cow.child.value = 99

        # Both should be finalized, child first (depth-first exit)
        assert exit_order.index("child") < exit_order.index("parent")


class TestCopyOnWriteSetAttrCOWValue:
    """Test __setattr__ with CopyOnWrite value."""

    def test_set_cow_value_unwraps(self):
        """Setting a CopyOnWrite value should unwrap and track it."""
        original = NestedConfig(inner=SimpleConfig(value=42))

        with CopyOnWrite(original) as cow:
            inner_cow = cow.inner
            # Set a COW-wrapped value on another attribute
            cow.inner = inner_cow
            # Should have triggered a copy
            assert cow._self_is_copy is True

        assert original.inner.value == 42


class TestCopyOnWriteSetItemCOWValue:
    """Test __setitem__ with CopyOnWrite value."""

    def test_set_cow_value_in_dict(self):
        """Setting a CopyOnWrite value in a dict should unwrap it."""
        original = {"key": SimpleConfig(value=42)}

        with CopyOnWrite(original) as cow:
            inner_cow = cow["key"]
            cow["key"] = inner_cow
            assert cow._self_is_copy is True

        assert original["key"].value == 42


class TestCopyOnWriteDelItem:
    """Test __delitem__ operations."""

    def test_delitem_dict_with_cached_child(self):
        """Deleting an item with a cached child should clean up parent tracking."""
        original = {"a": 1, "b": 2}

        with CopyOnWrite(original) as cow:
            # Access item to cache child
            _ = cow["a"]
            assert "__item_'a'" in cow._self_children
            # Delete the item
            del cow["a"]
            assert cow._self_is_copy is True
            assert "a" not in cow.unwrap


class TestCopyOnWriteHash:
    """Test __hash__ for CopyOnWrite."""

    def test_hash_is_identity_based(self):
        """Hash should be based on object id."""
        cow1 = CopyOnWrite(SimpleConfig(value=1))
        cow2 = CopyOnWrite(SimpleConfig(value=1))
        # Same wrapped value, but different hash
        assert hash(cow1) != hash(cow2)
        assert hash(cow1) == id(cow1)

    def test_cow_usable_in_set(self):
        """CopyOnWrite instances should be usable in sets."""
        cow1 = CopyOnWrite(SimpleConfig(value=1))
        cow2 = CopyOnWrite(SimpleConfig(value=1))
        s = {cow1, cow2}
        assert len(s) == 2


class TestCopyOnWriteDir:
    """Test __dir__ delegation."""

    def test_dir_delegates_to_wrapped(self):
        """dir() should delegate to wrapped object."""
        original = SimpleConfig(value=42)
        cow = CopyOnWrite(original)
        assert "value" in dir(cow)
        assert "name" in dir(cow)


class TestCopyOnWriteCallable:
    """Test __call__ with various scenarios."""

    def test_call_non_callable_raises(self):
        """Calling a non-callable COW should raise TypeError."""
        original = SimpleConfig(value=42)

        with CopyOnWrite(original) as cow:
            # Access 'value' (an int, not callable)
            value_cow = cow.value
            with pytest.raises(TypeError, match="not callable"):
                value_cow()

    def test_finalize_via_call_marks_parent(self):
        """Calling .finalize() should mark parent as finalized."""

        class Finalizable:
            def __init__(self):
                self.value = 1

            def finalize(self) -> Self:
                return copy.copy(self)

        original = Finalizable()

        with CopyOnWrite(original) as cow:
            cow.finalize()
            # Parent should be marked as finalized
            assert cow._self_is_finalized is True


class TestCopyOnWriteDebugExtended:
    """Test debug mode for various operations."""

    def test_debug_getattr(self, capsys: pytest.CaptureFixture[str]) -> None:
        """Debug mode should print on attribute access."""
        original = SimpleConfig(value=42)

        with CopyOnWrite(original, debug=True) as cow:
            _ = cow.value

        captured = capsys.readouterr()
        assert "get" in captured.out.lower()

    def test_debug_copy_skip(self, capsys: pytest.CaptureFixture[str]) -> None:
        """Debug mode should print when copy is skipped."""
        original = SimpleConfig(value=42)

        with CopyOnWrite(original, debug=True) as cow:
            cow.value = 1  # First copy
            cow.name = "x"  # Should skip (already copied)

        captured = capsys.readouterr()
        assert "skip" in captured.out.lower()

    def test_debug_delattr(self, capsys: pytest.CaptureFixture[str]) -> None:
        """Debug mode should print on attribute deletion."""

        class Deletable:
            def __init__(self):
                self.x = 1

        original = Deletable()

        with CopyOnWrite(original, debug=True) as cow:
            del cow.x

        captured = capsys.readouterr()
        assert "del" in captured.out.lower()

    def test_debug_getitem(self, capsys: pytest.CaptureFixture[str]) -> None:
        """Debug mode should print on item access."""
        original = {"a": 1}

        with CopyOnWrite(original, debug=True) as cow:
            _ = cow["a"]

        captured = capsys.readouterr()
        assert "get" in captured.out.lower()

    def test_debug_setitem(self, capsys: pytest.CaptureFixture[str]) -> None:
        """Debug mode should print on item set."""
        original = {"a": 1}

        with CopyOnWrite(original, debug=True) as cow:
            cow["a"] = 99

        captured = capsys.readouterr()
        assert "set" in captured.out.lower()

    def test_debug_delitem(self, capsys: pytest.CaptureFixture[str]) -> None:
        """Debug mode should print on item deletion."""
        original = {"a": 1, "b": 2}

        with CopyOnWrite(original, debug=True) as cow:
            del cow["a"]

        captured = capsys.readouterr()
        assert "del" in captured.out.lower()

    def test_debug_call(self, capsys: pytest.CaptureFixture[str]) -> None:
        """Debug mode should print on call."""

        class CallableObj:
            def method(self) -> int:
                return 42

        original = CallableObj()

        with CopyOnWrite(original, debug=True) as cow:
            cow.method()

        captured = capsys.readouterr()
        assert "call" in captured.out.lower()

    def test_debug_exit(self, capsys: pytest.CaptureFixture[str]) -> None:
        """Debug mode should print on exit."""
        original = SimpleConfig(value=42)

        with CopyOnWrite(original, debug=True):
            pass

        captured = capsys.readouterr()
        assert "exit" in captured.out.lower()


class TestCopyOnWriteSelfPrefixGuards:
    """Test _self_ prefix guards in __getattr__ and __delattr__."""

    def test_getattr_self_prefix_raises(self):
        """Accessing _self_ prefixed attrs via __getattr__ should raise."""
        cow = CopyOnWrite(SimpleConfig(value=42))
        # Directly call __getattr__ to bypass wrapt's attribute handling
        with pytest.raises(AttributeError, match="_self_bogus"):
            cow.__getattr__("_self_bogus")

    def test_delattr_self_prefix_delegates(self):
        """Deleting _self_ prefixed attrs should delegate to super."""
        cow = CopyOnWrite(SimpleConfig(value=42))
        with pytest.raises(AttributeError):
            del cow._self_bogus


class TestCopyOnWriteDelAttrWithChild:
    """Test __delattr__ with tracked children."""

    def test_delattr_removes_child_tracking(self):
        """Deleting an attribute should remove it from children cache."""
        original = NestedConfig(inner=SimpleConfig(value=42))

        with CopyOnWrite(original) as cow:
            # Access to create child
            inner_cow = cow.inner
            assert "inner" in cow._self_children
            # Delete attribute
            del cow.inner
            assert "inner" not in cow._self_children
            # Child's parent tracking should be updated
            assert (cow, "inner") not in inner_cow._self_parents


def test_delattr_self_prefix():
    """Test __delattr__ with _self_ prefix delegates to super."""

    @dataclasses.dataclass  # check-dataclass: ignore[kw_only,slots]
    class Inner:
        x: int = 1

    cow = CopyOnWrite(Inner())
    # Add a _self_ attribute dynamically, then delete it
    cow._self_custom = "test"
    assert cow._self_custom == "test"
    del cow._self_custom
    assert not hasattr(cow, "_self_custom")


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
