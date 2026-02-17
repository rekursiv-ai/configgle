"""Copy-on-write proxy for safe, efficient counterfactual mutation.

This module provides a CopyOnWrite wrapper that allows mutations to an object
tree while preserving the original. Copies are made lazily only when mutations
actually occur, and propagate up to parent objects automatically.

Example:
    ```python
    original = MyConfig()

    with CopyOnWrite(original) as cow:
        cow.nested.field = 42        # Only copies 'nested' and root
        cow.items.append(1)          # Copies 'items' list
        result = cow.unwrap          # Get the modified copy

    # original is unchanged
    # result has the modifications
    ```

"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Generic, Self, TypeVar, cast
from typing_extensions import override

import copy

import wrapt


__all__ = ["CopyOnWrite"]

_T = TypeVar("_T")


# We use Any for parent/children because:
# 1. They hold CopyOnWrite wrappers of heterogeneous types (Config, int, list, etc.)
# 2. We need to call methods like ._copy() on them (object doesn't have these)
# 3. Due to invariance, CopyOnWrite[int] is not assignable to CopyOnWrite[object]
if TYPE_CHECKING:
    _ParentSet = set[tuple[Any, str]]
    _ChildrenDict = dict[str, Any]


# Note: wrapt.ObjectProxy is generic in type stubs but not subscriptable at runtime.
# We use Generic[_T] to provide type parameters and declare __wrapped__: _T.
class CopyOnWrite(wrapt.ObjectProxy, Generic[_T]):  # pyright: ignore[reportMissingTypeArgument]
    """A proxy that copies objects lazily on first mutation.

    Wraps an object and intercepts all attribute/item mutations. When a mutation
    occurs, the object (and all its parents) are shallow-copied first, ensuring
    the original object tree remains unchanged.

    Attributes are accessed through child CopyOnWrite wrappers, enabling
    copy-on-write semantics for deeply nested mutations like `obj.a.b.c = 1`.

    """

    # Declare __wrapped__ with proper type to help pyright
    __wrapped__: _T

    # Use _self_ prefix to avoid conflicts with wrapped object attributes
    # (this is the wrapt convention)
    # Note: These use Any because parent/children can wrap any type
    _self_parents: _ParentSet
    _self_children: _ChildrenDict
    _self_is_copy: bool
    _self_is_finalized: bool
    _self_debug: bool

    def __init__(
        self,
        wrapped: _T,
        parent: CopyOnWrite[Any] | None = None,
        key: str = "",
        debug: bool = False,
    ) -> None:
        """Initialize a copy-on-write proxy.

        Args:
          wrapped: The object to wrap.
          parent: Parent CopyOnWrite proxy (for internal use in nested access).
          key: Attribute/item key used to access this from parent.
          debug: Enable debug printing of COW operations.

        """
        # wrapt.ObjectProxy.__init__ type is partially unknown in stubs
        super().__init__(wrapped)  # pyright: ignore[reportUnknownMemberType]
        if parent is None:
            self._self_parents = set()
        else:
            self._self_parents = {(parent, key)}
        self._self_children = {}
        self._self_is_copy = False
        self._self_is_finalized = False
        self._self_debug = debug

    # -------------------------------------------------------------------------
    # Context manager
    # -------------------------------------------------------------------------

    def __enter__(self) -> Self:
        """Enter context manager."""
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_value: BaseException | None,
        exc_traceback: object,
    ) -> None:
        """Exit context manager, calling finalize on wrapped objects."""
        # Exit children first (depth-first)
        for child in self._self_children.values():
            child.__exit__(exc_type, exc_value, exc_traceback)

        # Call finalize if present and not already finalized
        finalize_fn = getattr(self.__wrapped__, "finalize", None)
        if not self._self_is_finalized and callable(finalize_fn):
            finalized = finalize_fn()
            # Update parent references to point to finalized value
            for parent, key in self._self_parents:
                setattr(parent.__wrapped__, key, finalized)

        if self._self_debug:
            print(
                f"  exit: {type(self.__wrapped__).__name__} "
                f"is_copy={self._self_is_copy} "
                f"is_finalized={self._self_is_finalized}",
            )

    # -------------------------------------------------------------------------
    # Copy-on-write core
    # -------------------------------------------------------------------------

    def _copy(self) -> Self:
        """Lazily copy this object and propagate copies to parents."""
        if self._self_is_copy:
            if self._self_debug:
                print(
                    f"  copy: {type(self.__wrapped__).__name__} [SKIP - already copied]",
                )
            return self

        if self._self_debug:
            print(f"  copy: {type(self.__wrapped__).__name__}")

        # Copy parents first (propagate up)
        for parent, _ in self._self_parents:
            parent._copy()  # noqa: SLF001

        # Now copy this object
        self.__wrapped__ = copy.copy(self.__wrapped__)
        self._self_is_copy = True

        # Update parent references to point to our new copy
        for parent, key in self._self_parents:
            setattr(parent.__wrapped__, key, self.__wrapped__)

        return self

    # -------------------------------------------------------------------------
    # Attribute access - return wrapped children for nested COW
    # -------------------------------------------------------------------------

    def __getattr__(self, key: str) -> CopyOnWrite[Any]:
        if key.startswith("_self_"):
            # This shouldn't happen with wrapt, but just in case
            raise AttributeError(key)

        if self._self_debug:
            print(f"  get : {type(self.__wrapped__).__name__}.{key}")

        # Return cached child wrapper or create new one
        child: CopyOnWrite[Any] | None = self._self_children.get(key)
        if child is None:
            actual = getattr(self.__wrapped__, key)
            child = CopyOnWrite(actual, parent=self, key=key, debug=self._self_debug)
            self._self_children[key] = child
        return child

    @override
    def __setattr__(self, key: str, value: object) -> None:
        # Let wrapt handle its internal attributes
        if key.startswith("_self_") or key == "__wrapped__":
            super().__setattr__(key, value)
            return

        if self._self_debug:
            print(f"  set : {type(self.__wrapped__).__name__}.{key} = {value!r}")

        # Unwrap CopyOnWrite values
        actual_value: object
        if isinstance(value, CopyOnWrite):
            cow_value = cast(CopyOnWrite[Any], value)
            actual_value = cow_value.__wrapped__
            # Track parent relationship
            self._self_children[key] = cow_value
            cow_value._self_parents.add((self, key))  # noqa: SLF001
        else:
            actual_value = value
            # Remove from children cache since it's no longer wrapped
            self._self_children.pop(key, None)

        # Copy-on-write: copy before mutating
        self._copy()
        setattr(self.__wrapped__, key, actual_value)

    @override
    def __delattr__(self, key: str) -> None:
        if key.startswith("_self_"):
            super().__delattr__(key)
            return

        if self._self_debug:
            print(f"  del : {type(self.__wrapped__).__name__}.{key}")

        # Remove from children and update parent tracking
        child: CopyOnWrite[Any] | None = self._self_children.pop(key, None)
        if child is not None:
            child._self_parents.discard((self, key))  # noqa: SLF001

        # Copy-on-write: copy before mutating
        self._copy()
        delattr(self.__wrapped__, key)

    # -------------------------------------------------------------------------
    # Item access (for sequences, mappings)
    # -------------------------------------------------------------------------

    def __getitem__(self, key: object) -> CopyOnWrite[Any]:
        if self._self_debug:
            print(f"  get : {type(self.__wrapped__).__name__}[{key!r}]")

        # Use string representation of key for children cache
        cache_key = f"__item_{key!r}"
        child: CopyOnWrite[Any] | None = self._self_children.get(cache_key)
        if child is None:
            actual = cast(object, self.__wrapped__[key])  # pyright: ignore[reportIndexIssue]  # ty: ignore[not-subscriptable]
            child = CopyOnWrite(
                actual,
                parent=self,
                key=cache_key,
                debug=self._self_debug,
            )
            self._self_children[cache_key] = child
        return child

    def __setitem__(self, key: object, value: object) -> None:
        if self._self_debug:
            print(f"  set : {type(self.__wrapped__).__name__}[{key!r}] = {value!r}")

        # Unwrap CopyOnWrite values
        actual_value: object
        if isinstance(value, CopyOnWrite):
            actual_value = cast(CopyOnWrite[Any], value).__wrapped__
        else:
            actual_value = value

        # Invalidate cached child for this key
        cache_key = f"__item_{key!r}"
        self._self_children.pop(cache_key, None)

        # Copy-on-write: copy before mutating
        self._copy()
        self.__wrapped__[key] = actual_value  # pyright: ignore[reportIndexIssue]  # ty: ignore[invalid-assignment]

    def __delitem__(self, key: object) -> None:
        if self._self_debug:
            print(f"  del : {type(self.__wrapped__).__name__}[{key!r}]")

        # Remove from children cache
        cache_key = f"__item_{key!r}"
        child: CopyOnWrite[Any] | None = self._self_children.pop(cache_key, None)
        if child is not None:
            child._self_parents.discard((self, cache_key))  # noqa: SLF001

        # Copy-on-write: copy before mutating
        self._copy()
        del self.__wrapped__[key]  # pyright: ignore[reportIndexIssue]  # ty: ignore[not-subscriptable]

    # -------------------------------------------------------------------------
    # Method calls - copy before calling mutating methods
    # -------------------------------------------------------------------------

    def __call__(self, *args: object, **kwargs: object) -> CopyOnWrite[Any]:
        """Invoke wrapped callable, copying parent first if needed."""
        if self._self_debug:
            print(f"  call: {self.__wrapped__}")

        # Mark parent as finalized if this is a finalize() call
        for parent, key in self._self_parents:
            if key == "finalize":
                parent._self_is_finalized = True  # noqa: SLF001

        # Copy parents first (in case method mutates the parent object)
        for parent, _ in self._self_parents:
            parent._copy()  # noqa: SLF001

        # For bound methods, re-fetch from the copied parent to get the method
        # bound to the copy, not the original
        method: Any = self.__wrapped__
        for parent, key in self._self_parents:
            # Get the updated method from the copied parent
            method = getattr(parent.__wrapped__, key)
            break  # Only need the first parent for bound methods

        if not callable(method):
            raise TypeError(f"{method!r} is not callable")
        result = method(*args, **kwargs)

        # Wrap the result
        return CopyOnWrite(result, debug=self._self_debug)

    # -------------------------------------------------------------------------
    # Representation
    # -------------------------------------------------------------------------

    @property
    def unwrap(self) -> _T:
        """Return the underlying object.

        Returns:
          wrapped: The underlying object, possibly a copy if mutations occurred.

        """
        return self.__wrapped__

    @override
    def __repr__(self) -> str:
        return repr(self.__wrapped__)

    @override
    def __dir__(self) -> list[str]:
        return dir(self.__wrapped__)

    @override
    def __hash__(self) -> int:
        # Use identity-based hash so CopyOnWrite instances can be stored in sets
        # regardless of whether the wrapped object is hashable
        return id(self)
