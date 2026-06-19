# ruff: noqa: T201
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

from types import TracebackType
from typing import TYPE_CHECKING, Any, Self, cast, override

import copy

import wrapt


__all__ = ["CopyOnWrite"]

# Immutable leaf types returned raw (not proxied) by ``__getattr__``: they cannot
# be copy-on-write-mutated and must not leak a wrapper into a config built from a
# read (``Child.Config(c=self.x)``).
_LEAF_TYPES = (int, float, str, bytes, bool, type(None))

# We use Any for parent/children because:
# 1. They hold CopyOnWrite wrappers of heterogeneous types (Config, int, list, etc.)
# 2. We need to call methods like ._copy() on them (object doesn't have these)
# 3. Due to invariance, CopyOnWrite[int] is not assignable to CopyOnWrite[object]
if TYPE_CHECKING:
    _ParentSet = set[tuple[Any, str]]
    _ChildrenDict = dict[str, Any]


class CopyOnWrite[T](wrapt.ObjectProxy[T]):
    """A proxy that copies objects lazily on first mutation.

    Wraps an object and intercepts all attribute/item mutations. When a mutation
    occurs, the object (and all its parents) are shallow-copied first, ensuring
    the original object tree remains unchanged.

    Attributes are accessed through child CopyOnWrite wrappers, enabling
    copy-on-write semantics for deeply nested mutations like `obj.a.b.c = 1`.

    """

    # Declare __wrapped__ with proper type to help pyright
    __wrapped__: T

    # Use _self_ prefix to avoid conflicts with wrapped object attributes
    # (this is the wrapt convention)
    # Note: These use Any because parent/children can wrap any type
    _self_parents: _ParentSet
    _self_children: _ChildrenDict
    _self_is_copy: bool
    _self_is_finalized: bool
    _self_debug: bool

    def __new__(
        cls,
        wrapped: T,
        parent: CopyOnWrite[Any] | None = None,
        key: str = "",
        debug: bool = False,
    ) -> Self:
        # Anchors the constructor return type to Self. Without this, both
        # basedpyright and ty infer `CopyOnWrite(...)` as `ObjectProxy[Unknown]`
        # because wrapt-stubs' `ObjectProxy.__new__` hardcodes a non-Self return.
        #
        # We tried fixing this upstream by vendoring wrapt-stubs into
        # `loop/lib/typings/wrapt/` with `__new__(cls, ...) -> Self`. In isolated
        # probes the stub fix worked (subclasses of `ObjectProxy[T]` were typed
        # correctly), but inside this file basedpyright re-resolves T based on
        # downstream `CopyOnWrite(...)` call sites (e.g. `__getitem__` passes
        # `cast(object, ...)`), which collapsed the return back to
        # `ObjectProxy[Unknown]`. A local `__new__` override is the only place
        # that keeps inference stable across every call site here.
        del wrapped, parent, key, debug
        return cast("Self", super().__new__(cls))

    def __init__(
        self,
        wrapped: T,
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
        super().__init__(wrapped)
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

    if TYPE_CHECKING:

        @override
        def __enter__(self) -> T:
            """Bind the wrapped type ``T`` to the ``as`` target (type-level only).

            At runtime ``__enter__`` returns ``self`` -- the proxy -- so writes
            inside the block are intercepted (copy-on-write). To the type checker
            it returns ``T`` (the wrapped object), so ``with CopyOnWrite(cfg) as
            cfg:`` types ``cfg`` as the real object with its real field types (not
            ``CopyOnWrite[Any]``) -- field reads/writes need no casts. The proxy
            forwards every operation to the wrapped object, so the fiction is
            sound; ``__copy__`` unwraps when something copies the proxy.
            """
            ...

    else:

        @override
        def __enter__(self):
            return self

    @override
    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_value: BaseException | None,
        traceback: TracebackType | None,
    ) -> bool | None:
        """Exit context manager, finalizing copied children (never the root)."""
        # Propagate a body exception untouched: skip all finalization so the
        # original error surfaces, rather than being masked by a child finalize
        # that then raises on a half-built object.
        if exc_type is not None:
            return None

        # Exit children first (depth-first)
        for child in self._self_children.values():
            child.__exit__(exc_type, exc_value, traceback)

        # Finalize copied *children* only, re-pointing each parent at the result.
        #   - ``_self_parents``: the root (the object passed to
        #     ``CopyOnWrite(self)``) is never finalized here. Finalizing it would
        #     re-enter the very ``finalize`` that opened the proxy and recurse.
        #     The caller owns the root's finalize.
        #   - ``_self_is_copy``: a read-only child stays the shared original
        #     (finalizing it would mutate the original).
        finalize_fn = getattr(self.__wrapped__, "finalize", None)
        if (
            self._self_parents
            and self._self_is_copy
            and not self._self_is_finalized
            and not getattr(self.__wrapped__, "_finalized", False)
            and callable(finalize_fn)
        ):
            finalized = finalize_fn()
            # Update parent references to point to finalized value
            for parent, key in self._self_parents:
                setattr(parent.__wrapped__, key, finalized)

        if self._self_debug:
            print(
                f"  exit: {type(self.__wrapped__).__name__} "
                f"is_copy={self._self_is_copy} "
                f"is_finalized={self._self_is_finalized}"
            )

    # -------------------------------------------------------------------------
    # Attribute access - return wrapped children for nested COW
    # -------------------------------------------------------------------------

    @override
    def __getattr__(self, name: str) -> CopyOnWrite[Any]:
        if name.startswith("_self_"):
            # This shouldn't happen with wrapt, but just in case
            raise AttributeError(name)

        if self._self_debug:
            print(f"  get : {type(self.__wrapped__).__name__}.{name}")

        actual = getattr(self.__wrapped__, name)
        # Return immutable leaf values raw -- not wrapped. A leaf cannot be
        # copy-on-write-mutated (``cow.x.y = 1`` is meaningless for an int), and
        # returning a proxy here leaks it: ``Child.Config(c=self.x)`` would store
        # the wrapper, not the value. Everything else (Figs, containers, unknown
        # objects that might contain a Fig) stays proxied so deep mutation stays
        # isolated.
        if isinstance(actual, _LEAF_TYPES):
            return actual  # pyright: ignore[reportReturnType] -- leaves return raw, not a proxy

        # Return cached child wrapper or create new one
        child: CopyOnWrite[Any] | None = self._self_children.get(name)
        if child is None:
            child = CopyOnWrite(actual, parent=self, key=name, debug=self._self_debug)
            self._self_children[name] = child
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

    @override
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

    @override
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

    @override
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
        method: object = self.__wrapped__
        for parent, key in self._self_parents:
            # Get the updated method from the copied parent
            method = getattr(parent.__wrapped__, key)
            break  # Only need the first parent for bound methods

        if not callable(method):
            raise TypeError(f"{method!r} is not callable")
        # The proxy resolves methods dynamically, so ty cannot infer the
        # callable signature after the runtime callable() check.
        result = method(*args, **kwargs)  # ty: ignore[call-top-callable]

        # Wrap the result
        return CopyOnWrite(result, debug=self._self_debug)

    # -------------------------------------------------------------------------
    # Representation
    # -------------------------------------------------------------------------

    @property
    def unwrap(self) -> T:
        """Return the underlying object.

        Returns:
          wrapped: The underlying object, possibly a copy if mutations occurred.

        """
        return self.__wrapped__

    @override
    def __copy__(self) -> T:
        """Shallow-copy the *wrapped* object, returning it unwrapped.

        ``Maker.finalize`` opens with ``copy.copy(self)``; wrapt's proxy rejects
        a bare ``copy.copy`` (``object proxy must define __copy__``). Returning
        the unwrapped copy keeps the finalize chain on a real config (never the
        proxy) -- so ``with CopyOnWrite(self) as self: ...; return
        super().finalize()`` needs no manual unwrap. The proxy is scaffolding for
        deciding when to copy; once a copy is made it is shed.
        """
        return copy.copy(self.__wrapped__)

    @override
    def __deepcopy__(self, memo: dict[int, object]) -> T:
        """Deep-copy the *wrapped* object, returning it unwrapped.

        Same rationale as ``__copy__``: a proxy read of a nested config may be
        deep-copied by the finalize cascade; wrapt rejects ``copy.deepcopy`` on
        the proxy. Deep-copy the wrapped value instead.
        """
        return copy.deepcopy(self.__wrapped__, memo)

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
