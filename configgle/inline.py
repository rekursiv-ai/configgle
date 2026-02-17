"""Inline configuration classes for wrapping functions and partial functions."""

from __future__ import annotations

from collections.abc import Callable, MutableMapping, MutableSequence
from typing import TYPE_CHECKING, Generic, Protocol, Self, TypeVar, runtime_checkable
from typing_extensions import override

import copy
import dataclasses
import functools
import reprlib


if TYPE_CHECKING:
    from configgle.custom_types import DataclassLike, Makeable


@runtime_checkable
class _HasMake(Protocol):
    def make(self) -> object: ...


@runtime_checkable
class _HasFinalize(Protocol):
    def finalize(self) -> Self: ...


__all__ = ["InlineConfig", "PartialConfig"]


_T = TypeVar("_T")


@dataclasses.dataclass(slots=True, init=False, repr=True, weakref_slot=True)
class InlineConfig(Generic[_T]):
    """Config wrapper for arbitrary callables with deferred execution.

    Stores a function and its arguments, calling them when make() is invoked.
    Supports nested configs in args/kwargs which are finalized/made recursively.

    """

    parent_class: None = dataclasses.field(default=None, init=False, repr=False)
    func: Callable[..., _T]
    _finalized: bool = dataclasses.field(
        default=False,
        init=False,
        repr=False,
    )
    _args: MutableSequence[object] = dataclasses.field(
        default_factory=list[object],
        init=False,
        repr=False,
    )
    # This must be last or else you need to manually define __deepcopy__ and __copy__.
    _kwargs: MutableMapping[str, object] = dataclasses.field(
        default_factory=dict[str, object],
        init=False,
        repr=False,
    )

    def __init__(
        self,
        /,
        func: Callable[..., _T],
        *args: object,
        **kwargs: object,
    ) -> None:
        self.func = func
        self._finalized = False
        self._args = list(args)
        self._kwargs = kwargs  # Must be last.

    def make(self) -> _T:
        """Finalize and invoke the wrapped function.

        Returns:
          result: Result of calling func(*args, **kwargs).

        """
        r = self.finalize()
        # Dynamic dispatch to make() on args that may have it
        args = [v.make() if isinstance(v, _HasMake) else v for v in r._args]  # noqa: SLF001
        kwargs = {
            k: v.make() if isinstance(v, _HasMake) else v
            for k, v in r._kwargs.items()  # noqa: SLF001
        }
        return r.func(*args, **kwargs)

    def finalize(self) -> Self:
        """Create a finalized copy with nested configs finalized.

        Returns:
          finalized: Copy with _finalized=True and nested configs finalized.

        """
        r = copy.copy(self)
        # Dynamic dispatch to finalize() on args that may have it
        r._args = [v.finalize() if isinstance(v, _HasFinalize) else v for v in r._args]  # noqa: SLF001
        r._kwargs = {  # noqa: SLF001
            k: v.finalize() if isinstance(v, _HasFinalize) else v
            for k, v in r._kwargs.items()  # noqa: SLF001
        }
        r._finalized = True  # noqa: SLF001
        return r

    def update(
        self,
        source: DataclassLike | Makeable[object] | None = None,
        *,
        skip_missing: bool = False,
        **kwargs: object,
    ) -> Self:
        """Update config kwargs from source and/or kwargs.

        Args:
          source: Optional source object to copy attributes from.
          skip_missing: If True, skip kwargs keys that don't exist in _kwargs.
          **kwargs: Additional attribute overrides.

        Returns:
          self: Updated instance for method chaining.

        """
        del skip_missing  # InlineConfig doesn't have fixed attributes
        if source is not None:
            if dataclasses.is_dataclass(source):
                for field in dataclasses.fields(source):
                    self._kwargs[field.name] = getattr(source, field.name)
            else:
                # Try to copy attributes from non-dataclass source
                for key in dir(source):
                    if not key.startswith("_"):
                        try:
                            self._kwargs[key] = getattr(source, key)
                        except (AttributeError, TypeError):
                            continue

        for key, value in kwargs.items():
            self._kwargs[key] = value

        return self

    @override
    def __delattr__(self, key: str) -> None:
        try:
            del self._kwargs[key]
            return
        except KeyError:
            pass
        object.__delattr__(self, key)

    def __getattr__(self, key: str) -> object:
        try:
            return object.__getattribute__(self, "_kwargs")[key]
        except (TypeError, AttributeError, KeyError):
            pass
        return object.__getattribute__(self, key)

    @override
    def __setattr__(self, key: str, value: object) -> None:
        try:
            _ = object.__getattribute__(self, key)
            hasattr_ = True
        except (TypeError, AttributeError):
            hasattr_ = False
        if hasattr_:
            # This is the reason kwargs must come last--we're relying on the
            # non passthroughs already being set.
            object.__setattr__(self, key, value)
            return
        try:
            self._kwargs[key] = value
            return
        except AttributeError:
            pass
        object.__setattr__(self, key, value)

    @reprlib.recursive_repr()
    def __repr__(self) -> str:
        return (
            f"{type(self).__qualname__}("
            + ", ".join(
                [repr(self.func)]
                + [repr(v) for v in self._args]
                + [f"{k}={v!r}" for k, v in self._kwargs.items()],
            )
            + ")"
        )


class PartialConfig(InlineConfig[Callable[..., _T]]):
    """InlineConfig that returns a functools.partial instead of calling the function."""

    def __init__(
        self,
        /,
        func: Callable[..., _T],
        *args: object,
        **kwargs: object,
    ) -> None:
        super().__init__(functools.partial, func, *args, **kwargs)
