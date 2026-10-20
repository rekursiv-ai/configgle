"""Inline configuration classes for wrapping functions and partial functions."""

from __future__ import annotations

from collections.abc import Callable, MutableMapping, MutableSequence
from typing import Any, ClassVar, Self, cast, override

import dataclasses
import functools
import reprlib

from configgle.fig import Fig
from configgle.walk import finalize_value, make_value


__all__ = ["InlineConfig", "PartialConfig"]

_INLINE_CONFIG_SLOTS = frozenset(("func", "_finalized", "_args", "_kwargs"))


class InlineConfig[T](Fig, init=False, eq=False, repr=False, require_defaults=False):
    """Config wrapper for arbitrary callables with deferred execution.

    Stores a function and its arguments, calling them when make() is invoked.
    Supports nested configs in args/kwargs which are finalized/made recursively.

    """

    func: Callable[..., T]
    """The callable ``make`` invokes."""

    _args: MutableSequence[object] = dataclasses.field(default_factory=list[object])
    """Its positional arguments."""

    _kwargs: MutableMapping[str, object] = dataclasses.field(
        default_factory=dict[str, object],
    )
    """Its keyword arguments."""

    def __init__(
        self,
        /,
        func: Callable[..., T],
        *args: object,
        **kwargs: object,
    ) -> None:
        self.func = func
        self._args = list(args)
        self._kwargs = kwargs

    @override
    def make(self) -> T:
        """Finalize and invoke the wrapped function.

        Builds from ``finalized()`` so the original config is left untouched and
        an already-finalized tree is not finalized again, matching ``Maker.make``.

        Returns:
          result: Result of calling func(*args, **kwargs).

        """
        r = self.finalized()
        # One walk over both, so a value an argument and a keyword share stays one.
        args, kwargs = make_value(
            (r._args, r._kwargs),  # noqa: SLF001 -- The implementation must inspect its private config state.
            making={id(r)},
        )
        return r.func(*args, **kwargs)

    def deferred_call(self) -> tuple[object, list[object], dict[str, object]]:
        """Return the call this config stands for, as its author wrote it.

        Returns:
          func: The callable the config names.
          args: Its positional arguments.
          kwargs: Its keyword arguments.

        """
        return self.func, list(self._args), dict(self._kwargs)

    @override
    def finalize(self) -> Self:
        """Finalize nested configs in place and mark this config finalized.

        Mutates ``self`` (the copy ``make`` already isolated). Child finalize
        may itself return a fresh object (a non-in-place implementation), so its
        return is captured back into ``_args`` / ``_kwargs``.

        Returns:
          finalized: ``self``, with _finalized=True and nested configs finalized.

        """
        self._finalized = True
        self._args, self._kwargs = finalize_value((self._args, self._kwargs))
        return self

    @override
    def update(
        self,
        source: object = None,
        /,
        *,
        skip_missing: bool = False,
        **kwargs: object,
    ) -> Self:
        """Update config kwargs from source and/or kwargs.

        Args:
          source: Optional source object to copy attributes from.
          skip_missing: Skip source and keyword keys absent from ``_kwargs``.
          **kwargs: Additional attribute overrides.

        Returns:
          self: Updated instance for method chaining.

        """
        valid_keys = set(self._kwargs) if skip_missing else None
        if source is not None:
            if dataclasses.is_dataclass(source):
                for field in dataclasses.fields(source):
                    if valid_keys is not None and field.name not in valid_keys:
                        continue
                    self._kwargs[field.name] = getattr(source, field.name)
            else:
                # Copy data attributes from non-dataclass source,
                # skipping private attrs and callables (methods, classmethods, etc.)
                for key in dir(source):
                    if key.startswith("_"):
                        continue
                    if valid_keys is not None and key not in valid_keys:
                        continue
                    try:
                        val = cast(object, getattr(source, key))
                    except (AttributeError, TypeError):
                        continue
                    if not callable(val):
                        self._kwargs[key] = val

        for key, value in kwargs.items():
            if valid_keys is not None and key not in valid_keys:
                continue
            self._kwargs[key] = value

        return self

    @override
    def __eq__(self, other: object) -> bool:
        if type(other) is not type(self):
            return NotImplemented
        assert isinstance(other, InlineConfig)
        return (
            self.func == other.func
            and list(self._args) == list(other._args)
            and dict(self._kwargs) == dict(other._kwargs)
        )

    __hash__: ClassVar[None] = None  # pyright: ignore[reportIncompatibleMethodOverride] -- Mutable config: equal-by-value objects must not be hashable.

    @override
    def __delattr__(self, key: str) -> None:
        try:
            del self._kwargs[key]
            return
        except KeyError:
            pass
        object.__delattr__(self, key)

    def __getattr__(self, key: str) -> Any:  # noqa: ANN401 -- the dynamic-attribute hook; its result is unknowable by construction.  # pyright: ignore[reportAny,reportExplicitAny] -- The dynamic-attribute hook; its result is unknowable by construction.
        """Get a dynamic attribute from kwargs or instance."""
        try:
            return object.__getattribute__(self, "_kwargs")[key]  # pyright: ignore[reportAny] -- The dynamic-attribute hook returns whatever the wrapped callable accepts.
        except (TypeError, AttributeError, KeyError):
            raise AttributeError(key) from None

    @override
    def __setattr__(self, key: str, value: object) -> None:
        if key in _INLINE_CONFIG_SLOTS:
            object.__setattr__(self, key, value)
            return
        self._kwargs[key] = value

    @override
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


class PartialConfig[T](InlineConfig[Callable[..., T]]):
    """InlineConfig that returns a functools.partial instead of calling the function."""

    def __init__(
        self,
        /,
        func: Callable[..., T],
        *args: object,
        **kwargs: object,
    ) -> None:
        super().__init__(functools.partial, func, *args, **kwargs)

    @override
    def deferred_call(self) -> tuple[object, list[object], dict[str, object]]:
        _, args, kwargs = super().deferred_call()
        return args[0], args[1:], kwargs
