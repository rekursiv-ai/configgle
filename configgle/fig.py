"""Dataclass metaclasses and Maker base for the nested Config pattern."""

from __future__ import annotations

from collections.abc import (
    Iterator,
    Mapping,
    Sequence,
    Set as AbstractSet,
)
from types import CellType, MethodType
from typing import (
    TYPE_CHECKING,
    Any,
    ClassVar,
    Generic,
    Protocol,
    Self,
    cast,
    dataclass_transform,
    runtime_checkable,
)
from typing_extensions import TypeVar, override

import copy
import dataclasses


if TYPE_CHECKING:
    # ty_extensions is a phantom module built into the ty type checker.
    # typings/ty_extensions/ provides a local stub so basedpyright can resolve it too.
    from ty_extensions import Intersection


from configgle.custom_types import DataclassLike, Makeable
from configgle.pprinting import pformat


@runtime_checkable
class _HasFinalize(Protocol):
    """Non-generic protocol for isinstance checks in _finalize_value.

    Using the generic Makeable protocol in isinstance causes basedpyright to
    leak Unknown into the negative branch. This simple non-generic protocol
    avoids that issue.
    """

    def finalize(self) -> Self: ...


__all__ = [
    "Dataclass",
    "Fig",
    "Maker",
    "Makes",
]


_T = TypeVar("_T")
_ParentT = TypeVar("_ParentT", default=Any)


class _IPythonPrinter(Protocol):
    """Protocol for IPython's RepresentationPrinter."""

    def text(self, text: str) -> None: ...


class _MakerParentClassDescriptor:
    """Descriptor that narrows parent_class return type via Generic inference."""

    def __get__(
        self,
        obj: Makeable[_ParentT] | None,
        owner: type[Makeable[_ParentT]],
    ) -> type[_ParentT]:
        return owner._parent_class()  # noqa: SLF001  # pyright: ignore[reportAttributeAccessIssue,reportUnknownMemberType,reportUnknownVariableType]


class MakerMeta(type):
    """Metaclass for the nested Config pattern.

    This metaclass (when combined with _MakerParentClassDescriptor) uses
    descriptor protocol to automatically bind the Config's parent class (when
    Config is defined as a nested class). The binding uses MethodType dynamic
    binding rather than an attribute to prevent infinite recursion sometimes
    seen in broken pickling implementations.

    __set_name__ captures the parent class when a Maker is defined as a nested
    class attribute (e.g., `class Config(Fig):` inside `MyClass`).

    __get__ narrows the type so that `MyClass.Config` is seen as both the
    Config type and a `Makeable[MyClass]`, giving `make()` the correct
    return type.

    """

    def _parent_class(cls) -> type | None: ...

    def __set_name__(cls, owner: type[_ParentT], name: str) -> None:
        """Bind the parent class reference when this class is a nested attribute.

        Uses MethodType to make parent_class immutable while remaining
        compatible with cloudpickle.

        See: https://docs.python.org/3/library/types.html#types.MethodType

        """

        def _parent_class(cls: MakerMeta) -> type[_ParentT]:
            del cls
            return owner

        cls._parent_class = MethodType(_parent_class, cls)  # ty: ignore[invalid-assignment]
        if owner_name := getattr(owner, "__name__", ""):
            cls.__name__ = f"{owner_name}.{name}"

    def __get__(
        cls: type[_T],
        obj: object,
        owner: type[_ParentT],
    ) -> Intersection[
        type[_T],
        type[Makeable[_ParentT]],
    ]:
        return cls


class Maker(Generic[_ParentT], metaclass=MakerMeta):
    """Base class providing make/finalize/update capabilities for configs.

    When nested inside a parent class, enables the pattern:
        instance = ParentClass.Config(...).make()

    """

    __slots__: ClassVar[tuple[str, ...]] = ("_finalized",)
    make_with_kwargs: ClassVar[bool] = False
    if TYPE_CHECKING:

        @property
        def parent_class(self) -> type[_ParentT]: ...

    else:
        parent_class = _MakerParentClassDescriptor()

    def __init__(self) -> None:
        self._finalized = False

    def make(self) -> _ParentT:
        """Finalize config and instantiate the parent class.

        Returns:
          instance: Instance of the parent class.

        Raises:
          ValueError: If not nested in a parent class.

        """
        config = self.finalize()
        cls = config.parent_class
        if cls is None:  # pyright: ignore[reportUnnecessaryComparison]
            raise ValueError("Maker must be nested in a parent class")
        if getattr(type(config), "make_with_kwargs", False):
            kwargs = {
                f.name: getattr(config, f.name)
                for f in dataclasses.fields(
                    cast(DataclassLike, cast(object, config)),
                )
            }
            return cls(**kwargs)
        return cls(config)  # pyright: ignore[reportCallIssue]

    def finalize(self) -> Self:
        """Create a finalized copy with derived defaults applied.

        Override this method to compute derived field values before instantiation.

        Returns:
          finalized: A shallow copy with _finalized=True.

        """
        r = copy.copy(self)

        for name in _get_object_attribute_names(r):
            try:
                value = getattr(r, name)
            except AttributeError:
                continue
            finalized_value = _finalize_value(value)
            if finalized_value is not value:
                # Use object.__setattr__ to bypass frozen dataclass restrictions
                object.__setattr__(r, name, finalized_value)

        # Use object.__setattr__ to bypass frozen dataclass restrictions
        object.__setattr__(r, "_finalized", True)
        return r

    def update(
        self,
        source: DataclassLike | Makeable[object] | None = None,
        *,
        skip_missing: bool = False,
        **kwargs: Any,
    ) -> Self:
        """Update config attributes from source and/or kwargs.

        Args:
          source: Optional source object to copy attributes from.
          skip_missing: If True, skip kwargs keys that don't exist as attributes.
          **kwargs: Additional attribute overrides (use **mapping to pass a dict).

        Returns:
          self: Updated instance for method chaining.

        """
        # Build valid_keys set if needed for skip_missing
        valid_keys: set[str] | None = None
        if skip_missing:
            valid_keys = set(_get_object_attribute_names(self))

        # Apply source attributes (kwargs take precedence)
        if source is not None:
            for name in _get_object_attribute_names(source):
                # Skip if already in kwargs (kwargs override source)
                if name in kwargs:
                    continue
                # Skip if not a valid key
                if valid_keys is not None and name not in valid_keys:
                    continue
                try:
                    setattr(self, name, getattr(source, name))
                except AttributeError:
                    continue

        # Apply kwargs
        for k, v in kwargs.items():
            if valid_keys is not None and k not in valid_keys:
                continue
            setattr(self, k, v)

        return self

    def _repr_pretty_(self, p: _IPythonPrinter, cycle: bool) -> None:
        """IPython pretty printer hook for rich display in notebooks."""
        if cycle:
            p.text(f"{type(self).__name__}(...)")
            return

        p.text(pformat(self))


class _Default:
    __slots__: ClassVar[tuple[str, ...]] = ("value",)

    def __init__(self, value: object) -> None:
        self.value = value

    def __bool__(self) -> bool:
        return bool(self.value)

    @override
    def __repr__(self) -> str:
        return f"{self.value!r}"


class _DataclassParams:
    __mro__: ClassVar[list[type]]
    __name__: ClassVar[str]
    __slots__: ClassVar[tuple[str, ...]] = (
        "eq",
        "frozen",
        "init",
        "kw_only",
        "match_args",
        "order",
        "repr",
        "slots",
        "unsafe_hash",
        "weakref_slot",
    )

    def __init__(
        self,
        init: bool = True,
        repr: bool = True,
        eq: bool = True,
        order: bool = False,
        unsafe_hash: bool = False,
        frozen: bool = False,
        match_args: bool = True,
        # The following differs from dataclasses.dataclass.
        kw_only: bool = True,
        slots: bool = True,
        weakref_slot: bool = True,
    ) -> None:
        self.init = init
        self.repr = repr
        self.eq = eq
        self.order = order
        self.unsafe_hash = unsafe_hash
        self.frozen = frozen
        self.match_args = match_args
        self.kw_only = kw_only
        self.slots = slots
        self.weakref_slot = weakref_slot

    @override
    def __repr__(self) -> str:
        return (
            f"{type(self).__name__}("
            + ", ".join(f"{k}={self[k]!r}" for k in self.keys())
            + ")"
        )

    def __getitem__(self, key: str) -> bool:
        return getattr(self, key)

    def __iter__(self) -> Iterator[str]:
        seen = set[str]()
        for c in type(self).__mro__:
            slots = getattr(c, "__slots__", ())
            if isinstance(slots, str):
                slots = (slots,)
            for s in slots:
                if s in seen:
                    continue
                seen.add(s)
                yield s

    keys = __iter__

    @classmethod
    def create(
        cls,
        existing: _DataclassParams,
        **kwargs: bool | _Default,
    ) -> _DataclassParams:
        new = _DataclassParams()
        missing = object()
        for k in new:
            # Check kwargs first
            v = kwargs.get(k, missing)
            if v is missing or isinstance(v, _Default):
                # Fall back to existing
                v = getattr(existing, k, missing)
            if v is missing:
                continue
            setattr(new, k, bool(v))
        return new


_True = _Default(True)
_False = _Default(False)


class _DataclassMeta(type):
    __classcell__: CellType | None = None
    __dataclass_params__: _DataclassParams = _DataclassParams()

    def __new__(
        mcls: type[_DataclassMeta],
        name: str,
        bases: tuple[type, ...],
        attrs: dict[str, object],
        *,
        init: bool | _Default = _True,
        repr: bool | _Default = _True,
        eq: bool | _Default = _True,
        order: bool | _Default = _False,
        unsafe_hash: bool | _Default = _False,
        frozen: bool | _Default = _False,
        match_args: bool | _Default = _True,
        # The following differs from dataclasses.dataclass.
        kw_only: bool | _Default = _True,
        slots: bool | _Default = _True,
        weakref_slot: bool | _Default | None = None,
        require_defaults: bool = True,
    ) -> _DataclassMeta:
        cls = super().__new__(mcls, name, bases, attrs)
        if classcell := attrs.get("__classcell__"):
            cls.__classcell__ = cast(CellType, classcell)
        if "__slots__" in cls.__dict__:
            return cls
        kwargs = _DataclassParams.create(
            cls.__dataclass_params__,
            init=init,
            repr=repr,
            eq=eq,
            order=order,
            unsafe_hash=unsafe_hash,
            frozen=frozen,
            match_args=match_args,
            kw_only=kw_only,
            slots=slots,
            weakref_slot=slots if weakref_slot is None else weakref_slot,
        )
        cls = dataclasses.dataclass(cls, **kwargs)

        if require_defaults:
            current_annotations = cast(
                dict[str, object],
                attrs.get("__annotations__", {}),
            )
            for field in dataclasses.fields(cls):  # pyright: ignore[reportArgumentType]
                if field.name not in current_annotations:
                    continue
                if (
                    field.default is dataclasses.MISSING
                    and field.default_factory is dataclasses.MISSING
                ):
                    raise TypeError(
                        f"{name}.{field.name} must have a default value. "
                        f"Use require_defaults=False to disable this check.",
                    )

        cls.__dataclass_params__ = kwargs
        cls = cast(_DataclassMeta, cls)
        return cls


@dataclass_transform(kw_only_default=True)
class DataclassMeta(_DataclassMeta):
    """Public metaclass for creating dataclass-based config classes.

    This metaclass automatically applies @dataclass decorator with sensible
    defaults (kw_only=True, slots=True, etc.) to any class using it.

    """


class Dataclass(metaclass=DataclassMeta):
    """Base class that auto-applies @dataclass with sensible defaults."""

    __slots__: ClassVar[tuple[str, ...]] = ()


@dataclass_transform(kw_only_default=True)
class FigMeta(_DataclassMeta, MakerMeta):
    """Combined metaclass for Fig.

    This metaclass combines _DataclassMeta (automatic dataclass conversion) and
    MakerMeta (parent class tracking) to enable the nested Config pattern where
    Config classes can call .make() to instantiate their parent class.

    """


class Fig(Maker[_ParentT], metaclass=FigMeta):
    """Dataclass with make/finalize/update for the nested Config pattern.

    Example:
      >>> class MyClass:
      ...     class Config(Fig):
      ...         x: int = 0
      ...
      ...     def __init__(self, config: Config) -> None:
      ...         self.x = config.x
      ...
      >>> obj = MyClass.Config(x=1).make()

    """

    __slots__: ClassVar[tuple[str, ...]] = ()


class Makes(Generic[_ParentT]):
    """Type-only base for inherited Configs that fixes the make() return type.

    When a Config inherits from a parent Config, the make() return type is
    the parent's type, not the child's. Use Makes as the first base to
    re-specify the return type:

        class Animal:
            class Config(Fig["Animal"]):
                name: str = "animal"
            def __init__(self, config: Config):
                self.name = config.name

        class Dog(Animal):
            class Config(Makes["Dog"], Animal.Config):
                breed: str = "mutt"

        dog: Dog = Dog.Config(breed="mutt").make()  # returns Dog, not Animal

    At runtime, Makes["X"] contributes nothing to the MRO — it exists
    purely for static type checking.

    Workaround for Python's lack of Intersection types. If Intersection
    types are adopted, MakerMeta.__get__ could potentially narrow the
    inherited make() return type directly, making this class unnecessary.

    """

    if TYPE_CHECKING:

        @property
        def parent_class(self) -> type[_ParentT]: ...

        def make(self) -> _ParentT: ...

    def __class_getitem__(cls, params: object) -> object:
        class _NoMroAlias:
            __origin__ = cls
            __args__ = (params,)

            @staticmethod
            def __mro_entries__(bases: object) -> tuple[()]:
                return ()

        return _NoMroAlias()


_ValueT = TypeVar("_ValueT")  # Used internally for _finalize_value


_SKIP_ATTRS = frozenset(("__weakref__", "__dict__", "_finalized"))


def _get_object_attribute_names(obj: object) -> Iterator[str]:
    """Yield attribute names, excluding __weakref__, __dict__, and _finalized."""
    seen = set[str]()
    if hasattr(type(obj), "__slots__"):
        for cls in type(obj).__mro__:
            slots = getattr(cls, "__slots__", ())
            if isinstance(slots, str):
                slots = (slots,)
            for slot in slots:
                if slot not in seen and slot not in _SKIP_ATTRS:
                    seen.add(slot)
                    yield slot
    if hasattr(obj, "__dict__"):
        for key in sorted(vars(obj)):
            if key not in seen and key not in _SKIP_ATTRS:
                seen.add(key)
                yield key


def _finalize_value(value: _ValueT) -> _ValueT:
    """Recursively finalize nested Fig instances, preserving container types.

    Traverses sequences, mappings, sets, and objects with __slots__/__dict__
    to discover and finalize all Fig instances.

    Args:
      value: Value to finalize recursively.

    Returns:
      finalized_value: Finalized copy with all nested configs finalized.

    """
    if isinstance(value, _HasFinalize) and not getattr(value, "_finalized", False):
        return value.finalize()  # ty: ignore[invalid-return-type]

    # Skip classes, types, and primitives - they don't need finalization
    if isinstance(value, (type, int, float, str, bytes, bool, type(None))):
        return value

    if isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
        finalized_items: list[object] = [_finalize_value(v) for v in value]
        if isinstance(value, tuple):
            if type(value) is tuple:
                return tuple(finalized_items)  # pyright: ignore[reportReturnType]  # ty: ignore[invalid-return-type]
            # Namedtuple needs unpacking
            return type(value)(*finalized_items)  # pyright: ignore[reportArgumentType]
        finalized = finalized_items
    elif isinstance(value, Mapping):
        # Mapping key type is unknown at runtime
        finalized = {k: _finalize_value(v) for k, v in value.items()}  # pyright: ignore[reportUnknownVariableType]
    elif isinstance(value, AbstractSet):
        finalized = {_finalize_value(v) for v in value}
    else:
        r = copy.copy(value)

        for name in _get_object_attribute_names(r):
            try:
                attr_value = getattr(r, name)
            except AttributeError:
                continue
            finalized_attr_value = _finalize_value(attr_value)
            if finalized_attr_value is not attr_value:
                object.__setattr__(r, name, finalized_attr_value)

        return r

    # Reconstruct container with finalized items
    return type(value)(finalized)  # pyright: ignore[reportCallIssue,reportUnknownArgumentType,reportUnknownVariableType]
