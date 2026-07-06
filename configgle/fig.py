"""Dataclass metaclasses and Maker base for the nested Config pattern.

Configs are mutable dataclasses (``Fig``) nested in their parent class as
``Config``; you build the parent with ``ParentClass.Config(...).make()``.

Lifecycle
---------
A config has three phases:

1. **Construct + mutate.** ``cfg = Foo.Config(); cfg.lr = 0.01``. Plain
   attribute assignment; nothing derived yet.
2. **Finalize.** ``cfg.finalize()`` applies derived defaults (fields computed
   from other fields) and cascades into nested child configs. It mutates the
   receiver IN PLACE and returns it -- it does NOT copy.
3. **Build.** ``cfg.make()`` constructs the parent class from the finalized
   config.

The copy that keeps a source config pristine happens ONCE, at the boundary:
``make()`` and ``pprint`` run ``config.copy_tree().finalize()``. So the source
you hand to ``make()`` is never mutated, while ``finalize`` itself stays a pure
in-place hook.

The free functions ``make``/``copy_tree``/``update`` and the matching
``Maker`` methods are equivalent; ``finalize`` is the one overridable hook and
is method-only.

Key operations
--------------
- ``copy_tree(cfg)`` -- a "semi-deep" copy: every nested config and every
  mutable container holding configs is duplicated, while leaf values
  (primitives, tensors, loggers) are aliased. Immutable containers
  (tuple/frozenset) are preserved unless an element changed. This is the copy
  ``finalize`` needs before mutating; ``make``/``pprint`` apply it for you.
- ``finalize(cfg)`` -- apply derived defaults in place (see below).
- ``make(cfg)`` -- ``copy_tree().finalize()`` then construct the parent.
- ``update(cfg, source, **kwargs)`` -- in-place attribute overrides from a
  source object and/or keywords (kwargs win); returns the config for chaining.

Writing a ``finalize`` override -- pre / super / post
-----------------------------------------------------
``super().finalize()`` cascades into the children, so it splits the method into
a PRE phase (before children finalize -- push values down) and a POST phase
(after -- derive values up)::

    @override
    def finalize(self) -> Self:
        if self.channels_in == -1:
            self.channels_in = self.channels_out  # own derived field (pre)
        self.norm.channels_in = self.channels_in  # inject into a child (pre)
        self = super().finalize()                 # children finalize here
        self.out_dim = self.norm.out_dim          # derive from a child (post)
        return self

Inject into a child BEFORE the super call (the child finalizes with it); derive
from a child AFTER it (read its finalized value). Pushdown dominates, so
``super()`` is usually last -- but it need not be.
"""

from __future__ import annotations

from collections.abc import (
    Iterator,
)
from types import CellType, MethodType
from typing import (
    IO,
    TYPE_CHECKING,
    Any,
    ClassVar,
    Generic,
    Protocol,
    Self,
    cast,
    dataclass_transform,
    override,
)
from typing_extensions import TypeVar

import dataclasses


if TYPE_CHECKING:
    from ty_extensions import Intersection


from configgle.custom_types import (
    DataclassLike,
    Makeable,
)
from configgle.pprinting import (
    _DEFAULT_CONTINUATION_PIPE_THRESHOLD,
    _SHORT_SEQUENCE_MAX_WIDTH,
    pformat,
    pprint,
)
from configgle.serialize import (
    Hooks,
    deserialize,
    serialize,
)
from configgle.walk import (
    _copy_slots,
    _finalize_value,
    _get_object_attribute_names,
)


__all__ = [
    "Dataclass",
    "Fig",
    "Maker",
    "Makes",
    "make",
    "update",
]


_T = TypeVar("_T")

# The "correct" thing to do is:
#   _ParentT_co = TypeVar(
#       "_ParentT_co",
#       covariant=True,  # Covariance allows bare Fig to work for Intersections.
#       default=Any,  # Only matters for non-ty; its a lie but ergonomic.
#                     # The "truth" would be `default=object`.
#   )
# However, this is not possible until ty PRs
#   https://github.com/astral-sh/ruff/pull/26545
#   https://github.com/astral-sh/ruff/pull/26553
# are merged. For now we just do,
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
        return owner._parent_class()  # noqa: SLF001 -- reads the metaclass-bound parent reference; deliberate internal access  # pyright: ignore[reportAttributeAccessIssue,reportUnknownMemberType,reportUnknownVariableType] -- _parent_class is bound dynamically by MakerMeta.__set_name__, invisible to the checker.


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
        """Bind the parent class reference when Config is defined as a nested class.

        Python calls ``__set_name__`` on class-body attributes during class
        creation (PEP 487), so ``class Config(Fig): ...`` inside ``MyClass``
        triggers ``Config.__set_name__(MyClass, "Config")`` automatically.

        We store the parent reference via ``MethodType`` rather than a plain
        attribute to avoid reference cycles that break pickle/cloudpickle.

        See: https://docs.python.org/3/library/types.html#types.MethodType

        """

        def _returns_owner(_: MakerMeta) -> type[_ParentT]:
            return owner

        cls._parent_class = MethodType(_returns_owner, cls)  # ty: ignore[invalid-assignment] -- assigning a bound MethodType over the declared classmethod stub; intentional dynamic binding.
        # __set_name__ is only called when nested inside a class, so owner
        # is always a real class with __name__.
        cls.__name__ = f"{owner.__name__}.{name}"

    if TYPE_CHECKING:

        def __get__(
            cls: _T,
            obj: object,
            owner: type[_ParentT],
        ) -> Intersection[
            _T,
            # Returning Maker (vs Makeable) allows bare Fig to work for Intersections.
            #   type[Maker[_ParentT_co]],
            # However, this is not possible until ty PRs
            #   https://github.com/astral-sh/ruff/pull/26545
            #   https://github.com/astral-sh/ruff/pull/26553
            # are merged. For now we just do,
            type[Makeable[_ParentT]],
        ]:
            # This return has been reviewed extensively. Do not replace it with
            # casts or type-checker suppressions; package-local stubs define the
            # intended checker behavior for this descriptor path.
            return cls  # ty: ignore[invalid-return-type] -- `cls` is only `_T` at source level; the `& type[Makeable[_ParentT]]` half of the intersection is a design assertion that ty's real Intersection cannot prove. Configgle's `ty_extensions` polyfill (where `Intersection[A, B] = A`) hides this when run under package-local ty config; root ty sees the real intersection and rejects the return. See fig.py module docstring for the design rationale.


class Maker(Generic[_ParentT], metaclass=MakerMeta):
    """Base class providing make/finalize/update capabilities for configs.

    When nested inside a parent class, enables the pattern:
        instance = ParentClass.Config(...).make()

    """

    __slots__: ClassVar[tuple[str, ...]] = ("_finalized",)
    if TYPE_CHECKING:

        @property
        def parent_class(self) -> type[_ParentT]: ...

    else:
        parent_class = _MakerParentClassDescriptor()

    def __init__(self) -> None:
        self._finalized = False

    def make(self) -> _ParentT:
        """Finalize this config and instantiate its parent class.

        Returns:
          instance: Instance of the parent class.

        Raises:
          ValueError: If the config is not nested in a parent class.

        """
        return make(self)

    def copy_tree(self, visited: dict[int, object] | None = None) -> Self:
        """Copy this config's tree down to leaf values.

        The default implementation copies ``self`` and recurses through its
        fields (the shared ``_copy_slots`` walk, which the free ``copy_tree``
        also uses for plain data objects). Override to customize copy semantics
        for a config -- the dual of overriding ``finalize`` -- and
        thread ``visited`` through any ``super().copy_tree(visited)`` /
        ``copy_tree(value, visited)`` calls so shared and cyclic references stay
        consistent.

        Args:
          visited: Maps ``id(obj)`` to its copy, so a config reached twice (a
            shared sub-config, or a cycle) yields one copy, not several. The free
            ``copy_tree`` supplies it; callers may omit it.

        Returns:
          copied: A structural copy with nested configs and containers fresh and
            leaf values aliased.

        """
        if visited is None:
            visited = {}
        return cast(Self, _copy_slots(self, visited))

    def finalize(self) -> Self:
        """Apply derived defaults in place and mark this config finalized.

        Similar to ``__post_init__`` but deferred: only called automatically
        by ``make()`` and ``pprint``, not at construction time. This lets you
        mutate the config (``cfg.lr = 0.01``) before derived defaults are
        computed. Override to compute derived field values.

        Finalize mutates ``self`` and returns it -- it does **not** copy.
        Callers that must preserve the original (``make``, ``pprint``) call
        ``config.copy_tree().finalize()`` so the copy happens once, at the
        boundary, and ``finalize`` stays a pure in-place hook. Nested
        ``Finalizeable`` configs are finalized in place too (their own
        ``finalize`` runs); since the caller already copied the whole tree,
        no per-child copy is needed here.

        Overriding: ``super().finalize()`` cascades into the children, splitting
        the override into a PRE phase (before it -- push values down / inject
        into children) and a POST phase (after it -- derive values up from the
        now-finalized children). Pushdown is the common case, so ``super()`` is
        usually last, but it need not be. See the module docstring for the full
        lifecycle and an example.

        Returns:
          finalized: ``self``, mutated, with _finalized=True.

        """
        # Mark finalized BEFORE the cascade. A config is normally finalized once
        # (``make``/``pprint`` run ``copy_tree().finalize()`` on a fresh tree), so
        # this is the only write that matters -- but ``copy_tree`` preserves the
        # identity of a sub-config shared by two fields (a DAG), and the flag lets
        # ``_finalize_value`` finalize that shared node exactly once. The flag
        # also drives pprint's display (finalize=True skips marked configs).
        # ``object.__setattr__`` bypasses frozen dataclass restrictions.
        object.__setattr__(self, "_finalized", True)

        # Cascade into nested Finalizeable attrs. The caller copied the tree
        # before calling finalize, so mutation is isolated. A child finalize may
        # return a different object than it received, so the result is written
        # back onto this config.
        for name in _get_object_attribute_names(self):
            try:
                value = getattr(self, name)
            except AttributeError:
                continue
            finalized_value = _finalize_value(value)
            if finalized_value is not value:
                object.__setattr__(self, name, finalized_value)

        return self

    def update(
        self,
        source: DataclassLike | Makeable[object] | None = None,
        /,
        *,
        skip_missing: bool = False,
        **kwargs: Any,
    ) -> Self:
        """Update this config's attributes in place from a source and/or kwargs.

        Mutates ``self`` (no copy) and returns it for chaining. Unlike
        ``finalize``, ``update`` computes nothing derived -- it only assigns the
        given attributes; run ``finalize`` (or ``make``) afterward to apply
        derived defaults.

        Args:
          source: Optional object whose attributes are copied onto this config.
          skip_missing: Skip keys that are not declared attributes of this config.
          **kwargs: Attribute overrides; take precedence over ``source``.

        Returns:
          self: This config, updated, for method chaining.

        """
        return update(self, source, skip_missing=skip_missing, **kwargs)

    def serialize(self, *, hooks: Hooks | None = None) -> Any:
        """Serialize this config tree to an encodable dict tree.

        Returns a JSON-encodable structure (nested dicts/lists/primitives), not
        a string -- the caller picks the transport. The tree is
        transport-agnostic: hand it to ``json.dumps``, ``yaml.safe_dump``,
        ``msgpack``, or embed it in a larger structure. Captures the config
        as-is: ``serialize`` does not finalize, so derived defaults are left for
        a later ``finalize``/``make`` on the loaded tree.

        Example:
          >>> import json
          >>> s = json.dumps(cfg.serialize(), indent=2)  # object -> JSON string
          >>> cfg = SomeClass.Config.deserialize(json.loads(s))  # string -> config
          >>> obj = cfg.make()

        Args:
          hooks: Optional ``{type: (encode, decode)}`` map for leaves JSON cannot
            represent natively (tensors, arrays, etc.).

        Returns:
          tree: An encodable tree that ``deserialize`` reverses into live objects.

        """
        return serialize(self, hooks=hooks)

    @classmethod
    def deserialize(cls, tree: object, *, hooks: Hooks | None = None) -> Self:
        """Reconstruct a config from a tree produced by ``serialize``.

        Resolves config classes and callables by their recorded import path, so
        the defining modules must be importable. This imports and calls code
        named in the payload; deserialize only trusted data.

        Example:
          >>> import json
          >>> s = json.dumps(cfg.serialize(), indent=2)  # object -> JSON string
          >>> cfg = SomeClass.Config.deserialize(json.loads(s))  # string -> config
          >>> obj = cfg.make()

        Args:
          tree: An encodable tree produced by ``serialize`` (e.g. from
            ``json.loads`` of a stored string).
          hooks: The same ``{type: (encode, decode)}`` map used to serialize.

        Returns:
          config: The reconstructed config tree.

        """
        return cast(Self, deserialize(tree, hooks=hooks))

    def pformat(
        self,
        indent: int = 8,
        width: int = 80,
        depth: int | None = None,
        *,
        compact: bool = False,
        sort_dicts: bool = False,
        underscore_numbers: bool = True,
        finalize: bool = True,
        mask_memory_addresses: bool = True,
        extra_compact: bool = True,
        continuation_pipe: int = _DEFAULT_CONTINUATION_PIPE_THRESHOLD,
        hide_default_values: bool = True,
        short_sequence_max_width: int = _SHORT_SEQUENCE_MAX_WIDTH,
    ) -> str:
        """Format this config as a string with Fig-aware pretty printing.

        Args:
          indent: Spaces per indent level.
          width: Maximum line width.
          depth: Maximum nesting depth (None for unlimited).
          compact: Use compact format for sequences.
          sort_dicts: Sort dictionary keys.
          underscore_numbers: Use underscores in large numbers.
          finalize: Auto-finalize unfinalized configs before printing.
          mask_memory_addresses: Replace memory addresses with placeholder.
          extra_compact: Use extra compact formatting.
          continuation_pipe: Lines threshold for continuation pipes (0=always, -1=never).
          hide_default_values: Omit fields with default values.
          short_sequence_max_width: Max width for single-line sequences.

        Returns:
          formatted: Pretty-printed string representation.

        """
        return pformat(
            self,
            indent=indent,
            width=width,
            depth=depth,
            compact=compact,
            sort_dicts=sort_dicts,
            underscore_numbers=underscore_numbers,
            finalize=finalize,
            mask_memory_addresses=mask_memory_addresses,
            extra_compact=extra_compact,
            continuation_pipe=continuation_pipe,
            hide_default_values=hide_default_values,
            short_sequence_max_width=short_sequence_max_width,
        )

    def pprint(
        self,
        stream: IO[str] | None = None,
        indent: int = 8,
        width: int = 80,
        depth: int | None = None,
        *,
        compact: bool = False,
        sort_dicts: bool = False,
        underscore_numbers: bool = True,
        finalize: bool = True,
        mask_memory_addresses: bool = True,
        extra_compact: bool = True,
        continuation_pipe: int = _DEFAULT_CONTINUATION_PIPE_THRESHOLD,
        hide_default_values: bool = True,
        short_sequence_max_width: int = _SHORT_SEQUENCE_MAX_WIDTH,
    ) -> None:
        """Pretty-print this config with Fig-aware formatting.

        Args:
          stream: Output stream (defaults to sys.stdout).
          indent: Spaces per indent level.
          width: Maximum line width.
          depth: Maximum nesting depth (None for unlimited).
          compact: Use compact format for sequences.
          sort_dicts: Sort dictionary keys.
          underscore_numbers: Use underscores in large numbers.
          finalize: Auto-finalize unfinalized configs before printing.
          mask_memory_addresses: Replace memory addresses with placeholder.
          extra_compact: Use extra compact formatting.
          continuation_pipe: Lines threshold for continuation pipes (0=always, -1=never).
          hide_default_values: Omit fields with default values.
          short_sequence_max_width: Max width for single-line sequences.

        """
        pprint(
            self,
            stream=stream,
            indent=indent,
            width=width,
            depth=depth,
            compact=compact,
            sort_dicts=sort_dicts,
            underscore_numbers=underscore_numbers,
            finalize=finalize,
            mask_memory_addresses=mask_memory_addresses,
            extra_compact=extra_compact,
            continuation_pipe=continuation_pipe,
            hide_default_values=hide_default_values,
            short_sequence_max_width=short_sequence_max_width,
        )

    def _repr_pretty_(self, p: _IPythonPrinter, cycle: bool) -> None:
        """IPython pretty printer hook for rich display in notebooks."""
        if cycle:
            p.text(f"{type(self).__name__}(...)")
            return

        p.text(self.pformat())


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
    """Metaclass that auto-applies ``@dataclass`` to subclasses.

    Using a metaclass instead of requiring ``@dataclass`` on each Config
    ensures consistent defaults and lets subclasses inherit dataclass
    settings without repeating the decorator.

    ``kw_only=True`` because the primary usage pattern is dotted assignment
    (``cfg.lr = 0.01``) rather than positional construction, and keyword-only
    args prevent accidental positional misuse.

    ``slots=True`` because configs are allocated frequently and slots give
    both memory savings and faster attribute access, and — importantly —
    prevent typos from silently creating new attributes
    (``cfg.lrr = 0.01`` raises ``AttributeError``).
    """

    __classcell__: CellType | None = None
    __dataclass_params__: _DataclassParams = _DataclassParams()
    make_with_kwargs: ClassVar[bool]

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
        make_with_kwargs: bool | None = None,
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
            for field in dataclasses.fields(cls):  # pyright: ignore[reportArgumentType] -- cls is a dataclass at this point (just decorated above), but the checker sees the pre-decoration type.
                if field.name not in current_annotations:
                    continue
                if (
                    field.default is dataclasses.MISSING
                    and field.default_factory is dataclasses.MISSING
                ):
                    raise TypeError(
                        f"{name}.{field.name} has no default value. "
                        f"Add a default or use require_defaults=False to disable this check.",
                    )

        if make_with_kwargs is not None:
            cls.make_with_kwargs = make_with_kwargs

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


class FigMeta(_DataclassMeta, MakerMeta):
    """Combined metaclass for Fig.

    This metaclass combines _DataclassMeta (automatic dataclass conversion) and
    MakerMeta (parent class tracking) to enable the nested Config pattern where
    Config classes can call .make() to instantiate their parent class.

    """

    if TYPE_CHECKING:

        def __new__(
            mcls: type[FigMeta],
            name: str,
            bases: tuple[type, ...],
            attrs: dict[str, object],
            **kwargs: Any,
        ) -> FigMeta: ...


# @dataclass_transform is on Fig (not FigMeta) to work around a ty bug
# where Intersection[_T, type[Generic[TypeVar]]] in MakerMeta.__get__
# breaks dataclass_transform field inheritance when applied to the
# metaclass (https://github.com/astral-sh/ty/issues/3282). When fixed,
# move @dataclass_transform back to FigMeta and remove it from here.
@dataclass_transform(kw_only_default=True)
class Fig(Maker[_ParentT], metaclass=FigMeta):
    """Dataclass with make/finalize/update for the nested Config pattern.

    Build with ``ParentClass.Config(...).make()``. See the module docstring for
    the construct -> finalize -> make lifecycle, the role of ``copy_tree``, and
    how to write a ``finalize`` override (mutate first, ``super().finalize()``
    last).

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

            @classmethod
            def __mro_entries__(cls, bases: object) -> tuple[()]:
                return ()

        return _NoMroAlias()


def make[ParentT](config: Maker[ParentT]) -> ParentT:
    """Finalize a config and instantiate its parent class.

    Args:
      config: The config to finalize and build.

    Returns:
      instance: An instance of the config's parent class.

    Raises:
      ValueError: If the config is not nested in a parent class.

    """
    finalized = config.copy_tree().finalize()
    cls = finalized.parent_class
    if cls is None:  # pyright: ignore[reportUnnecessaryComparison] -- parent_class is non-None per its annotation, but a Maker not nested in a class has none at runtime; the guard is a real runtime check.
        raise ValueError("Maker must be nested in a parent class")
    if getattr(type(finalized), "make_with_kwargs", False):
        kwargs = {
            f.name: getattr(finalized, f.name)
            for f in dataclasses.fields(cast(DataclassLike, cast(object, finalized)))
        }
        return cls(**kwargs)
    return cls(finalized)  # pyright: ignore[reportCallIssue] -- the parent class accepts its own Config; the checker cannot link parent_class back to that constructor signature.


def update[MakerT: Maker[Any]](
    config: MakerT,
    source: DataclassLike | Makeable[object] | None = None,
    /,
    *,
    skip_missing: bool = False,
    **kwargs: Any,
) -> MakerT:
    """Update a config's attributes in place from a source and/or overrides.

    Args:
      config: The config to mutate.
      source: Optional object whose attributes are copied onto ``config``.
      skip_missing: Skip keys that are not declared attributes of ``config``.
      **kwargs: Attribute overrides; take precedence over ``source``.

    Returns:
      config: The same config, updated, for method chaining.

    """
    valid_keys: set[str] | None = None
    if skip_missing:
        valid_keys = set(_get_object_attribute_names(config))

    # Apply source attributes (kwargs take precedence).
    if source is not None:
        for name in _get_object_attribute_names(source):
            if name in kwargs:
                continue
            if valid_keys is not None and name not in valid_keys:
                continue
            try:
                setattr(config, name, getattr(source, name))
            except AttributeError:
                continue

    # Apply kwargs.
    for key, value in kwargs.items():
        if valid_keys is not None and key not in valid_keys:
            continue
        setattr(config, key, value)

    return config
