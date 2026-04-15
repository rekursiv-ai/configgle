"""Configgle: Composable config objects for building Python classes.

Configgle provides the nested Config pattern: define a ``Config`` dataclass
inside a class, set fields to configure behavior, call ``.make()`` to get
an instance::

    class MyModel:
        class Config(Fig):
            lr: float = 0.001
            layers: int = 3

        def __init__(self, config: Config):
            self.lr = config.lr

    model = MyModel.Config(lr=0.01).make()  # returns MyModel instance

Fig and Fig["X"]
----------------

``Fig`` is the main base class. It combines two metaclasses:

- ``DataclassMeta`` -- auto-applies ``@dataclass`` with opinionated defaults
  (``kw_only=True``, ``slots=True``, ``require_defaults=True``), so every field
  must have a default.
- ``MakerMeta`` -- uses ``__set_name__`` to capture the parent class when
  ``Config`` is defined as a nested class, enabling ``.make()`` to know what to
  construct.

``Fig["X"]`` is the type-parameterized form. The parameter tells the type
checker what ``.make()`` returns and is needed for all typecheckers except ty::

    class Dog:
        class Config(Fig["Dog"]):   # .make() -> Dog
            breed: str = "mutt"

        def __init__(self, config: Config):
            self.breed = config.breed

    dog: Dog = Dog.Config().make()  # type checker knows this is Dog

At runtime ``Fig["Dog"]`` and bare ``Fig`` behave identically --
``MakerMeta.__set_name__`` binds the parent class automatically when
``Config`` is nested. The type parameter is only needed for static type
narrowing.

Makes["X"] -- Inherited Configs
-------------------------------

When a child's ``Config`` inherits from a parent's ``Config``, ``.make()``
would return the parent type. ``Makes["X"]`` re-narrows it. Put it first
in the bases::

    class Animal:
        class Config(Fig["Animal"]):
            name: str = ""

        def __init__(self, config: Config):
            self.name = config.name

    class Dog(Animal):
        class Config(Makes["Dog"], Animal.Config):
            breed: str = "mutt"

    dog: Dog = Dog.Config().make()  # Dog, not Animal

At runtime ``Makes["X"]`` contributes nothing to the MRO -- it exists
purely for static type checking (needed for all type checkers except
``ty``, which infers the correct return type without it). Workaround
for Python's lack of Intersection types.

Maker -- Core Methods
---------------------

``Fig`` inherits from ``Maker``, which provides three methods:

- ``.make()`` -- finalize the config, then call ``parent_class(config)``
  to build the object.
- ``.finalize()`` -- shallow-copy and recursively finalize nested configs.
  Override this to compute derived defaults (see below).
- ``.update(source, **kwargs)`` -- merge attributes from another config or
  kwargs for config composition. Returns ``self`` for chaining.

Overriding finalize()
---------------------

Override ``finalize()`` to compute derived defaults. The contract:

1. Call ``super().finalize()`` first. This shallow-copies the config and
   recursively finalizes all nested ``Finalizeable`` attributes (configs
   inside lists, dicts, dataclass fields, etc.).
2. The returned object is a copy -- mutate it freely.
3. Return the mutated copy.

::

    from typing import Self, override
    import copy

    class Sandwich:
        class Config(Fig["Sandwich"]):
            bread: str = "sourdough"
            topping: Topping.Config | None = None

            @override
            def finalize(self) -> Self:
                self = super().finalize()
                # `self` is now a copy. Nested configs are already
                # finalized -- but we can re-finalize after mutation.
                if self.topping is not None:
                    self.topping = copy.copy(self.topping)
                    self.topping.portion = "double"
                    self.topping = self.topping.finalize()
                return self

If a nested config needs a field set *before* its own ``finalize()`` runs,
``copy.copy`` it, set the field, and call ``.finalize()`` on it yourself as
shown above. The base ``finalize()`` skips anything already finalized
(``_finalized=True``).

Positional Fields (kw_only=False)
---------------------------------

By default all fields are keyword-only. To allow leading positional fields,
pass ``kw_only=False`` to the class definition and use the ``KW_ONLY``
sentinel from ``dataclasses`` to mark where keyword-only fields begin::

    from dataclasses import KW_ONLY

    class Rectangle:
        class Config(Fig["Rectangle"], kw_only=False):
            width: float = 1.0       # positional OK
            height: float = 1.0      # positional OK
            _: KW_ONLY
            color: str = "blue"      # keyword-only
            filled: bool = True

        def __init__(self, config: Config):
            ...

    # Both work:
    Rectangle.Config(3.0, 4.0, color="red")
    Rectangle.Config(width=3.0, height=4.0)

The Mixin Pattern
-----------------

For reusable building blocks, define the mixin's Config with ``Fig``
and each concrete class's Config with ``Makes`` (neither type parameter
is needed for ``ty``, but is required by other type checkers). This lets
multiple concrete classes share config fields and ``__init__`` logic::

    class FlavorMixin:
        class Config(Fig["FlavorMixin"], kw_only=False):
            flavor: str = "vanilla"
            _: KW_ONLY
            sprinkles: bool = False

        def __init__(self, *args, config: Config, **kwargs):
            self.flavor = config.flavor
            self.sprinkles = config.sprinkles
            super().__init__(*args, **kwargs)

    class Cake(FlavorMixin):
        class Config(Makes["Cake"], FlavorMixin.Config):
            layers: int = 2

        def __init__(self, config: Config):
            super().__init__(config=config)
            self.layers = config.layers

    class Milkshake(FlavorMixin):
        class Config(Makes["Milkshake"], FlavorMixin.Config):
            thick: bool = True

        def __init__(self, config: Config):
            super().__init__(config=config)
            self.thick = config.thick

    cake = Cake.Config("chocolate", layers=3).make()
    shake = Milkshake.Config("strawberry", thick=False).make()

The mixin's ``__init__`` accepts ``*args, config=, **kwargs`` and
forwards unknowns via ``super().__init__(*args, **kwargs)``, letting
Python's MRO route them to the next class in the chain. Each concrete
class's ``__init__`` takes only ``config`` and passes ``config=`` to
``super().__init__``.

When combining a mixin with an existing base class (e.g., from a
third-party library), the concrete class lists both in its bases and
forwards the base class's required args through ``super()``::

    class FlavorWidget(FlavorMixin, tkinter.Button):
        class Config(Makes["FlavorWidget"], FlavorMixin.Config):
            text: str = "Click me"

        def __init__(self, config: Config):
            super().__init__(config=config, text=config.text)

CopyOnWrite
-----------

``CopyOnWrite`` is a ``wrapt.ObjectProxy``-based proxy for safe
counterfactual mutation. It wraps a config tree and lazily copies
objects only when (and where) mutations actually occur, propagating
copies upward to parents. The original tree is never modified.

The typical use case is inside a ``finalize()`` override where you need
to tweak a nested config without manually ``copy.copy``-ing every node
on the path::

    class Bakery:
        class Config(Fig["Bakery"]):
            cake: Cake.Config = field(default_factory=Cake.Config)
            num_ovens: int = 2

            @override
            def finalize(self) -> Self:
                self = super().finalize()
                with CopyOnWrite(self) as cow:
                    cow.cake.layers = self.num_ovens * 2
                    self = cow.unwrap
                return self

    # The original Cake.Config default is never modified.

Without ``CopyOnWrite``, the equivalent requires ``copy.copy`` at each
level of nesting. Nested attribute access returns child wrappers, so
``cow.a.b.c = 1`` lazily copies ``c``, ``b``, ``a``, and the root --
but only on first mutation. Subsequent writes to already-copied objects
are free. On context manager exit, any copied configs are automatically
finalized.

Other Components
----------------

``Dataclass`` -- Standalone base with the auto-dataclass metaclass but
without Maker/make(). For plain data objects.

``@autofig`` -- Decorator that auto-generates a ``Config`` from a class's
``__init__`` signature. Uses ``make_with_kwargs=True`` so ``make()`` passes
config fields as kwargs instead of passing the config object.

``InlineConfig`` / ``PartialConfig`` -- Config wrappers for callables.
``InlineConfig(fn, **kw).make()`` calls ``fn(**kw)``.
``PartialConfig(fn, **kw).make()`` returns ``functools.partial(fn, **kw)``.

``pprint`` / ``pformat`` -- Config-aware pretty printer. Hides defaults,
auto-finalizes, masks memory addresses. Available as both module-level
functions and as methods on any ``Maker`` subclass::

    from configgle import pformat
    print(pformat(cfg))  # module-level

    cfg.pprint()         # method — prints to stdout
    s = cfg.pformat()    # method — returns string

``Makeable`` -- Runtime-checkable ``Protocol`` defining the config
interface (``make()``, ``finalize()``, ``update()``). Also aliased as
``Configurable``.

Design Highlights
-----------------

- **Type safety** -- ``@dataclass_transform``, ``Generic``, ``Protocol``,
  and descriptor tricks give type checkers accurate ``.make()`` return types.
- **Composition** -- Configs nest naturally; ``finalize()`` recursively
  walks the tree.
- **Immutability-friendly** -- ``finalize()`` returns copies;
  ``CopyOnWrite`` enables mutation without touching originals.
- **Pickle/cloudpickle compatible** -- Parent class binding uses
  ``MethodType`` to avoid reference cycles during serialization.

Type Checking
-------------

Both ``ty`` and ``basedpyright`` are first-class supported. ``ty``
provides better inference for ``Intersection``-based return type narrowing
in ``MakerMeta.__get__``. ``basedpyright`` works well but occasionally
requires ``Makes`` annotations to achieve the same narrowing.
"""

from __future__ import annotations

from configgle.copy_on_write import CopyOnWrite
from configgle.custom_types import (
    Configurable,
    DataclassLike,
    Finalizeable,
    HasConfig,
    HasRelaxedConfig,
    Makeable,
    MutableNamespace,
    RelaxedConfigurable,
    RelaxedMakeable,
)
from configgle.decorator import autofig
from configgle.fig import Dataclass, Fig, Maker, Makes
from configgle.inline import InlineConfig, PartialConfig
from configgle.pprinting import pformat, pprint


__all__ = [
    "Configurable",
    "CopyOnWrite",
    "Dataclass",
    "DataclassLike",
    "Fig",
    "Finalizeable",
    "HasConfig",
    "HasRelaxedConfig",
    "InlineConfig",
    "Makeable",
    "Maker",
    "Makes",
    "MutableNamespace",
    "PartialConfig",
    "RelaxedConfigurable",
    "RelaxedMakeable",
    "autofig",
    "pformat",
    "pprint",
]
