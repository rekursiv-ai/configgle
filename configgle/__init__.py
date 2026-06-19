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

``Fig`` inherits from ``Maker``. The lifecycle is construct -> finalize ->
make:

- ``.make()`` -- ``copy_tree().finalize()`` then call ``parent_class(config)``
  to build the object. The source config you pass is never mutated.
- ``.finalize()`` -- apply derived defaults IN PLACE and recursively finalize
  nested configs; returns ``self``. Override this to compute derived defaults
  (see below). It does NOT copy.
- ``.copy_tree()`` -- a "semi-deep" copy: nested configs and mutable containers
  holding configs are duplicated; leaf values (primitives, tensors, loggers)
  are aliased; immutable containers are preserved unless an element changed.
  This is the copy ``make``/``pprint`` apply before finalizing, so the original
  stays pristine.
- ``.update(source, **kwargs)`` -- merge attributes from another config or
  kwargs in place, for composition. Returns ``self`` for chaining. ``update``
  only assigns; it computes nothing derived (run ``finalize`` afterward).

Overriding finalize() -- pre / super / post
--------------------------------------------

Override ``finalize()`` to compute derived defaults. ``super().finalize()`` is
the seam that cascades into the nested child configs, so the call splits the
method into two phases::

    @override
    def finalize(self) -> Self:
        # PRE  (runs BEFORE children finalize):
        #   push values DOWN into children, set own derived fields.
        self = super().finalize()   # cascade: children finalize here
        # POST (runs AFTER children finalize):
        #   derive values UP from the now-finalized children.
        return self

Put a value you INJECT into a child before ``super()`` (so the child finalizes
with it); put a value you DERIVE from a child after ``super()`` (so you read its
finalized result). Pushdown is by far the common case, so ``super()`` is usually
last -- but it is NOT required to be; the pull-up phase legitimately follows it.

::

    from typing import Self, override

    class Sandwich:
        class Config(Fig["Sandwich"]):
            bread: str = "sourdough"
            topping: Topping.Config | None = None

            @override
            def finalize(self) -> Self:
                if self.topping is not None:
                    self.topping.portion = "double"  # pushdown (pre)
                return super().finalize()

``finalize`` mutates in place; the copy that protects the original happens once
at the ``make``/``pprint`` boundary (``copy_tree().finalize()``), so a config is
finalized exactly once, on a fresh tree -- there is never a re-finalize. A
config passed to ``make()`` is left untouched.

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

``Maker`` subclasses also integrate with IPython/Jupyter via
``_repr_pretty_``, so configs display with Fig-aware formatting
in notebooks automatically.

Design Highlights
-----------------

- **Type safety** -- ``@dataclass_transform``, ``Generic``, ``Protocol``,
  and descriptor tricks give type checkers accurate ``.make()`` return types.
- **Composition** -- Configs nest naturally; ``finalize()`` recursively
  walks the tree.
- **Predictable copying** -- ``finalize()`` mutates in place; the copy that
  protects the original happens once at the ``make``/``pprint`` boundary via
  ``copy_tree()``.
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
