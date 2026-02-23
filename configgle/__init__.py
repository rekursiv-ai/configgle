"""Configgle: Tools for making configurable Python classes for A/B experiments.

Configgle is a library for building configurable Python classes -- designed for
scenarios like A/B experiments where you want to create objects from declarative
config objects with defaults, overrides, and nested composition.

Core Pattern
------------

The central idea is the nested Config pattern: you define a Config dataclass
inside a class, then instantiate via ``Config(...).make()``::

    class MyModel:
        class Config(Fig):
            lr: float = 0.001
            layers: int = 3

        def __init__(self, config: Config):
            self.lr = config.lr

    model = MyModel.Config(lr=0.01).make()  # returns MyModel instance

Key Components
--------------

``Fig`` -- The main base class. Combines two metaclasses:

- ``DataclassMeta`` -- auto-applies ``@dataclass`` with opinionated defaults
  (``kw_only=True``, ``slots=True``, ``require_defaults=True``), so every field
  must have a default.
- ``MakerMeta`` -- uses ``__set_name__`` to capture the parent class when
  ``Config`` is defined as a nested class, enabling ``.make()`` to know what to
  construct.

``Maker`` -- Provides three core methods:

- ``make()`` -- finalizes the config, then calls ``parent_class(config)`` to
  build the object.
- ``finalize()`` -- shallow-copies and recursively finalizes nested configs.
  Override this to compute derived defaults.
- ``update(source, **kwargs)`` -- merges attributes from another config or
  kwargs (for config composition/overriding).

``Makes`` -- A type-level-only mixin for inheritance. When ``Dog.Config``
inherits from ``Animal.Config``, ``Makes["Dog"]`` re-narrows the return type of
``.make()`` so it returns ``Dog``, not ``Animal``. It contributes nothing to the
runtime MRO.

``Dataclass`` -- A standalone base class that gives you the auto-dataclass
metaclass without the Maker/Config machinery. Useful for plain data objects with
the same opinionated defaults.

``@autofig`` -- Decorator that auto-generates a ``Config`` from a class's
``__init__`` signature. Uses ``make_with_kwargs=True`` so ``make()`` passes
config fields as kwargs instead of passing the config object.

``InlineConfig`` / ``PartialConfig`` -- Config wrappers for arbitrary callables.
``InlineConfig(fn, *args, **kwargs)`` stores a deferred function call;
``.make()`` invokes it. ``PartialConfig`` wraps into ``functools.partial``
instead.

``CopyOnWrite`` -- A ``wrapt.ObjectProxy``-based proxy for safe counterfactual
mutation. Wraps a config tree and lazily copies objects only when mutations
occur, propagating copies up to parents. Useful for "what if I change this one
field?" without touching the original.

``pprint`` / ``pformat`` -- A ``PrettyPrinter`` subclass with config-aware
formatting: hides default values, auto-finalizes before printing, masks memory
addresses, adds continuation pipes for long outputs, and collapses short
sequences.

Design Highlights
-----------------

- **Type safety** -- Heavy use of ``@dataclass_transform``, ``Generic``,
  ``Protocol``, and descriptor tricks to give type checkers (basedpyright)
  accurate return types for ``.make()``.
- **Composition** -- Configs nest naturally; ``finalize()`` recursively walks
  the tree to finalize sub-configs.
- **Immutability-friendly** -- ``finalize()`` returns copies; ``CopyOnWrite``
  enables mutation without touching originals.
- **Pickle/cloudpickle compatible** -- Parent class binding uses ``MethodType``
  to avoid infinite recursion during serialization.

Type Checking
-------------

Both ``ty`` and ``basedpyright`` are first-class supported. ``ty`` provides
better inference for the ``Intersection``-based return type narrowing in
``MakerMeta.__get__``, so ``.make()`` return types are resolved more accurately.
``basedpyright`` works well but occasionally requires ``Makes`` annotations to
achieve the same narrowing.
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
