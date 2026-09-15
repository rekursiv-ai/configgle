"""Decorator to auto-generate a Config dataclass from __init__ parameters."""

from __future__ import annotations

from dataclasses import field
from functools import partial
from types import SimpleNamespace
from typing import TYPE_CHECKING, cast, get_type_hints, overload

import importlib
import inspect
import sys

from configgle.custom_types import HasRelaxedConfig
from configgle.fig import Fig, FigMeta


if TYPE_CHECKING:
    from collections.abc import Callable


__all__ = ["autofig"]


@overload
def autofig[T](cls: type[T], /) -> type[HasRelaxedConfig[T]]: ...


@overload
def autofig[T](
    cls: None = None,
    /,
    *,
    require_defaults: bool = True,
    kw_only: bool = True,
) -> Callable[[type[T]], type[HasRelaxedConfig[T]]]: ...


def autofig[T](
    cls: type[T] | None = None,
    /,
    *,
    require_defaults: bool = True,
    kw_only: bool = True,
) -> type[HasRelaxedConfig[T]] | Callable[[type[T]], type[HasRelaxedConfig[T]]]:
    """Create a nested Config dataclass from __init__ parameters.

    The Config class gets parent_class (via MakerMeta), make() to instantiate
    the parent class via kwargs unpacking, finalize() for derived defaults, and
    update() for config merging.

    Constructors must be instance methods with keyword-capable parameters;
    positional-only and variadic parameters and Config method names are rejected.
    Defaults retain the constructor's object identities, including shared mutable
    defaults across fields and Config instances. Unresolved types retain
    their constructor scope and can resolve later. Invalid annotations fall back to
    object for that field. Annotation inspection can execute user code.

    Args:
      cls: The class to decorate (when used without arguments).
      require_defaults: If True, all Config fields must have defaults.
      kw_only: If True, all Config fields are keyword-only.

    Returns:
      decorated: The class with a nested Config dataclass attached.

    Raises:
      TypeError: A constructor parameter cannot be passed by keyword.

    Example:
      @autofig
      class Foo:
          def __init__(self, x: int = 0, y: str = "default"):
              self.x = x
              self.y = y

      # Now you can use:
      config = Foo.Config(x=10, y="hello")
      foo = config.make()  # Makes Foo(x=10, y="hello")

    """

    def decorator(cls_: type[T]) -> type[HasRelaxedConfig[T]]:
        return _autofig(cls_, require_defaults=require_defaults, kw_only=kw_only)

    if cls is None:
        # Called with arguments: @autofig(require_defaults=True)
        return decorator
    # Called without arguments: @autofig.
    return decorator(cls)


def _autofig[T](
    cls_: type[T],
    *,
    require_defaults: bool,
    kw_only: bool,
) -> type[HasRelaxedConfig[T]]:
    """Generate and attach a config using the constructor's annotations."""
    raw_constructor = inspect.getattr_static(cls_, "__init__")
    if isinstance(raw_constructor, (staticmethod, classmethod)):
        raise TypeError("autofig requires an instance-method __init__ constructor.")
    constructor = cast("Callable[..., object]", raw_constructor)
    if constructor is object.__init__:
        if cls_.__new__ is not object.__new__:
            raise TypeError("autofig requires an __init__ constructor, not __new__.")
        sig = inspect.Signature()
    elif sys.version_info >= (3, 14):
        sig = inspect.signature(  # pyright: ignore[reportUnreachable] -- The public checker targets 3.12; this branch runs on the monorepo's 3.14.
            constructor,
            annotation_format=importlib.import_module(
                "annotationlib"
            ).Format.FORWARDREF,
        )
    else:
        sig = inspect.signature(constructor)

    annotations: dict[str, object] = {}
    defaults_: dict[str, object] = {}
    owner = next(base for base in cls_.__mro__ if "__init__" in vars(base))
    localns = {
        name: value
        for base in reversed(owner.__mro__)
        for name, value in vars(base).items()
    }
    localns[owner.__name__] = owner
    for param in getattr(owner, "__type_params__", ()):
        localns[param.__name__] = param
    resolve = partial(
        _resolve_annotation,
        constructor=constructor,
        annotations={name: param.annotation for name, param in sig.parameters.items()},
        localns=localns,
    )

    for i, (param_name, param) in enumerate(sig.parameters.items()):
        if i == 0:
            if param.kind not in (
                inspect.Parameter.POSITIONAL_ONLY,
                inspect.Parameter.POSITIONAL_OR_KEYWORD,
            ):
                raise TypeError("autofig requires an explicit instance parameter.")
            continue
        if (
            hasattr(Fig, param_name)
            or param_name == "make_with_kwargs"
            or (param_name.startswith("__") and param_name.endswith("__"))
        ):
            raise TypeError(f"autofig parameter {param_name!r} conflicts with Config.")
        if param.kind not in (
            inspect.Parameter.POSITIONAL_OR_KEYWORD,
            inspect.Parameter.KEYWORD_ONLY,
        ):
            raise TypeError(
                f"autofig parameter {param_name!r} must accept a keyword argument; "
                f"{param.kind.description} parameters are unsupported."
            )
        try:
            annotations[param_name] = resolve(param_name)
        except NameError:
            # get_type_hints evaluates this expression in Config's namespace;
            # the resolver retains class locals and the constructor's live globals.
            annotations[param_name] = f"__autofig_resolve__({param_name!r})"
        if param.default is not inspect.Parameter.empty:
            defaults_[param_name] = field(
                default_factory=partial(_constructor_default, param.default)
            )

    Config = FigMeta(
        "Config",
        (Fig,),
        {
            "__annotations__": annotations,
            "__module__": cls_.__module__,
            "__qualname__": f"{cls_.__qualname__}.Config",
            "__autofig_resolve__": resolve,
            **defaults_,
        },
        require_defaults=require_defaults,
        kw_only=kw_only,
        make_with_kwargs=True,
    )

    Config.__set_name__(cls_, "Config")
    cls_.Config = Config  # pyright: ignore[reportAttributeAccessIssue] -- The decorator installs the generated Config attribute dynamically on the decorated class.  # ty: ignore[unresolved-attribute] -- The decorator installs the generated Config attribute dynamically on the decorated class.

    return cls_  # pyright: ignore[reportReturnType] -- The decorator adds Config at runtime, so the returned class satisfies HasRelaxedConfig beyond its source declaration.  # ty: ignore[invalid-return-type] -- The decorator adds Config at runtime, so the returned class satisfies HasRelaxedConfig beyond its source declaration.


def _resolve_annotation(
    name: str,
    *,
    constructor: Callable[..., object],
    annotations: dict[str, object],
    localns: dict[str, object],
) -> object:
    """Resolve one constructor annotation while retaining its original scope."""
    annotation = annotations[name]
    if annotation is inspect.Parameter.empty:
        return object
    try:
        return get_type_hints(
            SimpleNamespace(
                __annotations__={name: annotation},
                __type_params__=getattr(constructor, "__type_params__", ()),
                __no_type_check__=getattr(constructor, "__no_type_check__", False),
            ),
            globalns=getattr(inspect.unwrap(constructor), "__globals__", {}),
            localns=localns,
        ).get(name, object)
    except NameError:
        raise
    except Exception:  # noqa: BLE001 -- User annotation expressions must fail independently; missing names remain deferred.
        return object


def _constructor_default(value: object) -> object:
    """Adapt an existing constructor default to dataclasses' zero-argument factory."""
    return value
