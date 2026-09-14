"""Decorator to auto-generate a Config dataclass from __init__ parameters."""

from __future__ import annotations

from types import SimpleNamespace
from typing import TYPE_CHECKING, get_type_hints, overload

import contextlib
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

    Args:
      cls: The class to decorate (when used without arguments).
      require_defaults: If True, all Config fields must have defaults.
      kw_only: If True, all Config fields are keyword-only.

    Returns:
      decorated: The class with a nested Config dataclass attached.

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
    if sys.version_info >= (3, 14):
        sig = inspect.signature(  # pyright: ignore[reportUnreachable] -- The public checker targets 3.12; this branch runs on the monorepo's 3.14.
            cls_.__init__,
            annotation_format=importlib.import_module(
                "annotationlib"
            ).Format.FORWARDREF,
        )
    else:
        sig = inspect.signature(cls_.__init__)

    annotations: dict[str, object] = {}
    defaults_: dict[str, object] = {}
    localns = {
        name: value
        for base in reversed(cls_.__mro__)
        for name, value in vars(base).items()
    }
    localns[cls_.__name__] = cls_

    for i, (param_name, param) in enumerate(sig.parameters.items()):
        if i == 0:
            continue
        param_type = param.annotation
        if param_type is inspect.Parameter.empty:
            param_type = object
        else:
            with contextlib.suppress(NameError):
                # Resolve independently so one forward reference cannot erase
                # another parameter's already-resolvable class-local alias.
                param_type = get_type_hints(
                    SimpleNamespace(
                        __annotations__={param_name: param_type},
                        __type_params__=getattr(cls_.__init__, "__type_params__", ()),
                        __no_type_check__=getattr(
                            cls_.__init__, "__no_type_check__", False
                        ),
                    ),
                    globalns=getattr(inspect.unwrap(cls_.__init__), "__globals__", {}),
                    localns=localns,
                ).get(param_name, object)
        annotations[param_name] = param_type
        if param.default is not inspect.Parameter.empty:
            defaults_[param_name] = param.default

    Config = FigMeta(
        "Config",
        (Fig,),
        {
            "__annotations__": annotations,
            "__module__": cls_.__module__,
            **defaults_,
        },
        require_defaults=require_defaults,
        kw_only=kw_only,
        make_with_kwargs=True,
    )

    Config.__set_name__(cls_, "Config")
    cls_.Config = Config  # pyright: ignore[reportAttributeAccessIssue] -- The decorator installs the generated Config attribute dynamically on the decorated class.  # ty: ignore[unresolved-attribute] -- The decorator installs the generated Config attribute dynamically on the decorated class.

    return cls_  # pyright: ignore[reportReturnType] -- The decorator adds Config at runtime, so the returned class satisfies HasRelaxedConfig beyond its source declaration.  # ty: ignore[invalid-return-type] -- The decorator adds Config at runtime, so the returned class satisfies HasRelaxedConfig beyond its source declaration.
