from __future__ import annotations

from collections.abc import Callable
from types import ModuleType
from typing import (
    TYPE_CHECKING,
    Protocol,
    TypeVar,
    cast,
    get_args,
    get_type_hints,
    no_type_check,
)

import pickle
import sys

import pytest

from configgle.cli_override import apply_overrides
from configgle.decorator import autofig


if TYPE_CHECKING:
    from configgle.custom_types import HasRelaxedConfig
from configgle.fig import Fig


# ``ty`` regression canary: as of ty 0.0.49 (astral-sh/ty#143) class decorator
# return types are honored, so `.Config` resolves with no suppression here.
# If ty regresses, this line emits unresolved-attribute and ty check fails.
@autofig
class _Canary:
    def __init__(self, x: int = 0):
        self.x = x


_canary_config = _Canary.Config(x=1)


@pytest.mark.parametrize("child_name", ["Node", "Later"])
@pytest.mark.parametrize("future_annotations", [True, False])
def test_forward_reference_preserves_annotations(
    monkeypatch: pytest.MonkeyPatch,
    child_name: str,
    future_annotations: bool,
) -> None:
    """Forward references preserve scalar coercion and resolve after import."""
    preamble = "from __future__ import annotations\n" if future_annotations else ""
    if not future_annotations and sys.version_info < (3, 14):
        pytest.skip("Native deferred annotations require Python 3.14.")
    module = ModuleType("_autofig_forward_test")
    monkeypatch.setitem(sys.modules, module.__name__, module)
    source = (
        preamble
        + f"""
from configgle.decorator import autofig

@autofig
class Node:
    Count = int

    def __init__(self, count: Count = 1, child: {child_name} | None = None):
        self.count = count
        self.child = child

class Later:
    pass
"""
    )
    exec(  # noqa: S102 -- A fresh module must run the decorator before its forward references are bound.
        compile(source, "<autofig-forward-test>", "exec", dont_inherit=True),
        vars(module),
    )
    node = cast("type[HasRelaxedConfig[object]]", module.Node)
    config = node.Config()
    apply_overrides(config, ['count="7"'])
    assert _field(config, "count") == 7
    assert type(_field(config, "count")) is int
    child_hint = cast(object, get_type_hints(node.Config)["child"])
    child_type = cast(object, getattr(module, child_name))
    assert isinstance(child_type, type)
    assert child_hint == _union(child_type, type(None))
    assert _field(config.make(), "child") is None
    with pytest.raises(ValueError, match="count"):
        apply_overrides(config, ["count=invalid"])


def _field(obj: object, name: str) -> object:
    """Read a field the decorator generated at runtime, so the checker can't see it."""
    return cast(object, getattr(obj, name))


class _SupportsOr(Protocol):
    def __or__(self, other: object, /) -> object: ...


def _union(left: object, right: object) -> object:
    return cast(_SupportsOr, left) | right


def test_class_local_annotation_preserves_scalar_override_type() -> None:
    """Constructor annotations can refer to aliases in their owning class."""

    @autofig
    class Scoped:
        Count = int

        def __init__(self, count: Count = 1):
            self.count = count

    config = Scoped.Config()
    apply_overrides(config, ['count="7"'])
    assert _field(config, "count") == 7
    assert get_type_hints(Scoped.Config)["count"] is int


def test_inherited_class_local_annotation() -> None:
    """Inherited constructors retain the aliases defined by their base class."""

    class Base:
        Count = int

        def __init__(self, count: Count = 1):
            self.count = count

    @autofig
    class Derived(Base):
        pass

    config = Derived.Config()
    apply_overrides(config, ['count="7"'])
    assert _field(config, "count") == 7
    assert get_type_hints(Derived.Config)["count"] is int


@pytest.mark.parametrize("future_annotations", [True, False])
def test_inherited_annotation_ignores_subclass_shadow(
    monkeypatch: pytest.MonkeyPatch,
    future_annotations: bool,
) -> None:
    """Subclass aliases cannot change an inherited constructor's contract."""
    preamble = "from __future__ import annotations\n" if future_annotations else ""
    if not future_annotations and sys.version_info < (3, 14):
        pytest.skip("Native deferred annotations require Python 3.14.")
    module = ModuleType("_autofig_shadow_test")
    monkeypatch.setitem(sys.modules, module.__name__, module)
    source = (
        preamble
        + """
from configgle.decorator import autofig

class Base:
    Count = int

    def __init__(self, count: Count = 1, peer: Base | None = None):
        self.count = count
        self.peer = peer

class Shadow:
    Count = str

@autofig
class Derived(Shadow, Base):
    Count = str
"""
    )
    exec(  # noqa: S102 -- Separate compilation exercises both annotation evaluation models.
        compile(source, "<autofig-shadow-test>", "exec", dont_inherit=True),
        vars(module),
    )
    derived = cast("type[HasRelaxedConfig[object]]", module.Derived)
    config = derived.Config()
    apply_overrides(config, ['count="7"'])
    assert type(_field(config, "count")) is int
    assert _field(config.make(), "count") == 7
    base = cast(type, module.Base)
    assert get_type_hints(derived.Config)["peer"] == base | None


@pytest.mark.parametrize(
    "annotation", ["int | 1", "'int['", "int.missing", "int('bad')", "'1 / 0'"]
)
@pytest.mark.parametrize("future_annotations", [True, False])
def test_annotation_failure_isolated(
    monkeypatch: pytest.MonkeyPatch,
    annotation: str,
    future_annotations: bool,
) -> None:
    """An unusable annotation leaves other fields typed and configurable."""
    preamble = "from __future__ import annotations\n" if future_annotations else ""
    if not future_annotations and sys.version_info < (3, 14):
        pytest.skip("Native deferred annotations require Python 3.14.")
    module = ModuleType("_autofig_invalid_test")
    monkeypatch.setitem(sys.modules, module.__name__, module)
    source = (
        preamble
        + f"""
from configgle.decorator import autofig

@autofig
class Broken:
    Count = int

    def __init__(self, count: Count = 1, broken: {annotation} = None):
        self.count = count
        self.broken = broken
"""
    )
    exec(  # noqa: S102 -- Invalid annotations must reach the runtime decorator without static checking.
        compile(source, "<autofig-invalid-test>", "exec", dont_inherit=True),
        vars(module),
    )
    broken = cast("type[HasRelaxedConfig[object]]", module.Broken)
    config = broken.Config()
    apply_overrides(config, ['count="7"', 'broken="kept"'])
    assert type(_field(config, "count")) is int
    assert _field(config.make(), "count") == 7
    assert _field(config.make(), "broken") == "kept"
    assert get_type_hints(broken.Config)["broken"] is object
    with pytest.raises(ValueError, match="count"):
        apply_overrides(config, ["count=invalid"])


def test_constructor_type_parameter_is_preserved() -> None:
    """Resolving a parameter retains the constructor's generic type scope."""

    @autofig
    class Generic:
        def __init__[T](self, value: T | None = None, fallback: T | None = None):
            self.value = value
            self.fallback = fallback

    annotation = cast(object, get_type_hints(Generic.Config)["value"])
    assert isinstance(get_args(annotation)[0], TypeVar)
    assert Generic.Config().make().value is None


def test_constructor_type_check_opt_out_is_preserved() -> None:
    """An explicit typing opt-out retains the existing untyped field contract."""

    @autofig
    class Untyped:
        @no_type_check
        def __init__(self, count: int = 1):
            self.count = count

    assert get_type_hints(Untyped.Config)["count"] is object


def test_basic_decorator():
    @autofig
    class Foo:
        def __init__(self, x: int = 1, y: str = "default", z: float = 0.0):
            self.x = x
            self.y = y
            self.z = z

    assert Foo.Config.__bases__ == (Fig,)
    assert issubclass(Foo.Config, Fig)
    assert Foo.Config.parent_class == Foo

    config = Foo.Config(x=42, y="hello", z=3.14)
    assert _field(config, "x") == 42
    assert _field(config, "y") == "hello"
    assert _field(config, "z") == 3.14

    foo = config.make()
    assert foo.x == 42
    assert foo.y == "hello"
    assert foo.z == 3.14


def test_config_update():
    @autofig
    class Foo:
        def __init__(self, x: int = 0, y: str = ""):
            self.x = x
            self.y = y

    config = Foo.Config(x=1, y="a")
    config.update(x=99)
    assert _field(config, "x") == 99
    assert _field(config, "y") == "a"


def test_with_defaults():
    @autofig
    class Bar:
        def __init__(
            self,
            items: list[int] | None = None,
            name: str = "",
            count: int = 5,
        ):
            self.items = items if items is not None else []
            self.name = name
            self.count = count

    assert issubclass(Bar.Config, Fig)
    assert Bar.Config.parent_class == Bar

    config = Bar.Config(items=[1, 2, 3], name="test")
    assert _field(config, "items") == [1, 2, 3]
    assert _field(config, "name") == "test"
    assert _field(config, "count") == 5

    bar = config.make()
    assert bar.items == [1, 2, 3]
    assert bar.name == "test"
    assert bar.count == 5


def test_original_init_preserved():
    @autofig
    class Baz:
        def __init__(self, a: int = 0, b: str = ""):
            self.a = a
            self.b = b

    baz = Baz(a=10, b="direct")  # pyright: ignore[reportCallIssue] -- The decorator test passes a runtime-only fixture argument.  # ty: ignore[unknown-argument] -- The decorator test passes a runtime-only fixture argument.
    assert baz.a == 10  # pyright: ignore[reportAttributeAccessIssue, reportUnknownMemberType] -- The dynamic decorator fixture exposes members created at runtime.  # ty: ignore[unresolved-attribute] -- The dynamic decorator fixture exposes members created at runtime.
    assert baz.b == "direct"  # pyright: ignore[reportAttributeAccessIssue, reportUnknownMemberType] -- The dynamic decorator fixture exposes members created at runtime.  # ty: ignore[unresolved-attribute] -- The dynamic decorator fixture exposes members created at runtime.


def test_require_defaults():
    """Test that autofig with require_defaults=False allows parameters without defaults."""

    @autofig(require_defaults=False)
    class NoDefaults:
        def __init__(self, x: int):
            self.x = x

    # Should work - require_defaults=False allows parameters without defaults.
    config = NoDefaults.Config(x=42)
    instance = config.make()
    assert instance.x == 42


def test_autofig_with_broken_type_hints():
    """Test autofig when get_type_hints fails (e.g., unresolvable forward refs)."""
    # ``exec`` creates a class whose annotations reference 'Nonexistent' --
    # a name absent from the exec namespace -- so get_type_hints will raise.
    ns: dict[str, object] = {}
    exec(  # noqa: S102 -- The test deliberately exercises the dynamic decorator protocol.
        "class B:\n    def __init__(self, x: 'Nonexistent' = 0):\n        self.x = x\n",
        ns,
    )
    Cls = ns["B"]
    decorated = autofig(Cls)  # pyright: ignore[reportCallIssue, reportArgumentType, reportUnknownVariableType] -- The decorator test uses a dynamic callable fixture outside the stub's overloads.  # ty: ignore[no-matching-overload] -- exec erases the generated class type.
    config = decorated.Config(x=42)  # pyright: ignore[reportUnknownVariableType, reportUnknownMemberType] -- The dynamic decorator fixture has no static member metadata.
    assert config.make().x == 42  # pyright: ignore[reportUnknownMemberType] -- The dynamic decorator fixture has no static member metadata.


@pytest.mark.parametrize("future_annotations", [True, False])
def test_deferred_annotations_keep_constructor_scope(
    monkeypatch: pytest.MonkeyPatch,
    future_annotations: bool,
) -> None:
    """A forward reference retains aliases in the same annotation expression."""
    preamble = "from __future__ import annotations\n" if future_annotations else ""
    if not future_annotations and sys.version_info < (3, 14):
        pytest.skip("Native deferred annotations require Python 3.14.")
    module = _compile_module(
        preamble
        + """
from configgle.decorator import autofig

@autofig
class Node:
    class Child:
        pass

    def __init__(self, child: list[Child | Later] | None = None):
        self.child = child

class Later:
    pass
""",
        monkeypatch=monkeypatch,
    )
    node = cast("type[HasRelaxedConfig[object]]", module.Node)
    child_name = "Child"
    child = cast(type, getattr(node, child_name))
    later_name = "Later"
    later = cast(type, getattr(module, later_name))
    hint = cast(object, get_type_hints(node.Config)["child"])
    outer_args = get_args(hint)
    assert outer_args[1] is type(None)
    assert get_args(outer_args[0]) == (_union(child, later),)


@pytest.mark.parametrize("future_annotations", [True, False])
def test_class_generic_parameter_scope(
    monkeypatch: pytest.MonkeyPatch,
    future_annotations: bool,
) -> None:
    """Class type parameters have the same identity in generated annotations."""
    preamble = "from __future__ import annotations\n" if future_annotations else ""
    if not future_annotations and sys.version_info < (3, 14):
        pytest.skip("Native deferred annotations require Python 3.14.")
    module = _compile_module(
        preamble
        + """
from configgle.decorator import autofig

@autofig
class Node[T]:
    def __init__(self, value: T | None = None):
        self.value = value
""",
        monkeypatch=monkeypatch,
    )
    node = cast("type[HasRelaxedConfig[object]]", module.Node)
    type_params = cast(tuple[object, ...], node.__type_params__)
    assert get_args(get_type_hints(node.Config)["value"])[0] is type_params[0]


@pytest.mark.parametrize("signature", ["value: int = 1, /", "*args", "**kwargs"])
def test_unsupported_parameter_kinds_rejected(
    monkeypatch: pytest.MonkeyPatch,
    signature: str,
) -> None:
    """Reject unsupported argument binding before publishing a broken Config."""
    module = _compile_module(
        f"""
class Node:
    def __init__(self, {signature}):
        pass
""",
        monkeypatch=monkeypatch,
    )
    node = cast(type, module.Node)
    with pytest.raises(TypeError, match=r"autofig.*parameter"):
        autofig(node)
    assert not hasattr(node, "Config")


def test_inherited_forward_annotation_uses_defining_module(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """An inherited unresolved type resolves where the constructor was defined."""
    base = _compile_module(
        """
from __future__ import annotations
class Base:
    Count = int
    def __init__(self, value: Count | Later | None = None):
        self.value = value
""",
        monkeypatch=monkeypatch,
        name="_autofig_base_test",
    )
    derived = _compile_module(
        """
from _autofig_base_test import Base
from configgle.decorator import autofig
@autofig
class Derived(Base):
    Count = str
class Later:
    pass
""",
        monkeypatch=monkeypatch,
    )
    exec("class Later: pass", vars(base))  # noqa: S102 -- The referenced type becomes available after decoration.
    derived_cls = cast("type[HasRelaxedConfig[object]]", derived.Derived)
    later = cast(object, base.Later)
    assert get_type_hints(derived_cls.Config)["value"] == _union(
        _union(int, later), type(None)
    )


def test_config_round_trips_through_pickle(monkeypatch: pytest.MonkeyPatch) -> None:
    """A module-level autofig Config exposes its real import path."""
    module = _compile_module(
        """
from configgle.decorator import autofig
@autofig
class Node:
    def __init__(self, count: int = 1):
        self.count = count
""",
        monkeypatch=monkeypatch,
    )
    node = cast("type[HasRelaxedConfig[object]]", module.Node)
    config = node.Config()
    count_name = "count"
    setattr(config, count_name, 7)
    restored = cast(object, pickle.loads(pickle.dumps(config)))
    assert type(restored) is node.Config
    make_restored = cast("Callable[[], object]", _field(restored, "make"))
    assert _field(make_restored(), "count") == 7
    serialize = cast("Callable[[], object]", config.serialize)
    deserialize_name = "deserialize"
    deserialize = cast(
        "Callable[[object], object]", getattr(node.Config, deserialize_name)
    )
    decoded = deserialize(serialize())
    assert type(decoded) is node.Config
    make_decoded = cast("Callable[[], object]", _field(decoded, "make"))
    assert _field(make_decoded(), "count") == 7


def test_mutable_constructor_defaults_preserve_constructor_semantics(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Omitted defaults retain the same identity as ordinary constructor calls."""
    module = _compile_module(
        """
from configgle.decorator import autofig
@autofig
class Node:
    def __init__(self, values: list[int] = [1]):
        self.values = values
""",
        monkeypatch=monkeypatch,
    )
    node = cast("type[HasRelaxedConfig[object]]", module.Node)
    first, second = node.Config(), node.Config()
    direct = node()
    assert _field(first, "values") is _field(second, "values")
    assert _field(first, "values") is _field(direct, "values")
    cast(list[int], _field(first, "values")).append(2)
    assert _field(second, "values") == [1, 2]
    assert _field(direct, "values") == [1, 2]
    assert _field(first.make(), "values") == [1, 2]


def test_shared_constructor_defaults_keep_aliases(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Two parameters sharing one nested default also share it in the Config."""
    module = _compile_module(
        """
from configgle.decorator import autofig
shared = ([1],)
@autofig
class Node:
    def __init__(self, left: tuple[list[int]] = shared, right: tuple[list[int]] = shared):
        self.left = left
        self.right = right
""",
        monkeypatch=monkeypatch,
    )
    node = cast("type[HasRelaxedConfig[object]]", module.Node)
    direct = node()
    assert _field(direct, "left") is _field(direct, "right")
    config = node.Config()
    assert _field(config, "left") is _field(config, "right")
    assert _field(config, "left") is _field(direct, "left")
    cast(tuple[list[int]], _field(config, "left"))[0].append(2)
    assert _field(config, "right") == ([1, 2],)
    explicit = ([7],)
    overridden = node.Config(left=explicit)
    assert _field(overridden, "left") is explicit
    shared_name = "shared"
    assert _field(overridden, "right") is getattr(module, shared_name)


@pytest.mark.parametrize(
    "name",
    ["make", "finalize", "_finalized", "__autofig_resolve__", "make_with_kwargs"],
)
def test_reserved_parameter_names_are_rejected(
    monkeypatch: pytest.MonkeyPatch,
    name: str,
) -> None:
    """Constructor fields cannot replace Config's construction machinery."""
    module = _compile_module(
        f"""
class Node:
    def __init__(self, {name}: int = 1):
        pass
""",
        monkeypatch=monkeypatch,
    )
    node = cast(type, module.Node)
    with pytest.raises(TypeError, match=r"autofig.*conflict"):
        autofig(node)
    assert not hasattr(node, "Config")


def test_empty_constructor() -> None:
    """A class with object.__init__ can build through an empty Config."""

    class Empty:
        pass

    decorated = autofig(Empty)
    assert isinstance(decorated.Config().make(), Empty)


@pytest.mark.parametrize(
    "constructor",
    [
        "@staticmethod\n    def __init__(value: int = 1): pass",
        "@classmethod\n    def __init__(cls, value: int = 1): pass",
        "def __init__(*args): pass",
    ],
)
def test_unsupported_constructor_binding_rejected(
    monkeypatch: pytest.MonkeyPatch,
    constructor: str,
) -> None:
    """Unsupported receiver binding cannot silently discard a config parameter."""
    module = _compile_module(
        f"class Node:\n    {constructor}\n", monkeypatch=monkeypatch
    )
    node = cast(type, module.Node)
    with pytest.raises(TypeError, match="autofig"):
        autofig(node)
    assert not hasattr(node, "Config")


def _compile_module(
    source: str,
    *,
    monkeypatch: pytest.MonkeyPatch,
    name: str = "_autofig_test",
) -> ModuleType:
    module = ModuleType(name)
    monkeypatch.setitem(sys.modules, name, module)
    exec(  # noqa: S102 -- Dynamic source exercises runtime annotations without static-checker interference.
        compile(source, "<autofig-test>", "exec", dont_inherit=True),
        vars(module),
    )
    return module


if __name__ == "__main__":
    from configgle.lib.testing.main import test_main

    test_main(__file__)
