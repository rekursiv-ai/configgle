from __future__ import annotations

from types import ModuleType
from typing import TypeVar, get_args, get_type_hints, no_type_check

import sys

import pytest

from configgle.cli_override import apply_overrides
from configgle.decorator import autofig
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
    config = module.Node.Config()
    apply_overrides(config, ['count="7"'])
    assert config.count == 7
    assert type(config.count) is int
    assert (
        get_type_hints(module.Node.Config)["child"]
        == getattr(module, child_name) | None
    )
    assert config.make().child is None
    with pytest.raises(ValueError, match="count"):
        apply_overrides(config, ["count=invalid"])


def test_class_local_annotation_preserves_scalar_override_type() -> None:
    """Constructor annotations can refer to aliases in their owning class."""

    @autofig
    class Scoped:
        Count = int

        def __init__(self, count: Count = 1):
            self.count = count

    config = Scoped.Config()
    apply_overrides(config, ['count="7"'])
    assert config.count == 7
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
    assert config.count == 7
    assert get_type_hints(Derived.Config)["count"] is int


def test_constructor_type_parameter_is_preserved() -> None:
    """Resolving a parameter retains the constructor's generic type scope."""

    @autofig
    class Generic:
        def __init__[T](self, value: T | None = None, fallback: T | None = None):
            self.value = value
            self.fallback = fallback

    annotation = get_type_hints(Generic.Config)["value"]
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
    assert Foo.Config.parent_class == Foo

    config = Foo.Config(x=42, y="hello", z=3.14)
    assert config.x == 42
    assert config.y == "hello"
    assert config.z == 3.14

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
    assert config.x == 99
    assert config.y == "a"


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

    assert Bar.Config.parent_class == Bar

    config = Bar.Config(items=[1, 2, 3], name="test")
    assert config.items == [1, 2, 3]
    assert config.name == "test"
    assert config.count == 5

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


if __name__ == "__main__":
    from configgle.lib.testing.main import test_main

    test_main(__file__)
