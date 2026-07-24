# ty Missing Features / Limitations

This document tracks ty type checker limitations discovered while making configgle compatible with both ty and basedpyright.

**ty version tested:** 0.0.49

---

## 1. Decorator return types not honored

**Status:** Fixed in ty 0.0.49 ([astral-sh/ty#143](https://github.com/astral-sh/ty/issues/143), via [astral-sh/ruff#22375](https://github.com/astral-sh/ruff/pull/22375))

**Description:** ty previously did not honor decorator return type annotations. When a decorator declared it returns a different type than the input, ty ignored this and used the original class type. As of 0.0.49, ty honors the declared return type, so `.Config` access on `@autofig`-decorated classes resolves with no suppression.

**Example:**
```python
from typing import TypeVar, Protocol, ClassVar

_T = TypeVar("_T")


class HasConfig(Protocol[_T]):
    Config: ClassVar[type[_T]]


def decorator(cls: type[_T]) -> type[HasConfig[_T]]:
    cls.Config = type("Config", (), {})
    return cls  # type: ignore


@decorator
class Foo:
    pass


# basedpyright: Foo is type[HasConfig[Foo]] ✓
# ty: Foo is <class 'Foo'> ✗

Foo.Config  # ty error: Class `Foo` has no attribute `Config`
```

**Impact (historical):** Users of `@autofig` saw `unresolved-attribute` errors when accessing `.Config` on decorated classes. No longer occurs on ty ≥ 0.0.49.

---

## 2. TypeIs does not narrow to intersection type

**Status:** Minor (workaround available)

**Description:** `TypeIs` (PEP 742) should narrow a type to the intersection of the original type and the guard type. ty narrows to just the guard type, losing the original type information.

**Example:**
```python
from typing_extensions import TypeIs


class Finalizable(Protocol):
    def finalize(self) -> Self: ...


_T = TypeVar("_T")


def needs_finalize(x: object) -> TypeIs[Finalizable]:
    return hasattr(x, "finalize")


def process(value: _T) -> _T:
    if needs_finalize(value):
        # basedpyright: value is _T & Finalizable, returns _T ✓
        # ty: value is Finalizable, returns Finalizable ✗
        return value.finalize()
    return value
```

**Impact:** Functions that use `TypeIs` for type narrowing show return type mismatches.

**Workaround:** Add `# ty: ignore[invalid-return-type]` on affected return statements.

---

## 3. Protocol decorated with @dataclass flagged as invalid

**Status:** Fixed in configgle

**Description:** ty reports an error when a Protocol is decorated with `@dataclass`, even with `init=False, repr=False, eq=False`. This pattern is used to make protocols `runtime_checkable` with dataclass-like behavior.

**Example:**
```python
@runtime_checkable
@dataclasses.dataclass(init=False, repr=False, eq=False)
class DataclassLike(Protocol):
    pass


# ty error: Protocol class `DataclassLike` cannot be decorated with `@dataclass`
```

**Resolution:** Removed `@dataclass` decorator, added explicit `__dataclass_fields__` class variable instead.

---

## 4. `Final[T]` in Protocol requires value

**Status:** Fixed in configgle

**Description:** ty requires `Final` annotations to have values, even in Protocol definitions where they serve as interface declarations.

**Example:**
```python
class Configurable(Protocol):
    _finalized: Final[bool]  # ty error: Final symbol not assigned a value
```

**Resolution:** Changed to `_finalized: bool` (removed `Final`).

---

## 5. Generic proxy subscript operations on type variable

**Status:** Requires suppression

**Description:** When implementing a generic proxy class `Proxy[_T]` that forwards `__getitem__`/`__setitem__`/`__delitem__` to a wrapped object, ty reports errors because `_T` may not support these operations.

**Example:**
```python
class CopyOnWrite(Generic[_T]):
    __wrapped__: _T

    def __getitem__(self, key: object) -> object:
        return self.__wrapped__[key]  # ty error: Cannot subscript _T
```

**Impact:** Proxy patterns require suppression comments.

**Workaround:** Add `# ty: ignore[not-subscriptable]` or `# ty: ignore[invalid-assignment]`.

---

## 6. `hasattr()` does not narrow type

**Status:** Requires suppression

**Description:** ty does not narrow types based on `hasattr()` checks. After `if hasattr(x, "method")`, ty still sees `x` as the original type without the method.

**Example:**
```python
def process(v: object) -> object:
    if hasattr(v, "make"):
        return v.make()  # ty error: Object of type `object` is not callable
    return v
```

**Workaround:** Use `isinstance()` with a `runtime_checkable` Protocol, or add `# ty: ignore[call-non-callable]`.

---

## Summary Table

| Issue | Severity | Workaround Available |
|-------|----------|---------------------|
| Decorator return types | Fixed (0.0.49) | n/a |
| TypeIs intersection | Medium | Suppress comment |
| Protocol + @dataclass | Low | Remove decorator |
| Final in Protocol | Low | Remove Final |
| Generic proxy subscript | Medium | Suppress comment |
| hasattr narrowing | Medium | Use isinstance or suppress |
