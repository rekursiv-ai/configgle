# ty Missing Features / Limitations

Type-checker limitations found while making configgle work under `ty`,
`basedpyright`, and `pyrefly`, plus the upstream fixes that closed them.

`Intersection`-based `make()` typing (see `MakerMeta.__get__` in `fig.py`)
reaches corners of the type system most libraries never touch, so several
behaviors configgle depends on were fixed upstream rather than worked around
here.

**Required:** `ty>=0.0.60` (declared in `pyproject.toml`) -- the first release
carrying [ruff#26649](https://github.com/astral-sh/ruff/pull/26649), general
`type[Protocol]` support, without which configgle's `Intersection` does not
simplify. **Tested on:** 0.0.65 with `basedpyright` 1.39.9. Every status below
was re-verified against 0.0.65 on 2026-08-05 by running the example.

---

## Open

### 1. TypeIs does not narrow to intersection type

**Status:** Open. Requires suppression.

`TypeIs` (PEP 742) should narrow to the intersection of the declared type and
the guard type. `ty` narrows to just the guard type, losing the original.

```python
def process(value: _T) -> _T:
    if needs_finalize(value):
        # basedpyright: value is _T & Finalizable, returns _T OK
        # ty: value is Finalizable, returns Finalizable FAIL
        return value.finalize()
    return value
```

On 0.0.65: `warning[invalid-return-type] Return type does not match returned
value: expected _T@process, found Self@finalize`.

**Workaround:** `# ty: ignore[invalid-return-type]` on the affected return.

### 2. `hasattr()` does not narrow type

**Status:** Open. Requires suppression.

After `if hasattr(x, "method")`, `ty` still sees the original type.

```python
def process(v: object) -> object:
    if hasattr(v, "make"):
        return v.make()  # ty: Object of type `object` is not callable
    return v
```

On 0.0.65: `warning[call-non-callable]`.

**Workaround:** Use `isinstance()` with a `runtime_checkable` Protocol -- what
configgle does with `Finalizeable` and `MutableNamespace` in `custom_types.py`.

### 3. Generic proxy subscript operations on a type variable

**Status:** Open. Requires suppression.

A generic proxy forwarding `__getitem__`/`__setitem__` to a wrapped object
errors because `_T` may not support the operation.

```python
class CopyOnWrite(Generic[_T]):
    __wrapped__: _T

    def __getitem__(self, key: object) -> object:
        return self.__wrapped__[key]  # ty: Cannot subscript _T
```

On 0.0.65: `warning[not-subscriptable]`.

**Workaround:** `# ty: ignore[not-subscriptable]` or
`# ty: ignore[invalid-assignment]`.

### 4. Protocol decorated with `@dataclass` flagged as invalid

**Status:** Open upstream; avoided in configgle.

`ty` errors when a Protocol is decorated with `@dataclass`, even with
`init=False, repr=False, eq=False`. On 0.0.65:
`warning[invalid-dataclass] Protocol class ... cannot be decorated with
@dataclass`.

**Resolution here:** `DataclassLike` (`custom_types.py`) declares an explicit
`__dataclass_fields__` ClassVar instead of using the decorator.

### 5. TypeVars in `ClassVar` are spec-illegal but semantically needed

**Status:** Open by design; suppressed deliberately.

PEP 526 forbids type variables inside `ClassVar`, but a Protocol needs a
class-level attribute whose type varies per parameterization -- something the
type system cannot express. `ty`, `basedpyright`, and `pyrefly` all reject it;
all three suppressions are intentional. `pyrefly` additionally refuses to read
`Config` off such a Protocol at all ("Generic attribute ... is not visible on
the class"), so code that only needs the class's shape casts to a non-generic
Protocol with `Config: ClassVar[type[RelaxedMakeable[object]]]` instead. See `HasConfig`, `RelaxedMakeable`, and `HasRelaxedConfig` in
`custom_types.py`, where the alternatives (drop `ClassVar`, use `@property`)
are documented inline.

### 6. TypeVar polluted by a `T | None` annotation into a callable parameter

**Status:** Open --
[ty#4016](https://github.com/astral-sh/ty/issues/4016) (filed by us,
2026-07-16; labels: generics, bidirectional inference, callables).

A `T | None` return/default annotation leaks `None` into the solution for a
`TypeVar` bound by a callable parameter.

### 7. `pyrefly` ignores class decorator return types

**Status:** Open --
[pyrefly#1180](https://github.com/facebook/pyrefly/issues/1180), fix pending in
[pyrefly#2294](https://github.com/facebook/pyrefly/pull/2294).

The `pyrefly` counterpart of the closed `ty#143`: `@autofig class Foo` stays
`type[Foo]`, so `Foo.Config` is unresolved. Calling `autofig(Foo)` as a function
is typed correctly.

**Workaround:** `# pyrefly: ignore[missing-attribute]` where `.Config` is read
straight off a decorated class.

---

## Fixed upstream

Filed from configgle's use of `Intersection`, metaclass descriptors, and
covariant protocols. All verified closed.

| Issue | What it blocked | Closed |
|---|---|---|
| [ty#143](https://github.com/astral-sh/ty/issues/143) | Class decorator return types were ignored, so `@autofig`'s `.Config` was unresolvable. Fixed via [ruff#22375](https://github.com/astral-sh/ruff/pull/22375), released in 0.0.49 | 2026-05-19 |
| [ty#3279](https://github.com/astral-sh/ty/issues/3279) | A metaclass `__get__` under `TYPE_CHECKING` made the class unusable as a base -- exactly `MakerMeta`'s shape | 2026-04-15 |
| [ty#3282](https://github.com/astral-sh/ty/issues/3282) | Metaclass lookup through intersection-typed bases, so an inherited Config keeps its child fields | 2026-06-19 |
| [ty#3835](https://github.com/astral-sh/ty/issues/3835) | Panic on a recursive PEP-695 alias mixing covariant and invariant generics (0.0.52 regression); `Makeable[_T_co]` triggered it | 2026-06-23 |

Also fixed: `Final[T]` in a Protocol without a value once required an
assignment. It type-checks clean on 0.0.65, so configgle's `_finalized: bool`
no longer needs to avoid `Final` for that reason.

### Patches we contributed

| PR | What it does | Status |
|---|---|---|
| [ruff#26545](https://github.com/astral-sh/ruff/pull/26545) | Narrow `isinstance` against intersections containing an invalid member -- the `isinstance(x, Makeable)` path | Merged 2026-07-07, shipped in 0.0.57 |
| [jsonpickle#611](https://github.com/jsonpickle/jsonpickle/pull/611) | Escape reserved-tag dict keys before the picklability check (they were silently dropped) | Merged 2026-07-07 |
| [ruff#26553](https://github.com/astral-sh/ruff/pull/26553) | Filter trivial `object` constructors from `type[...]` intersection member lookup and calls | Closed unmerged |
| [ruff#26571](https://github.com/astral-sh/ruff/pull/26571) | Allow `type[C]` assignable to `type[Protocol]` via structural subtyping | Closed unmerged |

The last two were closed because `ty` gained general
[`type[Protocol]` support](https://github.com/astral-sh/ruff/pull/26649)
(merged 2026-07-14, shipped in 0.0.60), which simplifies the intersection and
dissolves both problems at the source rather than patching around them. As
carljm put it on #26553: "configgle no longer encounters the problem because
after #26649 the intersection (which involves a `type[Proto]`) now simplifies."
That is why the package floor is 0.0.60 and not the 0.0.49 that first resolved
`@autofig`.

The jsonpickle patch predates the decision to ship a bespoke serializer; see
[jsonpickle.md](jsonpickle.md) for why configgle keeps that wire format but not
that dependency.

---

## Summary

| Limitation | Severity | Workaround |
|---|---|---|
| TypeIs intersection | Medium | Suppress `invalid-return-type` |
| `hasattr` narrowing | Medium | `isinstance` + runtime_checkable Protocol |
| Generic proxy subscript | Medium | Suppress `not-subscriptable` |
| Protocol + `@dataclass` | Low | Declare `__dataclass_fields__` explicitly |
| TypeVar in `ClassVar` | Low | Deliberate suppression, all three checkers |
| TypeVar pollution (#4016) | Medium | None yet |
| `pyrefly` class decorators (#1180) | Low | Suppress `missing-attribute` at direct `.Config` reads |
