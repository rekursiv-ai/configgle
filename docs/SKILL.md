---
name: configgle
description: ALWAYS invoke this skill before defining a class with configurable parameters, any `Fig[...]`/`Makes[...]` subclass, `Makeable` pipelines, or nested Config hierarchies, and when reviewing such code. Do not write configgle classes -- invoke first.
---

_tl;dr:_  The ENTIRE PURPOSE OF CONFIGGLE is define 100% of the STATE OF AN
EXPERIMENT and achieve transparent, maintainable, self-documenting experiments
in the presence of 1000s or even millions of experiments. Configgle makes A/B
experimentation easy to: write, reproduce, and debug. For
debugging/understanding a `configgle.Fig` descendant, the single most important
technique is `cfg.pprint(hide_default_values=False)`.

## House rules (non-negotiable)

These are the project's stipulated guidelines for configgle. Code review
rejects violations. Mental model: **configgle is typed, dependency-injectable
YAML embedded in Python.** A variant is a different *value* or a different
*injected piece* -- never a new flag or a patched global.

If a value is wrong, **change the value** or **inject different capability** --
never add a global, a helper, or a build step to compute it. The config tree is
the source of truth; reaching for `scratch_dir(...)`, a `*_dir()` helper, or an
in-`finalize` build is fighting the pattern.

1. **Construct, then mutate.** `cfg = Foo.Config()`, then `cfg.foo.bar = 42` on
   later lines -- not `Foo.Config(foo=Bar.Config(bar=42))`. *Exception:* a
   generic/typed slot set at construction so the type narrows -- e.g.
   `TrainLoop.Config(step=TRMTrainStep.Config(), dataset=PuzzleDataset.Config())`.
2. **Set only diffs from defaults.** `cfg.x = <default>` is noise -- delete it.
   Confirm the default with `Foo.Config().pprint(hide_default_values=False)`.
3. **Never monkeypatch; tests assert the *finalized* config.** Inject the `Foo`
   you need into a slot, or set fixture values on the config. Missing a hook?
   *Add the hook* -- don't `monkeypatch.setattr` a `module.ensure_*` / `*_dir` a
   config calls. Want a fake dataset/model? Roll a tiny Config into the tree.
   Assert the *resolved* value (`runtime.scratch_dir = tmp_path`, finalize,
   assert the `Path`) -- never the raw `"{scratch_dir}/..."` template.
4. **Don't bloat a Config.** A variant is a new small injected Config, not a pile
   of flags/branches on an existing class. Need a test variant? Roll/inject one.
   This holds at every scale: a whole new problem is also values in existing
   slots, so reaching for a new module -- or a new shared layer -- to hold a
   different size or vocabulary is the same mistake as adding a flag.
5. **No global or environment state.** Don't define module-level variables or
   read environment variables to determine Config values; write every
   experiment-affecting value as a Config field. No `scratch_dir(...)` pulls,
   `DATA_ROOT = Path(scratch_dir(...))`, `os.environ`, or parameterless
   `ensure_*()` baking in a global path. Runtime data uses the fixed logical
   `/opt/scratch` namespace. Path fields type `str | Path | None`: a `Path` is
   literal, while a `str` may strictly expand `{run_dir}`,
   `{study_name}`, and `{experiment_name}` in the owning Config's `finalize`.
   Unknown or missing placeholders fail. Every templated TrainLoop Config
   carries all four run-context fields; parent propagation fills only `None`, so
   explicit child values win. *Tools* (`download_data.py` / `ensure.py` /
   builders) may call `scratch_dir(...)` with a comment
   "data-helper-only, not for experiments"; only a Config must never consume
   their globals.
6. **`finalize()` declares; it never *does*.** Pure derived-default wiring: push
   a value down into a child, or derive a parent field up from a finalized one.
   **No I/O, staging, network, disk/count read, or build** -- the rule is
   absolute. The same ban covers experiment factories: a factory returns a
   config and nothing more. The "doing" lives in `__init__`/`make()`; staging
   happens ahead of time. Need a data-derived count? Declare it as a literal
   field and let the *dataset's* `__init__` verify it against its own resolved
   `data_dir` (gated) -- never verify, from the model or finalize, a tree you
   don't own.
7. **`pprint()` is your eyes.**
   `cfg.pprint(finalize=False, hide_default_values=False)` for raw user input;
   `pprint()` for the finalized diff view. Use it to confirm a default (2),
   inspect a derived value (6), or verify a finalized experiment.
8. **The delta must be self-evident.** A reader scanning a factory should see
   WHAT CHANGED without computing anything. Read a value off the config
   (`cfg.max_epochs * cfg.num_steps_eval`) rather than restating a literal a
   parent already set; name the constant that carries meaning
   (`NUM_TRAIN_SAMPLES: Final`, not a bare `50_000`); and prefer a slightly
   longer inline expression over a helper that hides the change behind a call.
   DRY is subordinate to legibility here -- see "Readability beats DRY" below.
9. **A `Literal` naming implementations is a missing injection.** A field like
   `optimizer_kind: Literal["adamw", "muon"]` with a branch per value closes
   the set: a caller wanting Lion must patch the library. Declare
   `Makeable[Protocol]` and let them supply a `PartialConfig` -- see
   "Inject capability, don't enumerate it" below. `Literal` is right for a
   genuine mode (`"reflect"` vs `"constant"` padding), wrong for anything a
   third party might implement.

## Inject capability, don't enumerate it

The test: **could someone reasonably want a value not in your list?** If yes,
the field is an injection point, not an enumeration.

```python
# Bad: every new optimizer needs a branch here, so trying one means patching
# the library rather than writing an experiment.
optimizer_kind: Literal["adamw", "muon"] = "adamw"
learning_rate: float = 1e-3  # only meaningful for one branch
conv_momentum: float = 0.6  # only meaningful for the other

# Good: the config carries the callable and its arguments. The step never
# learns which optimizer it is running.
optimizer: Makeable[Callable[..., Optimizer]] = field(
    default_factory=CompositeOptimizer.Config,
)
```

Two consequences worth stating because they are easy to miss:

- The per-branch hyperparameters disappear from the Config. `conv_momentum`
  was noise for every run that used AdamW; it now lives where it applies, on
  the `Muon.Config` that carries it.
- The protocol is the contract. Declare what the slot must *do*
  (`(model) -> Optimizer`), not what it must *be*, so an implementation the
  library has never seen still fits.

Spell a deferred-constructor slot `Makeable[Callable[..., Optimizer]]`, never
`Makeable[partial[Optimizer]]`: typeshed declares `partial` invariant, so
`partial[Muon]` does not satisfy `partial[Optimizer]` and every concrete
optimizer is rejected. Never alias the resulting type -- a name hides the
protocol from the reader and from the checker's error message.

A slot holding a predicate (a selector, a filter) takes a comparable OBJECT,
not a closure. Two closures with identical bodies are unequal and their `repr`
carries an address, so a config holding one never equals its parent and the
`pprint` diff shows a change on every run.

Prefer **mutation over nested keyword arguments** when building the injected
config -- the same rule as house rule 1, and it matters more here because the
nesting gets deep:

```python
# Bad: a wall of nested calls; the reader tracks parens to see what belongs
# to which optimizer.
cfg.step.optimizer = CompositeOptimizer.Config(
    optimizers=[
        PartialConfig(torch.optim.SGD, lr=0.67, momentum=0.85),
        Muon.Config(lr=0.24, ns_steps=3),
    ],
)

# Good: same line count, one assignment per fact, and each line's PATH says
# what it configures -- no alias to resolve against a binding above.
sgd, muon = cfg.step.optimizer.optimizers = [
    PartialConfig(torch.optim.SGD),
    Muon.Config(),
]
sgd.lr = 0.67
sgd.momentum = 0.85
muon.lr = 0.24
muon.ns_steps = 3
```

`PartialConfig.__setattr__` writes into its kwargs and `__getattr__` reads them
back, so the dotted path keeps working at any depth and both forms build an
identical partial. Bind a local only to shorten a genuinely long path, never to
rename a short one -- `m.lr = 0.24` forces the reader to look up what `m` was;
`muon.lr = 0.24` does not.

The one-line chained assignment introduces a slot and names it in the same
breath, and applies to ordinary Config slots too:

```python
topk = cfg.metrics["accuracy"] = TopK.Config()
topk.k_values = [1]
```

The one place nesting is required is a `default_factory`, which must be a
single expression.

## Readability beats DRY

An experiment factory is read far more often than it is written, and what a
reader needs is the DIFF: which values this variant changes, and to what. Three
rules, in priority order when they conflict:

**Derive from the config, don't restate a literal.** A recomputed constant is
indistinguishable from a deliberate override, and it goes stale silently:

```python
# Bad: is 512 a change, or an echo of the parent? And if exp000's batch size
# moves, this silently computes a step count for a batch size nobody uses.
def exp001():
    cfg = exp000()
    cfg.max_steps = 8 * 50_000 // 512


# Good: reads the parent's value, so only `8` is this variant's contribution.
def exp001():
    cfg = exp000()
    cfg.max_epochs = 8
    cfg.max_steps = cfg.max_epochs * cfg.num_steps_eval
```

**Name a constant only when the name carries meaning the number doesn't.**
`NUM_TRAIN_SAMPLES = 50_000` earns its name -- it is a fact about the dataset,
fixed and non-obvious. A batch size does not: it is a tunable that belongs on
the config, where `check-globals` will insist it lives.

**Don't hoist shared setup into a helper.** A `set_epochs(cfg, 8)` that mutates
four fields is fewer lines and strictly worse: the reader must open it to learn
whether the schedule horizon moved. Inline the four assignments in each factory
even though they repeat. The duplication is the point -- each factory states its
own budget in full.

Corollary for reviewers: if you cannot tell what a factory changes without
opening its parent or a helper, the factory is wrong even when the values are
right.

## Configgle basics

All classes with configurable parameters in this repo use the nested
`Config` class pattern from the
[`configgle`](https://github.com/rekursiv-ai/configgle) library.

> **Reference docs.** Run `help(configgle)` for the
> canonical contract -- `Fig`, `Makes`, `Maker`, the `finalize()`
> contract, the mixin pattern, serialization, and edge cases. This skill
> covers project-specific patterns; it doesn't restate the reference.

## The `.make()` pattern

Configgle separates **"what to build"** (Config) from **"how it works"**
(the runtime class):

1. Caller mutates a Config tree -- only the diffs from defaults.
2. `config.make()` auto-`finalize()`s, then runs `parent_class(config)`.
3. `__init__(self, config)` is the one place that reads the config.
   **All construction work goes there.**

```python
from configgle import Fig, Makeable


class Sandwich:
    class Config(Fig["Sandwich"]):
        portion_grams: int = 50
        """Default per-topping portion in grams."""

        topping: Makeable[Topping] = field(default_factory=Topping.Config)
        """The topping; any config that builds a Topping works."""

    def __init__(self, config: Config):
        self.portion_grams = config.portion_grams
        self.topping = config.topping.make()


cfg = Sandwich.Config()
cfg.portion_grams = 80
sandwich = cfg.make()
```

Note what is NOT here: no `bread_kind: Literal["sourdough", "rye"]` naming the
choices the library happens to know. A scalar is a scalar and a choice is a
slot -- see house rule 9.

Rules:
- Use `Fig["ClassName"]` for the standard case; `Makes["ClassName"]`
  when a sub-class inherits a parent's Config but should still build
  itself (see `help(...)` for the mixin example).
- All Configs must be default-constructable. **Every field needs a
  default and a one-line docstring** -- no exceptions. The docstring
  goes on the line directly below the field. A missing default raises
  `TypeError` naming the field at class-creation time; the
  `require_defaults=False` escape hatch is not for experiment configs.
  Code review rejects Configs with undocumented fields.
- Configs are `slots=True`, so a misspelled assignment (`cfg.lrr = 0.01`)
  raises `AttributeError` instead of silently creating a field nothing
  reads. Do not add `__dict__` back.
- The `Fig["Dog"]` parameter is type-checker-only. `__set_name__` binds
  `parent_class` from the nesting, so bare `Fig` behaves identically at
  runtime -- a wrong string is a typing bug, never a build failure.
- Validate in `__init__`, never in `finalize()`. `pprint` still prints --
  it catches the raise and downgrades it to a `UserWarning` -- but it
  prints the tree you TYPED, since the finalize that would have filled
  the derived values is the thing that failed. So the config you most
  need to inspect is the one whose propagated widths, resolved paths,
  and inferred heads are all missing, and the reason is two warnings
  deep. Raising costs the finalized view, not the view.
- Never use `__post_init__` -- it doesn't run after users mutate fields.

When `__init__` grows past two or three structural decisions
(branching, derived intermediates, sub-object construction), factor
the pieces into `Makeable` slots so the wiring becomes one-line
`config.thing.make()` calls.

## `finalize()` -- inferred / "smart" defaults

`finalize()` exists for derived defaults: fields whose value depends
on other fields -- and *only* that (house rule 6: it declares, never does).
**Contract** (see `help(configgle)`):
`super().finalize()` cascades into the children, so it splits the method
into a **pre** phase (before children finalize -- push values down) and a
**post** phase (after -- derive values up):

```python
def finalize(self) -> Self:
    self.child.x = self.a  # pre: push down into child
    self = super().finalize()  # children finalize here
    self.total = self.child.y  # post: derive up from child
    return self
```

Inject into a child BEFORE `super()`; derive from a child AFTER it.
Pushdown dominates, so `super()` is usually last -- but it need not be.

`finalize()` mutates in place -- it does **not** copy. The copy that
protects the original happens once at the `make()` / `pprint` boundary
(`copy_tree().finalize()`), so a config is finalized once on a fresh
tree and the one passed to `make()` is untouched.

`make()` skips `finalize()` when `_finalized` is already set, which is what
lets a parent's `__init__` rebuild a child via `config.child.make()` without
re-running the child's derived defaults. A `finalize()` body may therefore
assume it runs once per tree -- but only for an UNMUTATED config. Mutating
after a finalize and finalizing again is out of contract and unguarded, so a
non-idempotent body (one that prepends a path prefix, appends to a list)
applies twice. Overriding `make()` means reproducing that guard; prefer
overriding `finalize()` or `copy_tree()`, which are the designed hooks.

### Sentinel-propagation pattern

To let a nested Makeable slot inherit a parent field, give the nested
Config a sentinel default (`-1` for ints, `None` for objects) and
overwrite it in the parent's `finalize()`:

```python
class Topping:
    class Config(Fig["Topping"]):
        portion_grams: int = -1
        """Portion size in grams; -1 = inherit from parent sandwich."""


class Sandwich:
    class Config(Fig["Sandwich"]):
        portion_grams: int = 50
        """Default per-topping portion in grams."""

        topping: Makeable[Topping] = field(default_factory=Topping.Config)
        """The topping; any subclass works as long as it builds a Topping."""

        @override
        def finalize(self) -> Self:
            if self.topping.portion_grams == -1:
                self.topping.portion_grams = self.portion_grams
            return super().finalize()
```

If the slot is `Makeable[Protocol]` and the Protocol doesn't include
the field, declare a small `@runtime_checkable` Protocol for what the
parent needs and narrow with `isinstance`:

```python
@runtime_checkable
class HasPortion(Protocol):
    portion_grams: int


if isinstance(self.topping, HasPortion) and self.topping.portion_grams == -1:
    self.topping.portion_grams = self.portion_grams
```

Avoid `getattr(slot, "field", None)` -- it bypasses the type checker
and hides the contract from readers.

## `Makeable[Protocol]` -- pluggable sub-components

Annotate a Config field as `Makeable[Protocol]` whenever the
sub-module should be swappable. The parent declares the *interface* it
needs, not the concrete class. For a collection of swappable parts,
pair with `MutableSequence`:

```python
garnishes: MutableSequence[Makeable[Garnish]] = field(
    default_factory=list[Makeable[Garnish]],
)
"""Garnishes stacked top-to-bottom in order."""
```

### `default_factory` for nested Config slots

Default with the class itself; Config classes are callable, so the
class **is** the factory:

```python
# Correct.
topping: Makeable[Topping] = field(default_factory=Topping.Config)

# Wrong: redundant lambda, ruff flags PLW0108.
topping: Makeable[Topping] = field(default_factory=lambda: Topping.Config())
```

Use the lambda form only when the default needs constructor arguments
the parent's `finalize()` won't fill in (assume `Topping.Config` has
`kind: Literal["cheese", "pickles", "lettuce"] = "cheese"`):

```python
topping: Makeable[Topping] = field(
    default_factory=lambda: Topping.Config(kind="pickles"),
)
```

## `pprint` -- the point of configgle

`cfg.pprint()` / `cfg.pformat()` walk the whole Config tree -- every
sub-Config, every derived default, every user override -- with one
call. Reach for it **before** reading the source when diagnosing
mismatched hyperparameters, snapshotting in a test, or debugging
`finalize()` propagation.

```python
cfg = Sandwich.Config()
cfg.pprint()  # to stdout
s = cfg.pformat()  # to string
```

**Two surprising defaults:**

- `finalize=True` (default) -- prints the *finalized* tree (derived
  defaults filled in). Pass `finalize=False` to see raw user input.
- `hide_default_values=True` (default) -- omits fields at their
  default. Pass `hide_default_values=False` for the full picture.

For a full finalized snapshot:

```python
cfg.pprint(hide_default_values=False)
```

Pass `finalize=False` only when inspecting raw user input before propagation.

To compare configs, call `udiff`; never run `difflib` over `pformat` output:

```python
print(base.udiff(variant))  # unified diff, defaults included
print(base.udiff(variant, mode="fields"))  # one row per changed field path
print(base.udiff(a, b, mode="sxs"))  # side by side; also "sxs-full"
```

`udiff` never mutates its sources. It takes every `pformat` option plus
`n` (context lines) and `color` (`None` colors only on a terminal);
`finalize=False` compares raw inputs.

See `help(configgle.pprint)` for the rest of the
knobs (`indent`, `width`, `depth`, `compact`,
`mask_memory_addresses`, etc.).

## Other Config methods

- `cfg.update(source=None, *, skip_missing=False, **kwargs)` -- overlay
  attributes from another config or kwargs in place; returns `self`.
  Supports nested dict overlay: `cfg.update(topping={"portion_grams": 80})`.
  Assigns only -- computes nothing derived; `finalize`/`make` does that.
- `cfg.copy_tree()` -- "semi-deep" copy: nested configs and mutable
  containers holding configs are duplicated; leaf values (primitives,
  tensors) are aliased. `make()`/`pprint` apply it before finalizing so
  the original stays pristine; reach for it directly to finalize a copy
  without touching the source (`cfg.copy_tree().finalize()`).
- `type(cfg).parent_class` -- the class `.make()` would construct. Useful
  for introspecting a `Makeable[Protocol]` slot without instantiating.

## Experiments often need to type-narrow Config slots

Experiment files routinely start from a base Config, then mutate
nested fields. Because slots like `step` are typed as
`Makeable[TrainStepProtocol]`, basedpyright doesn't know the concrete
subclass -- narrow with `isinstance` before the mutation:

```python
def exp001() -> TrainLoop.Config:
    cfg = exp000()
    assert isinstance(cfg.step, MyTrainStep.Config)
    cfg.step.compile = True
    assert isinstance(cfg.dataset, MyDataset.Config)
    cfg.dataset.drop_last = True
    return cfg
```

The `assert isinstance(...)` also catches the case where someone
reorganized the base experiment and silently swapped the slot type.

Add the assert **only** when the next line touches a field the narrowed
subclass adds. Asserting before setting a *base-class* field (one the
`Makeable[Protocol]` slot already exposes) is noise -- delete it.

Better still, narrow ONCE in a subclass so no factory needs the assert at all
(the `priml` skill's `Cifar10TrainLoop`):

```python
class Cifar10TrainLoop(Makes["TrainLoop"], TrainLoop.Config):
    step: Cifar10TrainStep.Config = field(default_factory=Cifar10TrainStep.Config)
```

Use the per-factory assert when the slot type genuinely varies across
experiments; use the subclass when a whole family shares one.

## Real Configs to read

The `priml` package is the reference consumer:

- `priml/model/*.py` -- `Conv2d`, `BatchNorm2d`, `Linear`, etc.;
  sentinel-default `channels_in` / `channels_out` with `finalize()`
  propagation.
- `priml/train/train_loop.py` -- deeply-nested `TrainLoop.Config` with
  real `Makeable[Protocol]` slots.
- `priml/baselines/*/{experiments,model,train_step}.py` -- per-study
  triple: `experiments.py` returns a fully-wired `TrainLoop.Config`;
  `model.py` / `train_step.py` define the domain-specific Configs.
