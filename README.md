# configgle🤭

[![PyPI version](https://img.shields.io/pypi/v/configgle.svg)](https://pypi.org/project/configgle/)
[![CI](https://github.com/rekursiv-ai/configgle/actions/workflows/package-validation.yml/badge.svg?branch=main)](https://github.com/rekursiv-ai/configgle/actions/workflows/package-validation.yml)
[![Python 3.12+](https://img.shields.io/badge/python-3.12%2B-blue.svg)](pyproject.toml)
[![License: Apache-2.0](https://img.shields.io/badge/license-Apache--2.0-blue.svg)](LICENSE)
[![Discord](https://img.shields.io/discord/1530237005311639592?logo=discord&logoColor=white&label=Discord&color=5865F2)](https://discord.gg/2GZFPPvCqn)

Type-safe hierarchical experiment configuration using pure Python dataclass factories and dependency injection.

## Quick Start

```bash
# Mac:
#   # Required for quick install.
#   brew install uv

# Ubuntu/Debian:
#   # Required for quick install.
#   sudo apt-get install -y curl
#   curl -LsSf https://astral.sh/uv/install.sh | sh

uv add configgle

# Alternatively: python -m pip install configgle
```

Hierarchical experiment configuration using pure Python dataclasses with typed
factory methods, covariant protocols, inheritance support, and tooling for
pretty printing, autodecorating, updating, and semi-deep copying.

## Example

```python
from configgle import Fig


class Model:
    class Config(Fig):
        channels_in: int = 256
        num_layers: int = 4

    def __init__(self, config: Config):
        self.config = config


# Create and modify config
cfg = Model.Config()
cfg.channels_in = 512

# Instantiate the parent class
model = cfg.make()
print(model.config)
assert isinstance(model, Model)
```

Configs are plain mutable dataclasses, so experiments are just functions that
tweak a baseline:

```python
def exp000() -> Model.Config:
    return Model.Config()


def exp001() -> Model.Config:
    cfg = exp000()
    cfg.channels_in = 512
    cfg.num_layers = 8
    return cfg
```

Or use `@autofig` to auto-generate the Config from `__init__`:

```python
from configgle import autofig
from torch import nn


@autofig
class Model(nn.Module):
    def __init__(self, channels_in: int = 256, num_layers: int = 4):
        super().__init__()
        self.channels_in = channels_in
        self.num_layers = num_layers


# Config is auto-generated from __init__ signature
model = Model.Config(channels_in=512).make()
print(model.channels_in)  # 512
```

## Features

### Type-safe `make()`

**tl;dr:** Both `ty` and `basedpyright` are first-class supported.
Unfortunately neither is perfect:

| | `ty` | `basedpyright` |
|---|:---:|:---:|
| Bare `Fig` infers parent type | ✅ | ❌ (`Any` fallback) |
| Explicit `Fig["Parent"]` specifies parent type | ✅ | ✅ |
| Inheritance infers parent type | ✅ | ❌ |
| Explicit `Makes["Child"]` narrows inferred inherited parent type | ✅ | ✅ |
| Inherited Config child fields | ✅ (workaround for [#3282](https://github.com/astral-sh/ty/issues/3282)) | ✅ |
| `@autofig` `.Config` access | ✅ (fixed [#143](https://github.com/astral-sh/ty/issues/143)) | ✅ |

**Details:**

When `Config` is defined as a nested class, `MakerMeta.__get__` uses the
descriptor protocol to infer the parent class automatically. The return type
of `__get__` is `Intersection[type[Config], type[Makeable[Parent]]]`, so
`make()` knows the exact return type with zero annotation effort:

```python
class Model:
    class Config(Fig):
        channels_in: int = 256

    def __init__(self, config: Config):
        self.channels_in = config.channels_in


model = Model.Config(channels_in=512).make()  # inferred as Model
```

Type checkers that support `Intersection` (like `ty`) resolve this fully --
bare `Fig` is all you need. For type checkers that don't yet support
`Intersection` (like `basedpyright`), parameterize with the parent class
name to give the checker the same information explicitly:

```python
class Model:
    class Config(Fig["Model"]):  # explicit type parameter only for basedpyright
        channels_in: int = 256

    def __init__(self, config: Config):
        self.channels_in = config.channels_in


model: Model = Model.Config(channels_in=512).make()  # returns Model, not object
```

Without `["Model"]`, non-`ty` checkers fall back to `Any` (so attribute access
works without typecheck suppressions).

`ty` gets full inference from `Intersection` -- bare `Fig` and inherited
configs just work. `basedpyright` doesn't support `Intersection` yet, so it
needs explicit `Fig["Parent"]` and `Makes["Child"]` annotations. `ty`
honors class decorator return types ([#143](https://github.com/astral-sh/ty/issues/143),
fixed in 0.0.49), so `@autofig`-decorated classes resolve `.Config` with no
suppression on both checkers. When `Intersection` lands in the
[type spec](https://github.com/python/typing/issues/213), `Makes` becomes
unnecessary and both checkers will infer everything from bare `Fig`.

Requires `ty>=0.0.60`. Several behaviors configgle relies on
were fixed upstream rather than worked around here -- see
[docs/ty_missing_features.md](docs/ty_missing_features.md) for the current
limitations and their provenance.

### Inheritance with `Makes` (only for `basedpyright`)

When a child class inherits a parent's Config, the `make()` return type would
normally be the parent. Use `Makes` to re-bind it (again, only needed for `basedpyright`):

```python
from configgle import Makes


class Animal:
    class Config(Fig["Animal"]):
        name: str = "animal"

    def __init__(self, config: Config):
        self.config = config
        self.name = config.name


class Dog(Animal):
    class Config(Makes["Dog"], Animal.Config):
        breed: str = "mutt"

    def __init__(self, config: Config):
        super().__init__(config)
        self.breed = config.breed


dog: Dog = Dog.Config(name="Rex", breed="labrador").make()  # returns Dog, not Animal
```

`Makes` contributes nothing to the MRO at runtime -- it exists purely for the
type checker (see the [type checker table](#type-safe-make) above). When
[Intersection](https://github.com/python/typing/issues/213) lands, `Makes`
becomes unnecessary.

### Covariant `Makeable` protocol

`Makeable[T]` is a covariant protocol satisfied by any `Fig`, `InlineConfig`,
or custom class exposing `make()`, `finalize()`, `update()`, plus the
`_finalized` and `parent_class` members (`Maker` and `InlineConfig` provide all
five). Because it's covariant, `Makeable[Dog]` is assignable to
`Makeable[Animal]`:

```python
from configgle import Makeable


def train(config: Makeable[Animal]) -> Animal:
    return config.make()


# All valid:
train(Animal.Config())
train(Dog.Config(breed="poodle"))
```

This makes it easy to write functions that accept any config for a class
hierarchy without losing type information.

### Nested config finalization -- pre / super / post

Override `finalize()` to compute derived fields. `super().finalize()` cascades
into the child configs, so it splits the method into a **pre** phase (before
children finalize -- push values down) and a **post** phase (after -- derive
values up):

```python
from configgle import Configurable  # Just an alias to Makeable.
from dataclasses import field


class Encoder:
    class Config(Fig):
        c_in: int = 256
        mlp: Configurable[nn.Module] = field(default_factory=MLP.Config)

        def finalize(self) -> Self:
            self.mlp.c_in = self.c_in  # pre: push down into the child
            self = super().finalize()  # children finalize here
            self.out = self.mlp.out  # post: derive up from the child
            return self
```

Inject into a child **before** `super()` (so it finalizes with the value);
derive from a child **after** (so you read its finalized result). Pushdown is
the common case, so `super()` is usually last -- but it need not be.

`finalize()` mutates in place; the copy that protects the original happens once
at the `make()` / `pprint` boundary (`copy_tree().finalize()`), so a config is
finalized exactly once on a fresh tree (never re-finalized) and the config
passed to `make()` is left untouched.

### `update()` for bulk mutation

Configs support bulk updates from another config or keyword arguments:

```python
cfg = Model.Config(channels_in=256)
cfg.update(channels_in=512, num_layers=8)

# Or copy from another config (kwargs take precedence):
cfg.update(other_cfg, num_layers=12)
```

### `InlineConfig` / `PartialConfig`

`InlineConfig` wraps an arbitrary callable and its arguments into a config
object with deferred execution. Use it for classes where all constructor
arguments are known at config time:

```python
from configgle import InlineConfig
import torch.nn as nn

cfg = InlineConfig(nn.Linear, in_features=256, out_features=128, bias=False)
cfg.out_features = 64  # attribute-style access to kwargs
layer = cfg.make()  # calls nn.Linear(in_features=256, out_features=64, bias=False)
y = layer(x)  # use the constructed module
```

`PartialConfig` is shorthand for `InlineConfig(functools.partial, fn, ...)`
-- use it for functions where some arguments aren't known at config time:

```python
from configgle import PartialConfig
import torch.nn.functional as F

cfg = PartialConfig(F.cross_entropy, label_smoothing=0.1)
loss_fn = cfg.make()  # returns functools.partial(F.cross_entropy, label_smoothing=0.1)
loss = loss_fn(
    logits, targets
)  # calls F.cross_entropy(logits, targets, label_smoothing=0.1)
```

Nested configs in args/kwargs are finalized and `make()`-d recursively, so
both compose naturally with `Fig` configs.

### `copy_tree()`

`copy_tree()` is a "semi-deep" copy: nested configs and mutable containers
holding configs are duplicated, while leaf values (primitives, tensors,
loggers) are aliased. `make()` and `pprint` apply it before finalizing so the
source config stays pristine. Reach for it directly to finalize a copy without
touching the source:

```python
finalized = cfg.copy_tree().finalize()  # cfg unchanged
```

### `pprint` / `pformat`

Config-aware pretty printing that hides default values, auto-finalizes before
printing, and scrubs memory addresses:

```python
from configgle import Configurable, Fig, pformat


class MLP:
    class Config(Fig):
        c_in: int = 256
        c_out: int = 256
        num_layers: int = 2
        dropout: float = 0.1
        use_bias: bool = True

    def __init__(self, config: Config): ...


class Model:
    class Config(Fig):
        channels_in: int = 256
        num_layers: int = 4
        mlp: Configurable[nn.Module] = field(default_factory=MLP.Config)
        output_mlp: Configurable[nn.Module] = field(default_factory=MLP.Config)

    def __init__(self, config: Config): ...


def exp001():
    cfg = Model.Config()
    cfg.channels_in = 512
    cfg.num_layers = 12
    cfg.mlp.c_in = 512
    cfg.mlp.c_out = 1024
    cfg.mlp.num_layers = 4
    cfg.mlp.dropout = 0.2
    cfg.mlp.use_bias = False
    cfg.output_mlp.c_in = 1024
    cfg.output_mlp.c_out = 256
    cfg.output_mlp.dropout = 0.3
    return cfg


print(pformat(exp001(), continuation_pipe=0))
# Model.Config(
#    channels_in=512,
#    num_layers=12,
#    mlp=MLP.Config(
#    │       c_in=512,
#    │       c_out=1_024,
#    │       num_layers=4,
#    │       dropout=0.2,
#    │       use_bias=False
#    ),
#    output_mlp=MLP.Config(c_in=1_024, dropout=0.3)
# )
```

Default values are hidden, continuation pipes show where nested blocks belong,
large numbers get underscores (`1_024`), and short sub-configs collapse onto
one line. `pformat` and `pprint` are also available as methods on any `Fig` config:

```python
cfg = exp001()
cfg.pprint()  # prints to stdout
s = cfg.pformat()  # returns string
```

### `serialize()` / `deserialize()`

`serialize()` returns a tree of plain Python containers (`dict`, `list`, `str`,
`int`, `float`, `bool`, `None`), not a string, so the caller picks the
transport:

```python
import json, yaml

tree = cfg.serialize()  # plain dict/list/primitives

json_text = json.dumps(tree, indent=2)
yaml_text = yaml.safe_dump(tree, sort_keys=False)

cfg = Model.Config.deserialize(yaml.safe_load(yaml_text))  # a real Model.Config
model = cfg.make()
```

`yaml.safe_dump` refuses any non-plain Python object, so a clean dump means the
tree contains nothing else. The same tree works with msgpack, a W&B run record,
or a field in a larger payload.

The round trip preserves what a plain dict loses: nested config classes,
polymorphic `Makeable` slots, tuples vs lists, DAG identity (a sub-config shared
by two fields stays shared), and cycles. The wire format is
[jsonpickle](https://jsonpickle.github.io)'s `py/*` tag vocabulary, so the tree
is legible to anyone who knows jsonpickle. `serialize()` does not finalize, so
the loaded config is raw and ready for `finalize()` / `make()`. Leaves JSON
cannot represent natively (tensors, arrays) take a
`hooks={type: (encode, decode)}` map.

Deserialization imports the modules named in the payload, so treat a serialized
config like `pickle`: load only trusted data.

### CLI overrides

`apply_overrides` edits a config from `PATH=VALUE` strings. Dotted paths reach
into nested configs, every hop is checked against the node's declared fields
(so a typo raises instead of silently creating an attribute), and the value is
cast to the field's declared type:

```python
from configgle import apply_overrides

cfg = Model.Config()
apply_overrides(cfg, ["channels_in=512", "mlp.dropout=0.2"])
```

`configgle/launch.py` wires that to argparse, so any factory function returning
a config is runnable as-is:

```python
# myproject/experiments.py
def baseline() -> Makeable[Trainer]:
    return Trainer.Config()
```

```sh
python -m configgle myproject.experiments.baseline --override mlp.dropout=0.2
```

The launcher is deliberately small -- it exists to make `--override` usable out
of the box and to show the pattern. A real project usually wants its own entry
point (hardware logging, distributed setup, run naming); build it on
`resolve_config` and `apply_overrides` rather than copying the file.

### `Dataclass` base

`Dataclass` provides the auto-dataclass metaclass (with the same opinionated
defaults as `Fig`: `kw_only=True`, `slots=True`, etc.) but without `Maker` or
`make()`. Use it for plain data objects that don't need the factory pattern.

### `@autofig` for zero-boilerplate configs

When you don't need a hand-written Config, `@autofig` generates one from
`__init__` (see [Example](#example) above).

### Pickling and cloudpickle

Configs are fully compatible with `pickle` and `cloudpickle`, including the
parent class reference. This is important for distributed workflows (e.g.,
sending configs across processes):

```python
import cloudpickle, pickle

cfg = Model.Config(channels_in=512)
cfg_ = pickle.loads(cloudpickle.dumps(cfg))
model = cfg_.make()  # parent_class is preserved
```

## Lineage

The nested-`Config` shape is not new. Two production training codebases use
it, and one cites the other:

- **[AXLearn](https://github.com/apple/axlearn)** (Apple, 2023). Its
  [ML API Style](https://github.com/apple/axlearn/blob/main/docs/ml_api_style.md)
  sets the rules configgle follows: every layer has a `Config` member class,
  a composite layer's config holds its children's configs as fields, and a
  child's `input_dim` is set by the parent. `axlearn/common/config.py`
  supplies `Configurable.Config.instantiate()`, `default_config()`, and
  `config_for_function` / `config_for_class` for signature-derived configs.
- **[torchtitan](https://github.com/pytorch/torchtitan)** (Meta, 2026-02-23).
  [PR #2386](https://github.com/pytorch/torchtitan/pull/2386) replaced its
  TOML job config with a `Configurable` base whose nested
  `@dataclass(kw_only=True, slots=True) class Config` builds the owner via
  `build()`. The author's note credits AXLearn's style doc. Model configs
  then grew methods -- `get_nparams_and_flops(model, seq_len)` -- which is
  the same move as a `finalize()` override: computation that belongs with
  the config, not the module.

configgle was written without knowledge of either and released 2026-02-02,
three weeks before torchtitan's refactor merged. It adds a `finalize()`
cascade for derived fields (AXLearn has validators; torchtitan's
`update_from_config` carries a `TODO` calling itself an encapsulation
violation), `pprint` as a diff against class defaults, `LateBound` for
references across the built tree, and ships as a library with no framework
attached. Both appear in the comparison below.

## Comparison

|                                             | [configgle] | [AXLearn] | [torchtitan] | [Hydra] | [Sacred] | [OmegaConf] | [Gin] | [ml_collections] | [Fiddle] | [Confugue] |
| ------------------------------------------- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| Python-based configuration                  |  ✅   |  ✅   |  ✅   |  🟡   |  🟡   |  🟡   |  ❌   |  ✅   |  ✅   |  🟡   |
| YAML-based configuration                    |  🚫   |  🚫   |  🚫   |  ✅   |  ✅   |  ✅   |  ❌   |  ❌   |  ❌   |  ✅   |
| Installable without a trainer               |  ✅   |  ❌   |  ❌   |  ✅   |  ✅   |  ✅   |  ✅   |  ✅   |  ✅   |  ✅   |
| Set a nested field from the command line    |  ✅   |  ❌   |  ✅   |  ✅   |  ✅   |  🟡   |  🟡   |  ✅   |  ✅   |  ❌   |
| Nested `Config` builds its class            |  ✅   |  ❌   |  ✅   |  ❌   |  ❌   |  ❌   |  ❌   |  ❌   |  ❌   |  ❌   |
| Type checker knows what `make()` returns    |  ✅   |  ✅   |  🟡   |  ❌   |  ❌   |  ❌   |  ❌   |  ❌   |  ✅   |  ❌   |
| Derived fields cascade to children          |  ✅   |  🟡   |  🟡   |  🟡   |  🟡   |  🟡   |  ❌   |  🟡   |  ❌   |  ❌   |
| Config schema generated from `__init__`     |  ✅   |  ✅   |  ❌   |  🟡   |  ❌   |  ❌   |  ❌   |  ❌   |  ✅   |  ❌   |
| Print only fields that differ from defaults |  ✅   |  ✅   |  ❌   |  ❌   |  ❌   |  ❌   |  🟡   |  ❌   |  🟡   |  ❌   |
| Weight tying by path                        |  ✅   |  ❌   |  🟡   |  ❌   |  ❌   |  ❌   |  ❌   |  ❌   |  🟡   |  ❌   |
| To plain dict and back to typed config      |  ✅   |  🟡   |  🟡   |  🟡   |  🟡   |  🟡   |  ❌   |  🟡   |  🟡   |  ❌   |
| Survives `pickle`/`cloudpickle`             |  ✅   |  🟡   |  🟡   |  🟡   |  🟡   |  ✅   |  🟡   |  🟡   |  ✅   |  🟡   |
| Commits in the last six months              |  ✅   |  ✅   |  ✅   |  ✅   |  🟡   |  ✅   |  ✅   |  ✅   |  ✅   |  ❌   |
| GitHub stars                                |  --   | 2.4k  | 5.7k  | 10.6k | 4.4k  | 2.4k  | 2.2k  | 1.0k  |  386  |  21   |

[configgle]: https://github.com/rekursiv-ai/configgle
[AXLearn]: https://github.com/apple/axlearn
[torchtitan]: https://github.com/pytorch/torchtitan
[Hydra]: https://github.com/facebookresearch/hydra
[Sacred]: https://github.com/IDSIA/sacred
[OmegaConf]: https://github.com/omry/omegaconf
[Gin]: https://github.com/google/gin-config
[ml_collections]: https://github.com/google/ml_collections
[Fiddle]: https://github.com/google/fiddle
[Confugue]: https://github.com/cifkao/confugue

✅ = yes, 🟡 = partial/caveat, ❌ = no, 🚫 = intentionally no.

Corrections welcome --
[open a PR](https://github.com/rekursiv-ai/configgle/pulls).

(🚫 is used only where a project states the reason in writing: configgle
here, and AXLearn and torchtitan on the YAML row.)

<details>
<summary><b>What each row means.</b></summary>

- **Python-based configuration.**
  - 🟡 Python is a second path beside the primary YAML one.
- **YAML-based configuration.** Independent of the row above, not its
  opposite: Gin is neither, it has its own `.gin` DSL.
  - 🚫 An experiment should manifest in exactly one Python function that returns
    its config. A YAML file splits that function in two, so reproducing a run
    means reconstructing which file, which defaults list, and which CLI
    overrides composed it. YAML (and JSON, etc) as transport is supported:
    `deserialize(yaml.safe_load(...))` returns a live typed config. It is just
    not where an experiment is written down.
  - 🚫 AXLearn's style doc: configs are Python objects "because config
    objects can be logged and manipulated." torchtitan #2386 dropped TOML
    because "everything needs to be registered explicitly before it can be
    used" and a typo like `training.stpes` "silently becomes a default
    value."
- **Installable without a trainer.** AXLearn's `config.py` and torchtitan's
  `configurable.py` ship inside their trainers and import them.
- **Set a nested field from the command line**, e.g.
  `--override mlp.dropout=0.2`.
  - 🟡 OmegaConf parses the flags but leaves you to merge them
    (`OmegaConf.from_cli`); Gin needs a separate flags integration.
  - ❌ AXLearn selects a named config function with `--config` and takes
    per-run values through its own `absl` flags; nested fields are not
    reachable from the command line.
- **Nested `Config` builds its class**, with no field or argument naming it.
  configgle's `MakerMeta.__get__` does this via the descriptor protocol;
  torchtitan's `__init_subclass__` sets `_owner`. AXLearn requires
  `default_config()` to stamp `klass`; Fiddle names the class at the call
  site (`fdl.Config(MyClass)`).
- **Type checker knows what `make()` returns.** configgle's inference is
  `ty`-only; `basedpyright` needs `Fig["Model"]` (see
  [Type-safe `make()`](#type-safe-make)).
  - 🟡 torchtitan's `build()` is unannotated and `_owner` is `ClassVar[type
    | None]`, so both checkers see `Any`.
- **Derived fields cascade to children**, e.g.
  `out_dim` following `channels_in`. configgle's `finalize()` is a hook you
  override; it runs on the parent, then on every child config.
  - 🟡 Recomputed only at a conversion boundary: Hydra and OmegaConf re-run
    `__post_init__`, Sacred re-executes config scopes, ml_collections has lazy
    `FieldReference`. torchtitan's `update_from_config` is called once from
    the trainer and carries a `TODO` calling itself an encapsulation
    violation.
  - 🟡 AXLearn validates on assignment (`register_validator`) and lets a
    parent `Config` set a child's `input_dim` in a method the caller runs
    (the pattern its style doc recommends); nothing recomputes on its own.
- **Config schema generated from `__init__`**, so adding an argument needs no
  config edit
  (configgle's `@autofig`, Fiddle's `@auto_config`, AXLearn's
  `config_for_class`).
  - 🟡 Hydra's `configen` is an experimental codegen tool, not a decorator.
- **Print only fields that differ from defaults** (`pprint`).
  - ✅ AXLearn's `debug_string(omit_default_values=...)` omits `None` and
    `REQUIRED` by default and takes any set of values to hide.
  - 🟡 Gin's `config_str` and Fiddle's `printing` render the whole tree.
- **Weight tying by path** -- a built object names a sibling; the reference
  lives in the config and is wired after the outermost build (`LateBound`),
  so a tied head prints as `TiedLinear.Config(tied="in_proj")` and no parent
  writes the alias.
  - 🟡 torchtitan has `enable_weight_tying: bool` on the root config, and
    `Decoder.__init__` assigns `tok_embeddings.weight = lm_head.weight`.
    Fiddle's `build` memoizes by config identity, so one `fdl.Config`
    instance referenced from two places builds one object; the tie is
    expressed by sharing the config, not by naming a path.
- **To plain dict and back to typed config** -- both halves: out to plain
  containers (*not* a string, so you pick the transport), and back to live
  *typed* objects.
  - 🟡 Fiddle and ml_collections emit only a string; Hydra, OmegaConf, and
    Sacred reload to untyped dicts. AXLearn and torchtitan have `to_dict`
    and no way back.
- **Survives `pickle`/`cloudpickle`.**
  - 🟡 Works for plain configs but not every construct, or needs cloudpickle.
    AXLearn and torchtitan configs are plain `attrs`/dataclass objects, so
    pickle works where every field value does; neither documents it.
- **Commits in the last six months.** Commits, not releases: Gin and Fiddle
  ship rarely but still take commits.
  - 🟡 Commits in the last year.
- **GitHub stars** -- for context. configgle's own count is on the repo
  page; most of these libraries have years of production use behind them.

</details>

<details>
<summary><b>How each library works.</b></summary>

Release dates, commit dates, and star counts verified 2026-08-05 (PyPI JSON API
and the GitHub repos/commits APIs); configgle itself was at 1.3.6, released the
same day. AXLearn and torchtitan were checked 2026-09-10.

**[AXLearn](https://github.com/apple/axlearn)** (Apple) -- not on PyPI; last commit 2026-07-08.
`axlearn/common/config.py` (first commit 2023-07-09) defines `Configurable`
with a nested `@config_class class Config(InstantiableConfig[C])` built on
`attrs`; `cfg.instantiate()` calls `self.klass(self)`. Configs are mutated
in place with `cfg.set(...)` or copied with `clone`, and every field is
validated on assignment. `config_for_function` / `config_for_class` derive a
config from a signature. `REQUIRED` marks a field with no default and raises
at `instantiate`. `visit` walks the tree; `to_dict` and `debug_string` are
output-only. Runs are selected by `--config <name>` naming a
`TrainerConfigFn`; nested overrides go through Python, not flags. Its
[ML API Style](https://github.com/apple/axlearn/blob/main/docs/ml_api_style.md)
is the design document the nested-`Config` pattern comes from.

**[torchtitan](https://github.com/pytorch/torchtitan)** (Meta) -- not on PyPI; last commit 2026-09-10.
`torchtitan/config/configurable.py` (added 2026-02-23 in
[#2386](https://github.com/pytorch/torchtitan/pull/2386), which credits
AXLearn) defines `Configurable` with a nested
`@dataclass(kw_only=True, slots=True) class Config`; `cfg.build()` calls
`self._owner(config=replace(self))`, with `_owner` wired by
`__init_subclass__`. `traverse(config_cls)` yields
`(fqn, config, parent, attr)` for in-place replacement of a subtree. CLI
overrides come from `tyro` as `section.key=value`. `to_dict` is output-only.
Derived values are applied by `update_from_config`, which the source marks
`TODO: This function violates encapsulation`. Model configs also carry
`get_nparams_and_flops(model, seq_len)`.

**[Hydra](https://github.com/facebookresearch/hydra)** (Meta) -- PyPI 1.3.4 released 2026-07-04; last commit 2026-08-04.
YAML-centric with optional "structured configs" (Python dataclasses registered
in a `ConfigStore`). Instantiation uses `hydra.utils.instantiate()`, which
resolves a `_target_` field -- typically a string import path, though a class
object is also accepted -- and returns `Any`. Composition is done via defaults
lists (usually YAML, optionally a `defaults` field on a dataclass), not class
inheritance; dataclass inheritance works at the schema level. `configen` is an
experimental code-generation tool (latest release v0.9.0.dev8) that produces
structured configs from class signatures.

**[Sacred](https://github.com/IDSIA/sacred)** -- PyPI 0.8.7 released 2024-11-26; last commit 2025-10-22.
Experiment management framework. Config is defined via `@ex.config` scopes
(local variables become config entries) or loaded from YAML/JSON/pickle files,
and overridden on the command line with `with 'a.b=5'`. Sacred auto-*injects*
config values into captured functions by parameter name (dependency injection),
but does not auto-*generate* configs from function signatures. Reuse is by
composition -- ingredients nest, and stacked config scopes override a
reusable ingredient's defaults -- rather than class inheritance. No typed
factory methods; `Experiment` objects are not picklable, though config *files*
may be pickles. Its config is a plain dict tree (`normalize_or_die` rejects keys
that collide with jsonpickle tags), so it dumps to JSON/YAML cleanly but reloads
to dicts, not typed objects.

**[OmegaConf](https://github.com/omry/omegaconf)** -- PyPI 2.3.1 released 2026-06-11; last commit 2026-07-29.
YAML-native configuration with a "structured config" mode that accepts
`@dataclass` schemas. Configs are `DictConfig` proxy objects at runtime, not
dataclass instances; `OmegaConf.to_object()` converts them back into real
instances (re-running `__post_init__` recursively as it goes). Supports
dataclass inheritance for schema definition. Good pickle support
(`__getstate__`/`__setstate__`). `to_object()` acts as a factory but is typed
`Any`, so callers lose static types. `OmegaConf.from_cli()` parses a dotlist
but leaves the merge to you. No auto-generation, no protocols.
`OmegaConf.to_container()` gives a plain dict, but `OmegaConf.create()` on it
yields `get_type(...) == dict`; recovering the class needs `merge` against the
structured schema, then `to_object()`. Hydra inherits this mechanism.

**[Gin](https://github.com/google/gin-config)** (Google) -- PyPI 0.5.0 released 2021-11-03; last commit 2026-07-02.
Global string-based registry. You decorate functions with `@gin.configurable`
and bind parameters via `.gin` files or `gin.bind_parameter('fn.param', val)`.
There are no config objects -- parameter values live in a global dict keyed by
`(scope, selector)`. No typed returns, and no config-object inheritance (though
`.gin` files compose via `include`, and scoped bindings inherit from the root
scope). The docs still state "Gin-configurable functions are not pickleable,"
but as of 2021 Gin wraps the metaclass `__call__` so that *instances* of
configurable classes pickle fine; a community PR proposing `__reduce__` was
closed unmerged. With no config objects there is nothing to serialize:
`config_str()` emits `.gin` text, and `get_bindings()` returns a flat
`Dict[str, Any]` of bindings rather than a config tree.

**[ml_collections](https://github.com/google/ml_collections)** (Google) -- PyPI 1.1.0 released 2025-04-17; last commit 2026-07-07.
Dict-like `ConfigDict` with dot-access, type-checking on mutation, and
`FieldReference` for lazy cross-references between values. Config files are
Python, not YAML (the library itself depends on PyYAML for printing).
`config_flags` gives `--config.foo.bar=3e-4` overrides for free. No factory
method or typed instantiation. Pickle works for plain configs, but
`FieldReference` operations that build lambdas internally (`.identity()` --
used by `get_oneway_ref()` -- and the `.to_int()`/`.to_float()`/`.to_str()`
casts) fail with standard pickle; cloudpickle handles them. Serialization is
one-way: `to_dict()`/`to_json()`/`to_yaml()` exist, but there is no `from_json`
or `from_dict` anywhere in `config_dict.py`.

**[Fiddle](https://github.com/google/fiddle)** (Google) -- PyPI 0.3.0 released 2024-04-09; last commit 2026-07-21.
Python-first. You build config graphs with `fdl.Config[MyClass]` objects and
call `fdl.build()` to instantiate them. `build(Config[T]) -> T` is typed via
`@overload`. Config modification is functional (`fdl.copy_with`) -- you don't
subclass a config to override values. `@auto_config` rewrites a factory
function's AST to produce a config graph automatically. Full
pickle/cloudpickle support, and `serialization.dump_json`/`load_json` round-trip
a config graph faithfully (though that module lives under `_src.experimental`).
`dump_json` returns a *string*, not containers -- the container tree it builds
internally (`Serialization(...).result`) is private, so YAML or msgpack means
re-parsing the JSON.

**[Confugue](https://github.com/cifkao/confugue)** -- PyPI 0.1.1 released 2020-04-22; last commit 2021-09-13.
YAML-based hierarchical configuration. The `configure()` method instantiates
objects from YAML dicts, with the class overridden via a `class:` key whose
value uses PyYAML's `!!python/name:` tag. Returns `Any`. Partial config
inheritance via YAML merge keys (`<<: *base`). No CLI, no auto-generation, no
protocols. Pickling is undocumented and untested -- configured instances do
pickle, but `bind()` results do not. Serialization is read-only:
`Configuration.from_yaml` loads, but there is no dump half.

</details>

## See also

Sibling projects in the [rekursiv-ai](https://github.com/rekursiv-ai) family:

- [sagent](https://github.com/rekursiv-ai/sagent) — The self-mutating multi-provider coding-agent CLI and typed Python library.
- [trackinizer](https://github.com/rekursiv-ai/trackinizer) — Centralized agent database for tracking inquiries, work, and the evidence behind conclusions.
- [wesearch](https://github.com/rekursiv-ai/wesearch) — Web search, resilient page fetch, and scholarly-paper lookup without a browser stack.
- [madcatter](https://github.com/rekursiv-ai/madcatter) — Rich-based Markdown renderer for the terminal; ships the `mdcat` CLI.
- [priml](https://github.com/rekursiv-ai/priml) — Composable PyTorch building blocks: models, optimizers, losses, and a step-based training loop.
- [copybarista](https://github.com/rekursiv-ai/copybarista) — Bidirectional source sync for publishing OSS-ready trees from a monorepo.
- [sudoku](https://github.com/rekursiv-ai/sudoku) — Sudoku-Extreme solved end to end with a 7M-parameter recursive transformer.

## Citing

If you find our work useful, please consider citing:

```bibtex
@misc{rekursivai2026configgle,
      title={Configgle - Type-safe hierarchical experiment configuration using pure Python dataclass factories and dependency injection.},
      author={Joshua V. Dillon},
      year={2026},
      howpublished={Github},
      url={https://github.com/rekursiv-ai/configgle},
}
```
