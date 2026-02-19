# configgle🤭
Hierarchical configuration using pure Python dataclasses, with typed factory
methods, covariant protocols, and full inheritance support.

## Installation

```bash
python -m pip install configgle
```

## Example

```python
from configgle import Fig

class Model:
    class Config(Fig):
        hidden_size: int = 256
        num_layers: int = 4

    def __init__(self, config: Config):
        self.config = config

# Create and modify config
cfg = Model.Config()
cfg.hidden_size = 512

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
    cfg.hidden_size = 512
    cfg.num_layers = 8
    return cfg
```

Or use `@autofig` to auto-generate the Config from `__init__`:

```python
from configgle import autofig
from torch import nn

@autofig
class Model(nn.Module):
    def __init__(self, hidden_size: int = 256, num_layers: int = 4):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers

# Config is auto-generated from __init__ signature
model = Model.Config(hidden_size=512).make()
print(model.hidden_size)  # 512
```

## Features

### Type-safe `make()`

When `Config` is defined as a nested class, `MakerMeta.__get__` uses the
descriptor protocol to infer the parent class automatically. The return type
of `__get__` is `Intersection[type[Config], type[Makeable[Parent]]]`, so
`make()` knows the exact return type with zero annotation effort:

```python
class Model:
    class Config(Fig):
        hidden_size: int = 256

    def __init__(self, config: Config):
        self.hidden_size = config.hidden_size

model = Model.Config(hidden_size=512).make()  # inferred as Model
```

Type checkers that support `Intersection` (like `ty`) resolve this fully --
bare `Fig` is all you need. For type checkers that don't yet support
`Intersection` (like `basedpyright`), parameterize with the parent class
name to give the checker the same information explicitly:

```python
class Model:
    class Config(Fig["Model"]):  # explicit type parameter only for basedpyright
        hidden_size: int = 256

    def __init__(self, config: Config):
        self.hidden_size = config.hidden_size

model: Model = Model.Config(hidden_size=512).make()  # returns Model, not object
```

Without `["Model"]`, non-`ty` checkers fall back to `Any` (so attribute access
works without typecheck suppressions).

Both `ty` and `basedpyright` are first-class supported. Here's the full
picture (including [`Makes`](#inheritance-with-makes), introduced next):

| | `ty` | `basedpyright` |
|---|:---:|:---:|
| Bare `Fig` infers parent type | ✅ | ❌ (`Any` fallback) |
| Inheritance infers parent type | ✅ | 🟡 (Needs `Makes["Child"]`) |
| Explict `Fig["Parent"]` | ✅ | ✅ |
| `@autofig` `.Config` access | ❌ ([#143](https://github.com/astral-sh/ty/issues/143)) | ✅ |

`ty` gets full inference from `Intersection` -- bare `Fig` and inherited
configs just work. `basedpyright` doesn't support `Intersection` yet, so it
needs explicit `Fig["Parent"]` and `Makes["Child"]` annotations. `ty` doesn't
yet support class decorator return types, so `@autofig`-decorated classes need
`# ty: ignore[unresolved-attribute]` to access `.Config`; `basedpyright`
handles this correctly. When `Intersection` lands in the
[type spec](https://github.com/python/typing/issues/213), `Makes` becomes
unnecessary and both checkers will infer everything from bare `Fig`.

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
or custom class with `make()`, `finalize()`, and `update()`. Because it's
covariant, `Makeable[Dog]` is assignable to `Makeable[Animal]`:

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

### Nested config finalization

Override `finalize()` to compute derived fields before instantiation. Nested
configs are finalized recursively:

```python
from configgle import Configurable  # Just an alias to Makeable.
from dataclasses import field

class Encoder:
    class Config(Fig):
        c_in: int = 256
        mlp: Configurable[nn.Module] = field(default_factory=MLP.Config)

        def finalize(self) -> Self:
            self = super().finalize()
            self.mlp.c_in = self.c_in  # propagate dimensions
            return self
```

### `update()` for bulk mutation

Configs support bulk updates from another config or keyword arguments:

```python
cfg = Model.Config(hidden_size=256)
cfg.update(hidden_size=512, num_layers=8)

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
cfg.out_features = 64     # attribute-style access to kwargs
layer = cfg.make()        # calls nn.Linear(in_features=256, out_features=64, bias=False)
y = layer(x)              # use the constructed module
```

`PartialConfig` is shorthand for `InlineConfig(functools.partial, fn, ...)`
-- use it for functions where some arguments aren't known at config time:

```python
from configgle import PartialConfig
import torch.nn.functional as F

cfg = PartialConfig(F.cross_entropy, label_smoothing=0.1)
loss_fn = cfg.make()      # returns functools.partial(F.cross_entropy, label_smoothing=0.1)
loss = loss_fn(logits, targets)  # calls F.cross_entropy(logits, targets, label_smoothing=0.1)
```

Nested configs in args/kwargs are finalized and `make()`-d recursively, so
both compose naturally with `Fig` configs.

### `CopyOnWrite`

`CopyOnWrite` wraps a config tree and lazily copies objects only when mutations
occur. Copies propagate up to parents automatically, so the original is never
touched. This is especially useful inside `finalize()`, where you want to
derive a variant of a shared sub-config without mutating the original:

```python
from configgle import CopyOnWrite, Fig

class Encoder:
    class Config(Fig):
        hidden_size: int = 256
        encoder: Configurable[nn.Module] = field(default_factory=MLP.Config)
        decoder: Configurable[nn.Module] = field(default_factory=MLP.Config)

        def finalize(self) -> Self:
            self = super().finalize()
            # encoder and decoder can share the same MLP.Config object.
            # CopyOnWrite lets us tweak the decoder's copy without
            # touching the encoder's (or the shared original).
            with CopyOnWrite(self) as cow:
                cow.decoder.c_out = self.hidden_size * 2
            return cow.unwrap
```

Only the mutated nodes (and their ancestors) are shallow-copied; everything
else stays shared.

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
        hidden_size: int = 256
        num_layers: int = 4
        mlp: Configurable[nn.Module] = field(default_factory=MLP.Config)
        output_mlp: Configurable[nn.Module] = field(default_factory=MLP.Config)
    def __init__(self, config: Config): ...

def exp001():
    cfg = Model.Config()
    cfg.hidden_size = 512
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
#    hidden_size=512,
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
one line.

### `@autofig` for zero-boilerplate configs

When you don't need a hand-written Config, `@autofig` generates one from
`__init__` (see [Example](#example) above).

### Pickling and cloudpickle

Configs are fully compatible with `pickle` and `cloudpickle`, including the
parent class reference. This is important for distributed workflows (e.g.,
sending configs across processes):

```python
import cloudpickle, pickle

cfg = Model.Config(hidden_size=512)
cfg_ = pickle.loads(cloudpickle.dumps(cfg))
model = cfg_.make()  # parent_class is preserved
```

## Comparison

| | [configgle](https://github.com/jvdillon/configgle) | [Hydra](https://github.com/facebookresearch/hydra) | [Sacred](https://github.com/IDSIA/sacred) | [OmegaConf](https://github.com/omry/omegaconf) | [Gin](https://github.com/google/gin-config) | [ml_collections](https://github.com/google/ml_collections) | [Fiddle](https://github.com/google/fiddle) | [Confugue](https://github.com/cifkao/confugue) |
|---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| Pure Python (no YAML/strings) | ✅ | ❌ | ❌ | 🟡 | ❌ | ✅ | ✅ | ❌ |
| Typed `make()`/`build()` return | ✅ | ❌ | ❌ | ❌ | ❌ | ❌ | ✅ | ❌ |
| Config inheritance | ✅ | 🟡 | ❌ | 🟡 | ❌ | ❌ | ❌ | 🟡 |
| Covariant protocol | ✅ | ❌ | ❌ | ❌ | ❌ | ❌ | ❌ | ❌ |
| Nested finalization | ✅ | ❌ | ❌ | ❌ | ❌ | ❌ | ❌ | ❌ |
| Copy-on-write | ✅ | ❌ | ❌ | ❌ | ❌ | ❌ | ❌ | ❌ |
| `pickle`/`cloudpickle` | ✅ | 🟡 | ❌ | ✅ | ❌ | 🟡 | ✅ | ❌ |
| Auto-generated configs | ✅ | 🟡 | ❌ | ❌ | ❌ | ❌ | ✅ | ❌ |
| GitHub stars | -- | 10.2k | 4.4k | 2.3k | 2.1k | 1.0k | 374 | 21 |

✅ = yes, 🟡 = partial, ❌ = no. Corrections welcome --
[open a PR](https://github.com/jvdillon/configgle/pulls).

### How each library works

**[Hydra](https://github.com/facebookresearch/hydra)** (Meta) --
YAML-centric with optional "structured configs" (Python dataclasses registered
in a `ConfigStore`). Instantiation uses `hydra.utils.instantiate()`, which
resolves a string `_target_` field to an import path -- the return type is
`Any`. Config composition is done via YAML defaults lists, not class
inheritance. Dataclass inheritance works at the schema level. `configen` is
an experimental code-generation tool (v0.9.0.dev8) that produces structured
configs from class signatures. Configs survive pickle trivially since
`_target_` is a string, not a class reference.

**[Sacred](https://github.com/IDSIA/sacred)** --
Experiment management framework. Config is defined via `@ex.config` scopes
(local variables become config entries) or loaded from YAML/JSON files. Sacred
auto-*injects* config values into captured functions by parameter name
(dependency injection), but does not auto-*generate* configs from function
signatures. No typed factory methods, no config inheritance, no pickle
support for the experiment/config machinery.

**[OmegaConf](https://github.com/omry/omegaconf)** --
YAML-native configuration with a "structured config" mode that accepts
`@dataclass` schemas. Configs are always wrapped in `DictConfig` proxy objects
at runtime (not actual dataclass instances). Supports dataclass inheritance
for schema definition. Good pickle support (`__getstate__`/`__setstate__`).
No factory method (`to_object()` returns `Any`), no auto-generation, no
protocols.

**[Gin](https://github.com/google/gin-config)** (Google) --
Global string-based registry. You decorate functions with `@gin.configurable`
and bind parameters via `.gin` files or `gin.bind_parameter('fn.param', val)`.
There are no config objects -- parameter values live in a global dict keyed by
dotted strings. No typed returns, no config inheritance. The docs state
"gin-configurable functions are not pickleable," though a 2020 PR added
`__reduce__` methods that improve support.

**[ml_collections](https://github.com/google/ml_collections)** (Google) --
Dict-like `ConfigDict` with dot-access, type-checking on mutation, and
`FieldReference` for lazy cross-references between values. Pure Python, no
YAML. No factory method or typed instantiation. Pickle works for plain configs,
but `FieldReference` operations that use lambdas internally (`.identity()`,
`.to_int()`) fail with standard pickle (cloudpickle handles them).

**[Fiddle](https://github.com/google/fiddle)** (Google) --
Python-first. You build config graphs with `fdl.Config[MyClass]` objects and
call `fdl.build()` to instantiate them. `build(Config[T]) -> T` is typed via
`@overload`. Config modification is functional (`fdl.copy_with`), not
inheritance-based -- there are no config subclasses. `@auto_config` rewrites a
factory function's AST to produce a config graph automatically. Full
pickle/cloudpickle support.

**[Confugue](https://github.com/cifkao/confugue)** --
YAML-based hierarchical configuration. The `configure()` method instantiates
objects from YAML dicts, with the class specified via a `!type` YAML tag.
Returns `Any`. Partial config inheritance via YAML merge keys (`<<: *base`).
No pickle support, no auto-generation, no protocols.

## Citing

If you find our work useful, please consider citing:

```bibtex
@misc{dillon2026configgle,
      title={Configgle - Hierarchical experiment configuration using pure Python dataclass factories and dependency injection.},
      author={Joshua V. Dillon},
      year={2026},
      howpublished={Github},
      url={https://github.com/jvdillon/configgle},
}
```

## License

Apache License 2.0
