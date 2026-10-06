# Changelog

All notable configgle changes are documented here. This project follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/).

Releases up to and including 1.4.0 are described in the
[GitHub release notes](https://github.com/rekursiv-ai/configgle/releases).

## Unreleased

### Removed

- **Breaking:** `configgle.testing` and its `assert_pprint_golden`, added in
  1.4.1. The helper moved to priml as
  `priml.testing.golden.assert_pprint_golden`, which takes the same
  `test_file`, `name`, and `config` arguments, writes the same
  `testdata/<name>.txt` goldens, and adds an optional `normalize`. It
  regenerates with `--regenerate-golden`; `CONFIGGLE_REGENERATE_GOLDEN=1` is no
  longer read. Import it from priml, or compare
  `config.pformat(finalize=True, mask_memory_addresses=True,
  hide_default_values=False)` against your own golden.

## 1.4.1 - 2026-10-03

### Added

- `traverse(root, cls)` and `Match`, exported from `configgle`, walk a config
  tree and yield every node that is an instance of `cls`. A `Match` holds the
  node, its dotted path (`blocks[0].attn`, `extras['aux']`), and
  `replace(new)`, which writes into the parent slot: an attribute, a list
  index, a dict key, or a tuple position. A shared node is yielded once. With
  `recurse=True`, matches nested inside a match are yielded too, parents
  first.
- `LateBound`, exported from `configgle`, lets a built object reference a
  sibling it could not see at construction, such as a tied weight named by
  path. After the outermost `make()` finishes, each `LateBound` reachable from
  the built object gets `bind(root)` once, with the root object. Subclassing
  is the opt-in; an unrelated method named `bind` is never called.
- `Maker.finalized()` and `InlineConfig.finalized()` return a finalized copy,
  skipping `finalize()` when the config is already finalized. `make()` builds
  from it, and an overridden `make` should too.
- `configgle.testing.assert_pprint_golden(test_file=..., name=..., config=...)`
  compares a config's finalized, full `pformat` with `testdata/<name>.txt`
  next to the test file. A missing golden is written and the assertion fails
  so it can be reviewed. `CONFIGGLE_REGENERATE_GOLDEN=1` rewrites existing
  goldens.
- `configgle.custom_json.encode_graph` and `decode_graph` serialize any object
  graph, not just a config. `decode_graph` refuses imports and `__reduce__`
  calls unless a `DecodeCapabilities` grants them. `Fig.deserialize` grants
  both, as before.

### Changed

- `make()` walks the built object once after the outermost build, looking for
  `LateBound` objects, even when none exist. A torch module is searched
  through `modules()`; any other object through its attributes and
  containers. The walk visits every Python object the built object holds, so
  its cost grows with them: building an object that holds two million small
  lists takes about a second longer than in 1.4.0.
- **Breaking:** `Config.update(source)` raises `AttributeError` when `source`
  has a field the config does not declare. 1.4.0 skipped such fields
  silently. Pass `skip_missing=True` to keep skipping them.
- **Breaking:** `@autofig` raises `TypeError` at decoration time for
  constructors it cannot turn into a `Config`: positional-only, `*args`, or
  `**kwargs` parameters; a parameter named after a `Config` member (`make`,
  `update`, `finalize`, ...) or a dunder; a `staticmethod` or `classmethod`
  `__init__`; and a class that defines only `__new__`.
- `@autofig` resolves each annotation separately. An unresolved forward
  reference is kept and resolved later, instead of every field's type
  becoming `object`, and an invalid annotation falls back to `object` for
  that field only. Defaults keep their identity, so a mutable default such as
  `[]` is now accepted (1.4.0 raised `ValueError`) and shared just as the
  constructor shares it.
- `@autofig` names the generated class `<Class>.Config`, in the decorated
  class's module. `pformat` prints `Foo.Config(...)` rather than
  `Config(...)`, and the config can now be pickled and serialized; both
  failed in 1.4.0.
- `serialize()` writes `pathlib` paths, datetimes, and UUIDs with the compact
  tags `py/path`, `py/datetime`, and `py/uuid`. Path payloads are now the
  same on Python 3.12 and 3.14. Payloads written by 1.4.0 still load, but
  1.4.0 reads the new tags back as plain dicts without raising, so upgrade
  every reader before any writer.
- `pformat` and `pprint` output changed in ways that affect snapshot tests:
  - Function reprs include the module: `<function mymodule.act at ...>`.
  - Masked memory addresses use one fixed placeholder, `0xdefacedeface`, on
    every platform. Only addresses after ` at ` and outside string literals
    are masked.
  - `hide_default_values` (on by default) hides only fields equal to a
    literal default. Fields with a `default_factory`, such as lists, dicts,
    and nested configs, are always shown. The printer no longer builds a
    default instance to compare against.
  - A config nested in a list, dict, or other container is finalized for
    display like a top-level one. 1.4.0 printed it unfinalized and warned
    about a "potentially unfinalized dataclass".
- Type annotations: `Maker.serialize()` returns `object` instead of `Any`,
  and `RelaxedMakeable.parent_class` is the property inherited from
  `Makeable` instead of a `ClassVar`.

### Removed

- **Breaking:** the `configgle.serialize` module and its `serialize`,
  `deserialize`, and `Hooks`. Use the methods (`cfg.serialize(hooks=...)`,
  `Cls.Config.deserialize(tree, hooks=...)`) or `configgle.custom_json`'s
  `encode_graph` and `decode_graph` with `GraphHooks`. To match the old
  `deserialize`, call `decode_graph(tree, hooks=...,
  capabilities=DecodeCapabilities(resolve=resolve_import, apply_reduce=True))`.
- **Breaking:** the older helpers in `configgle.custom_json`: `JsonCodec`,
  `dataclass_to_json`, `dataclass_from_json`, and the coercers `bool_val`,
  `int_val`, `float_val`, `str_val`, `list_val`, `dict_val`, `dicts_val`,
  and `datetime_val`. The module now provides typed codecs (`Codec` and its
  subclasses) with `decode`, `decode_or_none`, and `take`.

### Fixed

- Every config starts with `_finalized=False`, set at allocation, whether it
  was constructed, copied with `copy.copy`, unpickled, or deserialized. In
  1.4.0 the attribute was missing until `finalize()` ran.
- `finalize()` resets `_finalized` to `False` when a nested `finalize`
  raises. Before, the flag stayed `True`, and a later `make()` on a copy
  skipped finalization and built from underived defaults.
- `hide_default_values` works on configs with required fields. 1.4.0 showed
  every field of such a config.
- `pformat` no longer joins a multi-line value onto one line when that would
  change whitespace inside a string literal.
- `InlineConfig` equality compares the callable, arguments, and keywords.
  `==` raised `AttributeError` in 1.4.0.
- A command-line override of a `tuple` field produces a tuple: `t=[1,2]`
  gives `(1, 2)`, not a list.
- Defining a `Fig` subclass with class keyword arguments no longer raises
  `TypeError` in the metaclass.
