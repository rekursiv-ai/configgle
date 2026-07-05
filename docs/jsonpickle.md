# Why configgle ships a bespoke serializer instead of jsonpickle

`serialize`/`deserialize` are configgle's own structural walk, not a wrapper over
[jsonpickle](https://jsonpickle.github.io).

**We roll our own implementation, but the wire format is jsonpickle's.** configgle
emits jsonpickle's `py/*` tag vocabulary (`py/object`, `py/id`, `py/tuple`,
`py/reduce`, `json://` keys, positional references) deliberately, so a configgle
tree is legible to anyone who knows jsonpickle and either decoder can read the
other's output. What we replaced is the *implementation*, not the *format*.

We seriously evaluated adopting jsonpickle itself first. This document records
why we tried it, how far we got, and why we still went bespoke.

The full jsonpickle-backed implementation is preserved in git history (the
"Adopt jsonpickle-backed config serialization" commit); the next commit
("Replace jsonpickle path with native serializer") swaps it for the bespoke
version. Read those two commits for the actual code -- this doc is rationale
only, so it does not duplicate what the SHAs already capture.

## The effort we made to use jsonpickle

The jsonpickle-backed version was real, complete, and green. It:

- round-tripped the whole test suite plus the real `baselines/arcagi1` config;
- handled nested Figs, polymorphic slots, DAG identity, `torch.dtype`,
  `MappingProxyType`, tuples/sets, enums, hooks, and faithful finalized-state;
- routed Figs through `Maker.__getstate__`/`__setstate__` so jsonpickle would
  not choke on the read-only `parent_class` descriptor;
- subclassed `Pickler`/`Unpickler` to close jsonpickle's lenient-by-default gaps
  (escape reserved-tag data keys, reject lambdas / local classes / unpicklable
  leaves loudly, raise on a dangling back-reference);
- shipped hand-written type stubs (jsonpickle has none) and lazily imported the
  library so importing configgle stayed cheap.

We also fixed one of jsonpickle's bugs upstream:
<https://github.com/jsonpickle/jsonpickle/pull/611> (reserved-tag dict keys were
silently dropped before the picklability check).

In short: it was not a strawman. It worked.

## Why we chose bespoke anyway

The decisive factor was not "jsonpickle is broken" -- it is mature and
well-tested. It is that making jsonpickle meet a *config library's* contract
required so much of it that the reuse argument collapsed:

- **We were overriding its internals, not reusing its behavior.** Six
  underscore-prefixed `Pickler`/`Unpickler` methods, plus native pickle-state
  hooks on `Maker`, plus per-type handlers. Coupling to a library's private
  methods is fragile -- a patch release can rename them and silently break us.
- **Its default posture is lenient; ours must be strict.** jsonpickle silently
  drops unpicklable leaves, degrades types, and emits non-strict JSON
  (`Infinity`/`NaN`). A config library must fail loudly and preserve types.
  Every one of those gaps was closable, but only by re-adding the exact
  structural walking that adopting jsonpickle was meant to delete.
- **The line-count win was illusory.** The wrapper looked small until you count
  the stubs, the state hooks, the six overrides, and the two external
  dependencies it drags in. The bespoke serializer is larger in one file but
  removes all of that and depends on nothing.
- **Correctness.** With the bespoke serializer the full suite passes with zero
  `xfail`; the jsonpickle path carried known gaps (set/frozenset used as a cycle
  target) that only its unreleased version fixes, and even then only for
  non-Fig anchors.

For a configuration library whose entire job is to faithfully persist and reload
experiment configs, silent data loss, silent type degradation, and non-strict
JSON are disqualifying. The bespoke serializer handles every case correctly, is
covered by the test suite, and carries no runtime dependency -- while still
speaking jsonpickle's `py/*` wire format for interoperability.
