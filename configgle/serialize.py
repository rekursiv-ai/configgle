"""Structural (de)serialization for the config tree: ``serialize`` / ``deserialize``.

The third structural walk of the config lifecycle, alongside ``copy_tree`` and
``_finalize_value`` in ``walk.py``. It mirrors them shape-for-shape -- the same
Figs, slotted data objects, lists, dicts (keys included), sets, and tuples are
visited -- but instead of copying or finalizing it emits a JSON-encodable tree
(dicts, lists, primitives), and reverses that tree back into live objects.

``serialize`` returns the encodable **tree**, not a string: the caller picks the
transport (``json.dumps(cfg.serialize())``, YAML, msgpack, or embedding it in a
larger structure). ``deserialize`` reverses a tree the same way.

``serialize`` does **not** finalize: it captures the raw config exactly as the
caller built it (derived defaults are left to a later ``finalize``/``make`` on
the deserialized tree). The traversal reuses ``walk._get_object_attribute_names``
so field discovery stays identical to the copy/finalize walks.

Wire format
-----------
The wire format follows jsonpickle's conventions (https://jsonpickle.github.io):
plain data (primitives, finite floats, lists, str-keyed dicts) stays NATIVE
JSON; a typed value becomes a dict tagged with a ``py/`` key. This is a JSON
port of pickle's reduce protocol, and its tag vocabulary is jsonpickle's
``py/*`` tags -- so a configgle tree is legible to anyone who knows jsonpickle.

Tags (import paths are DOTTED, ``module.qualname``, like jsonpickle)::

    {"py/object": "mod.Cls", <state fields flat>}   a Fig / dataclass
    {"py/type": "mod.Cls"}                          a class, referenced
    {"py/function": "mod.fn"}                        a function, referenced
    {"py/tuple": [...]}                              a tuple
    {"py/set": [...]}                                a set
    {"py/b64": "<base64>"}                           bytes
    {"py/float": "inf"}                              a non-finite float
    {"py/reduce": [<callable>, <args>, ...]}         a ``__reduce__`` leaf
                                                      (frozenset, namedtuple,
                                                      OrderedDict, Path, Decimal,
                                                      Enum, dtype, any picklable)
    {"py/id": n}                                     a back-reference (below)
    {"lr": 0.01}                                     a plain str dict -- NATIVE
    [1, 2, 3]                                         a plain list -- NATIVE

A native str-keyed dict whose key would masquerade as a wire tag (starts with
``py/`` or ``json://``) is json:// escaped so decode never mistakes data for a
node. ``py/float`` (and ``py/b64``/``py/type``/``py/tuple``) are value leaves --
they consume no ``py/id`` encounter index.

References follow jsonpickle's POSITIONAL scheme: every mutable object emitted
is implicitly numbered by encounter order (0, 1, 2, ...) -- the FIRST full
encoding carries no marker. A repeat of the same object (a DAG or a cycle) is
``{"py/id": n}``, an index into that encounter order. The decoder rebuilds the
same order, so the index resolves to the identical object -- preserving shared
identity and terminating cycles, with no explicit id on the original.

Non-str dict keys use jsonpickle's ``keys`` convention: a key ``k`` is emitted
as ``"json://" + json.dumps(encode(k))`` (a plain str key stays as-is), so a
str-keyed dict is a native JSON object and an int/tuple-keyed one still round-
trips.

configgle extension: ``{"py/hook": [<type>, payload]}`` and
``{"py/inline": [<type>, {func, args, kwargs}]}`` cover the hook and
``InlineConfig`` cases jsonpickle has no tag for.

Identity
--------
A MUTABLE object is numbered by encounter index before its children are encoded;
a later occurrence is ``{"py/id": n}`` -- so one shared by two fields (a DAG) or
reached again via a cycle stays shared and any cycle terminates (mirroring
``copy_tree``'s ``visited`` map). Registered (identity-preserving): ``py/object``
(Fig), native ``list``/``dict``, ``py/set``, ``py/inline``, ``py/hook``, and a
``py/reduce`` object that is BUILT-THEN-MUTATED (its reduce carries state /
listitems / dictitems -- e.g. ``OrderedDict`` or a stateful object).

An IMMUTABLE value -- a bare ``tuple``/``frozenset``/namedtuple, and a
``py/reduce`` leaf whose reduce is a pure ``(callable, args)`` (frozenset, Path,
Decimal, enum, dtype) -- is NOT registered: it constructs atomically, round-trips
by VALUE (equal, not necessarily the same object), and cannot be a cycle TARGET
(it is not yet built while its own args decode, so a back-reference to it could
not resolve). A cycle that routes THROUGH an immutable still closes: its anchor
is the enclosing mutable node (e.g. a Fig inside a frozenset that points back at
the Fig -- entered via the Fig -- closes on the Fig's ``py/id``). Value leaves
(``py/tuple``, ``py/b64``, ``py/float``, ``py/type``, ``py/function``) likewise
consume no index.

A hooked leaf ITSELF is registered, so two fields holding the same hooked object
stay shared. Its ``encode``d PAYLOAD is opaque, though -- configgle does not walk
inside it -- so an object shared between a hook payload and the surrounding tree
is NOT deduplicated across that boundary (it round-trips by value, not identity).

Leaf taxonomy
-------------
Matches ``copy_tree``/``_finalize_value``: primitives, ``bytes``, non-finite
floats, and scalar subclasses (``IntEnum`` etc.) are all leaves. ``bytes`` is
base64 (``py/b64``); a non-finite float is a ``py/float`` value token; every
other scalar/opaque leaf goes through the ``__reduce__`` fallback (``py/reduce``).
configgle never imports the defining library (torch/numpy/...): it records import
paths and imports lazily at decode time.

Opaque leaves whose reduce carries mutable state, or a non-importable
reconstructor, are NOT auto-handled: supply a ``hooks`` mapping
``{type: (encode, decode)}``; without a hook such a leaf raises ``TypeError``.

Security
--------
``deserialize`` imports the modules named in the payload and calls the resolved
classes/functions. Treat a serialized config like ``pickle``: only deserialize
data you trust. There is no sandboxing.
"""

from __future__ import annotations

from collections.abc import (
    Callable,
    Mapping,
    Sequence,
    Set as AbstractSet,
)
from types import MappingProxyType
from typing import Any, Protocol, cast, runtime_checkable

import base64
import importlib
import json
import math

from configgle.inline import InlineConfig
from configgle.walk import _get_object_attribute_names


__all__ = [
    "deserialize",
    "serialize",
]

# The wire format uses jsonpickle's fixed ``py/*`` tag strings inline
# (https://jsonpickle.github.io); ``py/hook``/``py/inline``/``py/float`` are
# configgle additions. They are protocol constants, not tunables.


def _is_reserved_key(key: str) -> bool:
    """True if ``key`` would masquerade as a wire tag (``py/...`` or ``json://...``)."""
    return key.startswith(("py/", "json://"))


# Hook maps a concrete leaf type to (encode, decode) callables.
type Hooks = Mapping[type, tuple[Callable[[Any], Any], Callable[[Any], Any]]]


def serialize(obj: object, *, hooks: Hooks | None = None) -> Any:
    """Serialize a config tree (or any config value) to an encodable dict tree.

    Returns a JSON-encodable structure (nested dicts/lists/primitives), not a
    string -- the caller picks the transport
    (``json.dumps(serialize(cfg), indent=2)``, YAML, msgpack, or embedding it in
    a larger dict). The tree follows jsonpickle's ``py/*`` conventions. The
    config is captured as-is: ``serialize`` does not finalize, so derived
    defaults are left for a later ``finalize``/``make`` on the deserialized tree.

    Args:
      obj: The config, container, or value to serialize.
      hooks: Optional ``{type: (encode, decode)}`` map for leaves JSON cannot
        represent natively (tensors, arrays, etc.). ``encode`` turns the leaf
        into a JSON-encodable payload.

    Returns:
      tree: An encodable dict/list/primitive tree that ``deserialize`` reverses
        into live objects.

    Raises:
      TypeError: If an opaque leaf has no matching hook.

    """
    return _Encoder(hooks or {}).encode(obj)


def deserialize(tree: object, *, hooks: Hooks | None = None) -> Any:
    """Reconstruct live objects from a tree produced by ``serialize``.

    Config classes and callables are resolved by their recorded import path, so
    the modules that define them must be importable. This imports and calls code
    named in the payload; deserialize only trusted data (see module docstring).

    Args:
      tree: An encodable tree produced by ``serialize`` (e.g. from
        ``json.loads`` of a stored string).
      hooks: The same ``{type: (encode, decode)}`` map used to serialize; the
        ``decode`` half rebuilds each hooked leaf.

    Returns:
      obj: The reconstructed config tree (or value).

    """
    return _Decoder(hooks or {}).decode(tree)


@runtime_checkable
class _Named(Protocol):
    """A class or function: carries both ``__module__`` and ``__qualname__``."""

    __module__: str
    __qualname__: str


def _dotted_name(obj: object) -> str:
    """Return the dotted ``module.qualname`` path for a class or function."""
    # The single choke point for every recorded reference: rejecting a non-
    # importable path (``<locals>``) here makes a local class / lambda fail loudly
    # at serialize time rather than producing an un-loadable tree.
    if not isinstance(obj, _Named) or "<locals>" in obj.__qualname__:
        raise TypeError(
            f"Cannot serialize {obj!r}: it has no importable path "
            f"(module-level __qualname__). Local/lambda callables and local "
            f"classes/subclasses cannot be deserialized.",
        )
    path = f"{obj.__module__}.{obj.__qualname__}"
    try:
        resolved = _resolve(path)
    except (AttributeError, ImportError) as error:
        raise TypeError(
            f"Cannot serialize {obj!r}: import path {path!r} does not resolve "
            "to the same object.",
        ) from error
    if resolved is not obj:
        raise TypeError(
            f"Cannot serialize {obj!r}: import path {path!r} does not resolve "
            "to the same object.",
        )
    return path


def _resolve(path: str) -> Any:
    """Import and return the object named by a dotted ``module.qualname`` path."""
    # Import the longest importable prefix, then walk the remaining dotted parts
    # as attributes -- so ``mod.Foo.Config`` resolves even though ``mod.Foo`` is
    # not itself a module.
    parts = path.split(".")
    for split in range(len(parts) - 1, 0, -1):
        module_name = ".".join(parts[:split])
        try:
            module = importlib.import_module(module_name)
        except ImportError:
            continue
        obj: Any = module
        for part in parts[split:]:
            obj = getattr(obj, part)
        return obj
    raise ImportError(f"Cannot resolve path: {path!r}")


class _Encoder:
    """Encode a config tree into a jsonpickle-format JSON-encodable structure.

    Every mutable object is numbered by encounter order (``_seen``); a repeat is
    emitted as ``{"py/id": n}`` -- jsonpickle's positional reference scheme.
    """

    def __init__(self, hooks: Hooks) -> None:
        self._hooks = hooks
        # Maps id(obj) -> encounter index for objects already emitted.
        self._seen: dict[int, int] = {}
        # Objects kept alive so their id() is not reused before serialize ends.
        self._alive: list[object] = []

    def encode(self, value: object) -> Any:
        value_type = type(value)
        if value is None or value_type in (bool, int, str):
            return value
        if value_type is float:
            fvalue = cast(float, value)
            if not math.isfinite(fvalue):
                # inf/nan are not valid strict JSON. A NON-registering value tag
                # (like py/b64) -- NOT py/reduce -- so it consumes no encounter
                # index on either side (a float is a value leaf, never shared).
                return {"py/float": repr(fvalue)}
            return value
        if value_type is bytes:
            return {"py/b64": base64.b64encode(cast(bytes, value)).decode("ascii")}
        if isinstance(value, type):
            return {"py/type": _dotted_name(value)}

        # An already-emitted object (shared or cyclic): reference by index.
        # Checked before hooks so a shared/cyclic hooked leaf shares identity.
        ref = self._reference(value)
        if ref is not None:
            return ref

        hook = self._hooks.get(type(value))
        if hook is not None:
            return self._encode_hook(value, hook[0])

        if callable(value) and not isinstance(
            value, (InlineConfig, tuple, Sequence, Mapping, AbstractSet)
        ):
            return {"py/function": _dotted_name(value)}
        if isinstance(value, InlineConfig):
            return self._encode_inline(cast("InlineConfig[object]", value))
        if type(value) is tuple:
            # A namedtuple (tuple SUBCLASS) reduces instead, to keep its type;
            # only a bare tuple uses py/tuple.
            tup = cast("tuple[object, ...]", value)  # ty: ignore[redundant-cast]
            return {"py/tuple": self._encode_items(tup)}
        if type(value) is list:
            lst = cast("list[object]", value)
            self._register(lst)
            return [self.encode(v) for v in lst]
        if type(value) is set:
            # A set SUBCLASS reduces; only a bare set uses py/set.
            return self._encode_set(cast("AbstractSet[object]", value))
        if type(value) is dict:
            return self._encode_mapping(cast("Mapping[object, object]", value))
        # A list/dict/set SUBCLASS, a namedtuple, or a frozenset -- anything whose
        # exact type matters -- falls through to the __reduce__ fallback, which
        # records the type and reconstructs it faithfully.
        # A configgle config (a dataclass -- Fig / Dataclass) encodes as a
        # py/object with its named fields flat. A foreign slotted object (Path,
        # ...) is NOT a config: prefer its __reduce__, falling back to raw-slot
        # field encoding only if it has no usable reduce.
        if hasattr(type(value), "__dataclass_fields__"):
            return self._encode_object(value)

        reduced = self._encode_reducible(value)
        if reduced is not None:
            return reduced

        if _is_data_object(value):
            return self._encode_object(value)

        # MappingProxyType is a read-only dict view that CPython cannot pickle
        # (``__reduce__`` raises), but ``MappingProxyType(d)`` reconstructs it, so
        # emit that recipe as a py/reduce to preserve the wrapper type faithfully.
        # An atomic ``(callable, args)`` reduce -- NOT registered, so it round-trips
        # by value (like frozenset / Path); the inner dict still carries identity.
        if type(value) is MappingProxyType:
            # basedpyright sees MappingProxyType[Unknown, Unknown] here; the cast
            # gives dict() a concrete element type.
            proxy = cast("Mapping[object, object]", value)
            return {
                "py/reduce": [
                    {"py/type": "types.MappingProxyType"},
                    {"py/tuple": [self._encode_mapping(dict(proxy))]},
                ]
            }

        # An exotic container that neither reduces nor is a config (e.g. a local
        # list subclass) degrades to its BASE container by contents -- its wrapper
        # type is lost, its data round-trips. Keeps serialize total over odd
        # container types (configgle extension; jsonpickle would reduce or fail).
        if isinstance(value, Mapping):
            return self._encode_mapping(cast("Mapping[object, object]", value))
        if isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
            self._register(value)
            return [self.encode(v) for v in cast("Sequence[object]", value)]
        if isinstance(value, AbstractSet):
            return self._encode_set(cast("AbstractSet[object]", value))

        raise TypeError(
            f"Cannot serialize leaf of type {type(value).__name__!r}. "
            f"Pass hooks={{{type(value).__name__}: (encode, decode)}} to serialize().",
        )

    def _reference(self, value: object) -> dict[str, Any] | None:
        """Return a ``{"py/id": n}`` back-reference if already emitted."""
        seen = self._seen.get(id(value))
        if seen is None:
            return None
        return {"py/id": seen}

    def _register(self, value: object) -> int:
        """Assign ``value`` the next encounter index (before encoding children).

        Registering first is what makes cycles terminate: a back-reference
        reached while encoding children finds ``value`` already numbered and
        emits ``{"py/id": n}`` instead of recursing forever.
        """
        index = len(self._seen)
        self._seen[id(value)] = index
        self._alive.append(value)  # keep alive so id() is not reused.
        return index

    def _encode_hook(
        self, value: object, encode: Callable[[Any], Any]
    ) -> dict[str, Any]:
        self._register(value)
        return {"py/hook": [_dotted_name(type(value)), encode(value)]}

    def _encode_items(self, values: Sequence[object]) -> list[Any]:
        return [self.encode(v) for v in values]

    def _encode_set(self, value: AbstractSet[object]) -> dict[str, Any]:
        # jsonpickle tags both set and frozenset py/set. A set is mutable so it
        # is registered for identity; a frozenset round-trips by value.
        if isinstance(value, set):
            self._register(value)
        return {"py/set": [self.encode(v) for v in value]}

    def _encode_mapping(self, value: Mapping[object, object]) -> Any:
        items = list(value.items())
        str_keyed = all(isinstance(k, str) for k, _ in items)
        # A plain str-keyed dict is NATIVE JSON -- unless a key would masquerade
        # as a wire tag (starts with "py/" or "json://"), in which case EVERY key
        # is json:// escaped so the whole dict is unambiguously data.
        needs_escape = any(isinstance(k, str) and _is_reserved_key(k) for k, _ in items)
        self._register(value)
        if type(value) is dict and str_keyed and not needs_escape:
            return {cast(str, k): self.encode(v) for k, v in items}
        return {self._encode_key(k): self.encode(v) for k, v in items}

    def _encode_key(self, key: object) -> str:
        """Encode a dict key as a JSON string, jsonpickle ``json://`` style."""
        if isinstance(key, str) and not _is_reserved_key(key):
            return key
        return "json://" + json.dumps(self.encode(key))

    def _encode_inline(self, value: InlineConfig[object]) -> dict[str, Any]:
        self._register(value)
        return {
            "py/inline": [
                _dotted_name(type(value)),
                {
                    "func": self.encode(value.func),
                    "args": self._encode_items(value._args),  # noqa: SLF001
                    "kwargs": {
                        k: self.encode(v)
                        for k, v in value._kwargs.items()  # noqa: SLF001
                    },
                },
            ]
        }

    def _encode_object(self, value: object) -> dict[str, Any]:
        # jsonpickle py/object: the dotted class path plus state fields FLAT.
        self._register(value)
        payload: dict[str, Any] = {"py/object": _dotted_name(type(value))}
        for name in _get_object_attribute_names(value):
            try:
                attr = getattr(value, name)
            except AttributeError:
                continue
            payload[name] = self.encode(attr)
        return payload

    def _encode_reducible(self, value: object) -> dict[str, Any] | None:
        """Encode an opaque leaf via the pickle ``__reduce_ex__`` protocol.

        Emits jsonpickle's ``py/reduce`` form ``[<type node>, <args node>]``.
        Reads the object's own reduce output and records the reconstruction
        recipe as import paths, so configgle serializes third-party leaves
        (``torch.dtype``, ``pathlib.Path``, enums, ...) WITHOUT importing their
        libraries. Returns None on any failure so the caller raises the clear
        opaque-leaf ``TypeError``.
        """
        reduce = getattr(value, "__reduce_ex__", None)
        if reduce is None:
            return None
        try:
            reduced: object = reduce(2)
        except Exception:  # noqa: BLE001 -- reduce is best-effort; any failure = no fallback.
            return None

        # Form 1: a bare string is a global-name reference in the object's module.
        if isinstance(reduced, str):
            return {"py/type": f"{type(value).__module__}.{reduced}"}

        # Form 2: the full pickle reduce tuple
        # (callable, args, state?, listitems?, dictitems?). Emitted as
        # jsonpickle's py/reduce -- a list of the 2-5 encoded elements. Decode
        # replays them: obj = callable(*args); setstate(state); extend(listitems);
        # update(dictitems). This is what makes frozenset / namedtuple /
        # OrderedDict / enum round-trip faithfully.
        if not isinstance(reduced, tuple):
            return None
        parts = list(cast("tuple[object, ...]", reduced))  # ty: ignore[redundant-cast]
        if not (2 <= len(parts) <= 5) or not callable(parts[0]):
            return None
        if not isinstance(parts[1], tuple):
            return None
        # Elements 4 and 5 (listitems / dictitems) are ITERATORS; materialize
        # them to concrete sequences before encoding. dictitems -> list of pairs.
        if len(parts) >= 4 and parts[3] is not None:
            parts[3] = list(cast("Sequence[object]", parts[3]))
        if len(parts) >= 5 and parts[4] is not None:
            parts[4] = list(cast("Sequence[tuple[object, object]]", parts[4]))
        # A reduce object is registered for identity ONLY if it is built-then-
        # mutated (state / listitems / dictitems present) -- an existing object a
        # member can back-reference (a real cycle target, e.g. OrderedDict or a
        # stateful object). A pure (callable, args) reduce constructs ATOMICALLY,
        # so it is an immutable value (frozenset, namedtuple, tuple, Path, enum):
        # not a cycle target, round-trips by value, and MUST NOT be registered --
        # a member back-ref to it could not resolve (it is not yet built while its
        # args decode). Decode mirrors this: register iff the recipe has a tail.
        is_mutable_target = any(p is not None for p in parts[2:])
        # Snapshot the encounter state: a reduce that FAILS partway must roll
        # back EVERY registration it made, or a later slot-fallback would emit
        # py/id references to phantom nodes. Restore both maps exactly on failure.
        seen_snapshot = dict(self._seen)
        alive_len = len(self._alive)
        try:
            # Register (before children) only a mutable target, so a cyclic
            # listitem/dictitem pointing back at it closes as a py/id.
            if is_mutable_target:
                self._register(value)
            elements = [self.encode(p) for p in parts]
        except TypeError:
            self._seen = seen_snapshot
            del self._alive[alive_len:]
            return None  # non-importable reconstructor -> no fallback.
        # Trim trailing None elements (no-ops on decode) for compactness.
        while len(elements) > 2 and elements[-1] is None:
            elements.pop()
        return {"py/reduce": elements}


class _Decoder:
    """Reverse ``_Encoder`` output back into live objects.

    Rebuilds the encoder's encounter order in ``_built`` so a ``py/id`` index
    resolves to the identical object.
    """

    def __init__(self, hooks: Hooks) -> None:
        self._hooks = hooks
        # Objects by encounter index (jsonpickle positional references).
        self._built: list[object] = []

    def decode(self, data: object) -> Any:
        if isinstance(data, (int, float, str, bool, type(None))):
            return data
        if isinstance(data, list):
            # A native JSON array is a mutable list -- registered for identity.
            result: list[object] = []
            self._built.append(result)
            result.extend(self.decode(v) for v in cast("list[object]", data))
            return result
        if not isinstance(data, dict):
            raise TypeError(f"Unexpected JSON node: {type(data)!r}")
        node = cast("dict[str, Any]", data)

        if "py/id" in node:
            return self._built[cast(int, node["py/id"])]
        if "py/type" in node:
            return _resolve(cast(str, node["py/type"]))
        if "py/function" in node:
            return _resolve(cast(str, node["py/function"]))
        if "py/tuple" in node:
            return tuple(self.decode(v) for v in node["py/tuple"])
        if "py/set" in node:
            return self._decode_set(node)
        if "py/b64" in node:
            return base64.b64decode(cast(str, node["py/b64"]))
        if "py/float" in node:
            # A value leaf -- no _built append, symmetric with encode.
            return float(cast(str, node["py/float"]))
        if "py/reduce" in node:
            return self._decode_reduce(node)
        if "py/hook" in node:
            return self._decode_hook(node)
        if "py/inline" in node:
            return self._decode_inline(node)
        if "py/object" in node:
            return self._decode_object(node)
        # A plain (native) data dict -- registered for identity, keys unescaped.
        return self._decode_dict(node)

    def _decode_set(self, node: dict[str, Any]) -> object:
        result: set[object] = set()
        self._built.append(result)
        result.update(self.decode(v) for v in node["py/set"])
        return result

    def _decode_dict(self, node: dict[str, Any]) -> dict[object, object]:
        result: dict[object, object] = {}
        self._built.append(result)
        for key, val in node.items():
            result[self._decode_key(key)] = self.decode(val)
        return result

    def _decode_key(self, key: str) -> object:
        if key.startswith("json://"):
            return self.decode(json.loads(key[len("json://") :]))
        return key

    def _decode_reduce(self, node: dict[str, Any]) -> object:
        # Replay pickle's reduce protocol: (callable, args, state?, listitems?,
        # dictitems?). A built-then-mutated object (any tail element present) was
        # registered on encode as a cycle target, so reserve its slot BEFORE
        # decoding the parts and fill it in. A pure (callable, args) reduce is an
        # atomic immutable value -- NOT registered on encode -- so it must NOT
        # consume a _built index here either (index parity).
        elements = cast("list[Any]", node["py/reduce"])
        is_mutable_target = any(e is not None for e in elements[2:])
        index = -1
        if is_mutable_target:
            index = len(self._built)
            self._built.append(None)
        func = self.decode(elements[0])
        args = cast("tuple[object, ...]", self.decode(elements[1]))
        obj = func(*args)
        if is_mutable_target:
            self._built[index] = obj

        if len(elements) > 2 and elements[2] is not None:
            _apply_state(obj, self.decode(elements[2]))
        if len(elements) > 3 and elements[3] is not None:
            obj.extend(self.decode(elements[3]))
        if len(elements) > 4 and elements[4] is not None:
            for key, val in self.decode(elements[4]):
                obj[key] = val
        return obj

    def _decode_hook(self, node: dict[str, Any]) -> object:
        path, payload = cast("list[Any]", node["py/hook"])
        _, decode = self._hooks[_resolve(cast(str, path))]
        obj = decode(payload)
        self._built.append(obj)
        return obj

    def _decode_inline(self, node: dict[str, Any]) -> object:
        path, payload = cast("list[Any]", node["py/inline"])
        cls: Any = _resolve(cast(str, path))
        # Reconstruct the stored InlineConfig state directly rather than calling
        # the constructor: PartialConfig.__init__ rewraps its func in
        # functools.partial, but the serialized state is already the post-wrap
        # shape, so re-running __init__ would double-wrap.
        obj = cls.__new__(cls)
        self._built.append(obj)
        obj.func = self.decode(payload["func"])
        obj._finalized = False  # noqa: SLF001
        obj._args = [self.decode(v) for v in payload["args"]]  # noqa: SLF001
        obj._kwargs = {  # noqa: SLF001
            k: self.decode(v)
            for k, v in cast(dict[str, Any], payload["kwargs"]).items()
        }
        return obj

    def _decode_object(self, node: dict[str, Any]) -> object:
        cls: Any = _resolve(cast(str, node["py/object"]))
        # Build empty, register for cycles, then fill fields (a field may point
        # back at this node). Bypass __init__ so no field ordering is assumed.
        obj = cls.__new__(cls)
        self._built.append(obj)
        # _finalized is bookkeeping (skipped by the encoder). Restore it to a
        # freshly-constructed config's state -- but only for Makers (a plain
        # Dataclass has no such slot and would raise on assignment).
        if _has_finalized_slot(cls):
            object.__setattr__(obj, "_finalized", False)
        for name, value in node.items():
            if name == "py/object":
                continue
            object.__setattr__(obj, name, self.decode(value))
        return obj


def _apply_state(obj: object, state: object) -> None:
    """Apply a pickle reduce ``state`` to ``obj`` (``__setstate__`` or attrs)."""
    setstate = getattr(obj, "__setstate__", None)
    if setstate is not None:
        setstate(state)
        return
    # No __setstate__: state is a __dict__ mapping, or (slots_dict) / (dict, slots).
    dict_state: object = state
    slots_state: object = None
    if isinstance(state, tuple):
        # basedpyright keeps state as tuple[Unknown, ...]; ty narrows it.
        pair = cast("tuple[object, ...]", state)  # ty: ignore[redundant-cast]
        if len(pair) == 2:
            dict_state, slots_state = pair
    if isinstance(dict_state, dict):
        for key, val in cast("dict[str, object]", dict_state).items():
            object.__setattr__(obj, key, val)
    if isinstance(slots_state, dict):
        for key, val in cast("dict[str, object]", slots_state).items():
            object.__setattr__(obj, key, val)


def _is_data_object(value: object) -> bool:
    """True for a dataclass instance or any object carrying its own __slots__."""
    return _is_data_class(type(value))


def _is_data_class(cls: type) -> bool:
    """True for a dataclass or a class carrying its own ``__slots__``."""
    return bool(
        hasattr(cls, "__dataclass_fields__") or "__slots__" in cls.__dict__,
    )


def _has_finalized_slot(cls: type) -> bool:
    """True if ``cls`` declares a ``_finalized`` slot anywhere in its MRO (a Maker)."""
    for base in cls.__mro__:
        slots = getattr(base, "__slots__", ())
        if isinstance(slots, str):
            slots = (slots,)
        if "_finalized" in slots:
            return True
    return False
