from __future__ import annotations

from collections import OrderedDict, deque
from collections.abc import Callable
from dataclasses import field
from decimal import Decimal
from pathlib import Path
from types import MappingProxyType
from typing import Any, NamedTuple, Self, SupportsIndex, cast, override

import enum
import json
import pickle

import pytest

from configgle.fig import Dataclass, Fig
from configgle.inline import InlineConfig, PartialConfig
from configgle.serialize import Hooks, deserialize, serialize


# Module-level config classes: deserialize resolves by import path, so the target
# must be importable (a class nested in a test function cannot be).
class Leaf:
    class Config(Fig["Leaf"]):
        v: int = 0
        name: str = "leaf"

    def __init__(self, config: Config) -> None:
        self.v = config.v


class Nested:
    class Config(Fig["Nested"]):
        leaf: Leaf.Config = field(default_factory=Leaf.Config)
        scale: float = 1.0

    def __init__(self, config: Config) -> None:
        del config


class Containers:
    class Config(Fig["Containers"]):
        items: list[Leaf.Config] = field(default_factory=list[Leaf.Config])
        mapping: dict[str, Leaf.Config] = field(default_factory=dict[str, Leaf.Config])
        pair: tuple[int, str] = (1, "a")
        tags: frozenset[str] = frozenset()
        nums: list[int] = field(default_factory=lambda: [1, 2, 3])

    def __init__(self, config: Config) -> None:
        del config


class Animal:
    class Config(Fig["Animal"]):
        name: str = "animal"

    def __init__(self, config: Config) -> None:
        self.name = config.name


class Dog(Animal):
    class Config(Animal.Config):
        breed: str = "mutt"

    def __init__(self, config: Config) -> None:
        super().__init__(config)


class Holder:
    class Config(Fig["Holder"]):
        # Typed as the base Config; may hold a Dog.Config at runtime.
        animal: Animal.Config = field(default_factory=Animal.Config)

    def __init__(self, config: Config) -> None:
        del config


class Point(Dataclass):
    x: int = 0
    y: int = 0


class Coord(NamedTuple):
    lat: float
    lon: float


class WithCoord:
    class Config(Fig["WithCoord"]):
        coord: Coord = Coord(1.0, 2.0)

    def __init__(self, config: Config) -> None:
        del config


class DagRoot:
    class Config(Fig["DagRoot"]):
        a: Leaf.Config = field(default_factory=Leaf.Config)
        b: Leaf.Config = field(default_factory=Leaf.Config)

    def __init__(self, config: Config) -> None:
        del config


class Weight:
    def __init__(self, data: list[float]) -> None:
        self.data = data


class HasWeight:
    class Config(Fig["HasWeight"]):
        weight: Weight = field(default_factory=lambda: Weight([0.0]))

    def __init__(self, config: Config) -> None:
        del config


_WEIGHT_HOOKS: Hooks = {Weight: (lambda w: w.data, Weight)}


class _StatefulLeaf:
    """A leaf whose reduce carries state -> registered, so a DAG keeps identity."""

    def __init__(self, data: list[int]) -> None:
        self.data = data


class Unpicklable:
    """A leaf whose __reduce__ raises -- genuinely unserializable without a hook."""

    @override
    def __reduce_ex__(self, protocol: SupportsIndex) -> str | tuple[Any, ...]:
        del protocol
        raise TypeError("nope")


class SlottedFailingReduce:
    """A slotted leaf whose reduce registers a shared child, then fails to encode.

    Exercises JP-001: the failed reduce must roll back the child registration so
    the slot-fallback path (this class is slotted -> a data object) does not emit
    a phantom py/id.
    """

    __slots__ = ("shared",)

    def __init__(self, shared: list[int]) -> None:
        self.shared = shared

    @override
    def __reduce_ex__(self, protocol: SupportsIndex) -> tuple[Any, ...]:
        del protocol

        # Encoding these reduce args registers `shared`, then fails on the
        # non-importable local function -> the reduce fallback returns None.
        def _local(x: object) -> object:
            return x

        return (dict, (self.shared, _local))


class HasUnpicklable:
    class Config(Fig["HasUnpicklable"]):
        x: object = None

    def __init__(self, config: Config) -> None:
        del config


class Cyclic:
    class Config(Fig["Cyclic"], slots=False):
        peer: object = None
        v: int = 0

    def __init__(self, config: Config) -> None:
        del config


class TwoWeights:
    class Config(Fig["TwoWeights"]):
        a: Weight = field(default_factory=lambda: Weight([0.0]))
        b: Weight = field(default_factory=lambda: Weight([0.0]))

    def __init__(self, config: Config) -> None:
        del config


class WithBytes:
    class Config(Fig["WithBytes"]):
        blob: bytes = b""

    def __init__(self, config: Config) -> None:
        del config


class WithOrderedDict:
    class Config(Fig["WithOrderedDict"]):
        ordered: OrderedDict[str, int] = field(default_factory=OrderedDict[str, int])

    def __init__(self, config: Config) -> None:
        del config


class WithFloat:
    class Config(Fig["WithFloat"]):
        x: float = 0.0

    def __init__(self, config: Config) -> None:
        del config


class Color(enum.IntEnum):
    RED = 1
    BLUE = 2


class Suit(enum.StrEnum):
    HEARTS = "hearts"
    SPADES = "spades"


class PlainEnum(enum.Enum):
    A = enum.auto()
    B = enum.auto()


class WithEnums:
    class Config(Fig["WithEnums"]):
        color: Color = Color.RED  # config-globals: ignore -- enum member, not a global.
        suit: Suit = Suit.HEARTS  # config-globals: ignore -- enum member, not a global.

    def __init__(self, config: Config) -> None:
        del config


class WithPlainEnum:
    class Config(Fig["WithPlainEnum"]):
        e: object = None  # a plain (non-scalar) Enum: an opaque leaf

    def __init__(self, config: Config) -> None:
        del config


class WithReducibleLeaves:
    class Config(Fig["WithReducibleLeaves"]):
        # Third-party / stdlib leaves configgle does not know about, handled by
        # the __reduce__ fallback without importing their libraries.
        path: object = None
        dec: object = None
        proxy: object = None

    def __init__(self, config: Config) -> None:
        del config


class ReducesToTuple:
    """A leaf whose __reduce__ reconstructor is the builtin ``tuple``."""

    @override
    def __reduce_ex__(self, protocol: SupportsIndex) -> tuple[Any, ...]:
        del protocol
        return (tuple, ([1, 2],))


class ReducesToDict:
    """A leaf whose __reduce__ reconstructor is the builtin ``dict``."""

    @override
    def __reduce_ex__(self, protocol: SupportsIndex) -> tuple[Any, ...]:
        del protocol
        return (dict, ([("a", 1), ("b", 2)],))


class Derived:
    class Config(Fig["Derived"]):
        base: int = 2
        doubled: int = -1

        @override
        def finalize(self) -> Self:
            if self.doubled == -1:
                self.doubled = self.base * 2
            return super().finalize()

    def __init__(self, config: Config) -> None:
        del config


class Hashable:
    # eq=False makes the config hashable, so it can be a set/frozenset member
    # and still hold a field pointing back at the containing set (a cycle).
    class Config(Fig["Hashable"], eq=False):
        peers: object = None  # holds a frozenset or set pointing back at self
        tag: int = 0

    def __init__(self, config: Config) -> None:
        del config


class ImmutableDag:
    class Config(Fig["ImmutableDag"]):
        a: tuple[int, ...] = ()
        b: tuple[int, ...] = ()
        s: frozenset[int] = frozenset()
        t: frozenset[int] = frozenset()

    def __init__(self, config: Config) -> None:
        del config


def _roundtrip[T](cfg: T) -> T:
    # Round-trip through json.dumps/loads too, proving serialize() yields a
    # genuinely JSON-encodable tree (not just an in-memory structure).
    return deserialize(json.loads(json.dumps(serialize(cfg))))


def test_scalar_fields_roundtrip():
    cfg = Leaf.Config(v=5, name="hi")
    back = _roundtrip(cfg)
    assert isinstance(back, Leaf.Config)
    assert back.v == 5
    assert back.name == "hi"


def test_nested_fig_roundtrip():
    cfg = Nested.Config()
    cfg.leaf.v = 7
    cfg.scale = 2.5
    back = _roundtrip(cfg)
    assert isinstance(back, Nested.Config)
    assert isinstance(back.leaf, Leaf.Config)
    assert back.leaf.v == 7
    assert back.scale == 2.5


def test_roundtrip_result_is_makeable():
    """A deserialized config still builds its parent class."""
    back = _roundtrip(Leaf.Config(v=3))
    obj = back.make()
    assert isinstance(obj, Leaf)
    assert obj.v == 3


def test_no_finalize_on_serialize():
    """Serialization captures the raw (unfinalized) config, not a derived one."""
    cfg = Derived.Config(base=5)
    tree = serialize(cfg)
    # doubled stays at its sentinel -- finalize did not run. jsonpickle py/object
    # inlines fields flat alongside the py/object type key.
    assert tree["doubled"] == -1
    assert tree["py/object"].endswith("Derived.Config")


def test_list_and_dict_of_figs_roundtrip():
    cfg = Containers.Config()
    cfg.items = [Leaf.Config(v=1), Leaf.Config(v=2)]
    cfg.mapping = {"a": Leaf.Config(v=9)}
    back = _roundtrip(cfg)
    assert [leaf.v for leaf in back.items] == [1, 2]
    assert isinstance(back.items[0], Leaf.Config)
    assert back.mapping["a"].v == 9


def test_tuple_and_frozenset_roundtrip():
    cfg = Containers.Config()
    cfg.pair = (3, "z")
    cfg.tags = frozenset({"x", "y"})
    back = _roundtrip(cfg)
    assert back.pair == (3, "z")
    assert isinstance(back.pair, tuple)
    assert back.tags == frozenset({"x", "y"})
    assert isinstance(back.tags, frozenset)


class WithSet:
    class Config(Fig["WithSet"]):
        s: set[int] = field(default_factory=lambda: {1, 2})
        keyed: dict[int, str] = field(default_factory=lambda: {1: "a", 2: "b"})

    def __init__(self, config: Config) -> None:
        del config


def test_set_roundtrip():
    back = _roundtrip(WithSet.Config(s={7, 8, 9}))
    assert back.s == {7, 8, 9}
    assert isinstance(back.s, set)


def test_non_str_keyed_dict_roundtrip():
    """A dict with non-str keys serializes via an items list, not a JSON object."""
    back = _roundtrip(WithSet.Config(keyed={10: "x", 20: "y"}))
    assert back.keyed == {10: "x", 20: "y"}
    assert all(isinstance(k, int) for k in back.keyed)


def test_list_of_primitives_roundtrip():
    cfg = Containers.Config()
    cfg.nums = [4, 5, 6]
    back = _roundtrip(cfg)
    assert back.nums == [4, 5, 6]


def test_polymorphic_field_preserves_runtime_type():
    """A base-typed slot holding a subclass config deserializes as the subclass."""
    cfg = Holder.Config(animal=Dog.Config(breed="corgi"))
    back = _roundtrip(cfg)
    assert isinstance(back.animal, Dog.Config)
    assert back.animal.breed == "corgi"


def test_dag_identity_preserved():
    """A config shared by two fields stays shared after a round-trip."""
    shared = Leaf.Config(v=42)
    cfg = DagRoot.Config(a=shared, b=shared)
    assert cfg.a is cfg.b
    back = _roundtrip(cfg)
    assert back.a is back.b
    assert back.a.v == 42


def test_standalone_dataclass_roundtrip():
    """A Dataclass (no Maker) round-trips too."""
    back = _roundtrip(Point(x=3, y=4))
    assert isinstance(back, Point)
    assert (back.x, back.y) == (3, 4)


def test_namedtuple_roundtrip():
    back = _roundtrip(WithCoord.Config(coord=Coord(5.0, 6.0)))
    assert isinstance(back.coord, Coord)
    assert back.coord == Coord(5.0, 6.0)


def _plain_fn(a: int, b: int = 1) -> int:
    return a + b


def _stale_redefined_fn(value: int) -> int:
    return value + 1


def _live_redefined_fn(value: int) -> int:
    return value + 2


def test_class_and_function_reference_roundtrip():
    """A bare class or function round-trips via jsonpickle py/type / py/function.

    Both are references resolved by import path -- decode returns the same object.
    """
    tree = serialize({"fn": _plain_fn, "cls": dict})
    assert tree["cls"] == {"py/type": "builtins.dict"}
    assert tree["fn"] == {"py/function": f"{__name__}._plain_fn"}
    back = _roundtrip({"fn": _plain_fn, "cls": dict})
    assert back["fn"] is _plain_fn
    assert back["cls"] is dict


def test_stale_function_import_path_is_rejected(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A stale function cannot borrow a live function's import path."""
    monkeypatch.setattr(_stale_redefined_fn, "__name__", "_live_redefined_fn")
    monkeypatch.setattr(
        _stale_redefined_fn,
        "__qualname__",
        "_live_redefined_fn",
    )
    assert _stale_redefined_fn is not _live_redefined_fn
    with pytest.raises(TypeError, match="does not resolve to the same object"):
        serialize(_stale_redefined_fn)
    assert serialize(_live_redefined_fn) == {
        "py/function": f"{__name__}._live_redefined_fn",
    }


def test_inline_config_roundtrip():
    cfg = InlineConfig(_plain_fn, 10, b=20)
    back = _roundtrip(cfg)
    assert isinstance(back, InlineConfig)
    assert back.make() == 30


def test_partial_config_roundtrip():
    cfg = PartialConfig(_plain_fn, b=5)
    back = _roundtrip(cfg)
    assert isinstance(back, PartialConfig)
    partial = back.make()
    assert partial(a=10) == 15


def test_inline_config_with_nested_fig_roundtrip():
    cfg = InlineConfig(_plain_fn, Leaf.Config(v=100))
    back = _roundtrip(cfg)
    assert isinstance(back._args[0], Leaf.Config)
    assert back._args[0].v == 100


def test_hook_encodes_opaque_leaf():
    """A per-type hook serializes a leaf that JSON cannot represent natively."""
    cfg = HasWeight.Config(weight=Weight([1.0, 2.0]))
    tree = serialize(cfg, hooks=_WEIGHT_HOOKS)
    back = deserialize(json.loads(json.dumps(tree)), hooks=_WEIGHT_HOOKS)
    assert isinstance(back.weight, Weight)
    assert back.weight.data == [1.0, 2.0]


def test_unserializable_leaf_raises_without_hook():
    """An opaque leaf with no hook is a clear TypeError, not silent data loss."""

    class Opaque:
        pass

    class HasOpaque:
        class Config(Fig["HasOpaque"]):
            thing: object = field(default_factory=Opaque)

        def __init__(self, config: Config) -> None:
            del config

    with pytest.raises(TypeError, match="Opaque"):
        serialize(HasOpaque.Config())


def test_method_serialization_roundtrips():
    """Maker.serialize / deserialize class method mirror the free functions."""
    cfg = Leaf.Config(v=8)
    tree = cfg.serialize()
    back = Leaf.Config.deserialize(tree)
    assert isinstance(back, Leaf.Config)
    assert back.v == 8


def test_serialize_yields_json_encodable_tree():
    """Serialize returns a dict tree that json.dumps accepts unchanged."""
    tree = serialize(Leaf.Config(v=1))
    assert isinstance(tree, dict)
    # json.dumps must not raise -- the tree is genuinely JSON-encodable.
    assert isinstance(json.dumps(tree), str)


def test_deserialize_passes_through_non_config_top_level():
    """Deserialize of a bare scalar returns it (primitives pass through)."""
    assert deserialize(5) == 5
    assert deserialize("hi") == "hi"


def test_serialize_and_pickle_roundtrips_agree():
    """The serialize/deserialize path and the built-in pickle path agree.

    Configgle already supports pickle (parent_class restored via MethodType).
    This is a second, independent round-trip mechanism; both must reconstruct
    the same config so a caller can pick either without surprise -- including
    the hardest case, a base-typed slot holding a subclass config.
    """
    cfg = Holder.Config(animal=Dog.Config(breed="corgi"))

    via_pickle = pickle.loads(pickle.dumps(cfg))
    via_serialize = deserialize(serialize(cfg))

    # Both equal the original and each other.
    assert via_pickle == cfg
    assert via_serialize == cfg
    assert via_pickle == via_serialize
    # Both preserve the polymorphic runtime type (Dog, not Animal).
    assert isinstance(via_pickle.animal, Dog.Config)
    assert isinstance(via_serialize.animal, Dog.Config)


def test_picklable_leaf_roundtrips_without_a_hook():
    """A picklable leaf (Weight, via __reduce__) round-trips with no hook.

    The jsonpickle model reconstructs any object that pickles cleanly -- Weight
    reduces to (newobj, (cls,), {state}). Agrees with pickle.
    """
    cfg = HasWeight.Config(weight=Weight([1.0, 2.0]))
    via_pickle = pickle.loads(pickle.dumps(cfg))
    via_serialize = deserialize(serialize(cfg))
    assert via_serialize.weight.data == via_pickle.weight.data == [1.0, 2.0]

    # A hook still overrides the default reduce encoding when supplied.
    via_hook = deserialize(serialize(cfg, hooks=_WEIGHT_HOOKS), hooks=_WEIGHT_HOOKS)
    assert via_hook.weight.data == [1.0, 2.0]


# --- Regression tests for the review findings (CFG-1..CFG-9) ---


def test_self_cycle_roundtrips():
    """CFG-1: a self-referential config serializes without RecursionError.

    The module docstring and __init__ promise cycle support; the encoder must
    register a node before recursing into its children.
    """
    cfg = Cyclic.Config(v=5)
    cfg.peer = cfg  # a -> a
    back = _roundtrip(cfg)
    assert back.peer is back
    assert back.v == 5


def test_mutual_cycle_roundtrips():
    """CFG-1: a two-node cycle (a -> b -> a) round-trips with identity intact."""
    a = Cyclic.Config(v=1)
    b = Cyclic.Config(v=2)
    a.peer = b
    b.peer = a
    back = _roundtrip(a)
    peer = cast(Cyclic.Config, back.peer)
    assert peer.peer is back
    assert peer.v == 2


def test_cycle_through_list_roundtrips():
    """CFG-1: a cycle through a mutable container also terminates."""
    cfg = Cyclic.Config()
    cfg.peer = [cfg]  # a -> [a]
    back = _roundtrip(cfg)
    assert cast("list[object]", back.peer)[0] is back


def test_cycle_through_frozenset_member_roundtrips():
    """NEW-1 (rejected): a Fig inside its own frozenset field round-trips.

    The back-edge targets the Fig (reserved first), not the frozenset, so the
    reserve-first fig path closes the cycle. A frozenset cannot contain itself,
    so build-then-remember on the immutable container is never asked to honor a
    not-yet-built id.
    """
    a = Hashable.Config(tag=1)
    a.peers = frozenset({a})  # a in a.peers
    back = _roundtrip(a)
    assert next(iter(cast("frozenset[object]", back.peers))) is back
    assert back.tag == 1


def test_immutable_container_dag_roundtrips_by_value():
    """Immutable containers round-trip by VALUE, not object identity.

    A tuple/frozenset shared by two fields deserializes equal (``a == b``) but
    need not be the same object -- the documented contract, since immutables are
    value types and JSON has no identity concept. (Contrast: mutable list/dict/
    set/Fig preserve object identity.)
    """
    shared_tuple = (1, 2, 3)
    shared_set = frozenset({4, 5})
    cfg = ImmutableDag.Config(
        a=shared_tuple, b=shared_tuple, s=shared_set, t=shared_set
    )
    back = _roundtrip(cfg)
    assert back.a == back.b == (1, 2, 3)
    assert back.s == back.t == frozenset({4, 5})


def test_cycle_through_tuple_target_roundtrips():
    """NEW-2: a cycle whose edge targets a tuple terminates and round-trips.

    ``holder = (cfg,); cfg.peer = holder`` -- the back-edge points at the tuple.
    Immutables aren't id-shared, so the tuple is re-encoded by value; the cycle's
    mutable anchor (cfg) stays id-shared, so decode terminates without KeyError.
    """
    cfg = Cyclic.Config(v=1)
    holder: tuple[object, ...] = (cfg,)
    cfg.peer = holder  # cfg -> (cfg,) -> cfg
    back = cast("tuple[object, ...]", _roundtrip(holder))
    inner = cast(Cyclic.Config, back[0])
    assert cast("tuple[object, ...]", inner.peer)[0] is inner


def test_cycle_through_frozenset_from_mutable_anchor_roundtrips():
    """A cfg holding a frozenset that contains it, entered via the cfg, closes.

    The mutable Fig is the cycle anchor (reserve-first); the frozenset is a value
    edge. Entering via the frozenset itself is unsupported -- an immutable cannot
    be reserved-then-filled, so it cannot be a cycle TARGET (same limit as pickle).
    """
    cfg = Hashable.Config(tag=1)
    cfg.peers = frozenset({cfg})  # cfg -> frozenset({cfg}) -> cfg
    back = _roundtrip(cfg)
    inner = cast("frozenset[object]", back.peers)
    assert next(iter(inner)) is back


def test_cycle_through_set_target_roundtrips():
    """NEW-2: a mutable set IS a cycle target -- identity preserved on decode."""
    cfg = Hashable.Config(tag=2)
    holder: set[object] = {cfg}
    cfg.peers = holder  # cfg -> {cfg} -> cfg; set is mutable, so id-shared
    back = _roundtrip(holder)
    inner = cast(Hashable.Config, next(iter(back)))
    assert inner.peers is back


def test_int_and_str_enum_preserve_type():
    """NEW-4: IntEnum/StrEnum fields round-trip as the enum, not the base type.

    An int/str subclass is a leaf in copy_tree (aliased, keeps its type); the
    JSON transport would otherwise degrade it to a bare int/str, breaking enum
    dispatch downstream.
    """
    back = _roundtrip(WithEnums.Config(color=Color.BLUE, suit=Suit.SPADES))
    assert back.color is Color.BLUE
    assert back.suit is Suit.SPADES
    assert isinstance(back.color, Color)
    assert isinstance(back.suit, Suit)


def test_reducible_leaf_roundtrips_via_reduce_fallback():
    """An opaque leaf that pickles (Path, Decimal) round-trips with no hook.

    The __reduce__ fallback records the reconstruction recipe by import path, so
    configgle serializes third-party/stdlib leaves without importing them.
    Path uses the (callable, args) reduce form; both survive a JSON round-trip.
    """
    cfg = WithReducibleLeaves.Config()
    cfg.path = Path("relative/x/y")
    cfg.dec = Decimal("1.50")
    back = _roundtrip(cfg)
    assert back.path == Path("relative/x/y")
    assert isinstance(back.path, Path)
    assert back.dec == Decimal("1.50")
    assert isinstance(back.dec, Decimal)


def test_shared_reducible_leaf_roundtrips_by_value():
    """A reduce-fallback leaf shared by two fields round-trips by VALUE, not identity.

    Reduce leaves (Path/Decimal/dtype/enum) are immutable value types, so -- like
    tuple/frozenset -- they are equal after a round-trip but need not be the same
    object. (Contrast: a hooked leaf, which may wrap mutable state, keeps object
    identity.)
    """
    shared = Path("relative/shared")
    cfg = WithReducibleLeaves.Config(path=shared, dec=shared)
    back = _roundtrip(cfg)
    assert back.path == back.dec == Path("relative/shared")


def test_mapping_proxy_type_preserved():
    """A MappingProxyType round-trips as itself, not degraded to a plain dict.

    CPython cannot pickle ``mappingproxy`` (its ``__reduce__`` raises), but
    ``MappingProxyType(d)`` reconstructs it, so serialize records that recipe as a
    ``py/reduce`` -- preserving the read-only wrapper type, not just its contents.
    """
    cfg = WithReducibleLeaves.Config()
    cfg.proxy = MappingProxyType({"a": 1, "b": 2})
    back = _roundtrip(cfg)
    assert type(back.proxy) is MappingProxyType
    assert dict(cast("MappingProxyType[str, int]", back.proxy)) == {"a": 1, "b": 2}


def test_plain_enum_roundtrips_via_reduce_fallback():
    """A non-scalar Enum round-trips as the same member via the __reduce__ path.

    A plain ``Enum`` isn't int/str/float/bytes, but it pickles as
    ``(EnumClass, (value,))``, so the reduce fallback reconstructs the exact
    member (identity preserved -- enum members are singletons).
    """
    back = _roundtrip(WithPlainEnum.Config(e=PlainEnum.A))
    assert back.e is PlainEnum.A


def test_local_config_class_rejected_at_serialize():
    """NEW-3: a local (non-importable) config class fails loudly on serialize."""

    class Local:
        class Config(Fig["Local"]):
            x: int = 1

        def __init__(self, config: Config) -> None:
            del config

    with pytest.raises(TypeError, match="importable"):
        serialize(Local.Config())


def test_local_container_subclass_degrades_to_base():
    """A local container subclass round-trips by CONTENTS as the base container.

    Unlike a local *config class* (which rejects -- its type is the whole point),
    a container's contract is its contents: an unresolvable/local subclass type
    is dropped and the base container is rebuilt, keeping serialize total.
    """

    class LocalList(list[int]):
        pass

    back = _roundtrip(WithReducibleLeaves.Config(path=LocalList([1, 2])))
    assert back.path == [1, 2]
    assert type(cast("list[int]", back.path)) is list


def test_reduce_leaf_identity_split():
    """JP-003: stateful reduce objects share identity; immutable ones by value.

    A built-then-mutated reduce (a stateful object) is registered -> a DAG keeps
    object identity. An atomic (callable, args) reduce (frozenset) is a value ->
    equal-but-distinct, and never a cycle target (so it cannot leave a phantom).
    """
    stateful = _StatefulLeaf([1, 2])
    back_stateful = _roundtrip({"a": stateful, "b": stateful})
    assert back_stateful["a"] is back_stateful["b"]  # identity via py/id

    fs = frozenset({3, 4})
    back_fs = _roundtrip({"a": fs, "b": fs})
    assert back_fs["a"] == back_fs["b"] == frozenset({3, 4})  # value, not identity


def test_cycle_through_frozenset_target_terminates_by_value():
    """JP-003: a frozenset entered as the top-level cycle target terminates.

    The frozenset is an immutable value leaf (not registered), so it round-trips
    by value; the Fig inside it still closes its own cycle via py/id. No phantom
    None, no crash.
    """
    cfg = Hashable.Config(tag=1)
    cfg.peers = frozenset({cfg})
    back = cast("frozenset[object]", _roundtrip(cfg.peers))
    inner = cast(Hashable.Config, next(iter(back)))
    # inner's own back-edge to the (value-copied) frozenset is equal to `back`.
    assert cast("frozenset[object]", inner.peers) == back
    assert inner.tag == 1


def test_user_dict_with_pytag_key_roundtrips_as_data():
    """JP-002: a data dict whose key looks like a wire tag round-trips verbatim.

    A native str-keyed dict holding a "py/..." key is json:// escaped so decode
    never mistakes the data for a py/object / py/tuple / py/id node.
    """
    for key in (
        "py/id",
        "py/type",
        "py/function",
        "py/tuple",
        "py/set",
        "py/b64",
        "py/reduce",
        "py/hook",
        "py/inline",
        "py/object",
        "py/float",
        "json://x",
    ):
        data = {key: "user", "normal": 1}
        assert _roundtrip(data) == data, f"tag key {key} misrouted"


def test_non_finite_float_before_shared_mutable_keeps_refs():
    """JP-DRIFT-1: a non-finite float must not consume a positional id slot.

    An inf/nan uses a non-registering py/float tag; if it consumed a decode-only
    encounter index, every later py/id (a shared/cyclic mutable) would resolve
    off-by-one to a phantom node.
    """
    shared: list[int] = [1, 2]
    back = _roundtrip([float("inf"), shared, shared])
    assert back[0] == float("inf")
    assert back[1] is back[2]  # the shared list's py/id still resolves correctly
    assert back[1] == [1, 2]


def test_failed_reduce_rolls_back_child_registrations():
    """JP-001: a reduce that registers a child then fails must roll it back.

    Otherwise the slot-fallback encoding emits a py/id to a phantom node.
    """
    shared: list[int] = [1, 2]
    back = _roundtrip(SlottedFailingReduce(shared))
    assert back.shared == [1, 2]


def test_unshared_list_serializes_as_native_array():
    """TAK-001: a plain unshared list is native JSON, not a wrapped node."""
    assert serialize([1, 2, 3]) == [1, 2, 3]
    assert serialize({"a": [1, 2]}) == {"a": [1, 2]}


def test_shared_list_keeps_wrapper_for_identity():
    """A list shared by two fields is registered so a repeat emits ``py/id`` and stays shared."""
    shared: list[int] = [1, 2]
    cfg = Containers.Config()
    cfg.nums = shared
    # Same list object in a second field (typed loosely; runtime shares identity).
    object.__setattr__(cfg, "items", shared)
    back = _roundtrip(cfg)
    assert back.nums is cast("list[int]", back.items)
    assert back.nums == [1, 2]


def test_hooked_leaf_dag_identity_preserved():
    """CFG-2: a shared opaque (hooked) leaf stays shared after a round-trip."""
    shared = Weight([1.0])
    cfg = TwoWeights.Config(a=shared, b=shared)
    assert cfg.a is cfg.b
    back = deserialize(serialize(cfg, hooks=_WEIGHT_HOOKS), hooks=_WEIGHT_HOOKS)
    assert back.a is back.b


def test_deserialized_fig_has_finalized_flag():
    """CFG-3: a deserialized Fig carries _finalized like any constructed one."""
    back = _roundtrip(Leaf.Config(v=1))
    assert hasattr(back, "_finalized")
    assert back._finalized is False


def test_local_callable_rejected_at_serialize_boundary():
    """CFG-SER-002: a non-importable callable fails loudly on serialize, not decode.

    A lambda has __qualname__ with '<locals>' that deserialize cannot resolve;
    the failure must surface at the serialize boundary, not silently produce an
    un-loadable tree.
    """
    identity: Callable[[object], object] = lambda x: x  # noqa: E731  -- local lambda is the test subject (unimportable callable)
    cfg: InlineConfig[object] = InlineConfig(identity, 1)
    with pytest.raises((TypeError, ValueError), match=r"import path|module-level"):
        serialize(cfg)


def test_deserialize_passes_through_plain_dict():
    """A tagless dict (no ``py/*`` key) is literal data, returned as-is -- not an error."""
    assert deserialize({"no_tag": 1, "nested": {"x": 2}}) == {
        "no_tag": 1,
        "nested": {"x": 2},
    }


def test_reduce_reconstructor_is_builtin_container_roundtrips():
    """A leaf whose ``__reduce__`` reconstructor is a builtin container round-trips.

    The ``py/reduce`` recipe records ``builtins.tuple`` / ``builtins.dict`` as the
    reconstructor and replays it, rather than the container being misread as
    literal items by the class dispatch.
    """
    back_t = _roundtrip(WithReducibleLeaves.Config(path=ReducesToTuple()))
    assert back_t.path == (1, 2)
    back_d = _roundtrip(WithReducibleLeaves.Config(path=ReducesToDict()))
    assert back_d.path == {"a": 1, "b": 2}


def test_deserialize_rejects_dangling_ref_with_exception():
    """CFG-SER-004: malformed transport input raises, not assert (stripped by -O).

    A ``py/id`` to an index that was never defined is corrupt input; it must
    raise a real exception unconditionally.
    """
    with pytest.raises((KeyError, IndexError, ValueError)):
        deserialize({"py/id": 999})


def test_unserializable_leaf_error_names_current_api():
    """CFG-5: a truly unpicklable leaf raises a hooks-directed TypeError."""
    cfg = HasUnpicklable.Config()
    cfg.x = Unpicklable()
    with pytest.raises(TypeError) as excinfo:
        serialize(cfg)
    assert "hooks" in str(excinfo.value)


def test_bytes_leaf_roundtrips():
    """CFG-6: a bytes field round-trips (copy_tree/finalize already treat it as a leaf)."""
    back = _roundtrip(WithBytes.Config(blob=b"hello"))
    assert back.blob == b"hello"
    assert isinstance(back.blob, bytes)


def test_ordered_dict_subclass_type_preserved():
    """CFG-7: a dict subclass deserializes as its own type, matching copy_tree."""
    back = _roundtrip(WithOrderedDict.Config(ordered=OrderedDict([("a", 1), ("b", 2)])))
    assert type(back.ordered) is OrderedDict
    assert list(back.ordered.items()) == [("a", 1), ("b", 2)]


def test_non_finite_float_is_valid_strict_json():
    """CFG-9: serialize yields strict JSON -- json.dumps(allow_nan=False) must not raise."""
    tree = serialize(WithFloat.Config(x=float("inf")))
    # allow_nan=False rejects Infinity/NaN; the tree must survive it.
    dumped = json.dumps(tree, allow_nan=False)
    back = deserialize(json.loads(dumped))
    assert back.x == float("inf")


def test_reduce_leaf_with_listitems_roundtrips():
    """A reduce whose 4th element (listitems) fills the object round-trips.

    ``deque`` reduces to ``(deque, (), None, iter(items))`` -- the decoder must
    replay the listitems via ``extend``.
    """
    back = _roundtrip(WithReducibleLeaves.Config(path=deque([1, 2, 3])))
    assert back.path == deque([1, 2, 3])
    assert isinstance(back.path, deque)


class _CustomState:
    """A leaf with an explicit ``__setstate__`` (the reduce state-setter path)."""

    def __init__(self, value: int = 0) -> None:
        self.value = value

    @override
    def __getstate__(self) -> dict[str, int]:
        return {"value": self.value}

    def __setstate__(self, state: dict[str, int]) -> None:
        self.value = state["value"]


def test_reduce_leaf_with_custom_setstate_roundtrips():
    """A reduce whose object defines ``__setstate__`` restores via that method."""
    back = _roundtrip(WithReducibleLeaves.Config(path=_CustomState(7)))
    assert isinstance(back.path, _CustomState)
    assert back.path.value == 7


class _SlotsState:
    """A slotted leaf: its default reduce state is a ``(None, slots_dict)`` tuple."""

    __slots__ = ("a", "b")

    def __init__(self, a: int = 0, b: int = 0) -> None:
        self.a = a
        self.b = b


def test_reduce_leaf_with_slots_state_tuple_roundtrips():
    """A slotted leaf's ``(dict, slots)`` reduce-state tuple round-trips.

    Exercises the state-tuple branch of ``_apply_state`` (no ``__setstate__``).
    """
    back = _roundtrip(WithReducibleLeaves.Config(path=_SlotsState(a=1, b=2)))
    assert isinstance(back.path, _SlotsState)
    assert (back.path.a, back.path.b) == (1, 2)


def test_local_mapping_subclass_degrades_to_base_dict():
    """A Mapping with no usable reduce degrades to a plain ``dict`` by contents."""

    class LocalMap(dict[str, int]):
        pass

    back = _roundtrip(WithReducibleLeaves.Config(path=LocalMap({"a": 1, "b": 2})))
    assert back.path == {"a": 1, "b": 2}
    assert type(cast("dict[str, int]", back.path)) is dict


def test_deserialize_rejects_unresolvable_import_path():
    """A ``py/type`` naming a nonexistent module raises ``ImportError`` on decode."""
    with pytest.raises(ImportError, match="Cannot resolve path"):
        deserialize({"py/type": "no_such_module_xyz.Thing"})


class _NonReducibleSet(frozenset[int]):
    """A set subclass whose ``__reduce_ex__`` raises -- forces the set degrade path."""

    @override
    def __reduce_ex__(self, protocol: SupportsIndex) -> tuple[Any, ...]:
        del protocol
        raise TypeError("no reduce")


def test_non_reducible_set_degrades_to_base_set():
    """A set with no usable reduce degrades to a plain ``set`` by contents."""
    back = _roundtrip(WithReducibleLeaves.Config(path=_NonReducibleSet({1, 2, 3})))
    assert back.path == {1, 2, 3}
    assert type(cast("set[int]", back.path)) is set


class _BareStringReduce:
    """A singleton-like leaf whose ``__reduce__`` returns a bare global name."""

    @override
    def __reduce_ex__(self, protocol: SupportsIndex) -> str:
        del protocol
        return "_THE_BARE_STRING_SINGLETON"


_THE_BARE_STRING_SINGLETON = _BareStringReduce()


def test_bare_string_reduce_roundtrips_as_global_reference():
    """A ``__reduce__`` returning a bare name resolves to that module global."""
    back = _roundtrip(WithReducibleLeaves.Config(path=_THE_BARE_STRING_SINGLETON))
    assert back.path is _THE_BARE_STRING_SINGLETON


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
