"""Tests for ``configgle.codec``."""

from __future__ import annotations

from collections import OrderedDict
from dataclasses import dataclass, field, make_dataclass
from enum import Enum
from pathlib import Path, PurePosixPath
from types import MappingProxyType
from typing import (
    Annotated,
    Literal,
    Protocol,
    SupportsIndex,
    TypeVar,
    cast,
    override,
)
from uuid import UUID
from zoneinfo import ZoneInfo

import datetime as dt
import gc
import math
import threading
import weakref

import pytest

from configgle import codec
from configgle.codec import (
    Invalid,
    ReadError,
    from_plain,
    immutable,
    mutable,
    parse,
    to_plain,
)


_T = TypeVar("_T")


def _round_trip(value: object) -> object:
    return from_plain(to_plain(value), object, allow_imports=True)


class Color(Enum):
    RED = "red"
    BLUE = "blue"


class Plain:
    """A class with no reduce hooks of its own."""

    def __init__(self, x: int) -> None:
        self.x = x


class Slotted:
    """Slots plus a ``__dict__`` member, so both state paths are walked."""

    __slots__ = ("__dict__", "a")

    def __init__(self, a: int, b: int) -> None:
        self.a = a
        self.b = b


class Stateful:
    """Owns its pickled state through ``__setstate__``."""

    def __init__(self, value: int) -> None:
        self.value = value

    @override
    def __reduce__(self) -> tuple[object, tuple[int], dict[str, int]]:
        return (Stateful, (0,), {"value": self.value})

    def __setstate__(self, state: dict[str, int]) -> None:
        self.value = state["value"] * 10


@dataclass(frozen=True, kw_only=True, slots=True)
class Point:
    x: int
    y: float = 0.0


@dataclass(kw_only=True, slots=True)
class Node:
    name: str
    children: list[Node] = field(default_factory=list["Node"])


class Shape(Protocol):
    def area(self) -> float: ...


@dataclass(frozen=True, kw_only=True, slots=True)
class Square:
    side: float

    def area(self) -> float:
        return self.side**2


@dataclass(frozen=True, kw_only=True, slots=True)
class Holder:
    shape: Shape


class Deferred:
    """Owns the ``py/inline`` protocol."""

    def __init__(self, func: object, *args: object, **kwargs: object) -> None:
        self.func = func
        self.args = list(args)
        self.kwargs = dict(kwargs)

    def __custom_json_inline__(
        self,
    ) -> tuple[object, list[object], dict[str, object]]:
        return self.func, self.args, self.kwargs

    def __custom_json_inline_init__(
        self,
        func: object,
        args: list[object],
        kwargs: dict[str, object],
    ) -> None:
        self.func = func
        self.args = list(args)
        self.kwargs = dict(kwargs)


class Tensor:
    """A leaf only a hook can encode."""

    def __init__(self, values: list[float]) -> None:
        self.values = values

    __slots__ = ("values",)


def _tensor_values(tensor: Tensor) -> list[float]:
    return tensor.values


type Ints = list[int]


_HOOKS: codec.Hooks = MappingProxyType({Tensor: (_tensor_values, Tensor)})


class TestImmutableMutable:
    def test_immutable_freezes_every_level(self) -> None:
        frozen = immutable({"a": [1, {"b": [2]}]})

        assert isinstance(frozen, MappingProxyType)
        inner = frozen["a"]
        assert isinstance(inner, tuple)
        assert isinstance(inner[1], MappingProxyType)
        assert inner[1]["b"] == (2,)

    def test_mutable_thaws_every_level(self) -> None:
        thawed = mutable(MappingProxyType({"a": (1, MappingProxyType({"b": (2,)}))}))

        assert thawed == {"a": [1, {"b": [2]}]}

    def test_both_always_copy(self) -> None:
        source = {"a": [1]}

        thawed = mutable(source)

        assert thawed == source
        assert thawed is not source
        assert thawed["a"] is not source["a"]

    @pytest.mark.parametrize("leaf", [None, True, 1, 1.5, "s"])
    def test_leaves_pass_through(self, leaf: codec.Plain) -> None:
        assert immutable(leaf) is leaf
        assert mutable(leaf) is leaf


class TestToPlain:
    def test_tags_name_python_types(self) -> None:
        tree = to_plain(
            {
                "t": (1, 2),
                "s": {2, 1},
                "f": frozenset({3}),
                "b": b"\x00",
                "p": Path("/a"),
                "u": UUID(int=1),
                "c": 1 + 2j,
            },
        )

        assert tree == {
            "t": {"py/tuple": [1, 2]},
            "s": {"py/set": [1, 2]},
            "f": {"py/frozenset": [3]},
            "b": {"py/b64": "AA=="},
            "p": {"py/path": "/a"},
            "u": {"py/uuid": "00000000-0000-0000-0000-000000000001"},
            "c": {"py/complex": [1.0, 2.0]},
        }

    def test_json_dialect_tags_non_finite_floats(self) -> None:
        assert to_plain([math.inf, 1.5]) == [{"py/float": "inf"}, 1.5]

    def test_python_dialect_keeps_non_finite_floats(self) -> None:
        tree = to_plain([math.inf], dialect="python")

        assert tree == [math.inf]

    def test_immutable_output_is_read_only(self) -> None:
        tree = to_plain({"a": [(1,)]}, mutable=False)

        assert isinstance(tree, MappingProxyType)
        assert tree["a"] == (MappingProxyType({"py/tuple": (1,)}),)

    def test_shared_objects_become_references(self) -> None:
        shared = [1]

        assert to_plain([shared, shared]) == [[1], {"py/id": 1}]

    def test_cycles_terminate(self) -> None:
        cycle: list[object] = []
        cycle.append(cycle)

        assert to_plain(cycle) == [{"py/id": 0}]

    def test_tag_lookalike_and_non_str_keys_are_escaped(self) -> None:
        tree = to_plain({"py/x": 1, 2: 3, "json://y": 4})

        assert tree == {'json://"py/x"': 1, "json://2": 3, 'json://"json://y"': 4}

    def test_dataclass_names_its_class(self) -> None:
        assert to_plain(Point(x=1)) == {
            "py/object": f"{__name__}.Point",
            "x": 1,
            "y": 0.0,
        }

    def test_enum_encodes_by_reduce(self) -> None:
        assert to_plain(Color.RED) == {
            "py/reduce": [{"py/type": f"{__name__}.Color"}, {"py/tuple": ["red"]}],
        }

    def test_set_order_is_deterministic(self) -> None:
        assert to_plain({"b", "a", "c"}) == {"py/set": ["a", "b", "c"]}

    def test_named_zone_is_kept(self) -> None:
        moment = dt.datetime(2026, 1, 2, 3, tzinfo=ZoneInfo("America/Chicago"))

        assert to_plain(moment) == {
            "py/datetime": "2026-01-02T03:00:00-06:00[America/Chicago]",
        }

    def test_a_local_class_cannot_be_named(self) -> None:
        class Local:
            pass

        with pytest.raises(TypeError, match="<locals>"):
            to_plain(Local)

    def test_an_unknown_leaf_names_the_hook_to_add(self) -> None:
        with pytest.raises(TypeError, match=r"hooks=\{lock:"):
            to_plain(threading.Lock())


class TestRoundTrip:
    @pytest.mark.parametrize(
        "value",
        [
            1.5,
            math.inf,
            -math.inf,
            pytest.param(b"\x00\xffbytes", id="bytes"),
            PurePosixPath("/a/b"),
            Path("/a/b"),
            UUID(int=7),
            dt.datetime(2026, 9, 26, 12, tzinfo=dt.UTC),
            dt.datetime(2026, 1, 2, 3, tzinfo=ZoneInfo("Europe/Paris")),
            (1, (2, 3)),
            {1, 2},
            frozenset({1, 2}),
            {"k": [1, {"j": 2}]},
            {(1, 2): "tuple key", 3: "int key", "py/x": "tag key"},
            MappingProxyType({"a": 1}),
            1 + 2j,
            Color.BLUE,
            Point(x=3, y=4.5),
            max,
            Path,
        ],
        ids=repr,
    )
    def test_values_survive(self, value: object) -> None:
        restored = _round_trip(value)

        assert restored == value
        assert type(restored) is type(value)

    def test_nan_survives(self) -> None:
        restored = _round_trip(math.nan)

        assert isinstance(restored, float)
        assert math.isnan(restored)

    def test_an_ordered_mapping_keeps_its_order(self) -> None:
        restored = _round_trip(OrderedDict([("b", 1), ("a", 2)]))

        assert isinstance(restored, OrderedDict)
        items = cast("OrderedDict[str, int]", restored).items()
        assert list(items) == [("b", 1), ("a", 2)]

    def test_a_plain_object_restores_its_attributes(self) -> None:
        restored = _round_trip(Plain(3))

        assert isinstance(restored, Plain)
        assert restored.x == 3

    def test_slots_and_dict_members_both_restore(self) -> None:
        restored = _round_trip(Slotted(1, 2))

        assert isinstance(restored, Slotted)
        assert (restored.a, restored.b) == (1, 2)

    def test_setstate_owns_the_restored_state(self) -> None:
        restored = _round_trip(Stateful(4))

        assert isinstance(restored, Stateful)
        assert restored.value == 40

    def test_shared_objects_stay_shared(self) -> None:
        shared = {"a": 1}

        restored = _round_trip([shared, (shared,), Node(name="n")])

        assert isinstance(restored, list)
        assert restored[1][0] is restored[0]

    def test_a_dataclass_cycle_survives(self) -> None:
        parent = Node(name="p")
        parent.children.append(parent)

        restored = _round_trip(parent)

        assert isinstance(restored, Node)
        assert restored.children[0] is restored

    def test_inline_values_round_trip(self) -> None:
        restored = _round_trip(Deferred(max, 1, key=abs))

        assert isinstance(restored, Deferred)
        assert (restored.func, restored.args, restored.kwargs) == (
            max,
            [1],
            {"key": abs},
        )

    def test_hooks_encode_and_decode_leaves(self) -> None:
        tree = to_plain([Tensor([1.0, math.inf])], hooks=_HOOKS)

        restored = from_plain(tree, object, hooks=_HOOKS, allow_imports=True)

        assert tree == [
            {"py/hook": [f"{__name__}.Tensor", [1.0, {"py/float": "inf"}]]},
        ]
        assert isinstance(restored, list)
        assert isinstance(restored[0], Tensor)
        assert restored[0].values == [1.0, math.inf]

    def test_immutable_trees_decode_alike(self) -> None:
        value = {"a": (1, {2}), "b": Point(x=1)}

        tree = to_plain(value, mutable=False)

        assert from_plain(tree, object, allow_imports=True) == value

    def test_a_protocol_field_reads_by_tag(self) -> None:
        tree = to_plain(Holder(shape=Square(side=2.0)))

        restored = from_plain(tree, Holder, allow_imports=True)

        assert restored == Holder(shape=Square(side=2.0))


class TestTypedDecode:
    def test_a_dataclass_reads_untagged_fields(self) -> None:
        assert from_plain({"x": 1, "y": 2}, Point) == Point(x=1, y=2.0)

    def test_nested_generics_read(self) -> None:
        target = dict[str, list[tuple[int, str]]]

        assert from_plain({"a": [[1, "b"]]}, target) == {"a": [(1, "b")]}

    def test_typed_non_str_keys_read(self) -> None:
        assert from_plain({"1": "a"}, dict[str, str]) == {"1": "a"}
        assert from_plain({"json://1": "a"}, dict[int, str]) == {1: "a"}

    def test_an_enum_reads_its_value(self) -> None:
        assert from_plain("red", Color) is Color.RED

    def test_a_literal_keeps_bool_and_int_apart(self) -> None:
        assert from_plain(1, Literal[True, 1]) == 1
        assert from_plain(True, Literal[True, 1]) is True
        with pytest.raises(ReadError):
            from_plain(2, Literal[True, 1])

    def test_an_optional_reads_null_and_its_member(self) -> None:
        assert from_plain(None, int | None) is None
        assert from_plain(3, int | None) == 3

    def test_a_union_picks_the_matching_kind(self) -> None:
        assert from_plain("1", int | str) == "1"
        assert from_plain([1], str | list[int]) == [1]
        assert from_plain(1, float | int) == 1
        assert type(from_plain(1, float | int)) is int

    def test_a_union_tries_members_in_order(self) -> None:
        assert from_plain("/a", UUID | Path) == Path("/a")

    def test_a_float_accepts_an_int_and_a_tag(self) -> None:
        assert from_plain(1, float) == 1.0
        assert from_plain({"py/float": "-inf"}, float) == -math.inf

    def test_a_pep695_alias_resolves(self) -> None:
        assert from_plain([1], Ints) == [1]

    @pytest.mark.parametrize(
        ("data", "target"),
        [
            (True, int),
            (1.5, int),
            ("1", int),
            (1, str),
            (1, bool),
            (None, int),
            (1, type(None)),
            ("x", float),
            ({"py/float": "1.0"}, float),
            ([1], dict[str, int]),
            ({"a": 1}, list[int]),
            ([1, 2], tuple[int]),
            ("!", bytes),
            ("not-a-uuid", UUID),
            ("2026-13-01", dt.datetime),
            ("2026-01-01T00:00:00[Europe/Paris]", dt.datetime),
            ("2026-01-01T00:00:00+00:00[Nowhere/Zone]", dt.datetime),
            ("green", Color),
            ([1], Point),
            ({"x": 1, "z": 2}, Point),
            ({"py/tuple": [1]}, list[int]),
            (1, Path),
        ],
        ids=repr,
    )
    def test_mismatches_raise(
        self,
        data: codec.PlainTree,
        target: object,
    ) -> None:
        with pytest.raises(ReadError):
            from_plain(data, target)

    def test_a_bad_part_is_kept_as_invalid(self) -> None:
        with pytest.raises(ReadError) as raised:
            from_plain({"a": 1, "b": "x"}, dict[str, int])

        assert raised.value.bad == {
            ("b",): Invalid(raw="x", reason="cannot read 'x' as int"),
        }
        assert raised.value.partial == {
            "a": 1,
            "b": Invalid(raw="x", reason="cannot read 'x' as int"),
        }

    def test_a_bad_dataclass_field_keeps_the_rest(self) -> None:
        with pytest.raises(ReadError) as raised:
            from_plain({"x": "one", "y": 2}, Point)

        partial = raised.value.partial
        assert isinstance(partial, Point)
        assert partial.y == 2
        assert isinstance(partial.x, Invalid)
        assert list(raised.value.bad) == [("x",)]


class TestCapabilities:
    def test_imports_need_allow_imports(self) -> None:
        tree = to_plain(Plain(1))

        with pytest.raises(ReadError, match="allow_imports"):
            from_plain(tree, object)

    @pytest.mark.parametrize(
        "tree",
        [
            {"py/type": "pathlib.Path"},
            {"py/function": "builtins.max"},
            {"py/reduce": [{"py/type": "builtins.int"}, {"py/tuple": ["1"]}]},
        ],
        ids=repr,
    )
    def test_every_executable_tag_is_gated(self, tree: codec.MutablePlainTree) -> None:
        with pytest.raises(ReadError, match="allow_imports"):
            from_plain(tree, object)

    def test_a_path_resolving_to_a_class_decodes(self) -> None:
        assert from_plain(
            {"py/type": "pathlib.PurePosixPath"},
            object,
            allow_imports=True,
        ) is (PurePosixPath)

    def test_an_unknown_path_is_rejected(self) -> None:
        with pytest.raises(ReadError, match="no_such_module_xyz"):
            from_plain(
                {"py/type": "no_such_module_xyz.thing"},
                object,
                allow_imports=True,
            )

    def test_import_free_tags_need_no_capability(self) -> None:
        tree = to_plain({"t": (1,), "s": {1}, "m": MappingProxyType({"a": 1})})

        assert from_plain(tree, object) == {
            "t": (1,),
            "s": {1},
            "m": MappingProxyType({"a": 1}),
        }


class TestMalformedTrees:
    @pytest.mark.parametrize("ref", [-1, 99, True, "0"])
    def test_a_bad_reference_is_rejected(self, ref: codec.Plain) -> None:
        with pytest.raises(ReadError, match="py/id"):
            from_plain([{"py/id": ref}], object)

    @pytest.mark.parametrize(
        "tree",
        [
            {"py/unknown": 1},
            {"py/tuple": [1], "extra": 2},
            {"py/tuple": 1},
            {"py/b64": 1},
            {"py/reduce": [1]},
            {"py/inline": [1]},
            {"py/hook": ["x.Y", 1]},
            {"json://{": 1},
            {"py/id": 0, "x": 1},
        ],
        ids=repr,
    )
    def test_malformed_tags_are_rejected(self, tree: codec.MutablePlainTree) -> None:
        with pytest.raises(ReadError):
            from_plain(tree, object, allow_imports=True)

    def test_a_tag_must_fit_its_target(self) -> None:
        with pytest.raises(ReadError, match="as <class 'int'>"):
            from_plain({"py/tuple": [1]}, int)


class Unreducible(dict[str, int]):
    """A dict subclass whose reduce refuses."""

    @override
    def __reduce_ex__(self, protocol: SupportsIndex) -> str | tuple[object, ...]:
        raise TypeError("no reduce")


class UnreducibleList(list[int]):
    """A list subclass whose reduce refuses."""

    @override
    def __reduce_ex__(self, protocol: SupportsIndex) -> str | tuple[object, ...]:
        raise TypeError("no reduce")


class UnreducibleSet(set[int]):
    """A set subclass whose reduce refuses."""

    @override
    def __reduce_ex__(self, protocol: SupportsIndex) -> str | tuple[object, ...]:
        raise TypeError("no reduce")


class NamedReduce:
    """Reduces to a name, which has no plain form here."""

    @override
    def __reduce_ex__(self, protocol: SupportsIndex) -> str | tuple[object, ...]:
        return "NamedReduce"


class LocalReduce:
    """Reduces through a local callable, so state encoding takes over."""

    __slots__ = ("__dict__", "a")

    def __init__(self, a: int, b: int) -> None:
        self.a = a
        self.b = b

    @override
    def __reduce_ex__(self, protocol: SupportsIndex) -> str | tuple[object, ...]:
        def rebuild() -> None:
            pass

        return (rebuild, ())


class Unbuildable:
    """Refuses construction, so a reduce replay fails."""

    def __init__(self, x: int) -> None:
        if x < 0:
            raise ValueError("negative")
        self.x = x


@dataclass(frozen=True, kw_only=True, slots=True)
class Positive:
    n: int

    def __post_init__(self) -> None:
        if self.n <= 0:
            raise ValueError("n must be positive")


class Level(Enum):
    LOW = 1
    HIGH = 2


def _reduce(*parts: codec.MutablePlainTree) -> codec.MutablePlainTree:
    return {"py/reduce": list(parts)}


class TestDecodeEdges:
    def test_bool_reads_a_bool(self) -> None:
        assert from_plain(True, bool) is True

    def test_complex_reads_a_real(self) -> None:
        assert from_plain(2, complex) == 2 + 0j

    def test_a_variadic_tuple_reads_any_length(self) -> None:
        assert from_plain([1, 2, 3], tuple[int, ...]) == (1, 2, 3)

    def test_a_literal_enum_member_reads(self) -> None:
        assert from_plain(2, Literal[Level.HIGH]) is Level.HIGH

    def test_a_union_prefers_the_data_kind(self) -> None:
        assert from_plain({"a": 1}, int | dict[str, int]) == {"a": 1}

    def test_an_annotated_target_reads_its_type(self) -> None:
        assert from_plain(1, Annotated[int, "meta"]) == 1

    def test_a_typevar_target_reads_as_is(self) -> None:
        assert from_plain([1], _T) == [1]

    def test_a_tagged_value_fits_a_protocol(self) -> None:
        tree = to_plain(Square(side=1.0))

        assert from_plain(tree, Shape, allow_imports=True) == Square(side=1.0)

    def test_a_tagged_value_fits_a_literal(self) -> None:
        tree = to_plain(Color.RED)

        assert from_plain(tree, Literal[Color.RED], allow_imports=True) is Color.RED
        with pytest.raises(ReadError):
            from_plain(tree, Literal[Color.BLUE], allow_imports=True)

    def test_a_tagged_value_fits_a_typevar(self) -> None:
        assert from_plain({"py/tuple": [1]}, _T) == (1,)

    def test_reduce_items_replay(self) -> None:
        ordered = _reduce(
            {"py/type": "collections.OrderedDict"},
            {"py/tuple": []},
            None,
            None,
            [{"py/tuple": ["a", 1]}],
        )
        listed = _reduce({"py/type": "builtins.list"}, {"py/tuple": []}, None, [1, 2])

        assert from_plain(ordered, object, allow_imports=True) == OrderedDict(a=1)
        assert from_plain(listed, object, allow_imports=True) == [1, 2]

    def test_a_reference_to_an_unbuilt_dataclass_is_invalid(self) -> None:
        with pytest.raises(ReadError) as raised:
            from_plain({"name": "n", "children": [{"py/id": 0}]}, Node)

        assert list(raised.value.bad) == [("children", 0)]

    @pytest.mark.parametrize(
        ("data", "target"),
        [
            ("1j", complex),
            (1, UUID),
            (1, dt.datetime),
            ("2026-01-01T00:00:00+00:00[Europe/Paris", dt.datetime),
            (2, Literal[Level.LOW]),
            ("x", set[int]),
            ("x", frozenset[int]),
            ([[1]], set[object]),
            ([[1]], frozenset[object]),
            ({"json://[1]": 1}, dict[object, int]),
            ({"n": 0}, Positive),
            ({"x": 1}, Plain),
            ("x", int | float),
            ({"py/type": 1}, object),
            ({"py/type": "builtins.max"}, type),
            ({"py/type": "pathlib.NoSuchThing"}, object),
            ({"py/function": 1}, object),
            ({"py/function": "math.pi"}, object),
            ({"py/object": 1}, object),
            ({"py/object": "builtins.max"}, object),
            (
                {"py/inline": ["builtins.int", {"func": 1, "args": 1, "kwargs": 1}]},
                object,
            ),
            (
                {
                    "py/inline": [
                        "builtins.int",
                        {
                            "func": 1,
                            "args": list[codec.MutablePlainTree](),
                            "kwargs": dict[str, codec.MutablePlainTree](),
                        },
                    ],
                },
                object,
            ),
            ({"py/reduce": 1}, object),
            (_reduce(1, {"py/tuple": []}), object),
            (
                _reduce({"py/type": f"{__name__}.Unbuildable"}, {"py/tuple": [-1]}),
                object,
            ),
            (_reduce({"py/type": "builtins.int"}, {"py/tuple": []}, None, [1]), object),
            (
                _reduce({"py/type": "builtins.int"}, {"py/tuple": []}, None, None, []),
                object,
            ),
            (
                _reduce(
                    {"py/type": "builtins.dict"},
                    {"py/tuple": []},
                    None,
                    None,
                    [1],
                ),
                object,
            ),
        ],
        ids=repr,
    )
    def test_malformed_input_raises(
        self,
        data: codec.MutablePlainTree,
        target: object,
    ) -> None:
        with pytest.raises(ReadError):
            from_plain(data, target, allow_imports=True)

    def test_an_unresolvable_annotation_raises(self) -> None:
        broken = make_dataclass("Broken", [("x", "NoSuchName")])

        with pytest.raises(ReadError, match="field types"):
            from_plain({"x": 1}, broken)


class TestEncodeEdges:
    def test_a_shared_set_member_object_stays_shared(self) -> None:
        first = Point(x=1)

        restored = _round_trip([{first, Point(x=2)}, first])

        assert isinstance(restored, list)
        assert restored[1] in restored[0]

    def test_a_bound_method_round_trips_by_reduce(self) -> None:
        restored = _round_trip([1].append)

        assert callable(restored)
        restored(2)
        assert getattr(restored, "__self__", None) == [1, 2]

    def test_a_failed_reduce_falls_back_to_state(self) -> None:
        restored = _round_trip(LocalReduce(1, 2))

        assert isinstance(restored, LocalReduce)
        assert (restored.a, restored.b) == (1, 2)

    @pytest.mark.parametrize(
        ("value", "expected"),
        [
            (Unreducible(a=1), {"a": 1}),
            (UnreducibleList([1]), [1]),
            (UnreducibleSet({1}), {"py/set": [1]}),
        ],
        ids=repr,
    )
    def test_an_unreducible_container_encodes_by_kind(
        self,
        value: object,
        expected: codec.MutablePlainTree,
    ) -> None:
        assert to_plain(value) == expected

    def test_a_named_reduce_has_no_plain_form(self) -> None:
        with pytest.raises(TypeError, match="NamedReduce"):
            to_plain(NamedReduce())

    def test_an_import_path_naming_another_object_is_rejected(self) -> None:
        def shadow() -> None:
            pass

        shadow.__qualname__ = "max"
        shadow.__module__ = "builtins"

        with pytest.raises(TypeError, match="does not resolve"):
            to_plain(shadow)

    def test_an_unimportable_module_is_rejected(self) -> None:
        def orphan() -> None:
            pass

        orphan.__qualname__ = "orphan"
        orphan.__module__ = "no_such_module_xyz"

        with pytest.raises(TypeError, match="does not resolve"):
            to_plain(orphan)


class Mode(Enum):
    FAST = "fast"
    SLOW = 2


class TestParse:
    @pytest.mark.parametrize("text", ["123", "1.5", "run_a", "true", "null"])
    def test_a_str_target_keeps_the_text(self, text: str) -> None:
        assert parse(text, str) == text

    def test_a_quoted_json_string_reads_as_its_text(self) -> None:
        assert parse('"a b"', str) == "a b"

    @pytest.mark.parametrize(
        ("text", "target", "expected"),
        [
            ("7", int, 7),
            ("3e-4", float, 3e-4),
            ("2", float, 2.0),
            ("true", bool, True),
            ("[1, 2]", list[int], [1, 2]),
            ('{"a": 1}', dict[str, int], {"a": 1}),
            ("[1, 2]", tuple[int, int], (1, 2)),
            ("null", int | None, None),
            ("7", int | str, 7),
            ("run_a", int | str, "run_a"),
            ("123", str | None, "123"),
            ("fast", Mode, Mode.FAST),
            ("2", Mode, Mode.SLOW),
            ("a", Literal["a", "b"], "a"),
            ("/data/x", Path, Path("/data/x")),
            ('"/data/x"', Path, Path("/data/x")),
            ("00000000-0000-0000-0000-000000000001", UUID, UUID(int=1)),
            (
                "2026-01-02T03:04:05+00:00",
                dt.datetime,
                dt.datetime(2026, 1, 2, 3, 4, 5, tzinfo=dt.UTC),
            ),
        ],
        ids=repr,
    )
    def test_text_reads_as_its_target(
        self,
        text: str,
        target: object,
        expected: object,
    ) -> None:
        value = parse(text, target)

        assert value == expected
        assert type(value) is type(expected)

    @pytest.mark.parametrize(
        ("text", "expected"),
        [
            ("inf", math.inf),
            ("-inf", -math.inf),
            ("Infinity", math.inf),
            (" inf ", math.inf),
        ],
    )
    def test_non_finite_spellings_read_as_floats(
        self,
        text: str,
        expected: float,
    ) -> None:
        assert parse(text, float) == expected

    def test_nan_reads_as_a_float(self) -> None:
        assert math.isnan(parse("nan", float))

    def test_a_finite_number_is_not_a_spelling(self) -> None:
        value = parse("1e3", float | str)

        assert value == 1000.0
        assert type(value) is float

    @pytest.mark.parametrize(
        ("text", "target", "match"),
        [
            ("abc", int, "cannot read 'abc' as int"),
            ("[1]", int, r"cannot read \[1\] as int"),
            ("{}", int, "cannot read {} as int"),
            ("1.5", int, "cannot read 1.5 as int"),
            ("yes", bool, "cannot read 'yes' as bool"),
            ("inf", int, "cannot read inf as int"),
            ("[1, 2, 3]", tuple[int, int], "expected 2 items"),
            ("medium", Mode, "cannot read 'medium' as"),
        ],
        ids=repr,
    )
    def test_text_no_reading_fits_raises(
        self,
        text: str,
        target: object,
        match: str,
    ) -> None:
        with pytest.raises(ReadError, match=match):
            parse(text, target)


class TestCaches:
    def test_a_local_dataclass_stays_collectible(self) -> None:
        @dataclass(kw_only=True, slots=True)
        class Local:
            x: int

        assert from_plain({"x": 1}, Local) == Local(x=1)
        ref = weakref.ref(Local)
        del Local
        gc.collect()

        assert ref() is None


if __name__ == "__main__":
    from configgle.lib.testing.main import test_main

    test_main(__file__)
