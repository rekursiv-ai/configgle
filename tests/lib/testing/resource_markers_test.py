"""Tests for the shared pytest resource-marker rollups."""

from __future__ import annotations

from importlib import machinery, util
from pathlib import Path
from types import SimpleNamespace
from typing import TYPE_CHECKING, Final, cast

import subprocess
import sys
import textwrap

import pytest

from configgle.lib.testing import resource_markers
from configgle.lib.testing.resource_markers import (
    apply_golden_marker,
    apply_resource_markers,
    module_path,
    pytest_collection_modifyitems,
    resource_marker_aliases,
    resource_marker_timeout,
)


if TYPE_CHECKING:
    from collections.abc import Iterator


_CWD: Final = Path(__file__).resolve().parent


@pytest.fixture(autouse=True)
def fresh_call_graph() -> Iterator[None]:
    """Forget cached sources: each test writes its own modules under tmp_path."""
    caches = (
        resource_markers.module_path,
        resource_markers._module_source,
        resource_markers._function_facts,
        resource_markers._imports_golden_support,
        resource_markers._test_reaches_golden,
    )
    for cached in caches:
        cached.cache_clear()
    yield
    for cached in caches:
        cached.cache_clear()


def _run_pytest(tmp_path: Path, *, bindings: int) -> subprocess.CompletedProcess[str]:
    """Collect a tree carrying an unregistered marker under ``bindings`` hooks."""
    conftest = textwrap.dedent(
        """
        import sys

        sys.path.insert(0, {root!r})

        from configgle.lib.testing.resource_markers import pytest_collection_modifyitems

        __all__ = ["pytest_collection_modifyitems"]
        """,
    ).format(root=str(_CWD.parents[2]))
    package = tmp_path / "pkg"
    package.mkdir()
    (package / "conftest.py").write_text(conftest, encoding="utf-8")
    if bindings > 1:
        (tmp_path / "conftest.py").write_text(conftest, encoding="utf-8")
    (package / "probe_test.py").write_text(
        textwrap.dedent(
            """
            import pytest

            @pytest.mark.network_notreal
            def test_probe():
                pass
            """,
        ),
        encoding="utf-8",
    )
    return subprocess.run(
        [
            sys.executable,
            "-m",
            "pytest",
            "pkg/probe_test.py",
            "--collect-only",
            "-q",
            "-p",
            "no:cacheprovider",
        ],
        cwd=tmp_path,
        capture_output=True,
        text=True,
        check=False,
    )


@pytest.mark.cli_python_subprocess
def test_unknown_resource_marker_reports_a_usage_error(tmp_path: Path) -> None:
    """An unregistered resource marker must report, not crash the runner.

    The check runs inside ``pytest_collection_modifyitems``. pytest turns an
    arbitrary exception escaping a hook into an INTERNALERROR traceback, which
    buries the one line naming the offending marker; ``UsageError`` is the
    exception it renders as a plain message instead.
    """
    result = _run_pytest(tmp_path, bindings=1)

    output = result.stdout + result.stderr
    assert result.returncode == pytest.ExitCode.USAGE_ERROR
    assert "INTERNALERROR" not in output
    assert "Unknown resource marker: network_notreal" in output


@pytest.mark.cli_python_subprocess
def test_double_bound_hook_reports_a_usage_error(tmp_path: Path) -> None:
    """Two conftests binding the hook report the same way as one.

    A root conftest and a package conftest both bind it whenever the export
    ships tests outside the package directory, so the second binding must not
    change how a bad marker surfaces.
    """
    result = _run_pytest(tmp_path, bindings=2)

    output = result.stdout + result.stderr
    assert result.returncode == pytest.ExitCode.USAGE_ERROR
    assert "INTERNALERROR" not in output
    assert "Unknown resource marker: network_notreal" in output


def test_collection_hook_registers_the_golden_marker() -> None:
    registered: list[tuple[str, str]] = []

    config = _Config(registered)
    pytest_collection_modifyitems(cast(pytest.Config, cast(object, config)), [])
    pytest_collection_modifyitems(cast(pytest.Config, cast(object, config)), [])

    assert registered == [("markers", "golden: tests that assert against a golden")]


@pytest.mark.parametrize(("markexpr", "marked"), [("golden", True), ("", False)])
def test_golden_marks_are_derived_only_when_selected(
    tmp_path: Path,
    markexpr: str,
    *,
    marked: bool,
) -> None:
    module = tmp_path / "selected_test.py"
    module.write_text(
        "from numerics.testing.bfb import assert_bfb_against_golden as check\n"
        "\ndef test_selected():\n    check()\n",
        encoding="utf-8",
    )
    item = _Item(test_name="test_selected")
    item.module = type("Module", (), {"__file__": str(module)})

    pytest_collection_modifyitems(
        cast(pytest.Config, cast(object, _Config([], markexpr=markexpr))),
        cast(list[pytest.Item], cast(object, [item])),
    )

    assert ("golden" in [m.name for m in item.iter_markers()]) is marked


def test_unknown_alias_lookup_reports_a_usage_error() -> None:
    """``resource_marker_aliases`` rejects an unknown marker the same way.

    It is reached from the same hook, so a bare exception here escapes with the
    same INTERNALERROR shape as the collection check.
    """
    with pytest.raises(
        pytest.UsageError,
        match="Unknown resource marker: network_nope",
    ):
        resource_marker_aliases("network_nope")


def test_indirect_golden_caller_is_marked(tmp_path: Path) -> None:
    module = tmp_path / "bridge_test.py"
    module.write_text(
        "import numerics.testing.bfb as bfb\n"
        "\ndef test_indirect():\n    bfb.assert_bfb_against_golden()\n",
        encoding="utf-8",
    )
    item = _Item(test_name="test_indirect")
    item.module = type("Module", (), {"__file__": str(module)})

    apply_golden_marker([item])

    assert [marker.name for marker in item.iter_markers()] == ["golden"]


def test_imported_name_call_is_marked(tmp_path: Path) -> None:
    module = tmp_path / "name_test.py"
    module.write_text(
        "from numerics.testing.bfb import assert_bfb_against_golden as check\n"
        "\ndef test_name():\n    check()\n",
        encoding="utf-8",
    )
    item = _Item(test_name="test_name")
    item.module = type("Module", (), {"__file__": str(module)})

    apply_golden_marker([item])

    assert [marker.name for marker in item.iter_markers()] == ["golden"]


def test_importing_bfb_without_calling_an_assertion_is_not_marked(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    (tmp_path / "bfb.py").write_text(
        "def host_agnostic_numerics():\n    pass\n",
        encoding="utf-8",
    )
    module = tmp_path / "plain_test.py"
    module.write_text(
        "from bfb import host_agnostic_numerics\n"
        "\ndef test_plain():\n    host_agnostic_numerics()\n",
        encoding="utf-8",
    )
    monkeypatch.setattr(sys, "path", [str(tmp_path), *sys.path])
    item = _Item(test_name="test_plain")
    item.module = type("Module", (), {"__file__": str(module)})

    apply_golden_marker([item])

    assert list(item.iter_markers()) == []


def test_test_loading_a_checked_in_tensor_golden_is_marked(tmp_path: Path) -> None:
    module = tmp_path / "direct_test.py"
    module.write_text(
        "import torch\n"
        "\ndef test_direct():\n"
        '    torch.load(_CWD / "testdata" / "g.pt")\n',
        encoding="utf-8",
    )
    item = _Item(test_name="test_direct")
    item.module = type("Module", (), {"__file__": str(module)})

    apply_golden_marker([item])

    assert [marker.name for marker in item.iter_markers()] == ["golden"]


def test_test_reading_a_module_level_golden_path_is_marked(tmp_path: Path) -> None:
    module = tmp_path / "constant_test.py"
    module.write_text(
        "from numerics.testing.golden import read_tensors\n"
        '\nGOLDEN = _CWD / "testdata" / "data.pt"\n'
        "\ndef test_constant():\n    read_tensors(GOLDEN)\n",
        encoding="utf-8",
    )
    item = _Item(test_name="test_constant")
    item.module = type("Module", (), {"__file__": str(module)})

    apply_golden_marker([item])

    assert [marker.name for marker in item.iter_markers()] == ["golden"]


def test_call_through_a_from_imported_submodule_is_marked(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    package = tmp_path / "pkg"
    package.mkdir()
    (package / "__init__.py").write_text("", encoding="utf-8")
    (package / "records.py").write_text(
        'def golden_path(name):\n    return _CWD / "testdata" / name\n',
        encoding="utf-8",
    )
    module = tmp_path / "submodule_test.py"
    module.write_text(
        "from pkg import records\n"
        '\ndef test_submodule():\n    records.golden_path("g.pt")\n',
        encoding="utf-8",
    )
    monkeypatch.setattr(sys, "path", [str(tmp_path), *sys.path])
    monkeypatch.setattr(
        "configgle.lib.testing.resource_markers._CWD",
        tmp_path / "a" / "b",
    )
    item = _Item(test_name="test_submodule")
    item.module = type("Module", (), {"__file__": str(module)})

    apply_golden_marker([item])

    assert [marker.name for marker in item.iter_markers()] == ["golden"]


def test_test_writing_a_temporary_checkpoint_is_not_marked(tmp_path: Path) -> None:
    module = tmp_path / "checkpoint_test.py"
    module.write_text(
        "import torch\n"
        "\ndef test_checkpoint(tmp_path):\n"
        '    torch.load(tmp_path / "ckpt.pt")\n',
        encoding="utf-8",
    )
    item = _Item(test_name="test_checkpoint")
    item.module = type("Module", (), {"__file__": str(module)})

    apply_golden_marker([item])

    assert list(item.iter_markers()) == []


def test_a_test_module_moved_away_after_import_is_not_marked(tmp_path: Path) -> None:
    item = _Item(test_name="test_gone")
    item.module = type("Module", (), {"__file__": str(tmp_path / "gone_test.py")})

    apply_golden_marker([item])

    assert list(item.iter_markers()) == []


def test_module_path_ignores_invalid_and_external_specs(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def invalid_spec(name: str) -> object:
        del name
        raise ValueError("module __spec__ is None")

    monkeypatch.setattr(util, "find_spec", invalid_spec)
    assert module_path("torch._C._jit_tree_views") is None

    external = tmp_path / "external.py"
    external.write_text("", encoding="utf-8")
    spec = util.spec_from_file_location("external", external)
    assert spec is not None

    def external_spec(name: str) -> machinery.ModuleSpec:
        del name
        return spec

    monkeypatch.setattr(util, "find_spec", external_spec)
    assert module_path("external") is None


def test_openml_cold_fetch_keeps_its_hosted_runner_timeout_budget() -> None:
    """A cold OpenML download needs the previously established 120s budget."""
    assert resource_marker_timeout("network_openml") >= 120


def test_e2b_auth_keeps_integration_selection_and_network_deadline() -> None:
    """Selecting the E2B resource must preserve the real-network test tier."""
    item = _Item("network_e2b")
    apply_resource_markers([item], resource_markers=("network_e2b",))
    assert item.get_closest_marker("integration") is not None
    timeout = item.get_closest_marker("timeout")
    assert timeout is not None
    assert timeout.args == (resource_marker_timeout("network_e2b"),)


def test_resource_marker_family_rejects_malformed_names() -> None:
    with pytest.raises(ValueError, match="separator"):
        resource_markers.resource_marker_family("network")
    with pytest.raises(ValueError, match="family"):
        resource_markers.resource_marker_family("unknown_probe")


def test_unknown_resource_marker_is_rejected() -> None:
    with pytest.raises(pytest.UsageError, match="network_missing"):
        resource_markers._fail_on_unknown_resource_markers(
            {"network_missing"},
            resource_markers=set(),
        )


def test_golden_call_graph_import_edges_are_resolved(tmp_path: Path) -> None:
    direct = tmp_path / "direct.py"
    direct.write_text(
        "import numerics.testing.bfb as bfb\n",
        encoding="utf-8",
    )
    assert resource_markers._function_facts(direct, "assert_anything") == (True, ())

    imported = tmp_path / "imported.py"
    imported.write_text(
        "from numerics.testing.bfb import assert_bfb_against_golden\n",
        encoding="utf-8",
    )
    assert resource_markers._function_facts(
        imported,
        "assert_bfb_against_golden",
    ) == (True, ())
    aliased = tmp_path / "aliased.py"
    aliased.write_text(
        "from numerics.testing.bfb import assert_bfb_against_golden as check\n",
        encoding="utf-8",
    )
    assert resource_markers._function_facts(aliased, "check") == (True, ())


def test_import_graph_without_golden_support_is_false(tmp_path: Path) -> None:
    module = tmp_path / "plain.py"
    module.write_text("import pathlib\n", encoding="utf-8")
    assert resource_markers._imports_golden_support(module) is False


def test_import_graph_skips_a_seen_module(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    root = tmp_path / "root.py"
    root.write_text("import bridge\n", encoding="utf-8")
    bridge = tmp_path / "bridge.py"
    bridge.write_text("import bridge\n", encoding="utf-8")

    def find_bridge(name: str) -> Path | None:
        return bridge if name == "bridge" else None

    monkeypatch.setattr(resource_markers, "module_path", find_bridge)
    assert resource_markers._imports_golden_support(root) is False


def test_import_graph_follows_local_modules_to_golden_support(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    root = tmp_path / "root.py"
    bridge = tmp_path / "bridge.py"
    root.write_text("import bridge\n", encoding="utf-8")
    bridge.write_text("import numerics.testing.bfb\n", encoding="utf-8")

    def find_bridge(name: str) -> Path | None:
        return bridge if name == "bridge" else None

    monkeypatch.setattr(resource_markers, "module_path", find_bridge)
    assert resource_markers._imports_golden_support(root) is True


class _Config:
    """Minimal stand-in for the config surface the collection hook reads."""

    def __init__(
        self,
        registered: list[tuple[str, str]],
        *,
        markexpr: str = "",
    ) -> None:
        self.markers: list[str] = []
        self.registered = registered
        self.markexpr = markexpr

    def getini(self, name: str) -> list[str]:
        if name == "markers":
            return self.markers
        return []

    def getoption(self, name: str, default: object = None) -> object:
        return self.markexpr if name == "markexpr" else default

    def addinivalue_line(self, name: str, value: str) -> None:
        self.registered.append((name, value))
        self.markers.append(value)


class _Item:
    """Minimal stand-in exposing the marker surface the rollup touches."""

    def __init__(self, *marks: str, test_name: str = "test_probe") -> None:
        self.module: object | None = None
        self.obj = SimpleNamespace(__name__=test_name)
        self.own_markers: list[pytest.Mark] = [
            cast(pytest.MarkDecorator, getattr(pytest.mark, mark)).mark
            for mark in marks
        ]

    def iter_markers(self, name: str | None = None) -> Iterator[pytest.Mark]:
        for marker in self.own_markers:
            if name is None or marker.name == name:
                yield marker

    def get_closest_marker(self, name: str) -> pytest.Mark | None:
        for marker in self.own_markers:
            if marker.name == name:
                return marker
        return None

    def add_marker(
        self,
        marker: str | pytest.MarkDecorator,
        *,
        append: bool = True,
    ) -> None:
        del append
        assert isinstance(marker, pytest.MarkDecorator)
        self.own_markers.append(marker.mark)


def test_applying_markers_twice_leaves_one_alias() -> None:
    """Rolling up twice must not stack duplicate alias marks on an item.

    A root conftest and a package conftest both bind this hook, so an item is
    walked once per binding. Without idempotence the aliases accumulate, and the
    duplicates travel with the item into every later report.
    """
    item = _Item("network_github")
    for _ in range(2):
        apply_resource_markers([item], resource_markers=("network_github",))

    names = [marker.name for marker in item.iter_markers()]
    assert names.count("integration") == 1


def test_applying_markers_twice_leaves_one_live_llm_skip(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The live-LLM skip is added once, however many bindings walk the item.

    Every mark this function adds has to survive a second pass unchanged --
    an alias, a timeout, and a skip alike -- because a duplicate skip reason
    travels into every later report exactly as a duplicate alias does.
    """
    monkeypatch.delenv("RUN_REAL_LLM", raising=False)
    item = _Item("cli_real_llm")
    for _ in range(2):
        apply_resource_markers([item], resource_markers=("cli_real_llm",))

    names = [marker.name for marker in item.iter_markers()]
    assert names.count("skip") == 1


def test_applying_markers_twice_leaves_one_ci_skip(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The CI skip is added once, however many bindings walk the item."""
    monkeypatch.setenv("CI", "1")
    monkeypatch.delenv("RUN_INTEGRATION", raising=False)
    item = _Item("gpu_nvidia")
    for _ in range(2):
        apply_resource_markers([item], resource_markers=("gpu_nvidia",))

    names = [marker.name for marker in item.iter_markers()]
    assert names.count("skip") == 1


if __name__ == "__main__":
    from configgle.lib.testing.main import test_main

    test_main(__file__)
