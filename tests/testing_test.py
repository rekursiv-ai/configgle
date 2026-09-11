"""Tests for Configgle pprint golden assertions."""

from __future__ import annotations

from pathlib import Path
from typing import Self, override

import subprocess
import sys
import textwrap

import pytest

from configgle.fig import Fig, Maker
from configgle.testing import assert_pprint_golden


class _DefaultsOnly:
    class Config(Fig["_DefaultsOnly"]):
        default_value: int = 3
        """A value retained in the full golden."""

        long_default: str = "a default long enough to force dataclass pprint dispatch"
        """A long default retained in the full golden."""

    def __init__(self, config: Config) -> None:
        del config


class _Example:
    class Config(Fig["_Example"]):
        inherited: int = -1
        """A value filled during finalization."""

        @override
        def finalize(self) -> Self:
            self.inherited = 7
            return super().finalize()

    def __init__(self, config: Config) -> None:
        del config


@pytest.fixture(autouse=True)
def clear_regeneration(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("CONFIGGLE_REGENERATE_GOLDEN", raising=False)


def test_assert_pprint_golden_reads_full_finalized_config(tmp_path: Path) -> None:
    test_file = tmp_path / "owner_test.py"
    testdata = tmp_path / "testdata"
    testdata.mkdir()
    golden = testdata / "example.txt"
    golden.write_text(
        _Example.Config().pformat(hide_default_values=False) + "\n",
        encoding="utf-8",
    )

    assert_pprint_golden(
        test_file=str(test_file),
        name="example",
        config=_Example.Config(),
    )

    assert "inherited=7" in golden.read_text(encoding="utf-8")


def test_assert_pprint_golden_pins_rendering_policy(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    seen: dict[str, object] = {}

    def pformat(config: Maker[object], **kwargs: object) -> str:
        del config
        seen.update(kwargs)
        return "rendered"

    monkeypatch.setattr(Maker, "pformat", pformat)
    test_file = tmp_path / "owner_test.py"
    testdata = tmp_path / "testdata"
    testdata.mkdir()
    (testdata / "example.txt").write_text("rendered\n", encoding="utf-8")

    assert_pprint_golden(
        test_file=str(test_file),
        name="example",
        config=_Example.Config(),
    )

    assert seen == {
        "finalize": True,
        "hide_default_values": False,
        "mask_memory_addresses": True,
    }


def test_assert_pprint_golden_writes_full_unchanged_defaults(tmp_path: Path) -> None:
    test_file = tmp_path / "owner_test.py"

    with pytest.raises(AssertionError, match="Missing golden regenerated"):
        assert_pprint_golden(
            test_file=str(test_file),
            name="defaults",
            config=_DefaultsOnly.Config(),
        )

    rendered = (tmp_path / "testdata" / "defaults.txt").read_text(encoding="utf-8")
    assert "default_value=3" in rendered
    assert "long_default='a default long enough" in rendered


def test_assert_pprint_golden_regenerates_missing_then_fails(tmp_path: Path) -> None:
    test_file = tmp_path / "owner_test.py"

    with pytest.raises(AssertionError, match="Missing golden regenerated"):
        assert_pprint_golden(
            test_file=str(test_file),
            name="example",
            config=_Example.Config(),
        )

    rendered = (tmp_path / "testdata" / "example.txt").read_text(encoding="utf-8")
    assert "inherited=7" in rendered


def test_assert_pprint_golden_reports_mismatch_without_rewriting(
    tmp_path: Path,
) -> None:
    test_file = tmp_path / "owner_test.py"
    testdata = tmp_path / "testdata"
    testdata.mkdir()
    golden = testdata / "example.txt"
    golden.write_text("stale\n", encoding="utf-8")

    with pytest.raises(
        AssertionError,
        match="CONFIGGLE_REGENERATE_GOLDEN=1",
    ) as exc_info:
        assert_pprint_golden(
            test_file=str(test_file),
            name="example",
            config=_Example.Config(),
        )

    message = str(exc_info.value)
    assert "-stale" in message
    assert "+_Example.Config" in message
    assert golden.read_text(encoding="utf-8") == "stale\n"


def test_assert_pprint_golden_rejects_mismatch_under_optimized_python(
    tmp_path: Path,
) -> None:
    test_file = tmp_path / "owner_test.py"
    testdata = tmp_path / "testdata"
    testdata.mkdir()
    (testdata / "example.txt").write_text("stale\n", encoding="utf-8")
    script = textwrap.dedent(
        f"""
        from configgle.fig import Fig
        from configgle.testing import assert_pprint_golden

        class Example:
            class Config(Fig["Example"]):
                value: int = 1

            def __init__(self, config: Config) -> None:
                del config

        assert_pprint_golden(
            test_file={str(test_file)!r},
            name="example",
            config=Example.Config(),
        )
        """
    )

    result = subprocess.run(  # noqa: S603 -- fixed argv; the script is a literal built above
        [sys.executable, "-O", "-c", script],
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode != 0
    assert "AssertionError" in result.stderr


def test_assert_pprint_golden_regenerates_from_environment(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    test_file = tmp_path / "owner_test.py"
    testdata = tmp_path / "testdata"
    testdata.mkdir()
    golden = testdata / "example.txt"
    golden.write_text("stale\n", encoding="utf-8")
    monkeypatch.setenv("CONFIGGLE_REGENERATE_GOLDEN", "1")

    assert_pprint_golden(
        test_file=str(test_file),
        name="example",
        config=_Example.Config(),
    )

    assert "inherited=7" in golden.read_text(encoding="utf-8")


if __name__ == "__main__":
    from configgle.lib.testing.main import test_main

    test_main(__file__)
