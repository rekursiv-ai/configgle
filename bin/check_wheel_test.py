from __future__ import annotations

from typing import TYPE_CHECKING
from zipfile import ZipFile

import pytest

from bin import check_wheel


if TYPE_CHECKING:
    from pathlib import Path


def test_main_accepts_wheel_with_exact_required_entries(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A wheel passes when every required package entry is present."""
    dist = tmp_path / "dist"
    dist.mkdir()
    with ZipFile(dist / "configgle-1.0.whl", "w") as archive:
        for name in (
            "configgle/__init__.py",
            "configgle/py.typed",
            "ty_extensions/__init__.py",
        ):
            archive.writestr(name, "")
    monkeypatch.chdir(tmp_path)
    assert check_wheel.main() == 0


def test_main_reports_sorted_missing_entries(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A wheel failure names every missing entry in sorted order."""
    dist = tmp_path / "dist"
    dist.mkdir()
    with ZipFile(dist / "configgle-1.0.whl", "w") as archive:
        archive.writestr("configgle/__init__.py", "")
    monkeypatch.chdir(tmp_path)
    with pytest.raises(
        SystemExit,
        match=r"^wheel is missing required wheel entries: configgle/py\.typed, ty_extensions/__init__\.py$",
    ):
        check_wheel.main()


if __name__ == "__main__":
    from configgle.lib.testing.main import test_main

    test_main(__file__)
