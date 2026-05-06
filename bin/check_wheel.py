"""Validate Configgle wheel contents."""

from pathlib import Path

import zipfile


def main() -> None:
    wheel = next(Path("dist").glob("configgle-*.whl"))
    with zipfile.ZipFile(wheel) as archive:
        names = set(archive.namelist())

    missing = sorted(
        {
            "configgle/__init__.py",
            "configgle/py.typed",
            "ty_extensions/__init__.py",
        }
        - names
    )
    if missing:
        raise SystemExit(
            "wheel is missing required wheel entries: " + ", ".join(missing)
        )


if __name__ == "__main__":
    main()
