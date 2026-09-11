#!/bin/sh
# ruff: noqa: EXE003, D300 -- Polyglot shell/Python script.
# fmt: off
'''' 2>/dev/null #
exec uv --quiet --project "$(dirname "$0")" run --frozen --no-sync python3 "$0" "$@"
Validate Configgle wheel contents.
'''
# fmt: on

from pathlib import Path

import zipfile


def main() -> int:
    """Run the program; return the process exit code.


    Returns:
      result: The int.

    """
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
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
# vim: ft=python
