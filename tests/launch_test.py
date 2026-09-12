from __future__ import annotations

from dataclasses import field
from pathlib import Path

import os
import subprocess
import sys
import textwrap

import pytest

from configgle import Fig, Makeable, launch
from configgle.launch import resolve_config


class Child:
    class Config(Fig["Child"]):
        lr: float = 1e-3

    def __init__(self, config: Config):
        self.lr = config.lr


class Trainer:
    class Config(Fig["Trainer"]):
        steps: int = 100
        child: Child.Config = field(default_factory=Child.Config)

    def __init__(self, config: Config):
        self.config = config


def baseline() -> Makeable[Trainer]:
    """Return a fresh config, as the launcher expects a factory to."""
    return Trainer.Config()


not_a_callable = 42


def returns_non_config() -> int:
    return 42


def test_resolve_config_returns_the_factory_result() -> None:
    config = resolve_config(f"{__name__}.baseline")
    assert isinstance(config, Trainer.Config)
    assert config.steps == 100


def test_resolve_config_returns_a_fresh_config_each_call() -> None:
    """Two resolves must not share mutable state, or one run leaks into the next."""
    first = resolve_config(f"{__name__}.baseline")
    second = resolve_config(f"{__name__}.baseline")
    assert isinstance(first, Trainer.Config)
    assert isinstance(second, Trainer.Config)
    assert first is not second
    assert first.child is not second.child


def test_resolve_config_undotted_path_raises() -> None:
    with pytest.raises(ValueError, match="is not a dotted path"):
        resolve_config("baseline")


def test_resolve_config_missing_module_raises() -> None:
    with pytest.raises(ImportError, match="Cannot import module"):
        resolve_config("configgle_no_such_module.baseline")


def test_resolve_config_missing_attribute_raises() -> None:
    with pytest.raises(AttributeError, match="has no attribute"):
        resolve_config(f"{__name__}.no_such_function")


def test_resolve_config_non_callable_raises() -> None:
    with pytest.raises(TypeError, match="is not callable"):
        resolve_config(f"{__name__}.not_a_callable")


def test_resolve_config_non_config_return_raises() -> None:
    with pytest.raises(TypeError, match="not a config"):
        resolve_config(f"{__name__}.returns_non_config")


def test_resolve_config_composes_with_overrides_and_make() -> None:
    """The launcher's whole contract: resolve -> override -> make."""
    from configgle import apply_overrides  # noqa: PLC0415

    config = resolve_config(f"{__name__}.baseline")
    apply_overrides(config, ["steps=5", "child.lr=3e-4"])
    trainer = config.make()
    assert isinstance(trainer, Trainer)
    assert trainer.config.steps == 5
    assert trainer.config.child.lr == 3e-4


# --- The module docstring's worked example, executed ---------------------------
#
# The docstring is the only place a new user learns the pattern, so it is run
# rather than trusted: the indented block is extracted verbatim, written to a
# module, and launched through the real CLI. A drifted example fails here.


def _docstring_example() -> str:
    """Return the dedented `myproject/experiments.py` block from launch.__doc__."""
    doc = launch.__doc__ or ""
    body = doc.split("    # myproject/experiments.py\n", 1)[1]
    body = body.split("\nRun it,", 1)[0]
    # The docstring shows the public import path; in the monorepo the package is
    # nested. Normalizing here (rather than at each call site) keeps this file
    # byte-identical after the export rewrites `configgle` -> `configgle`;
    # a per-call `.replace` becomes a no-op whose line then re-wraps, and
    # `ruff format --check` fails in the exported tree.
    return textwrap.dedent(body).replace(
        "from configgle import",
        f"from {launch.__name__.rsplit('.', 1)[0]} import",
    )


def test_docstring_example_is_runnable(tmp_path: Path) -> None:
    """The documented example must import, resolve, override, and make."""
    (tmp_path / "myexample.py").write_text(_docstring_example())

    env = {**os.environ, "PYTHONPATH": f"{tmp_path}{os.pathsep}{Path.cwd()}"}
    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "configgle",
            "myexample.baseline",
            "--override",
            "lr=3e-4",
            "--override",
            "steps=5",
        ],
        capture_output=True,
        text=True,
        check=False,
        env=env,
    )

    assert result.returncode == 0, result.stderr
    # The example's `run()` prints its resolved config; the overrides must land.
    assert "training 5 steps at lr=0.0003" in result.stdout


def test_docstring_example_defaults_run_unchanged(tmp_path: Path) -> None:
    """Without overrides the example runs on its declared defaults."""
    (tmp_path / "myexample.py").write_text(_docstring_example())
    env = {**os.environ, "PYTHONPATH": f"{tmp_path}{os.pathsep}{Path.cwd()}"}
    result = subprocess.run(
        [sys.executable, "-m", "configgle", "myexample.baseline"],
        capture_output=True,
        text=True,
        check=False,
        env=env,
    )
    assert result.returncode == 0, result.stderr
    assert "training 100 steps at lr=0.001" in result.stdout


if __name__ == "__main__":
    from configgle.lib.testing.main import test_main

    test_main(__file__)
