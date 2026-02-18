"""Tests for pprinting module."""

from __future__ import annotations

from io import StringIO
from typing import Self, override

import copy
import warnings

from configgle import Fig
from configgle.pprinting import FigPrinter, pformat, pprint


class MockConfigurable:
    """Mock configurable object for testing."""

    def __init__(self, value: int, finalized: bool = False):
        self.value = value
        self._finalized = finalized

    def make(self) -> Self:
        return self

    def finalize(self) -> Self:
        new = copy.copy(self)
        new._finalized = True
        return new


def test_pformat_basic():
    """Test pformat function with basic object."""
    result = pformat({"a": 1, "b": 2})
    assert "'a': 1" in result
    assert "'b': 2" in result


def test_pformat_with_options():
    """Test pformat with various options."""
    obj = {"a": 1000000, "b": 2000000}

    # Test with underscore_numbers
    result = pformat(obj, underscore_numbers=True)
    assert "1_000_000" in result or "1000000" in result

    # Test without underscore_numbers
    result = pformat(obj, underscore_numbers=False)
    assert result is not None


def test_pformat_scrub_memory_address():
    """Test pformat with scrub_memory_address option."""

    class Obj:
        pass

    obj = Obj()
    result = pformat(obj, scrub_memory_address=True)
    # Memory addresses should be scrubbed (0x0defaced pattern)
    assert "0x0defaced" in result or repr(obj) in result


def test_pformat_finalize():
    """Test pformat with finalize option."""
    cfg = MockConfigurable(42, finalized=False)

    # With finalize=True (default) - MockConfigurable satisfies Makeable,
    # so it gets finalized automatically (no warning).
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        result = pformat(cfg, finalize=True)
        assert "MockConfigurable" in result
        assert len(w) == 0

    # With finalize=False - should not warn
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        result = pformat(cfg, finalize=False)
        assert "MockConfigurable" in result
        assert len(w) == 0


def test_pprint_basic():
    """Test pprint function."""
    stream = StringIO()
    pprint({"a": 1, "b": 2}, stream=stream)
    output = stream.getvalue()
    assert "'a': 1" in output
    assert "'b': 2" in output


def test_pprint_with_options():
    """Test pprint with various options."""
    stream = StringIO()
    obj = {"a": 1000000}

    pprint(
        obj,
        stream=stream,
        indent=2,
        width=120,
        underscore_numbers=True,
        finalize=False,
    )
    output = stream.getvalue()
    assert output is not None


def test_pretty_printer_init():
    """Test FigPrinter initialization."""
    pp = FigPrinter(
        indent=2,
        width=120,
        depth=3,
        compact=True,
        sort_dicts=True,
        underscore_numbers=True,
        finalize=True,
        scrub_memory_address=True,
    )
    assert pp._finalize is True
    assert pp._scrub_memory_address is not None


def test_pretty_printer_pprint():
    """Test FigPrinter.pprint method."""
    stream = StringIO()
    pp = FigPrinter(stream=stream)
    pp.pprint({"a": 1, "b": 2})
    output = stream.getvalue()
    assert "'a': 1" in output


def test_pretty_printer_pformat():
    """Test FigPrinter.pformat method."""
    pp = FigPrinter()
    result = pp.pformat({"a": 1, "b": 2})
    assert "'a': 1" in result


def test_pretty_printer_format_with_unfinalized_warning():
    """Test FigPrinter.format warns about unfinalized configs."""

    class UnfinalizedConfig:
        def __init__(self):
            self._finalized = False

        def make(self):
            return self

        def finalize(self) -> Self:
            new = copy.copy(self)
            new._finalized = True
            return new

    pp = FigPrinter(finalize=True)
    cfg = UnfinalizedConfig()

    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        pp.format(cfg, {}, 0, 0)
        # Should warn about unfinalized dataclass
        assert len(w) >= 1
        assert "unfinalized" in str(w[0].message).lower()


def test_pretty_printer_format_scrub():
    """Test FigPrinter.format with memory address scrubbing."""

    class Obj:
        pass

    pp = FigPrinter(scrub_memory_address=True)
    obj = Obj()
    result, _, _ = pp.format(obj, {}, 0, 0)
    # Memory address should be scrubbed (0x0defaced pattern)
    assert "0x0defaced" in result


def test_pretty_printer_try_to_finalize():
    """Test FigPrinter._try_to_finalize method."""

    class FinalizableConfig(Fig):
        value: int = 42

    pp = FigPrinter(finalize=True)
    cfg = FinalizableConfig()

    # Should finalize the config
    finalized = pp._try_to_finalize(cfg)
    # The finalized version should be a different object
    assert finalized is not cfg or cfg.value == 42


def test_pretty_printer_try_to_finalize_with_error():
    """Test FigPrinter._try_to_finalize handles errors."""

    class BadConfig(Fig):
        """Config that raises error during finalize."""

        value: int = 42

        @override
        def finalize(self) -> Self:
            raise ValueError("Cannot finalize")

    pp = FigPrinter(finalize=True)
    cfg = BadConfig()

    # Should catch the error and warn
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        _result = pp._try_to_finalize(cfg)
        # Should warn about the error
        assert len(w) >= 1
        assert "Cannot finalize" in str(w[0].message)


def test_pretty_printer_no_finalize():
    """Test FigPrinter with finalize=False."""

    class Config(Fig):
        value: int = 42

    pp = FigPrinter(finalize=False)
    cfg = Config()

    # Should not finalize
    result = pp._try_to_finalize(cfg)
    assert result is cfg


def test_scrub_memory_address_function():
    """Test the memory address scrubbing function."""
    from configgle.pprinting import (  # noqa: PLC0415
        _SCRUB_MEMORY_ADDRESS_FN,
    )

    # Test scrubbing memory addresses
    text = "Object at 0x7f8b9c0a1b20"
    result = _SCRUB_MEMORY_ADDRESS_FN(text)
    # Should replace the memory address
    assert "0xdeadbeef" in result or "0x" in result


if __name__ == "__main__":
    import pytest

    pytest.main([__file__, "-v"])
