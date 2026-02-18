"""Configgle: Tools for making configurable Python classes for A/B experiments."""

from __future__ import annotations

from configgle.copy_on_write import CopyOnWrite
from configgle.custom_types import (
    Configurable,
    DataclassLike,
    HasConfig,
    HasRelaxedConfig,
    Makeable,
    RelaxedConfigurable,
    RelaxedMakeable,
)
from configgle.decorator import autofig
from configgle.fig import Dataclass, Fig, Maker, Makes
from configgle.inline import InlineConfig, PartialConfig
from configgle.pprinting import pformat, pprint


__all__ = [
    "Configurable",
    "CopyOnWrite",
    "Dataclass",
    "DataclassLike",
    "Fig",
    "HasConfig",
    "HasRelaxedConfig",
    "InlineConfig",
    "Makeable",
    "Maker",
    "Makes",
    "PartialConfig",
    "RelaxedConfigurable",
    "RelaxedMakeable",
    "autofig",
    "pformat",
    "pprint",
]
