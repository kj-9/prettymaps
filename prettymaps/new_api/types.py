"""Experimental new api for prettymaps."""

import json
from dataclasses import dataclass
from pathlib import Path
from typing import NewType

import geopandas as gp

# maybe dataclass is more suitable
GeoDataFrames = NewType("GeoDataFrames", dict[str, gp.GeoDataFrame])
Layers = NewType("Layers", dict[str, dict])  # osm layer for query
Style = NewType("Style", dict[str, dict])  # style for layer


@dataclass
class Preset:
    """Useful set of parameters for getting and plotting gdfs."""

    # name: str # preset name, currently not encoded in json
    layers: Layers
    style: Style
    radius: float
    circle: bool | None = None
    dilate: float | None = None


def presets_directory() -> Path:
    """Returns presets directory."""
    return Path(__file__).resolve().parent / "presets"


def read_preset(name: str) -> Preset:
    """Read a preset from the presets folder (prettymaps/presets/)."""
    with open(presets_directory() / f"{name}.json") as f:
        # Load params from JSON file
        preset_dict = json.load(f)

    return Preset(**preset_dict)
