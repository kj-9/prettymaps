"""Tests for presets."""

from dataclasses import asdict

from prettymaps import presets
from prettymaps.draw import read_preset
from prettymaps.new_api.types import presets_directory
from prettymaps.new_api.types import read_preset as new_read_preset

NUM_PRESETS = 7


def test_presets():
    """Unit tests for presets."""
    assert len(presets()) == NUM_PRESETS


def test_presets_can_be_load():
    """Rgression test for presets between new and old api."""
    for json_file in presets_directory().iterdir():
        preset_name = json_file.name.removesuffix(".json")

        _ = read_preset(preset_name)  # old api
        _ = asdict(new_read_preset(preset_name))  # new api
