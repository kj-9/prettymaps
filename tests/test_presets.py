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


def test_presets_new_api():
    """Rgression test for presets between new and old api."""
    for json_file in presets_directory().iterdir():
        preset_name = json_file.name.removesuffix(".json")

        preset_from_old = read_preset(preset_name)
        preset_from_new = asdict(new_read_preset(preset_name))

        # only check keys in preset_from_old
        # since preset_from_old may not have all keys but preset_from_new always has all keys (None as default value)
        for key in preset_from_old.keys():
            assert preset_from_old[key] == preset_from_new[key]
