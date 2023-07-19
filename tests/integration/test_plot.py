"""integration / regression tests for plotting.

this test needs pytest-regression package.
run by `pythom -m test tests/`
force regen image by `pythom -m test tests/ --force-regen`
see also: https://pytest-regressions.readthedocs.io/en/latest/index.html.
"""

import pytest

from prettymaps import plot
from prettymaps.new_api.get import (
    Perimeter,
    Shape,
    TransformArg,
    get_gdfs,
    transform_gdfs,
)
from prettymaps.new_api.plot import (
    PlotArg,
    plot_gdfs,
)
from prettymaps.new_api.types import read_preset

# fixed plot parameters
DIFF_THRESHOLD = 3  # % of difference between images
DILATE = (0,)
RADIOUS = 300  # restrict radius for speed

# preset and query
preset_name_and_query = [
    (
        "tijuca",
        "barra da tijuca",
    ),  # using this style since not inlucde palette key which introduce randomness of color
    ("minimal", "barra da tijuca"),
]


@pytest.mark.parametrize(
    "preset_name_and_query",
    preset_name_and_query,
    ids=[preset for preset, _ in preset_name_and_query],
)
def test_regression_plot_from_old(
    image_regression, original_datadir, preset_name_and_query
):
    """Tests old plot api: prettymaps.plot is generating same plot as before."""
    preset_name, query = preset_name_and_query
    image = original_datadir / f"tmp-{preset_name}.png"
    print(f"{str(image)=}")
    plot(query, radius=RADIOUS, save_as=str(image), preset=preset_name)
    image_regression.check(image.read_bytes(), diff_threshold=DIFF_THRESHOLD)


@pytest.mark.parametrize(
    "preset_name_and_query",
    preset_name_and_query,
    ids=[preset for preset, _ in preset_name_and_query],
)
def test_regression_plot_from_new_to_old(
    image_regression, original_datadir, preset_name_and_query
):
    """Tests regression of plot from new api to plot from old api.

    Yields plot from new api and compare image from old api (which is produced and git checked in by previous `test_regression_plot_from_old`.)
    """
    preset_name, query = preset_name_and_query
    preset = read_preset(preset_name)

    image = original_datadir / f"tmp-{preset_name}-new-api.png"
    print(str(image))

    gdfs = get_gdfs(
        preset.layers,
        Perimeter.from_geocode_point(
            query, Shape("square", RADIOUS), dilate=preset.dilate
        ),
    )

    # do nothing
    gdfs_transformed = transform_gdfs(gdfs, TransformArg())
    assert gdfs_transformed == gdfs

    plot_gdfs(
        gdfs_transformed,
        plot_arg=PlotArg(layers=preset.layers, style=preset.style, save_as=str(image)),
    )

    reg_image = f"test_regression_plot_from_old_{preset_name}_"  # base image name in tests/test_plot dir to compare
    image_regression.check(
        image.read_bytes(), basename=reg_image, diff_threshold=DIFF_THRESHOLD
    )
