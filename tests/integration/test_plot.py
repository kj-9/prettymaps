"""integration / regression tests for plotting.

this test needs pytest-regression package.
run by `pythom -m test tests/`
force regen image by `pythom -m test tests/ --force-regen`
see also: https://pytest-regressions.readthedocs.io/en/latest/index.html.
"""

from prettymaps import plot
from prettymaps.new_api import (
    GetArg,
    PlotArg,
    TransformArg,
    get_gdfs,
    plot_gdfs,
    read_preset,
    transform_gdfs,
)

# plot parameters
DIFF_THRESHOLD = 3  # % of difference between images
QUERY = "barra da tijuca"
DILATE = (0,)
PRESET = "tijuca"  # using this style since not inlucde palette key which introduce randomness of color
RADIOUS = 300  # restrict radius for speed

# regression image name
REG_IMAGE = "plot_from_old_api"


def test_regression_plot_from_old(image_regression, original_datadir):
    """Tests old plot api: prettymaps.plot is generating same plot as before."""
    image = original_datadir / "tmp-tijuca-300.png"
    print(f"{str(image)=}")
    plot(QUERY, radius=RADIOUS, save_as=str(image), preset=PRESET)
    image_regression.check(
        image.read_bytes(), basename=REG_IMAGE, diff_threshold=DIFF_THRESHOLD
    )


def test_regression_plot_from_new_and_from_old(image_regression, original_datadir):
    """Tests regression of plot from new api to plot from old api."""
    preset = read_preset(PRESET)

    image = original_datadir / "tmp-tijuca-300-new-api.png"
    print(str(image))

    gdfs = get_gdfs(
        GetArg(query=QUERY, radius=RADIOUS, dilate=preset.dilate, layers=preset.layers)
    )

    # do nothing
    gdfs_transformed = transform_gdfs(gdfs, TransformArg())
    assert gdfs_transformed == gdfs

    plot_gdfs(
        gdfs_transformed,
        plot_arg=PlotArg(layers=preset.layers, style=preset.style, save_as=str(image)),
    )

    image_regression.check(
        image.read_bytes(), basename=REG_IMAGE, diff_threshold=DIFF_THRESHOLD
    )
