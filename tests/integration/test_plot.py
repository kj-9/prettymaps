"""
needs pytest-regression package.
run by `pythom -m test tests/`
force regen image by `pythom -m test tests/ --force-regen`
see also: https://pytest-regressions.readthedocs.io/en/latest/index.html
"""

from prettymaps import plot
from prettymaps.new_api import get_gdfs, transform_gdfs, plot_gdfs, GetArg, read_preset
from pathlib import Path

DIFF_THRESHOLD=3 # % of difference between images

def plot_new(datadir: Path) -> Path:
    preset=read_preset('default')

    image = datadir / "macau-circle-300-new-api.png"
    print(str(image))

    gdfs = get_gdfs(
        GetArg(query="Praça Ferreira do Amaral, Macau", radius=300, dilate=preset.dilate, layers=preset.layers, circle=True)
    )

    # do nothing
    gdfs_transformed = transform_gdfs(gdfs)
    assert gdfs_transformed == gdfs

    plot_gdfs(gdfs_transformed, layers=preset.layers, style=preset.style, save_as=str(image))

    return image



def test_plot(image_regression, original_datadir):
    print(f"{__file__=}")
    print(f"{original_datadir=}")
    
    image = original_datadir / "macau-circle-300.png"
    print(str(image))
    plot(
        "Praça Ferreira do Amaral, Macau",
        circle=True,
        radius=300,
        save_as=str(image),
    )
    image_regression.check(image.read_bytes(), basename='plot_old', diff_threshold=DIFF_THRESHOLD)

    image_from_new = plot_new(original_datadir)
    image_regression.check(image_from_new.read_bytes(), basename='plot_old', diff_threshold=DIFF_THRESHOLD)
