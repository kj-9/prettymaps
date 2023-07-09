from prettymaps import plot, presets


def test_presets():
    assert len(presets()) == 7


def test_plot(image_regression, datadir):
    """
    needs pytest-regression package.
    run by `pythom -m test tests/`
    force regen image by `pythom -m test tests/ --force-regen`
    see also: https://pytest-regressions.readthedocs.io/en/latest/index.html
    """
    image = datadir / "macau-circle-300.png"
    print(str(image))
    plot(
        "Praça Ferreira do Amaral, Macau",
        circle=True,
        radius=300,
        save_as=str(image),
    )
    image_regression.check(image.read_bytes(), diff_threshold=5)
