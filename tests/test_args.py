"""Test for arg types."""

from prettymaps.new_api import PlotArg, TransformArg


def test_transform_arg():
    """Unit test for TransformArg."""
    arg = TransformArg(1, 2, 3, 4, 5)
    x, y, scale_x, scale_y, rotation = arg.to_tuple()

    assert x == arg.x
    assert y == arg.y
    assert scale_x == arg.scale_x
    assert scale_y == arg.scale_y
    assert rotation == arg.rotation


def test_plot_arg():
    """Unit test for PlotArg."""
    arg = PlotArg(1, 2, 3, 4, 5)
    layers, style, ax, figsize, credit, show, save_as = arg.to_tuple()

    assert layers == arg.layers
    assert style == arg.style
    assert ax == arg.ax
    assert figsize == arg.figsize
    assert credit == arg.credit
    assert show == arg.show
    assert save_as == arg.save_as
