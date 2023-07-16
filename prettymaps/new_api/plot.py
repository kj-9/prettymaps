"""Experimental new api for prettymaps."""

import warnings
from collections.abc import Iterable
from copy import deepcopy
from typing import NamedTuple

import geopandas as gp
import matplotlib
import numpy as np
import osmnx as ox
import shapely.affinity
import shapely.ops
from matplotlib import pyplot as plt
from matplotlib.patches import Path, PathPatch
from shapely.geometry import (
    GeometryCollection,
    LineString,
    MultiLineString,
    MultiPolygon,
    Point,
    Polygon,
    box,
)
from shapely.geometry.base import BaseGeometry

from .types import GeoDataFrames, Layers, Style


class PlotArg(NamedTuple):
    """Dataclass represents arguments for plot_gdfs."""

    layers: Layers
    style: Style
    ax: plt.Axes | None = None
    figsize: tuple[float, float] = (12, 12)
    credit: None = None
    show: bool = True
    save_as: str | None = None


def create_background(gdfs: GeoDataFrames, style: Style) -> BaseGeometry:
    """Create a background layer given a collection of GeoDataFrames.

    Args:
        gdfs: Dictionary of GeoDataFrames
        style: Dictionary of matplotlib style parameters

    Returns:
        background geometry
    """
    # Create background
    background_pad = 1.1
    if "background" in style and "pad" in style["background"]:
        background_pad = style["background"].pop("pad")

    background = shapely.affinity.scale(
        box(
            *shapely.ops.unary_union(ox.project_gdf(gdfs["perimeter"]).geometry).bounds
        ),
        background_pad,
        background_pad,
    )

    if "background" in style and "dilate" in style["background"]:
        background = background.buffer(style["background"].pop("dilate"))

    return background


def override_params(default_dict: dict, new_dict: dict) -> dict:
    """Override parameters in 'default_dict' with additional parameters from 'new_dict'.

    Args:
        default_dict (dict): Default dict to be overriden with 'new_dict' parameters
        new_dict (dict): New dict to override 'default_dict' parameters

    Returns:
        dict: default_dict overriden with new_dict parameters
    """
    final_dict = deepcopy(default_dict)

    for key in new_dict.keys():
        if type(new_dict[key]) == dict:
            if key in final_dict:
                final_dict[key] = override_params(final_dict[key], new_dict[key])
            else:
                final_dict[key] = new_dict[key]
        else:
            final_dict[key] = new_dict[key]

    return final_dict


def draw_text(params: dict[str, dict], background: BaseGeometry) -> None:
    """Draw text with content and matplotlib style parameters specified by 'params' dictionary.

    params['text'] should contain the message to be drawn.

    Args:
        params (Dict[str, dict]): matplotlib style parameters for drawing text. params['text'] should contain the message to be drawn.
        background (BaseGeometry): Background layer
    """
    # Override default osm_credit dict with provided parameters
    params = override_params(
        dict(
            text="\n".join(
                [
                    "data © OpenStreetMap contributors",
                    "github.com/marceloprates/prettymaps",
                ]
            ),
            x=0,
            y=1,
            horizontalalignment="left",
            verticalalignment="top",
            bbox=dict(boxstyle="square", fc="#fff", ec="#000"),
            fontfamily="Ubuntu Mono",
        ),
        params,
    )
    x, y, text = (params.pop(k) for k in ["x", "y", "text"])

    # Get background bounds
    xmin, ymin, xmax, ymax = background.bounds

    x = np.interp([x], [0, 1], [xmin, xmax])[0]
    y = np.interp([y], [0, 1], [ymin, ymax])[0]

    plt.text(x, y, text, **params)


def PolygonPatch(shape: BaseGeometry, **kwargs) -> PathPatch:
    """_summary_.

    Args:
        shape (BaseGeometry): Shapely geometry
        kwargs: parameters for matplotlib's PathPatch constructor

    Returns:
        PathPatch: matplotlib PatchPatch created from input shapely geometry
    """
    # Init vertices and codes lists
    vertices, codes = [], []
    for poly in shape.geoms if isinstance(shape, Iterable) else [shape]:
        # Get polygon's exterior and interiors
        exterior = np.array(poly.exterior.xy)
        interiors = [np.array(interior.xy) for interior in poly.interiors]
        # Append to vertices and codes lists
        vertices += [exterior] + interiors
        codes += list(
            map(
                # Ring coding
                lambda p: [Path.MOVETO]
                + [Path.LINETO] * (p.shape[1] - 2)
                + [Path.CLOSEPOLY],
                [exterior] + interiors,
            )
        )
    # Generate PathPatch
    return PathPatch(
        Path(np.concatenate(vertices, 1).T, np.concatenate(codes)), **kwargs
    )


def graph_to_shapely(gdf: gp.GeoDataFrame, width: float = 1.0) -> BaseGeometry:
    """Given a GeoDataFrame containing a graph (street newtork),.

    convert them to shapely geometries by applying dilation given by 'width'.

    Args:
        gdf (gp.GeoDataFrame): input GeoDataFrame containing graph (street network) geometries
        width (float, optional): Line geometries will be dilated by this amount. Defaults to 1..

    Returns:
        BaseGeometry: Shapely
    """

    def highway_to_width(highway):
        if (type(highway) == str) and (highway in width):
            return width[highway]
        elif isinstance(highway, Iterable):
            for h in highway:
                if h in width:
                    return width[h]
            return np.nan
        else:
            return np.nan

    # Annotate GeoDataFrame with the width for each highway type
    gdf["width"] = gdf.highway.map(highway_to_width) if type(width) == dict else width

    # Remove rows with inexistent width
    gdf.drop(gdf[gdf.width.isna()].index, inplace=True)

    with warnings.catch_warnings():
        # Supress shapely.errors.ShapelyDeprecationWarning
        warnings.simplefilter("ignore", shapely.errors.ShapelyDeprecationWarning)
        if not all(gdf.width.isna()):
            # Dilate geometries based on their width
            gdf.geometry.update(
                gdf.apply(lambda row: row.geometry.buffer(row.width), axis=1)
            )

    return shapely.ops.unary_union(gdf.geometry)


def geometries_to_shapely(
    gdf: gp.GeoDataFrame,
    point_size: float | None = None,
    line_width: float | None = None,
) -> GeometryCollection:
    """Convert geometries in GeoDataFrame to shapely format.

    Args:
        gdf (gp.GeoDataFrame): Input GeoDataFrame
        point_size (Optional[float], optional): Point geometries (1D) will be dilated by this amount. Defaults to None.
        line_width (Optional[float], optional): Line geometries (2D) will be dilated by this amount. Defaults to None.

    Returns:
        GeometryCollection: Shapely geometries computed from GeoDataFrame geometries
    """
    geoms = gdf.geometry.tolist()
    collections = [x for x in geoms if type(x) == GeometryCollection]
    points = [x for x in geoms if type(x) == Point] + [
        y for x in collections for y in x.geoms if type(y) == Point
    ]
    lines = [x for x in geoms if type(x) in [LineString, MultiLineString]] + [
        y
        for x in collections
        for y in x.geoms
        if type(y) in [LineString, MultiLineString]
    ]
    polys = [x for x in geoms if type(x) in [Polygon, MultiPolygon]] + [
        y for x in collections for y in x.geoms if type(y) in [Polygon, MultiPolygon]
    ]

    # Convert points into circles with radius "point_size"
    if point_size:
        points = [x.buffer(point_size) for x in points] if point_size > 0 else []
    if line_width:
        lines = [x.buffer(line_width) for x in lines] if line_width > 0 else []

    return GeometryCollection(list(points) + list(lines) + list(polys))


def gdf_to_shapely(
    layer: str,
    gdf: gp.GeoDataFrame,
    width: dict | float | None = None,
    point_size: float | None = None,
    line_width: float | None = None,
    **kwargs,
) -> GeometryCollection:
    """Convert a dict of GeoDataFrames to a dict of shapely geometries.

    Args:
        layer (str): Layer name
        gdf (gp.GeoDataFrame): Input GeoDataFrame
        width (Optional[Union[dict, float]], optional): Street network width. Can be either a dictionary or a float. Defaults to None.
        point_size (Optional[float], optional): Point geometries (1D) will be dilated by this amount. Defaults to None.
        line_width (Optional[float], optional): Line geometries (2D) will be dilated by this amount. Defaults to None.

    Returns:
        GeometryCollection: Output GeoDataFrame
    """
    # Project gdf
    try:
        gdf = ox.project_gdf(gdf)
    except:
        pass

    if layer in ["streets", "railway", "waterway"]:
        geometries = graph_to_shapely(gdf, width)
    else:
        geometries = geometries_to_shapely(
            gdf, point_size=point_size, line_width=line_width
        )

    return geometries


def plot_gdf(
    layer: str,
    gdf: gp.GeoDataFrame,
    ax: matplotlib.axes.Axes,
    palette: list[str] | None = None,
    width: dict | float | None = None,
    union: bool = False,
    dilate_points: float | None = None,
    dilate_lines: float | None = None,
    hatch_c=None,
    **kwargs,
) -> None:
    """Plot a layer.

    Args:
        layer (str): layer name
        gdf (gp.GeoDataFrame): GeoDataFrame
        ax (matplotlib.axes.Axes): matplotlib axis object
        palette (Optional[List[str]], optional): Color palette. Defaults to None.
        width (Optional[Union[dict, float]], optional): Street widths. Either a dictionary or a float. Defaults to None.
        union (bool, optional): Whether to join geometries. Defaults to False.
        dilate_points (Optional[float], optional): Amount of dilation to be applied to point (1D) geometries. Defaults to None.
        dilate_lines (Optional[float], optional): Amount of dilation to be applied to line (2D) geometries. Defaults to None.

    Raises:
        Exception: _description_
    """
    # Convert GDF to shapely geometries
    geometries = gdf_to_shapely(
        layer, gdf, width, point_size=dilate_points, line_width=dilate_lines
    )
    geometries = geometries.geoms if isinstance(geometries, Iterable) else [geometries]

    # Unite geometries
    if union:
        geometries = shapely.ops.unary_union(geometries)

    if (palette is None) and ("fc" in kwargs) and (type(kwargs["fc"]) != str):
        palette = kwargs.pop("fc")

    # Plot shapes
    for shape in geometries:
        if type(shape) in [Polygon, MultiPolygon]:
            # Plot main shape (without silhouette)
            ax.add_patch(
                PolygonPatch(
                    shape,
                    lw=0,
                    ec=hatch_c if hatch_c else kwargs["ec"] if "ec" in kwargs else None,
                    fc=kwargs["fc"]
                    if "fc" in kwargs
                    else np.random.choice(palette)
                    if palette
                    else None,
                    **{k: v for k, v in kwargs.items() if k not in ["lw", "ec", "fc"]},
                ),
            )
            # Plot just silhouette
            ax.add_patch(
                PolygonPatch(
                    shape,
                    fill=False,
                    **{k: v for k, v in kwargs.items() if k not in ["hatch", "fill"]},
                )
            )
        elif type(shape) == LineString:
            ax.plot(
                *shape.xy,
                c=kwargs["ec"] if "ec" in kwargs else None,
                **{
                    k: v
                    for k, v in kwargs.items()
                    if k in ["lw", "lt", "dashes", "zorder"]
                },
            )
        elif type(shape) == MultiLineString:
            for c in shape.geoms:
                ax.plot(
                    *c.xy,
                    c=kwargs["ec"] if "ec" in kwargs else None,
                    **{
                        k: v
                        for k, v in kwargs.items()
                        if k in ["lw", "lt", "dashes", "zorder"]
                    },
                )


def plot_gdfs(gdfs: GeoDataFrames, plot_arg: PlotArg) -> None:
    """Plot gdfs."""
    layers, style, ax, figsize, credit, show, save_as = plot_arg

    # Create background GeoDataFrame
    background = create_background(gdfs, style)

    # Init matplotlib figure
    if ax is None:
        plt.figure(figsize=figsize)
        ax = plt.subplot(111, aspect="equal")

    # Draw layers in matplotlib mode
    for layer, gdf in gdfs.items():
        if layer in layers:
            plot_gdf(
                layer,
                gdf,
                ax,
                width=layers[layer].get("width"),
                **(style.get(layer, {})),
            )

    # Draw background
    if "background" in style:
        zorder = (
            style["background"].pop("zorder") if "zorder" in style["background"] else -1
        )
        ax.add_patch(
            PolygonPatch(
                background,
                **{k: v for k, v in style["background"].items() if k != "dilate"},
                zorder=zorder,
            )
        )

    # Draw credit message
    if credit:
        draw_text(credit, background)

    # Ajust figure and create PIL Image
    # Adjust axis
    ax.axis("off")
    ax.axis("equal")
    ax.autoscale()
    # Adjust padding
    plt.subplots_adjust(left=0, bottom=0, right=1, top=1, wspace=0, hspace=0)
    # Save result
    if save_as:
        plt.savefig(save_as)
    if not show:
        plt.close()
