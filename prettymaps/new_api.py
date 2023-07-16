"""Experimental new api for prettymaps."""

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Literal, NamedTuple, NewType, TypeAlias

import geopandas as gp
import numpy as np
import osmnx as ox
from matplotlib import pyplot as plt
from shapely.affinity import (
    rotate,
    scale,
    translate,
)
from shapely.geometry import (
    GeometryCollection,
    Point,
    Polygon,
    box,
)
from shapely.ops import unary_union

from .draw import Plot, PolygonPatch, create_background, draw_text, plot_gdf

# types
Query: TypeAlias = str | tuple[float, float] | gp.GeoDataFrame
"""Your query.
Example:
    - "Porto Alegre"
    - (-30.0324999, -51.2303767) (lat/long coordinates)
    - You can also provide a custom GeoDataFrame boundary as input
"""

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


class Shape(NamedTuple):
    """Shape parameters for creating perimeter from given point."""

    type: Literal["circle", "square"]
    radius: float = 1000  # defaults to 1km.


@dataclass
class Perimeter:
    """Dataclass of perimeter geodataframe for input for `get_gdfs()`."""

    boundary: gp.GeoDataFrame

    @staticmethod
    def _dilate(boundary, dilate):
        # Apply dilation
        boundary = ox.project_gdf(boundary)
        boundary.geometry = boundary.geometry.buffer(dilate)
        return boundary

    @classmethod
    def from_point(
        cls,
        point: tuple[float, float],
        shape: Shape,
        rotation: float = 0,
        dilate: float = 0,
    ) -> "Perimeter":
        """Create perimeter from point of lat,lng."""
        lat, lng = point

        # Create GeoDataFrame from point
        boundary = ox.project_gdf(
            gp.GeoDataFrame(geometry=[Point((lng, lat))], crs="EPSG:4326")
        )

        if shape.type == "circle":  # Circular shape
            # use .buffer() to expand point into circle
            boundary.geometry = boundary.geometry.buffer(shape.radius)
        elif shape.type == "square":  # Square shape
            x, y = np.concatenate(boundary.geometry[0].xy)
            r = shape.radius
            boundary = gp.GeoDataFrame(
                geometry=[
                    rotate(
                        Polygon(
                            [
                                (x - r, y - r),
                                (x + r, y - r),
                                (x + r, y + r),
                                (x - r, y + r),
                            ]
                        ),
                        rotation,
                    )
                ],
                crs=boundary.crs,
            )
        else:
            raise NotImplementedError(f"{shape=} is not implemented.")

        if dilate:
            boundary = cls._dilate(boundary, dilate)
        boundary = boundary.to_crs(4326)

        return cls(boundary=boundary)

    @classmethod
    def from_geocode_point(
        cls,
        query: str,
        shape: Shape,
        rotation: float = 0,
        dilate: float = 0,
    ) -> "Perimeter":
        """From point geocoded by query string to perimiter."""
        point = ox.geocode(query)

        return cls.from_point(
            point,
            shape=shape,
            rotation=rotation,
            dilate=dilate,
        )

    @classmethod
    def from_geocode_gdf(cls, query: str, dilate: float = 0):
        """Create perimeter from boundary returned from OSM."""
        boundary = ox.geocode_to_gdf(query)

        if dilate:
            boundary = cls._dilate(boundary, dilate)
        boundary = boundary.to_crs(4326)

        return cls(boundary=boundary)

    @classmethod
    def from_gdf(cls, gdf: gp.GeoDataFrame) -> "Perimeter":
        """Create perimeter from user provided gdf.

        just use gdf asis for perimeter.
        """
        return cls(boundary=gdf)


# Get a GeoDataFrame
def _get_gdf(
    layer,
    boundary,
    tags=None,
    osmid=None,
    custom_filter=None,
    **ignore_kwargs,
) -> gp.GeoDataFrame:
    # Apply tolerance to the boundary
    # boundary_with_tolerance = ox.project_gdf(boundary).buffer(boundary_TOLERANCE).to_crs(4326)
    # boundary_with_tolerance = unary_union(boundary_with_tolerance.geometry).buffer(0)

    boundary = unary_union(boundary.to_crs(4326).geometry)

    # Fetch from boundary's bounding box, to avoid missing some geometries
    bbox = box(*boundary.bounds)

    if layer in ["streets", "railway", "waterway"]:
        graph = ox.graph_from_polygon(
            bbox,
            custom_filter=custom_filter,
            truncate_by_edge=True,
        )
        gdf = ox.graph_to_gdfs(graph, nodes=False)
    elif layer == "coastline":
        # Fetch geometries from OSM
        gdf = ox.geometries_from_polygon(
            bbox, tags={tags: True} if type(tags) == str else tags
        )
    elif osmid is None:
        # Fetch geometries from OSM
        gdf = ox.geometries_from_polygon(
            bbox, tags={tags: True} if type(tags) == str else tags
        )
    else:
        gdf = ox.geocode_to_gdf(osmid, by_osmid=True)

    # Intersect with boundary
    gdf.geometry = gdf.geometry.intersection(boundary)
    # gdf = gdf[~gdf.geometry.is_empty]
    gdf.drop(gdf[gdf.geometry.is_empty].index, inplace=True)

    return gdf


def get_gdfs(layers: Layers, perimeter: Perimeter) -> GeoDataFrames:
    """Fetch GeoDataFrames given query and a dictionary of layers."""
    # Get other layers as GeoDataFrames
    gdfs = {"perimeter": perimeter.boundary}
    gdfs.update(
        {
            layer: _get_gdf(layer, perimeter.boundary, **kwargs)
            for layer, kwargs in layers.items()
            if layer != "perimeter"
        }
    )

    return GeoDataFrames(gdfs)


class TransformArg(NamedTuple):
    """Dataclass represents arguments for get_gdfs.

    x: x-axis translation. Defaults to 0.
    y: y-axis translation. Defaults to 0.
    scale_x: x-axis scale. Defaults to 1.
    scale_y: y-axis scale. Defaults to 1.
    rotation: rotation angle (in radians). Defaults to 0.
    """

    x: float = 0
    y: float = 0
    scale_x: float = 1
    scale_y: float = 1
    rotation: float = 0


def transform_gdfs(gdfs: GeoDataFrames, transform_arg: TransformArg) -> GeoDataFrames:
    """Apply geometric transformations to dictionary of GeoDataFrames."""
    # if just default arg, do nothing
    if transform_arg == TransformArg():
        return gdfs

    # Project geometries
    gdfs = GeoDataFrames(
        {
            name: ox.project_gdf(gdf) if len(gdf) > 0 else gdf
            for name, gdf in gdfs.items()
        }
    )
    # Create geometry collection from gdfs' geometries
    collection = GeometryCollection(
        [GeometryCollection(list(gdf.geometry)) for gdf in gdfs.values()]
    )

    # desturct arg
    x, y, scale_x, scale_y, rotation = transform_arg

    # Translation, scale & rotation
    collection = translate(collection, x, y)
    collection = scale(collection, scale_x, scale_y)
    collection = rotate(collection, rotation)
    # Update geometries
    for i, layer in enumerate(gdfs):
        gdfs[layer].geometry = list(collection.geoms[i].geoms)
        # Reproject
        if len(gdfs[layer]) > 0:
            gdfs[layer] = ox.project_gdf(gdfs[layer], to_crs="EPSG:4326")

    return gdfs


class PlotArg(NamedTuple):
    """Dataclass represents arguments for plot_gdfs."""

    layers: Layers
    style: Style
    ax: plt.Axes | None = None
    figsize: tuple[float, float] = (12, 12)
    credit: None = None
    show: bool = True
    save_as: str | None = None


def plot_gdfs(gdfs: GeoDataFrames, plot_arg: PlotArg) -> Plot:
    """Plot gdfs."""
    # returnは色々ありうる。Plotデータクラス、PILオブジェクト。副作用として画像保存させるか、それは別メソッドにするか。

    layers, style, ax, figsize, credit, show, save_as = plot_arg

    # 7. Create background GeoDataFrame and get (x,y) bounds
    background, xmin, ymin, xmax, ymax, dx, dy = create_background(gdfs, style)

    # 2. Init matplotlib figure
    if ax is None:
        fig = plt.figure(figsize=figsize)
        ax = plt.subplot(111, aspect="equal")

    # 8.2. Draw layers in matplotlib mode
    for layer in gdfs:
        if layer in layers:
            plot_gdf(
                layer,
                gdfs[layer],
                ax,
                width=layers[layer]["width"] if "width" in layers[layer] else None,
                **(style[layer] if layer in style else {}),
            )

    # 9. Draw background
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

    # 10. Draw credit message
    if credit is not None:
        draw_text(credit, background)

    # 11. Ajust figure and create PIL Image
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

    return Plot(gdfs, fig, ax, background)
