"""Experimental new api for prettymaps."""

import re
import json
import numpy as np
from abc import ABC, abstractmethod
from copy import deepcopy
from dataclasses import dataclass
from pathlib import Path
from typing import NewType, TypeAlias

import geopandas as gp
import osmnx as ox
from matplotlib import pyplot as plt
from shapely.affinity import (
    rotate,
    scale,
    translate,
)
from shapely.geometry import (
    GeometryCollection,
    box,
    Point,
    Polygon,
)
from shapely.ops import unary_union

from .draw import Plot, PolygonPatch, create_background, draw_text, plot_gdf


class Tuplable(ABC):
    """Tubplable: abstract class to have `to_tuple`.

    Abstract class serializable `self` to tuple.
    Intended to implement `to_tuple` method which return each attribute of `self` in tuple.
    Motivation is enabling type inference by write down each attribute when destruction like assignment of class attributes.
    Usually, serialization mechanism like dataclass, pyserde, pydantic returns `Any` type, not useful for type check.
    See also: https://github.com/kj-9/prettymaps/issues/14
    """

    @abstractmethod
    def to_tuple(self) -> tuple:
        """Serialize instance to tuple. useful for destruction assignment of attributes.

        Implement like: `return (self.a, self.b, ...)`, elements ordered by definition in class
        """
        pass


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


@dataclass
class GetArg(Tuplable):
    """Dataclass represents arguments for get_gdfs."""

    query: Query
    layers: Layers
    radius: float | None = None
    dilate: float | None = None  # maybe float
    rotation: float = 0  # rotation angle (in radians). Defaults to 0
    circle: bool = False

    def to_tuple(self):
        """Serialize instance to tuple. useful for destruction assignment of attributes."""
        return (
            self.query,
            self.layers,
            self.radius,
            self.dilate,
            self.rotation,
            self.circle,
        )


# Get a GeoDataFrame
def _get_gdf(
    layer,
    perimeter,
    perimeter_tolerance=0,
    tags=None,
    osmid=None,
    custom_filter=None,
    **kwargs,
) -> gp.GeoDataFrame:
    # Apply tolerance to the perimeter
    perimeter_with_tolerance = (
        ox.project_gdf(perimeter).buffer(perimeter_tolerance).to_crs(4326)
    )
    perimeter_with_tolerance = unary_union(perimeter_with_tolerance.geometry).buffer(0)

    # Fetch from perimeter's bounding box, to avoid missing some geometries
    bbox = box(*perimeter_with_tolerance.bounds)

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

    # Intersect with perimeter
    gdf.geometry = gdf.geometry.intersection(perimeter_with_tolerance)
    # gdf = gdf[~gdf.geometry.is_empty]
    gdf.drop(gdf[gdf.geometry.is_empty].index, inplace=True)

    return gdf

# Parse query (by coordinates, OSMId or name)
def parse_query(query):
    if isinstance(query, GeoDataFrame):
        return "polygon"
    elif isinstance(query, tuple):
        return "coordinates"
    elif re.match("""[A-Z][0-9]+""", query):
        return "osmid"
    else:
        return "address"

# Get circular or square boundary around point
def get_boundary(query, radius, circle=False, rotation=0):

    # Get point from query
    point = query if parse_query(query) == "coordinates" else ox.geocode(query)
    # Create GeoDataFrame from point
    boundary = ox.project_gdf(
        GeoDataFrame(geometry=[Point(point[::-1])], crs="EPSG:4326")
    )

    if circle:  # Circular shape
        # use .buffer() to expand point into circle
        boundary.geometry = boundary.geometry.buffer(radius)
    else:  # Square shape
        x, y = np.concatenate(boundary.geometry[0].xy)
        r = radius
        boundary = GeoDataFrame(
            geometry=[
                rotate(
                    Polygon(
                        [(x - r, y - r), (x + r, y - r),
                         (x + r, y + r), (x - r, y + r)]
                    ),
                    rotation,
                )
            ],
            crs=boundary.crs,
        )

    # Unproject
    boundary = boundary.to_crs(4326)

    return boundary

# Get perimeter from query
def get_perimeter(
    query, radius=None, by_osmid=False, circle=False, dilate=None, rotation=0, **kwargs
):

    if radius:
        # Perimeter is a circular or square shape
        perimeter = get_boundary(
            query, radius, circle=circle, rotation=rotation)
    else:
        # Perimeter is a OSM or user-provided polygon
        if parse_query(query) == "polygon":
            # Perimeter was already provided
            perimeter = query
        else:
            # Fetch perimeter from OSM
            perimeter = ox.geocode_to_gdf(
                query,
                by_osmid=by_osmid,
                **kwargs,
            )

    # Apply dilation
    perimeter = ox.project_gdf(perimeter)
    if dilate is not None:
        perimeter.geometry = perimeter.geometry.buffer(dilate)
    perimeter = perimeter.to_crs(4326)

    return perimeter


def get_gdfs(get_arg: GetArg) -> GeoDataFrames:
    """Fetch GeoDataFrames given query and a dictionary of layers."""
    query, layers, radius, dilate, rotation, circle = get_arg.to_tuple()

    # override layers
    override_args = ["circle", "dilate"]
    for layer in layers:
        for arg in override_args:
            if arg not in layers[layer]:
                layers[layer][arg] = locals()[arg]

                

    perimeter_kwargs = {}
    if "perimeter" in layers:
        perimeter_kwargs = deepcopy(layers["perimeter"])
        perimeter_kwargs.pop("dilate")

    # Get perimeter
    perimeter = get_perimeter(
        query, radius=radius, rotation=-rotation, dilate=dilate, **perimeter_kwargs
    )

    # Get other layers as GeoDataFrames
    gdfs = {"perimeter": perimeter}
    gdfs.update(
        {
            layer: _get_gdf(layer, perimeter, **kwargs)
            for layer, kwargs in layers.items()
            if layer != "perimeter"
        }
    )

    return GeoDataFrames(gdfs)


@dataclass
class TransformArg(Tuplable):
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

    def to_tuple(self):
        """Serialize instance to tuple. useful for destruction assignment of attributes."""
        return (self.x, self.y, self.scale_x, self.scale_y, self.rotation)


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
    x, y, scale_x, scale_y, rotation = transform_arg.to_tuple()

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


@dataclass
class PlotArg(Tuplable):
    """Dataclass represents arguments for plot_gdfs."""

    layers: Layers
    style: Style
    ax: plt.Axes | None = None
    figsize: tuple[float, float] = (12, 12)
    credit: None = None
    show: bool = True
    save_as: str | None = None

    def to_tuple(self):
        """Serialize instance to tuple. useful for destruction assignment of attributes."""
        return (
            self.layers,
            self.style,
            self.ax,
            self.figsize,
            self.credit,
            self.show,
            self.save_as,
        )


def plot_gdfs(gdfs: GeoDataFrames, plot_arg: PlotArg) -> Plot:
    """Plot gdfs."""
    # returnは色々ありうる。Plotデータクラス、PILオブジェクト。副作用として画像保存させるか、それは別メソッドにするか。

    layers, style, ax, figsize, credit, show, save_as = plot_arg.to_tuple()

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
