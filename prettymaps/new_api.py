"""Experimental new api for prettymaps."""

import json
from abc import ABC, abstractmethod
from copy import deepcopy
from dataclasses import dataclass
from pathlib import Path
from typing import NewType, TypeAlias

import geopandas as gp
import osmnx as ox
import shapely.affinity
import shapely.ops
from shapely.geometry import (
    GeometryCollection,
)

from .draw import Plot, PolygonPatch, create_background, draw_text, plot_gdf, plt
from .fetch import get_gdf, get_perimeter


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
            layer: get_gdf(layer, perimeter, **kwargs)
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
    collection = shapely.affinity.translate(collection, x, y)
    collection = shapely.affinity.scale(collection, scale_x, scale_y)
    collection = shapely.affinity.rotate(collection, rotation)
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
