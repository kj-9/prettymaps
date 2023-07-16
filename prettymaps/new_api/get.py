"""Experimental new api for prettymaps."""

from dataclasses import dataclass
from typing import Literal, NamedTuple

import geopandas as gp
import numpy as np
import osmnx as ox
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
from shapely.geometry.base import BaseGeometry
from shapely.ops import unary_union

from .types import GeoDataFrames, Layers


class Shape(NamedTuple):
    """Shape parameters for creating perimeter from given point."""

    type: Literal["circle", "square"]
    radius: float = 1000  # defaults to 1km.


@dataclass
class Perimeter:
    """Dataclass of perimeter geometry for input for `get_gdfs()`."""

    gdf: gp.GeoDataFrame
    geometry: BaseGeometry

    @staticmethod
    def _dilate(gdf, dilate):
        # Apply dilation
        gdf = ox.project_gdf(gdf)
        gdf.geometry = gdf.geometry.buffer(dilate)
        return gdf

    @staticmethod
    def _to_geometry(gdf: gp.GeoDataFrame) -> BaseGeometry:
        """gdf: gdf projected to crs 4326."""
        # Apply tolerance to the gdf
        # gdf_with_tolerance = ox.project_gdf(gdf).buffer(gdf_TOLERANCE).to_crs(4326)
        # gdf_with_tolerance = unary_union(gdf_with_tolerance.geometry).buffer(0)

        return unary_union(gdf.geometry)

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
        perimeter = ox.project_gdf(
            gp.GeoDataFrame(geometry=[Point((lng, lat))], crs="EPSG:4326")
        )

        if shape.type == "circle":  # Circular shape
            # use .buffer() to expand point into circle
            perimeter.geometry = perimeter.geometry.buffer(shape.radius)
        elif shape.type == "square":  # Square shape
            x, y = np.concatenate(perimeter.geometry[0].xy)
            r = shape.radius
            perimeter = gp.GeoDataFrame(
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
                crs=perimeter.crs,
            )
        else:
            raise NotImplementedError(f"{shape=} is not implemented.")

        if dilate:
            perimeter = cls._dilate(perimeter, dilate)

        perimeter = perimeter.to_crs(4326)

        return cls(gdf=perimeter, geometry=cls._to_geometry(perimeter))

    @classmethod
    def from_geocode_point(
        cls,
        query: str,
        shape: Shape,
        rotation: float = 0,
        dilate: float = 0,
    ) -> "Perimeter":
        """Create perimeter from a point geocoded by OSM."""
        point = ox.geocode(query)

        return cls.from_point(
            point,
            shape=shape,
            rotation=rotation,
            dilate=dilate,
        )

    @classmethod
    def from_geocode_gdf(cls, query: str, dilate: float = 0):
        """Create perimeter from a geodataframe represents the boundary of the object geocoded by OSM.

        This function may return perimeter which is not circle nor square,
        bound to some area (e.g. distirict), returned from OSM.
        """
        perimeter = ox.geocode_to_gdf(query)

        if dilate:
            perimeter = cls._dilate(perimeter, dilate)

        perimeter = perimeter.to_crs(4326)
        return cls(gdf=perimeter, geometry=cls._to_geometry(perimeter))

    @classmethod
    def from_gdf(cls, gdf: gp.GeoDataFrame) -> "Perimeter":
        """Create perimeter from user provided gdf.

        just use gdf asis for perimeter.
        """
        gdf = gdf.to_crs(4326)
        return cls(gdf=gdf, geometry=cls._to_geometry(gdf))


# Get a GeoDataFrame
def _get_gdf(
    layer,
    perimeter: Perimeter,
    tags=None,
    osmid=None,
    custom_filter=None,
    **ignore_kwargs,
) -> gp.GeoDataFrame:
    # Fetch from boundary's bounding box, to avoid missing some geometries
    bbox = box(*perimeter.geometry.bounds)

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
    gdf.geometry = gdf.geometry.intersection(perimeter.geometry)
    # gdf = gdf[~gdf.geometry.is_empty]
    gdf.drop(gdf[gdf.geometry.is_empty].index, inplace=True)

    return gdf


def get_gdfs(layers: Layers, perimeter: Perimeter) -> GeoDataFrames:
    """Fetch GeoDataFrames given query and a dictionary of layers."""
    # Get other layers as GeoDataFrames
    gdfs = {"perimeter": perimeter.gdf}
    gdfs.update(
        {
            layer: _get_gdf(layer, perimeter, **kwargs)
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
