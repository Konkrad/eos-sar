import abc
from dataclasses import dataclass
from typing import Optional

import numpy as np
import shapely
from numpy.typing import NDArray
from typing_extensions import override

from eos.dem import DEM, DEMSource
from eos.sar.model import SensorModel
from eos.sar.roi import Roi

Arrayf64 = NDArray[np.float64]


class RoiProvider(abc.ABC):
    """Abstract base class for strategies that determine a region of interest (Roi) in a sensor model's image."""

    @abc.abstractmethod
    def get_roi(
        self, proj_model: SensorModel, dem_source: DEMSource
    ) -> tuple[Roi, Arrayf64, Arrayf64]:
        """
        Determine the region of interest in the image of `proj_model`.

        Parameters
        ----------
        proj_model : eos.sar.model.SensorModel
            Sensor model used to project/localize points to determine the roi.
        dem_source : eos.dem.DEMSource
            Source used to fetch a DEM as needed to project points.

        Returns
        -------
        roi : eos.sar.roi.Roi
            The determined region of interest.
        rows : ndarray
            Row coordinates of the points (e.g. corners) used to derive the roi.
        cols : ndarray
            Column coordinates of the points (e.g. corners) used to derive the roi.
        """


def _geometry_to_geocoords(geometry: shapely.Geometry) -> list[tuple[float, float]]:
    if geometry.geom_type == "Polygon":
        geom = geometry.exterior
    else:
        # in practice only linear rings
        geom = geometry
    return geom.coords[:]


def _geometry_to_roi(
    geom_coords: list[tuple[float, float]],
    proj_model: SensorModel,
    dem: DEM,
    min_width: int = 1024,
    min_height: int = 512,
) -> tuple[Roi, Arrayf64, Arrayf64]:
    lons = np.asarray([c[0] for c in geom_coords])
    lats = np.asarray([c[1] for c in geom_coords])
    alts = np.nan_to_num(dem.elevation(lons, lats))
    rows, cols, _ = proj_model.projection(lons, lats, alts)
    roi = Roi.from_bounds_tuple(Roi.points_to_bbox(rows, cols))
    col, row, w, h = roi.to_roi()
    nh = max(h, min_height)
    nw = max(w, min_width)
    ncol = int(col + (w - nw) / 2)
    nrow = int(row + (h - nh) / 2)
    return Roi(ncol, nrow, nw, nh), rows, cols


@dataclass(frozen=True)
class GeometryRoiProvider(RoiProvider):
    """Determine the roi covering a given geometry, padded to a minimum width/height.

    Parameters
    ----------
    geometry : shapely.Geometry
        Geometry (in epsg:4326) whose projected extent defines the roi.
    dem_fetch_buffer : float, optional
        Buffer (in degrees) applied to `geometry`'s bounds when fetching the
        DEM used for the projection. The default is 0.1.
    min_width : int, optional
        Minimum width in pixels of the resulting roi. The default is 1024.
    min_height : int, optional
        Minimum height in pixels of the resulting roi. The default is 512.
    """

    geometry: shapely.Geometry
    dem_fetch_buffer: float = 0.1
    min_width: int = 1024
    min_height: int = 512

    @override
    def get_roi(
        self, proj_model: SensorModel, dem_source: DEMSource
    ) -> tuple[Roi, Arrayf64, Arrayf64]:
        """See RoiProvider.get_roi."""
        dem_bounds = self.geometry.buffer(self.dem_fetch_buffer).bounds
        dem = dem_source.fetch_dem(dem_bounds)

        # Get the geometry coords
        geo_coords = _geometry_to_geocoords(self.geometry)

        # Get the roi
        roi, rows, cols = _geometry_to_roi(
            geo_coords,
            proj_model,
            dem=dem,
            min_width=self.min_width,
            min_height=self.min_height,
        )
        return roi, rows, cols


@dataclass(frozen=True)
class CentroidRoiProvider(RoiProvider):
    """Determine a fixed-size roi centered on the projection of a given (lon, lat) point.

    Parameters
    ----------
    point : tuple of float
        (lon, lat) point, in epsg:4326, on which the roi is centered.
    w : int
        Width in pixels of the roi.
    h : int
        Height in pixels of the roi.
    dem_fetch_buffer_round_point : float, optional
        Buffer (in degrees) applied around `point` when fetching the DEM
        used for the projection. The default is 0.1.
    """

    point: tuple[float, float]
    w: int
    h: int
    dem_fetch_buffer_round_point: float = 0.1

    @override
    def get_roi(
        self, proj_model: SensorModel, dem_source: DEMSource
    ) -> tuple[Roi, Arrayf64, Arrayf64]:
        """See RoiProvider.get_roi."""
        dem_bounds = (
            shapely.Point(*self.point).buffer(self.dem_fetch_buffer_round_point).bounds
        )
        dem = dem_source.fetch_dem(dem_bounds)

        lon, lat = self.point
        alt = dem.elevation(lon, lat)
        assert isinstance(alt, (float, np.floating))
        row, col, _ = proj_model.projection(lon, lat, alt)
        orig = int(col - self.w / 2), int(row - self.h / 2)
        roi = Roi(*orig, self.w, self.h)
        rows, cols = roi.to_bounding_points()
        return roi, rows, cols


@dataclass(frozen=True)
class PrescribedRoiProvider(RoiProvider):
    """Return a roi given directly by the caller (or the whole image if none is given).

    Parameters
    ----------
    roi : eos.sar.roi.Roi, optional
        The roi to return. If None, the whole extent of the sensor model's
        image is used. The default is None.
    make_valid : bool, optional
        If True, `roi` is clipped to the bounds of the sensor model's image.
        The default is False.
    """

    roi: Optional[Roi] = None
    make_valid: bool = False

    @override
    def get_roi(
        self, proj_model: SensorModel, dem_source: DEMSource
    ) -> tuple[Roi, Arrayf64, Arrayf64]:
        """See RoiProvider.get_roi. `dem_source` is unused."""
        parent_shape = proj_model.h, proj_model.w
        if self.roi is None:
            roi = Roi(0, 0, parent_shape[1], parent_shape[0])
        else:
            if self.make_valid:
                roi = self.roi.make_valid(parent_shape)
            else:
                roi = self.roi

        rows, cols = roi.to_bounding_points()

        return roi, rows, cols
