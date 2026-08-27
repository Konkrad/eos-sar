from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Union

import numpy as np
import pyproj
from numpy.typing import ArrayLike
from typing_extensions import override

from eos.products.terrasarx.metadata import TSXMetadata, parse_tsx_metadata
from eos.sar import coordinates
from eos.sar.model import Arrayf64, CoordArrayLike, SensorModel
from eos.sar.model_helper import GenericSensorModelHelper
from eos.sar.orbit import Orbit
from eos.sar.projection_correction import Corrector


@dataclass(frozen=True)
class TSXModel(SensorModel):
    """SensorModel implementation for TerraSAR-X (TSX-1/TDX-1/PAZ-1) products.

    Wraps a :class:`~eos.sar.model_helper.GenericSensorModelHelper` built from
    the product's orbit and timing to implement the projection/localization
    interface defined by :class:`~eos.sar.model.SensorModel`.
    """

    generic_model: GenericSensorModelHelper
    # for SensorModel:
    w: int
    h: int
    orbit: Orbit
    wavelength: float

    @staticmethod
    def from_metadata(
        meta: TSXMetadata, orbit: Orbit, corrector: Corrector = Corrector()
    ) -> TSXModel:
        """Build a `TSXModel` from parsed product metadata and an orbit.

        Parameters
        ----------
        meta : TSXMetadata
            Metadata parsed with `parse_tsx_metadata`.
        orbit : Orbit
            Orbit to use for the sensor model.
        corrector : Corrector, optional
            Coordinate corrector applied by the underlying generic sensor
            model. Defaults to an identity `Corrector`.

        Returns
        -------
        TSXModel
            Sensor model ready for projection/localization.
        """
        coordinate = coordinates.SLCCoordinate(
            first_row_time=meta.image_start,
            first_col_time=meta.slant_range_time,
            azimuth_frequency=meta.azimuth_frequency,
            range_frequency=meta.range_frequency,
        )

        tolerance = 0.001
        projection_tolerance = float(tolerance / np.linalg.norm(orbit.sv[0].velocity))
        approx_centroid_lon, approx_centroid_lat = np.mean(meta.approx_geom, axis=0)

        generic_model = GenericSensorModelHelper(
            orbit=orbit,
            coordinate=coordinate,
            azt_init=meta.image_start,
            projection_tolerance=projection_tolerance,
            localization_tolerance=tolerance,
            max_iterations=20,
            coord_corrector=corrector,
            approx_centroid_lon=approx_centroid_lon,
            approx_centroid_lat=approx_centroid_lat,
        )

        return TSXModel(
            generic_model=generic_model,
            w=meta.width,
            h=meta.height,
            orbit=orbit,
            wavelength=meta.wavelength,
        )

    @override
    def to_azt_rng(self, row: ArrayLike, col: ArrayLike) -> tuple[Arrayf64, Arrayf64]:
        """Convert row/col image coordinates to azimuth time/range.

        Delegates to the underlying `GenericSensorModelHelper`.
        """
        return self.generic_model.to_azt_rng(row, col)

    @override
    def to_row_col(self, azt: ArrayLike, rng: ArrayLike) -> tuple[Arrayf64, Arrayf64]:
        """Convert azimuth time/range to row/col image coordinates.

        Delegates to the underlying `GenericSensorModelHelper`.
        """
        return self.generic_model.to_row_col(azt, rng)

    @override
    def projection(
        self,
        x: CoordArrayLike,
        y: CoordArrayLike,
        alt: CoordArrayLike,
        crs: Union[str, pyproj.CRS] = "epsg:4326",
        vert_crs: Optional[Union[str, pyproj.CRS]] = None,
        azt_init: Optional[ArrayLike] = None,
        as_azt_rng: bool = False,
    ) -> tuple[CoordArrayLike, CoordArrayLike, CoordArrayLike]:
        """Project a 3D point into image coordinates.

        See `eos.sar.model.SensorModel.projection`. Delegates to the
        underlying `GenericSensorModelHelper`.
        """
        return self.generic_model.projection(
            x, y, alt, crs, vert_crs, azt_init, as_azt_rng
        )

    @override
    def localization(
        self,
        row: CoordArrayLike,
        col: CoordArrayLike,
        alt: CoordArrayLike,
        crs: Union[str, pyproj.CRS] = "epsg:4326",
        vert_crs: Optional[Union[str, pyproj.CRS]] = None,
        x_init: Optional[ArrayLike] = None,
        y_init: Optional[ArrayLike] = None,
        z_init: Optional[ArrayLike] = None,
    ) -> tuple[CoordArrayLike, CoordArrayLike, CoordArrayLike]:
        """Localize a point in the image at a certain altitude.

        See `eos.sar.model.SensorModel.localization`. Delegates to the
        underlying `GenericSensorModelHelper`.
        """
        return self.generic_model.localization(
            row, col, alt, crs, vert_crs, x_init, y_init, z_init
        )


def main(xml_annotation_file_path):
    """
    Example usage
    """
    metadata = parse_tsx_metadata(xml_annotation_file_path)
    orbit = Orbit(sv=metadata.state_vectors, degree=11)
    model = TSXModel.from_metadata(metadata, orbit)
    return model


if __name__ == "__main__":
    import fire  # type: ignore

    fire.Fire(main)
