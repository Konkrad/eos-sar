from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np
from numpy.typing import ArrayLike, NDArray

from eos.products.capella.metadata import (
    CapellaSLCMetadata,
)
from eos.products.capella.polynomial import CapellaPolynomial2D


@dataclass(frozen=True)
class CapellaDoppler:
    """Doppler centroid model for a Capella SLC product.

    Wraps the 2D Doppler centroid polynomial (as a function of azimuth time
    since the first line and slant range) together with the timing/range
    parameters needed to convert row/col image coordinates to azimuth
    time/range.
    """

    poly_2d: CapellaPolynomial2D
    starting_range: float
    range_pixel_size: float
    delta_line_time: float

    @classmethod
    def from_metadata(cls, metadata: CapellaSLCMetadata) -> CapellaDoppler:
        """Build a `CapellaDoppler` from a `CapellaSLCMetadata`.

        Parameters
        ----------
        metadata : CapellaSLCMetadata
            Parsed Capella SLC metadata.

        Returns
        -------
        CapellaDoppler
        """
        return CapellaDoppler(
            CapellaPolynomial2D.from_poly_meta(metadata.fdop_cen_poly2d_meta),
            metadata.starting_range,
            metadata.range_pixel_size,
            metadata.delta_line_time,
        )

    def get_from_delta_azt_rng(
        self,
        delta_azt_from_im_start: NDArray[np.float64],
        rng: NDArray[np.float64],
        *,
        grid_eval: bool = False,
    ) -> NDArray[np.float64]:
        """Evaluate the Doppler centroid polynomial at azimuth time/range points.

        Parameters
        ----------
        delta_azt_from_im_start : NDArray[np.float64]
            Azimuth time(s) since the first line, in seconds.
        rng : NDArray[np.float64]
            Slant range(s), in meters.
        grid_eval : bool, optional
            If True, evaluate on the outer-product grid of the two inputs
            instead of pointwise. Defaults to False.

        Returns
        -------
        NDArray[np.float64]
            Doppler centroid frequency (Hz).
        """
        if grid_eval:
            fdop_cen = self.poly_2d.evaluate_grid(delta_azt_from_im_start, rng)
        else:
            assert delta_azt_from_im_start.shape == rng.shape, (
                f"{delta_azt_from_im_start.shape}!={rng.shape}"
            )
            fdop_cen = self.poly_2d.evaluate(delta_azt_from_im_start, rng)

        return fdop_cen

    def to_delta_azt(self, row: ArrayLike) -> NDArray[np.float64]:
        """Convert row(s) in the Doppler frame to azimuth time since the first line."""
        return np.asarray(row) * self.delta_line_time

    def to_rng(self, col: ArrayLike) -> NDArray[np.float64]:
        """Convert col(s) in the Doppler frame to slant range."""
        return self.starting_range + self.range_pixel_size * np.asarray(col)

    def get_from_row_col(
        self,
        row_roi: ArrayLike,
        col_roi: ArrayLike,
        roi_origin_in_doppler_frame: tuple[int, int] = (0, 0),
        matrix_to_doppler_frame_roi: Optional[NDArray[np.float64]] = None,
        *,
        grid_eval: bool = False,
    ) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
        """Evaluate the Doppler centroid at row/col coordinates of a region of interest.

        `row_roi`/`col_roi` are expressed in a (possibly cropped/resampled)
        ROI frame; they are first mapped back to the Doppler polynomial's
        native frame (translated by `roi_origin_in_doppler_frame`, and, if
        given, transformed by `matrix_to_doppler_frame_roi`) before
        evaluating the polynomial.

        Parameters
        ----------
        row_roi : ArrayLike
            Row coordinate(s) in the ROI.
        col_roi : ArrayLike
            Column coordinate(s) in the ROI, same shape as `row_roi` unless
            `grid_eval` is True.
        roi_origin_in_doppler_frame : tuple[int, int], optional
            (col, row) origin of the ROI in the Doppler polynomial's native
            frame. Defaults to (0, 0).
        matrix_to_doppler_frame_roi : NDArray[np.float64], optional
            If given, a 3x3 homogeneous transform matrix mapping
            (row, col, 1) points of the ROI to the Doppler polynomial's
            native frame, applied before the origin translation. If None,
            the ROI is assumed to already be in that frame.
        grid_eval : bool, optional
            If True, evaluate on the outer-product grid of `row_roi` and
            `col_roi` instead of pointwise. Defaults to False.

        Returns
        -------
        fdop_cen : NDArray[np.float64]
            Doppler centroid frequency (Hz) at each point.
        delta_azt : NDArray[np.float64]
            Azimuth time since the first line (seconds) at each point.
        """
        row_roi = np.asarray(row_roi)
        col_roi = np.asarray(col_roi)

        if not grid_eval:
            assert row_roi.shape == col_roi.shape, f"{row_roi.shape}!={col_roi.shape}"

        col_orig, row_orig = roi_origin_in_doppler_frame

        if matrix_to_doppler_frame_roi is None:
            # we are already in the Doppler frame
            # only need to apply the origin translation
            # and deal with grid eval
            delta_azt = self.to_delta_azt(row_roi + row_orig)

            fdop_cen = self.get_from_delta_azt_rng(
                delta_azt, self.to_rng(col_roi + col_orig), grid_eval=grid_eval
            )

            return fdop_cen, delta_azt

        else:
            # Since we need to apply a matrix to get to the Doppler frame
            # If grid eval is true, we start by a meshgrid
            # (a regular grid will become irregular after applying the matrix,
            # so grid eval will not give any computational advantage as we need
            # to polyval2d and not polygrid2d anyway
            if grid_eval:
                assert len(col_roi.shape) == len(row_roi.shape) == 1, (
                    "arrays should be 1D, got {len(col_roi.shape)}D array and {len(row_roi.shape)}D array"
                )
                cols_roi, rows_roi = np.meshgrid(col_roi, row_roi)
            else:
                cols_roi = col_roi.copy()
                rows_roi = row_roi.copy()

            # homogeneous coordinates
            # will have rows_roi.shape + (3, 1)
            # so that we can do a stacked matmul
            points = np.stack(
                [
                    rows_roi,
                    cols_roi,
                    np.ones_like(rows_roi),
                ],
                axis=-1,
            )[..., None]

            # grid at src
            points = np.matmul(matrix_to_doppler_frame_roi, points)[..., :2, 0]
            # points has shape : rows_roi.shape + (, 2)

            rows_roi = points[..., 0]
            cols_roi = points[..., 1]

            del points

            delta_azt = self.to_delta_azt(rows_roi + row_orig)

            # apply the origin translation
            fdop_cen = self.get_from_delta_azt_rng(
                delta_azt, self.to_rng(cols_roi + col_orig), grid_eval=False
            )

            return fdop_cen, delta_azt
