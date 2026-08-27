"""Coordinates conversion between image (row/column denoted as `row` \
    and `col`) and sar (azimuth time and range denoted as `azt` and `rng`)."""

import abc
from dataclasses import dataclass
from typing import Optional

import numpy as np
from numpy.typing import ArrayLike, NDArray
from typing_extensions import override

from eos.sar import const
from eos.sar.srgr import SRGRConverter

Arrayf64 = NDArray[np.float64]


class TwoDCoordinate(abc.ABC):
    """Abstract base class converting between image (row, col) and SAR (azimuth time, range) coordinates."""

    @abc.abstractmethod
    def to_azt_rng(self, row: ArrayLike, col: ArrayLike) -> tuple[Arrayf64, Arrayf64]:
        """Convert image (row, col) coordinates to (azimuth time, range).

        Parameters
        ----------
        row : ndarray or scalar
            Row coordinate(s) in the image.
        col : ndarray or scalar
            Column coordinate(s) in the image.

        Returns
        -------
        azt, rng : ndarray or scalar
            Azimuth time and (slant) range.
        """

    @abc.abstractmethod
    def to_row_col(self, azt: ArrayLike, rng: ArrayLike) -> tuple[Arrayf64, Arrayf64]:
        """Convert (azimuth time, range) coordinates to image (row, col).

        Parameters
        ----------
        azt : ndarray or scalar
            Azimuth time.
        rng : ndarray or scalar
            (Slant) range.

        Returns
        -------
        row, col : ndarray or scalar
            Row and column coordinate(s) in the image.
        """


@dataclass(frozen=True)
class SLCCoordinate(TwoDCoordinate):
    """Linear (row, col) <-> (azimuth time, slant range) conversion for a Single Look Complex (SLC) product.

    Parameters
    ----------
    first_row_time : float
        Azimuth time of the first row (in s).
    first_col_time : float
        Two-way travel time of the first column (in s).
    azimuth_frequency : float
        Pulse Repetition Frequency, i.e. sampling frequency along rows (in Hz).
    range_frequency : float
        Range sampling frequency along columns (in Hz).
    """

    first_row_time: float
    first_col_time: float
    azimuth_frequency: float
    range_frequency: float

    def to_azt(self, row: ArrayLike) -> Arrayf64:
        """Convert row coordinate(s) to azimuth time."""
        row = np.asarray(row)
        azt = row / self.azimuth_frequency + self.first_row_time
        return azt

    def to_rng(self, col: ArrayLike, azt: Optional[ArrayLike] = None) -> Arrayf64:
        """Convert column coordinate(s) to slant range. `azt` is unused (kept for interface compatibility)."""
        col = np.asarray(col)
        rng = (
            (col / self.range_frequency + self.first_col_time)
            * const.LIGHT_SPEED_M_PER_SEC
            / 2
        )
        return rng

    @override
    def to_azt_rng(self, row: ArrayLike, col: ArrayLike) -> tuple[Arrayf64, Arrayf64]:
        """Convert (row, col) image coordinates to (azimuth time, slant range)."""
        azt = self.to_azt(row)
        rng = self.to_rng(col)
        return azt, rng

    def to_row(self, azt: ArrayLike) -> Arrayf64:
        """Convert azimuth time to row coordinate(s)."""
        azt = np.asarray(azt)
        row = (azt - self.first_row_time) * self.azimuth_frequency
        return row

    def to_col(self, rng: ArrayLike, azt: Optional[ArrayLike] = None) -> Arrayf64:
        """Convert slant range to column coordinate(s). `azt` is unused (kept for interface compatibility)."""
        rng = np.asarray(rng)
        col = (
            2 * rng / const.LIGHT_SPEED_M_PER_SEC - self.first_col_time
        ) * self.range_frequency
        return col

    @override
    def to_row_col(self, azt: ArrayLike, rng: ArrayLike) -> tuple[Arrayf64, Arrayf64]:
        """Convert (azimuth time, slant range) to (row, col) image coordinates."""
        row = self.to_row(azt)
        col = self.to_col(rng)
        return row, col


@dataclass(frozen=True)
class GRDCoordinate(TwoDCoordinate):
    """Linear (row, col) <-> (azimuth time, slant range) conversion for a Ground Range Detected (GRD) product.

    Ground range is linear in the column coordinate; conversion to/from slant
    range is delegated to `srgr`.

    Parameters
    ----------
    first_row_time : float
        Azimuth time of the first row (in s).
    azimuth_time_interval : float
        Time between consecutive rows (in s).
    range_pixel_spacing : float
        Ground range spacing between consecutive columns (in m).
    srgr : SRGRConverter
        Slant range / ground range converter.
    """

    # NOTE: the function signature is slightly different than in SLCCoordinate
    # because the azt is required for the to_col, not optional

    first_row_time: float
    azimuth_time_interval: float
    range_pixel_spacing: float
    srgr: SRGRConverter

    def to_azt(self, row: ArrayLike) -> Arrayf64:
        """Convert row coordinate(s) to azimuth time."""
        row = np.asarray(row)
        azt = row * self.azimuth_time_interval + self.first_row_time
        return azt

    def to_rng(self, col: ArrayLike, azt: ArrayLike) -> Arrayf64:
        """Convert column coordinate(s) (via ground range) to slant range at the given azimuth time."""
        col = np.asarray(col)
        gr = col * self.range_pixel_spacing
        rng = self.srgr.gr_to_rng(gr, azt)
        return rng

    @override
    def to_azt_rng(self, row: ArrayLike, col: ArrayLike) -> tuple[Arrayf64, Arrayf64]:
        """Convert (row, col) image coordinates to (azimuth time, slant range)."""
        azt = self.to_azt(row)
        rng = self.to_rng(col, azt)
        return azt, rng

    def to_row(self, azt: ArrayLike) -> Arrayf64:
        """Convert azimuth time to row coordinate(s)."""
        azt = np.asarray(azt)
        row = (azt - self.first_row_time) / self.azimuth_time_interval
        return row

    def to_col(self, rng: ArrayLike, azt: ArrayLike) -> Arrayf64:
        """Convert slant range (via ground range) to column coordinate(s) at the given azimuth time."""
        gr = self.srgr.rng_to_gr(rng, azt)
        col = gr / self.range_pixel_spacing
        return col

    @override
    def to_row_col(self, azt: ArrayLike, rng: ArrayLike) -> tuple[Arrayf64, Arrayf64]:
        """Convert (azimuth time, slant range) to (row, col) image coordinates."""
        row = self.to_row(azt)
        col = self.to_col(rng, azt)
        return row, col
