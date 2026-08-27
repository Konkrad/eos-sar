from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray
from typing_extensions import override

from eos.products.capella.doppler_info import CapellaDoppler
from eos.sar.regist import (
    SarResample,
)
from eos.sar.roi import Roi


@dataclass(frozen=True)
class CapellaResample(SarResample):
    """`SarResample` implementation for Capella SLC products.

    Deramps/reramps using the Doppler centroid model of the source image
    (`src_doppler`) evaluated on the source and destination grids.
    """

    matrix: NDArray[np.float64]
    """
    matrix that goes from the destination frame who's origin is (0, 0) to the source frame defined by src_roi_in_img.
    """
    src_roi_in_img: Roi
    """
    Region from source image to read, before deramping and resampling
    """
    dst_shape: tuple[int, int]
    """
    The shape of the destination image
    """
    src_doppler: CapellaDoppler
    """
    object to compute the doppler frequency in the source frame 
    """

    def deramping_phase(self) -> NDArray[np.float64]:
        """Compute the deramping phase (radians) over `src_roi_in_img`, on the source grid."""
        fdop, delta_azt = self.src_doppler.get_from_row_col(
            np.arange(self.src_roi_in_img.h),
            np.arange(self.src_roi_in_img.w),
            self.src_roi_in_img.get_origin(),
            grid_eval=True,
        )

        deramping_phase = -2 * np.pi * fdop * delta_azt[:, None]

        return deramping_phase

    @override
    def deramp(self, src_array: NDArray[np.complex64]) -> NDArray[np.complex64]:
        """Deramp `src_array` by removing the Doppler-induced phase (see `deramping_phase`)."""
        return src_array * np.exp(1j * self.deramping_phase()).astype(np.complex64)

    def reramping_phase(self) -> NDArray[np.float64]:
        """Compute the reramping phase (radians) over the destination grid.

        Evaluated by mapping the destination grid to the source frame
        through `matrix`.
        """
        h, w = self.dst_shape
        fdop, delta_azt = self.src_doppler.get_from_row_col(
            np.arange(h),
            np.arange(w),
            self.src_roi_in_img.get_origin(),
            self.matrix,
            grid_eval=True,
        )

        reramping_phase = 2 * np.pi * fdop * delta_azt

        return reramping_phase

    @override
    def reramp(self, dst_array: NDArray[np.complex64]) -> NDArray[np.complex64]:
        """Reramp `dst_array` by restoring the Doppler-induced phase (see `reramping_phase`)."""
        return dst_array * np.exp(1j * self.reramping_phase()).astype(np.complex64)
