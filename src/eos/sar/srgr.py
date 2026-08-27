import abc

import numpy as np
from numpy.typing import ArrayLike, NDArray

Arrayf64 = NDArray[np.float64]


class SRGRConverter(abc.ABC):
    """Abstract base class for slant range / ground range (SRGR) converters."""

    @abc.abstractmethod
    def gr_to_rng(self, gr: ArrayLike, azt: ArrayLike) -> Arrayf64:
        """Convert ground range to slant range.

        Parameters
        ----------
        gr : ndarray or scalar
            Ground range value(s).
        azt : ndarray or scalar
            Azimuth time(s) at which to perform the conversion.

        Returns
        -------
        ndarray
            Slant range value(s).
        """

    @abc.abstractmethod
    def rng_to_gr(self, rng: ArrayLike, azt: ArrayLike) -> Arrayf64:
        """Convert slant range to ground range.

        Parameters
        ----------
        rng : ndarray or scalar
            Slant range value(s).
        azt : ndarray or scalar
            Azimuth time(s) at which to perform the conversion.

        Returns
        -------
        ndarray
            Ground range value(s).
        """
