from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np
import numpy.polynomial.chebyshev as T
import numpy.polynomial.legendre as L
import numpy.polynomial.polynomial as P
from numpy.typing import NDArray

from eos.products.capella.metadata import CapellaPolynomialMeta


@dataclass(frozen=True)
class CapellaPolynomial1D:
    """A 1D polynomial (standard, Chebyshev, or Legendre basis).

    coefficients : 1D array of the polynomial's coefficients, in the basis
    given by `poly_type`.
    """

    poly_type: Literal["standard", "chebyshev", "legendre"]
    coefficients: NDArray[np.float64]

    def __post_init__(self):
        assert self.poly_type in ["standard", "chebyshev", "legendre"]
        # assert 1D array
        assert len(self.coefficients.shape) == 1

    @classmethod
    def from_poly_meta(cls, poly_meta: CapellaPolynomialMeta) -> CapellaPolynomial1D:
        """Build a `CapellaPolynomial1D` from a `CapellaPolynomialMeta`.

        Parameters
        ----------
        poly_meta : CapellaPolynomialMeta
            Polynomial metadata parsed from a Capella product; must be 1D.

        Returns
        -------
        CapellaPolynomial1D
        """
        coefs = np.array(poly_meta.coefficients)

        return CapellaPolynomial1D(poly_meta.poly_type, coefs)

    def evaluate(self, x):
        """Evaluate the polynomial at `x`, using the basis given by `poly_type`."""
        if self.poly_type == "standard":
            return P.polyval(x, self.coefficients)
        elif self.poly_type == "chebyshev":
            return T.chebval(x, self.coefficients)
        elif self.poly_type == "legendre":
            return L.legval(x, self.coefficients)


@dataclass(frozen=True)
class CapellaPolynomial2D:
    """A 2D polynomial (standard, Chebyshev, or Legendre basis).

    coefficients : 2D array of the polynomial's coefficients, in the basis
    given by `poly_type`.
    """

    poly_type: Literal["standard", "chebyshev", "legendre"]
    coefficients: NDArray[np.float64]

    def __post_init__(self):
        assert self.poly_type in ["standard", "chebyshev", "legendre"]
        # assert 2D array
        assert len(self.coefficients.shape) == 2

    @classmethod
    def from_poly_meta(cls, poly_meta: CapellaPolynomialMeta) -> CapellaPolynomial2D:
        """Build a `CapellaPolynomial2D` from a `CapellaPolynomialMeta`.

        Parameters
        ----------
        poly_meta : CapellaPolynomialMeta
            Polynomial metadata parsed from a Capella product; must be 2D.

        Returns
        -------
        CapellaPolynomial2D
        """
        coefs = np.array(poly_meta.coefficients)
        return CapellaPolynomial2D(poly_meta.poly_type, coefs)

    def evaluate(self, x, y):
        """Evaluate the polynomial at paired points `(x, y)`.

        `x` and `y` must be broadcastable to the same shape; the polynomial
        is evaluated pointwise, not on a grid (see `evaluate_grid`).
        """
        if self.poly_type == "standard":
            return P.polyval2d(x, y, self.coefficients)
        elif self.poly_type == "chebyshev":
            return T.chebval2d(x, y, self.coefficients)
        elif self.poly_type == "legendre":
            return L.legval2d(x, y, self.coefficients)

    def evaluate_grid(self, x, y):
        """Evaluate the polynomial on the outer-product grid of `x` and `y`.

        Returns an array of shape `x.shape + y.shape`, unlike `evaluate`
        which evaluates pointwise.
        """
        if self.poly_type == "standard":
            return P.polygrid2d(x, y, self.coefficients)
        elif self.poly_type == "chebyshev":
            return T.chebgrid2d(x, y, self.coefficients)
        elif self.poly_type == "legendre":
            return L.leggrid2d(x, y, self.coefficients)
