from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Optional

import numpy as np
import tqdm
from numpy.typing import ArrayLike, NDArray
from scipy.optimize import minimize
from typing_extensions import override

from teosar import psutils


@dataclass(frozen=True)
class Grid:
    """Precomputed model predictions over a parameter grid (see `BaseModel.predict_grid`)."""

    values: NDArray
    parent_model: Model

    def get_values(self) -> NDArray:
        """Return the grid of predicted values."""
        return self.values

    def get_last_dim_size(self):
        """Return the size of the last (observation) axis of `values`."""
        return self.values.shape[-1]


class BaseModel(ABC):
    """A parametric model of a phase signal, used to search for the best-fitting parameters."""

    def get_dimension(self):
        """Return the number of unknown parameters of this model."""
        return len(self.get_theta_list())

    @abstractmethod
    def get_theta_list(self) -> list[NDArray]:
        """Return the tested values for each unknown parameter (one array per parameter)."""
        ...

    def get_theta_from_indices(self, indices: list[int]):
        """Return the parameter values at `indices` into each of `get_theta_list()`'s arrays."""
        assert len(indices) == self.get_dimension(), (
            "Should provide one index per element in theta_list"
        )
        return [theta[idx] for idx, theta in zip(indices, self.get_theta_list())]

    @abstractmethod
    def predict_grid(self) -> Grid:
        """
        Predicts the model for a meshgrid defined by each value taken in theta_list
        and each value of the feature observation.

        Returns
        -------
        Grid.

        """
        ...

    @abstractmethod
    def predict_single(self, theta_single: list[float]):
        """
        Predicts the model for a single value per theta

        Parameters
        ----------
        theta_single: list[float]
                    d float elements

        Returns
        -------
        prediction: (Nobs,)
        """
        ...


@dataclass
class Model(BaseModel):
    """Model class

    Parameters
    ----------
    const:
        Nconst
    feat:
        Nfeat x Nobs
    theta_list:
        d elems size Nj j=1...d

    """

    const: list[float]
    feat: NDArray[np.float64]
    theta_list: list[NDArray]

    @override
    def get_theta_list(self) -> list[NDArray]:
        """Return `self.theta_list`."""
        return self.theta_list


class SeasonalModel(Model):
    r"""
    Implements C.S.\sin(\frac{2\pi}{T}(T_i - T_0)),
    where T_i is the feature i=1...Nobservations,
    Cv is a known constant (for conversion to phase for example),
    S, T, T_0 are the unkown parameters of the sinusoidal signal
    """

    def __init__(self, C: float, Ti: NDArray[np.float64], S, T, T0):
        """
        Parameters
        ----------
        C : float
            Known constant (e.g. a conversion factor to phase).
        Ti : NDArray[np.float64]
            Observation times/features, shape (Nobs,).
        S, T, T0 : array-like
            Tested values for the amplitude, period, and phase offset
            unknown parameters, respectively.
        """
        theta_list = [S, T, T0]
        super().__init__([C], Ti[None, :], theta_list)

    @override
    def predict_grid(self) -> Grid:
        """Predict the sinusoidal signal for every combination of `S`, `T`, `T0`."""
        S, T, T0 = self.theta_list
        T = np.asarray(T)
        Ti = self.feat[0]
        res = -2 * np.pi * np.subtract.outer(T0, Ti)  # len(T0) x Nobs
        res = np.sin(np.multiply.outer(1 / T, res))  # len(T) x len(T0) x Nobs
        res = self.const[0] * np.multiply.outer(
            S, res
        )  # len(S) x len(T) * len(T0) x Nobs
        grid = Grid(res, self)
        return grid

    @override
    def predict_single(self, theta_single: list[float]):
        """Predict the sinusoidal signal for a single `(S, T, T0)` value."""
        assert len(theta_single) == 3, "Expected 3 values as theta"
        S, T, T0 = theta_single
        Ti = self.feat[0]
        return np.sin(2 * np.pi * (Ti - T0) / T) * S * self.const[0]


class LinearTermModel(Model):
    """
    1 constant, 1 feature, 1 unkown
    Implement const * feat * theta
    """

    def __init__(
        self,
        const: float,
        feat: NDArray[np.float64],
        theta: ArrayLike,  # Nobs
    ):
        """
        Parameters
        ----------
        const : float
            Known constant multiplier.
        feat : NDArray[np.float64]
            Observation feature (e.g. time or spatial coordinate), shape (Nobs,).
        theta : ArrayLike
            Tested values for the unknown linear coefficient.
        """
        theta_list = [np.asarray(theta)]
        super().__init__([const], feat[None, :], theta_list)

    @override
    def predict_grid(self) -> Grid:
        """Predict `const * feat * theta` for every tested `theta` value."""
        # len(theta) x Nobs
        res = self.const[0] * np.multiply.outer(self.theta_list[0], self.feat[0])
        grid = Grid(res, self)
        return grid

    @override
    def predict_single(self, theta_single: list[float]):
        """Predict `const * feat * theta` for a single `theta` value."""
        assert len(theta_single) == 1, "Expected single element in list"

        return self.const[0] * theta_single[0] * self.feat[0]


def expand_dims(array, pos, d, total_dims):
    """
    Insert size-1 axes into `array` so its `d` existing axes sit at position `pos`
    within `total_dims` axes.

    Used by `CompoundModel` to broadcast each sub-model's parameter grid
    into the combined parameter grid.

    Parameters
    ----------
    array : ndarray
        Array with `d` axes to reposition.
    pos : int
        Target starting position of `array`'s axes.
    d : int
        Number of axes in `array` (excluding the trailing observation
        axis, which is left in place).
    total_dims : int
        Total number of parameter axes in the combined grid.

    Returns
    -------
    ndarray
        `array` with size-1 axes inserted so its `d` axes occupy positions
        `[pos, pos + d)`.
    """
    dims_to_expand = [a for a in range(pos)] + [a for a in range(pos + d, total_dims)]
    return np.expand_dims(array, tuple(dims_to_expand))


class CompoundModel(BaseModel):
    """Combines several `BaseModel`s (e.g. planar + seasonal) into one additive model.

    The combined model's unknown parameters are the concatenation of each
    sub-model's parameters, and its prediction is the sum of each
    sub-model's prediction.
    """

    def __init__(self, models: list[BaseModel]):
        """
        Parameters
        ----------
        models : list[BaseModel]
            Sub-models to combine additively.
        """
        self.models = models
        self.dims = [mod.get_dimension() for mod in self.models]
        self.begin_positions = np.cumsum([0] + self.dims[:-1])
        self.d = np.sum(self.dims)

    @override
    def predict_grid(self) -> Grid:
        """Predict the sum of each sub-model's prediction, over the combined parameter grid."""
        pred = 0
        for mod, pos, d in zip(self.models, self.begin_positions, self.dims):
            pred = pred + expand_dims(mod.predict_grid().get_values(), pos, d, self.d)
        assert isinstance(pred, np.ndarray)
        return Grid(pred, self)

    @override
    def predict_single(self, theta_single: list[float]):
        """Predict the sum of each sub-model's prediction, for a single combined parameter value."""
        assert len(theta_single) == self.d, (
            f"Expected {self.d} elements in theta_single"
        )
        pred = 0
        for mod, pos, d in zip(self.models, self.begin_positions, self.dims):
            pred = pred + mod.predict_single(theta_single[pos : pos + d])
        return pred

    # override base method
    def get_dimension(self):
        """Return the total number of unknown parameters across all sub-models."""
        return self.d

    @override
    def get_theta_list(self) -> list[NDArray]:
        """Return the concatenation of each sub-model's tested parameter values."""
        theta_list = []
        for mod in self.models:
            theta_list += mod.get_theta_list()
        return theta_list


@dataclass
class ExhaustiveGamma:
    """Result of an exhaustive (grid) search for the coherence-maximizing model parameters.

    abs_gamma : coherence amplitude (`|gamma|`) at each point of the tested
        parameter grid.
    max_indices : grid indices of the maximum of `abs_gamma`.
    max_theta : parameter values at `max_indices`.
    max_gamma_abs : `abs_gamma` at `max_indices`.
    max_gamma_angle : phase (bias) of the complex coherence at `max_indices`.
    """

    abs_gamma: NDArray[np.float32]
    max_indices: list[int]
    max_theta: list[float]
    max_gamma_abs: float
    max_gamma_angle: float

    def get_bounds(
        self, theta_list: list[NDArray]
    ) -> list[tuple[Optional[float], Optional[float]]]:
        """Return, for each parameter, the grid values adjacent to `max_indices`.

        Used as bounds for a local refinement optimization around the
        exhaustive search's maximum.

        Parameters
        ----------
        theta_list : list[NDArray]
            Tested values for each parameter (same as used for the grid
            search).

        Returns
        -------
        list[tuple[Optional[float], Optional[float]]]
            `(lower, upper)` bound for each parameter: the grid values
            immediately below/above `max_theta`, or None at the edge of
            the grid.
        """
        # bounds
        bounds: list[tuple[Optional[float], Optional[float]]] = []
        assert len(self.max_indices) == len(theta_list), (
            "theta_list must have same size as max_indices"
        )
        for m, theta in zip(self.max_indices, theta_list):
            if m > 0:
                l = theta[m - 1]
            else:
                l = None
            if m < len(theta) - 1:
                u = theta[m + 1]
            else:
                u = None

            bounds.append((l, u))

        return bounds


@dataclass
class Periodogram:
    """Estimates model parameters that best explain a wrapped phase time series.

    Implements a periodogram-style search: coherence is evaluated
    exhaustively over a parameter grid (`exhaustive_gamma`), then refined
    locally around the best grid point (`refinement`).
    """

    phi_wrapped: NDArray[np.float64]  # N obs
    weights: Optional[NDArray[np.float64]] = None  # N obs

    def __init__(self, phi_wrapped: list[float], weights: Optional[list[float]] = None):
        """
        Parameters
        ----------
        phi_wrapped : list[float]
            Wrapped phase observations, one per date. nan values are
            allowed and are given zero weight.
        weights : list[float], optional
            Per-observation weight. Defaults to uniform weights. Weights
            are normalized to sum to 1, and nan observations are always
            given zero weight.
        """
        nan_mask = np.isnan(phi_wrapped)
        assert not np.all(nan_mask), "All values are nan, can't do periodogram"

        # might contain nans
        # replace nan with arbitrary value, weight=à will eliminate this from sum
        self.phi_wrapped = np.array(np.nan_to_num(phi_wrapped))

        # if weights is not None:
        # weights = np.array(weights)
        # else:
        # weights = np.ones((len(self.phi_wrapped),))
        weights = (
            np.ones((len(self.phi_wrapped),)) if weights is None else np.array(weights)
        )

        # put nan value weights to zero
        weights[nan_mask] = 0
        # normalize weights
        weights /= np.sum(weights)
        self.weights = weights

    def exhaustive_gamma(self, grid: Grid) -> ExhaustiveGamma:
        """
        Evaluate the (weighted) complex coherence over every point of `grid` and find its maximum.

        Parameters
        ----------
        grid : Grid
            Model predictions over the tested parameter grid (see
            `BaseModel.predict_grid`).

        Returns
        -------
        ExhaustiveGamma
            Coherence amplitude over the grid and the parameters/coherence
            at its maximum.
        """
        assert grid.get_last_dim_size() == len(self.phi_wrapped), (
            "Grid last dimension size should match phase array size"
        )

        tmp = np.exp(1j * (-grid.get_values() + self.phi_wrapped))
        cmpx_gamma = np.sum(tmp * self.weights, axis=-1)
        del tmp

        abs_gamma = np.abs(cmpx_gamma)
        max_indices = np.unravel_index(np.argmax(abs_gamma), abs_gamma.shape)
        max_gamma_angle = np.angle(cmpx_gamma[max_indices])
        del cmpx_gamma
        max_theta = grid.parent_model.get_theta_from_indices(max_indices)  # type: ignore
        max_gamma_abs = abs_gamma[max_indices]

        return ExhaustiveGamma(
            abs_gamma,
            max_indices,  # type: ignore
            max_theta,
            max_gamma_abs,
            max_gamma_angle,
        )

    def get_gamma_single(self, model: BaseModel, theta_single: list[float]):
        """Compute the (weighted) complex coherence of `model` at a single parameter value."""
        pred_per_obs = model.predict_single(theta_single)
        tmp = np.exp(1j * (self.phi_wrapped - pred_per_obs))
        gamma = np.sum(tmp * self.weights)
        return gamma

    def refinement(
        self, model: BaseModel, exhaustive_gamma: ExhaustiveGamma, no_failure=False
    ):
        """
        Locally refine the exhaustive grid search's maximum with a bounded optimizer.

        Maximizes `|gamma|` with L-BFGS-B, starting from
        `exhaustive_gamma.max_theta` and bounded by
        `exhaustive_gamma.get_bounds`.

        Parameters
        ----------
        model : BaseModel
            Model whose parameters are being estimated.
        exhaustive_gamma : ExhaustiveGamma
            Result of the exhaustive grid search, used as the starting
            point and bounds.
        no_failure : bool, optional
            If True, return the optimizer's result even if it did not
            report convergence. Defaults to False.

        Returns
        -------
        theta_opt : ndarray
            Refined parameter values.
        gamma_abs_opt : float
            Coherence amplitude at `theta_opt`.

        Raises
        ------
        ArithmeticError
            If the optimizer did not converge and `no_failure` is False.
        """
        estimated = exhaustive_gamma.max_theta
        bounds = exhaustive_gamma.get_bounds(model.get_theta_list())

        # refinement
        def to_minimize(x):
            return -np.abs(self.get_gamma_single(model, x))

        res = minimize(
            to_minimize,
            estimated,
            method="L-BFGS-B",
            options={"disp": False},
            bounds=bounds,
        )
        if res.success or no_failure:
            return res.x, -res.fun
        else:
            raise ArithmeticError("Did not converge")


def get_test_vals(max_val, half_n_samples):
    """
    Build a symmetric array of test values in `[-max_val, max_val]`, including 0.

    Parameters
    ----------
    max_val : float
        Maximum (and, negated, minimum) tested value.
    half_n_samples : int
        Number of samples on `[0, max_val]` (inclusive).

    Returns
    -------
    ndarray
        Sorted array of `2 * half_n_samples - 1` values from `-max_val` to
        `max_val`, symmetric around 0.
    """
    test_vals = np.linspace(0, max_val, half_n_samples)
    return np.hstack([-test_vals[:0:-1], test_vals])


# here specialize some cases for ease of use


def get_planar_model(xx, yy, max_slopes=(5e-2, 0.2), min_half_samples=10):
    """for fitting spatially affine phase"""
    n_ps = len(xx)

    def get_slope(max_slope):
        half_n_samples = max(min_half_samples, int(np.sqrt(n_ps) / 4))  # 25 for 10 000
        return get_test_vals(max_slope, half_n_samples)

    slope_x = get_slope(max_slopes[0])
    slope_y = get_slope(max_slopes[1])

    model = CompoundModel(
        [LinearTermModel(1, yy, slope_y), LinearTermModel(1, xx, slope_x)]
    )

    return model


@dataclass(frozen=True)
class PlanarPeriodogramResult:
    """Result of fitting a spatially-planar phase model for one date (see `planar_periodogram`).

    exhaustive_gamma : result of the exhaustive grid search.
    slopes : refined (row, col) planar slopes.
    bias : refined phase bias (radians).
    gamma_opt : coherence amplitude at the refined solution.
    """

    exhaustive_gamma: ExhaustiveGamma
    slopes: tuple[float, float]
    bias: float
    gamma_opt: float


def planar_periodogram(
    model: BaseModel, phi_wrapped_ts: np.ndarray, no_failure=False
) -> list[PlanarPeriodogramResult]:
    """
    Fit a spatially-planar phase model to each date of a wrapped phase time series.

    For each date, runs an exhaustive grid search followed by a local
    refinement (see `Periodogram.exhaustive_gamma`/`refinement`) to find
    the planar slopes and bias that best explain the observed phase.

    Parameters
    ----------
    model : BaseModel
        Planar phase model (see `get_planar_model`), shared across dates.
    phi_wrapped_ts : ndarray
        Wrapped phase time series, shape (n_dates, n_obs).
    no_failure : bool, optional
        Passed through to `Periodogram.refinement`. Defaults to False.

    Returns
    -------
    list[PlanarPeriodogramResult]
        One fit result per date.
    """
    grid = model.predict_grid()
    results = []
    for i in tqdm.trange(len(phi_wrapped_ts)):
        period = Periodogram(phi_wrapped_ts[i])
        exhaustive_gamma = period.exhaustive_gamma(grid)
        slopes, gamma_opt = period.refinement(
            model, exhaustive_gamma, no_failure=no_failure
        )
        assert isinstance(slopes, tuple)
        assert len(slopes) == 2
        slopes: tuple[float, float]

        cmpx_gamma_opt = period.get_gamma_single(model, list(slopes))
        bias = np.angle(cmpx_gamma_opt)
        diff = abs(gamma_opt - np.abs(cmpx_gamma_opt))
        threshold = 0.01
        assert diff < threshold, (
            f"something wrong with gamma opt, {diff} greater than {threshold} "
        )
        results.append(
            PlanarPeriodogramResult(exhaustive_gamma, slopes, bias, gamma_opt)
        )

    return results


def compensate_planar_phase(phi_ts, xx, yy, slopes, biases):
    """
    Subtract a per-date fitted planar phase from a wrapped phase time series.

    Parameters
    ----------
    phi_ts : ndarray
        Wrapped phase time series, shape (n_dates, n_obs).
    xx, yy : ndarray
        Spatial coordinates (e.g. col, row) of each observation, shape (n_obs,).
    slopes : Sequence[tuple[float, float]]
        Fitted (row, col) planar slopes for each date, as produced by
        `planar_periodogram`.
    biases : Sequence[float]
        Fitted phase bias (radians) for each date.

    Returns
    -------
    ndarray
        Wrapped phase time series with the fitted plane subtracted at each
        date, same shape as `phi_ts`.
    """
    model = CompoundModel([LinearTermModel(1, yy, [0]), LinearTermModel(1, xx, [0])])

    phi_compensated = np.zeros_like(phi_ts)
    for i in tqdm.trange(len(phi_ts)):
        prediction = model.predict_single(slopes[i]) + biases[i]
        phi_compensated[i] = psutils.wrap(phi_ts[i] - prediction)

    return phi_compensated
