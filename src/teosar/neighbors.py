import numpy as np
import scipy
import scipy.sparse
from numpy.typing import NDArray

from teosar import psutils


def get_neighbors(
    ps_col: NDArray[np.int32],
    ps_row: NDArray[np.int32],
    distance_threshold: float,
    resolution_x: float,
    resolution_y: float,
) -> NDArray[np.bool_]:
    """
    Compute the pairwise "is neighbor" boolean matrix between persistent scatterers.

    Two PS points are considered neighbors if their ground distance (pixel
    offsets scaled by `resolution_x`/`resolution_y`) is strictly less than
    `distance_threshold`. A point is never its own neighbor.

    Parameters
    ----------
    ps_col : ndarray of int32
        Column of each PS.
    ps_row : ndarray of int32
        Row of each PS, same length as `ps_col`.
    distance_threshold : float
        Maximum ground distance (in the same unit as `resolution_x`/
        `resolution_y`) for two PS to be considered neighbors.
    resolution_x : float
        Ground resolution in the column direction.
    resolution_y : float
        Ground resolution in the row direction.

    Returns
    -------
    ndarray of bool
        Square (n_ps, n_ps) matrix, True where the pair is within
        `distance_threshold` (diagonal is always False).
    """
    num_PS = len(ps_col)
    # find neighbors
    distance_x = np.abs(ps_col.reshape([1, -1]) - ps_col.reshape([-1, 1]))
    distance_y = np.abs(ps_row.reshape([1, -1]) - ps_row.reshape([-1, 1]))

    distance_in_meters = np.sqrt(
        (resolution_x * distance_x) ** 2 + (resolution_y * distance_y) ** 2
    )
    is_at_ok_distance = distance_in_meters < distance_threshold

    # remove yourself
    for i in range(num_PS):
        is_at_ok_distance[i, i] = False

    return is_at_ok_distance


def phi_ps_neighbors(phi_sparse, is_at_ok_distance):
    """
    Average, per PS and per date, the wrapped phase of its neighbors.

    For each PS, its neighbors' phases (from `is_at_ok_distance`) are
    circularly averaged (via the complex exponential sum) at each date;
    a PS with no neighbors is left as nan.

    Parameters
    ----------
    phi_sparse : ndarray
        Wrapped phase per date and PS, shape (n_dates, n_ps).
    is_at_ok_distance : scipy.sparse.csr_array
        Sparse (n_ps, n_ps) boolean neighbor matrix, as produced by
        `get_neighbors`.

    Returns
    -------
    ndarray
        Circularly-averaged neighbor phase, shape (n_dates, n_ps), nan
        for PS with no neighbors.
    """
    output = np.full(phi_sparse.shape, np.nan)
    for k in range(phi_sparse.shape[1]):
        neighbors = is_at_ok_distance.indices[
            is_at_ok_distance.indptr[k] : is_at_ok_distance.indptr[k + 1]
        ]
        phi_neighbors = phi_sparse[:, neighbors]

        if len(neighbors):
            phi_neighbors = np.angle(np.nansum(np.exp(1j * phi_neighbors), axis=-1))
            output[:, k] = phi_neighbors

    return output


def compute_phi_neighbors(
    ps_col, ps_row, phi_sparse_ts, resolution_x, resolution_y, distance_threshold=300
):
    """
    Compute the circularly-averaged neighbor phase for each PS and date.

    Combines `get_neighbors` and `phi_ps_neighbors`.

    Parameters
    ----------
    ps_col, ps_row : ndarray
        Column and row of each PS.
    phi_sparse_ts : array-like
        Wrapped phase per date and PS, shape (n_dates, n_ps).
    resolution_x, resolution_y : float
        Ground resolution in the column/row direction.
    distance_threshold : float, optional
        Maximum ground distance for two PS to be considered neighbors.
        The default is 300.

    Returns
    -------
    ndarray
        Circularly-averaged neighbor phase, shape (n_dates, n_ps).
    """
    is_at_ok_distance = scipy.sparse.csr_array(
        get_neighbors(ps_col, ps_row, distance_threshold, resolution_x, resolution_y)
    )

    phi_sparse_ts = np.array(phi_sparse_ts)
    phi_neighbors = phi_ps_neighbors(phi_sparse_ts, is_at_ok_distance)

    return phi_neighbors


def compute_ps_vs_neighbors(
    ps_col, ps_row, phi_sparse_ts, resolution_x, resolution_y, distance_threshold=300
):
    """
    Compute, per PS and date, the wrapped phase difference to its neighbors' average.

    Parameters
    ----------
    ps_col, ps_row : ndarray
        Column and row of each PS.
    phi_sparse_ts : array-like
        Wrapped phase per date and PS, shape (n_dates, n_ps).
    resolution_x, resolution_y : float
        Ground resolution in the column/row direction.
    distance_threshold : float, optional
        Maximum ground distance for two PS to be considered neighbors.
        The default is 300.

    Returns
    -------
    ndarray
        Wrapped phase difference between each PS and its neighbors'
        circularly-averaged phase, shape (n_dates, n_ps).
    """
    phi_sparse_ts = np.array(phi_sparse_ts)
    phi_neighbors = compute_phi_neighbors(
        ps_col,
        ps_row,
        phi_sparse_ts,
        resolution_x,
        resolution_y,
        distance_threshold=distance_threshold,
    )
    outs = psutils.wrap(phi_sparse_ts - phi_neighbors)
    return outs
