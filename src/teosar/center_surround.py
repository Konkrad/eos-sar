from dataclasses import dataclass

import numpy as np
import scipy
import scipy.sparse
from numpy.typing import NDArray

from teosar import periodogram, psutils


@dataclass(frozen=True)
class MaskedPS:
    """Persistent scatterer (PS) candidate masks split by `search_window`/`guard_window`.

    ps_in : PS candidates inside the guard window.
    ps_bw : PS candidates in the search window but outside the guard window
        ("border window").
    ps_out : PS candidates outside the search window.
    """

    ps_in: NDArray[np.bool_]
    ps_bw: NDArray[np.bool_]
    ps_out: NDArray[np.bool_]


def get_ps_masks_on_win(search_window, guard_window, ps_candidates_full_array):
    """Split a full-image PS candidate mask into search/guard-window regions.

    Parameters
    ----------
    search_window : psutils.Window
        Outer window defining the neighborhood ("surround") used to fit a
        local model.
    guard_window : psutils.Window
        Inner window (the "center"), excluded from the surround fit.
    ps_candidates_full_array : ndarray of bool
        Full-image mask of PS candidates.

    Returns
    -------
    MaskedPS
        PS candidate masks split into inside the guard window, inside the
        search window but outside the guard window, and outside the search
        window.
    """
    parent_shape = ps_candidates_full_array.shape
    mask_search = search_window.get_mask(parent_shape)
    mask_guard = guard_window.get_mask(parent_shape)
    mask_search_outside_guard = np.logical_and(mask_search, ~mask_guard)

    ps_bw = np.logical_and(ps_candidates_full_array, mask_search_outside_guard)

    ps_in = np.logical_and(ps_candidates_full_array, mask_guard)

    ps_out = np.logical_and(ps_candidates_full_array, ~mask_search)

    return MaskedPS(ps_in, ps_bw, ps_out)


def compensate_plane_from_surround(
    search_window,
    guard_window,
    ps_candidates_full_array,
    phi_ts,
    max_slopes=(5e-2, 5e-2),
    min_half_samples=10,
    *,
    no_failure=False,
    result_only_inside=False,
):
    """
    Estimate and remove a per-date planar phase trend fitted on a surrounding window.

    For each date (interferogram) in `phi_ts`, a planar phase model (an
    affine function of row/col) is fitted, via a periodogram search, on the
    PS candidates that fall in `search_window` but outside `guard_window`
    (the "surround"). The fitted plane is then subtracted from the PS
    candidates in the guard window (and optionally the surround as well),
    to compensate a local phase trend (e.g. residual orbital/atmospheric
    ramp) before further PS processing of the center window.

    Parameters
    ----------
    search_window : psutils.Window
        Outer window over which the planar phase model is fitted.
    guard_window : psutils.Window
        Inner window (the "center") whose PS candidates are compensated.
    ps_candidates_full_array : ndarray of bool
        Full-image mask of PS candidates.
    phi_ts : array-like
        Wrapped phase time series, shape (n_interferograms, h, w).
    max_slopes : tuple of float, optional
        Maximum tested (row, col) slopes for the planar model. The default
        is (5e-2, 5e-2).
    min_half_samples : int, optional
        Minimum number of tested slope samples on each side of zero. The
        default is 10.
    no_failure : bool, optional
        Passed through to `periodogram.planar_periodogram`. The default is
        False.
    result_only_inside : bool, optional
        If True, only return compensated values for PS candidates inside
        `guard_window`; otherwise also include those in the surround
        (`search_window` minus `guard_window`). The default is False.

    Returns
    -------
    phi_img_compensated_ts : ndarray
        Compensated phase time series rasterized back to `phi_ts`'s
        (h, w) shape, nan outside the compensated PS candidates.
    masked_ps : MaskedPS
        PS candidate masks split by `search_window`/`guard_window`.
    period_results : list
        Per-date planar periodogram fit results
        (`periodogram.PlanarPeriodogramResult`).
    """
    # masking in windows
    print("Masking PS")
    masked_ps = get_ps_masks_on_win(
        search_window, guard_window, ps_candidates_full_array
    )

    ps_bw_sparse = scipy.sparse.coo_array(masked_ps.ps_bw)
    col_bw = ps_bw_sparse.col
    row_bw = ps_bw_sparse.row

    ps_in_sparse = scipy.sparse.coo_array(masked_ps.ps_in)
    col_in = ps_in_sparse.col
    row_in = ps_in_sparse.row

    phi_ts = np.array(phi_ts)  # Ninterfs x h x w
    phi_wrapped_bw = phi_ts[:, row_bw, col_bw]  # Ninterfs x Nps_bw

    # periodogram per date
    # get the model
    xx = col_bw - search_window.col
    yy = row_bw - search_window.row
    model = periodogram.get_planar_model(
        xx, yy, max_slopes=max_slopes, min_half_samples=min_half_samples
    )
    # solving
    print("Fitting planar periodogram per date")
    period_results = periodogram.planar_periodogram(
        model, phi_wrapped_bw, no_failure=no_failure
    )

    if result_only_inside:
        row, col = row_in, col_in
    else:
        row, col = np.concatenate((row_in, row_bw)), np.concatenate((col_in, col_bw))

    # subtract fitted plane at ps inside guard window
    print("Compensating plane")
    xx = col - search_window.col
    yy = row - search_window.row
    phi_wrapped = phi_ts[:, row, col]  # Ninterfs x Nps

    slopes = [p.slopes for p in period_results]
    biases = [p.bias for p in period_results]

    phi_ps_compensated_ts = periodogram.compensate_planar_phase(
        phi_wrapped, xx, yy, slopes, biases
    )

    # convert to raster
    parent_shape = phi_ts[0].shape
    phi_img_compensated_ts = np.array(
        [
            psutils.sparse_data_to_raster(phi_ps, row, col, parent_shape)
            for phi_ps in phi_ps_compensated_ts
        ]
    )

    return phi_img_compensated_ts, masked_ps, period_results
