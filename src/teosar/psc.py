import numpy as np
from numpy.typing import NDArray

# Use the DA method to get PS candidates


def get_PS_candidates_DA(
    amplitudes: NDArray, threshold: float = 0.25
) -> tuple[NDArray[np.float64], NDArray[np.bool_], NDArray[np.bool_]]:
    """
    Compute DA = sigma_A / mean_A, A being the amplitude.
    Threshold
    Apply local minimum filter
    """

    sigma_A = np.std(
        amplitudes, axis=0
    )  # Which variant of std computation should we use ?
    mean_A = np.mean(amplitudes, axis=0)
    DA = sigma_A / mean_A

    PS_candidates_basic = DA < threshold

    # retrieve local minima of DA
    local_minimum = get_local_min(DA)

    PS_candidates_local_min = np.logical_and(PS_candidates_basic, DA == local_minimum)

    return DA, PS_candidates_basic, PS_candidates_local_min


def get_local_min(array: NDArray) -> NDArray:
    """
    Compute, for each pixel, the minimum value in its 3x3 neighborhood (edges clamped).

    Parameters
    ----------
    array : ndarray
        2D (or higher, broadcast over the first two axes) input array.

    Returns
    -------
    ndarray
        Array of the same shape as `array`, with each pixel replaced by the
        minimum of its 3x3 neighborhood (pixels outside the array bounds
        are ignored, not treated as smaller).
    """
    pos_list = [
        [-1, -1],
        [-1, 0],
        [-1, 1],
        [0, -1],
        [0, 0],
        [0, 1],
        [1, -1],
        [1, 0],
        [1, 1],
    ]

    h, w = array.shape[0], array.shape[1]
    shifted = np.inf * np.ones((len(pos_list), h, w))

    for i, pos in enumerate(pos_list):
        shifted[
            i, max(0, pos[0]) : min(h, h + pos[0]), max(0, pos[1]) : min(w, w + pos[1])
        ] = array[
            max(0, -pos[0]) : min(h, h - pos[0]), max(0, -pos[1]) : min(w, w - pos[1])
        ]

    local_minimum = np.min(shifted, axis=0)

    return local_minimum
