from typing import Any, Optional

import numpy as np
from numpy.typing import NDArray

import eos.dem
from eos.sar import (  # type: ignore
    model,
    roi,
    simulator,
)


def normalize(
    raster: NDArray[Any],
    simulation: NDArray[np.float32],
    shadow_threshold: float = 0.05,
    shadow_value: Optional[float] = 0.0,
) -> NDArray[Any]:
    """Radiometrically terrain-correct a raster by a simulated backscatter image.

    Parameters
    ----------
    raster : ndarray
        Raster to normalize (e.g. amplitude/intensity image).
    simulation : ndarray
        Simulated backscatter raster, co-registered with `raster`, used as the
        normalization reference.
    shadow_threshold : float, optional
        Simulated values below this threshold are considered layover/shadow.
        The default is 0.05.
    shadow_value : float, optional
        Value assigned to pixels flagged as shadow (below `shadow_threshold`).
        If None, shadow pixels are left as normalized (possibly very large)
        values. The default is 0.0.

    Returns
    -------
    ndarray
        The radiometrically terrain-corrected raster.
    """
    normalized = np.sqrt(np.abs(raster) ** 2 / (simulation + 1e-30))

    if shadow_value is not None:
        normalized[simulation < shadow_threshold] = shadow_value

    # TODO: check if it requires normalization with incidence angle as well
    return normalized


class RadiometricTerrainCorrector:
    """
    Warning: see eos.sar.simulator.SARSimulator notes. (= avoid large ROI)

    For parameter `simulator_kwargs`, see eos.sar.simulator.SARSimulator.
    """

    def __init__(
        self,
        proj_model: model.SensorModel,
        dem: eos.dem.DEM,
        roi: roi.Roi,
        simulator_kwargs={},
    ):
        self.simulator = simulator.SARSimulator(proj_model, dem, **simulator_kwargs)
        self.roi = roi
        self._simulation = None

    def apply(self, raster: np.ndarray):
        """Apply radiometric terrain correction to a raster.

        Parameters
        ----------
        raster : ndarray
            Raster to correct, matching the shape of the simulated
            backscatter image over `self.roi`.

        Returns
        -------
        ndarray
            The radiometrically terrain-corrected raster.
        """
        sim = self.get_simulation()
        assert raster.shape == sim.shape
        return normalize(raster, sim)

    def get_simulation(self) -> NDArray[np.float32]:
        """Return the simulated backscatter raster over `self.roi`, computing and caching it on first call."""
        if self._simulation is None:
            self._simulation = self.simulator.simulate(self.roi).astype(np.float32)
        assert self._simulation is not None
        return self._simulation
