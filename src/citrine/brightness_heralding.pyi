import citrine
import numpy as np
from dataclasses import dataclass
from numpy.typing import NDArray

__all__ = ['photon_pair_coupling_filter', 'single_photon_coupling_filter', 'brightness_and_heralding', 'calculate_optimisation_grid', 'make_lookup_table', 'BrightnessHeraldingLUT']

def photon_pair_coupling_filter(xi: float, alpha_sq: float, phi_0: float, np_over_ns: float, np_over_ni: float, rho_bounds: tuple[float, float] = (0, 2), theta_bounds: tuple[float, float] = ...) -> float: ...
def single_photon_coupling_filter(xi: float, alpha_sq: float, phi_0: float, np_over_ns: float, np_over_ni: float, rho_bounds: tuple[float, float] = (0, 2), theta_bounds: tuple[float, float] = ...) -> float: ...
def brightness_and_heralding(xi: float, alpha: float, phi_0: float, n_p: float, n_i: float, n_s: float, rho_max: float = 2.0) -> tuple[float, float, float]: ...
def calculate_optimisation_grid(xi: float, alpha_range: float | NDArray[np.floating], phi0_range: float | NDArray[np.floating], n_p: float, n_s: float, n_i: float, rho_max: float = 2.0, progress_bar: bool = True): ...
def make_lookup_table(xi_values: float | NDArray[np.floating], alpha: float | NDArray[np.floating], phi_0: float | NDArray[np.floating], crystal: citrine.Crystal, wavelength_pump: citrine.Wavelength, wavelength_signal: citrine.Wavelength, wavelength_idler: citrine.Wavelength, name: str | None = None, temp_dir: str | None = None, rho_max: float = 2.0, temperature: float = None, overwrite: bool = False) -> dict: ...

@dataclass(frozen=True)
class BrightnessHeraldingLUT:
    refractive_index: dict[str, float]
    wavelength: dict[str, citrine.Wavelength]
    temperature: float
    xi: float | NDArray[np.floating]
    alpha: float | NDArray[np.floating]
    phi: float | NDArray[np.floating]
    K1: NDArray[np.floating]
    K2: NDArray[np.floating]
    Heralding: NDArray[np.floating]
    @classmethod
    def from_file(cls, path: str): ...
    def interpolate(self, target_xi_values: float | list[float] | NDArray[np.floating]) -> tuple[NDArray[np.floating], NDArray[np.floating], NDArray[np.floating]]: ...
