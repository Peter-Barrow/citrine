import os
import datetime
import pickle
import numpy as np
from numba import njit
from typing import Tuple, Union, Dict, Optional, List
from numpy.typing import NDArray
from scipy.integrate import quad, nquad
from scipy.interpolate import RegularGridInterpolator
from dataclasses import dataclass
import citrine


_HAVE_TQDM = False
try:
    from tqdm import tqdm

    _HAVE_TQDM = True
except ImportError:
    pass

__all__ = [
    'photon_pair_coupling_filter',
    'single_photon_coupling_filter',
    'brightness_and_heralding',
    'calculate_optimisation_grid',
    'make_lookup_table',
]


def photon_pair_coupling_filter(
    xi: float,
    alpha_sq: float,
    phi_0: float,
    np_over_ns: float,
    np_over_ni: float,
    rho_bounds: Tuple[float, float] = (0, 2),
    theta_bounds: Tuple[float, float] = (0, 2 * np.pi),
) -> float:
    @njit()
    def _Q2_integrand(
        theta: float,
        rho_s: float,
        rho_i: float,
        xi: float,
        alpha_sq: float,
        phi_0: float,
        np_over_ns: float,
        np_over_ni: float,
    ) -> float:
        rho_s_sq = rho_s**2
        rho_i_sq = rho_i**2
        rho_rho_cos = 2 * rho_s * rho_i * np.cos(theta)

        # Term 1: Pump-target mode overlap (Pump beam + Spatial filtering in Fig.1)
        term1 = np.exp(-(1 + alpha_sq) * (rho_s_sq + rho_i_sq))

        # Term 2: Signal-idler correlation (SPDC in crystal in Fig. 1)
        term2 = np.exp(-2 * rho_s * rho_i * np.cos(theta))

        # When np_over_ns = np_over_ni = 1:
        # (1-2np/ns) = (1-2) = -1 and (1-2np/ni) = (1-2) = -1
        phase = (phi_0 / 2) + xi * (
            ((1 - (2 * np_over_ns)) * rho_s_sq)
            + ((1 - (2 * np_over_ni)) * rho_i_sq)
            + rho_rho_cos
        )
        term3 = np.sinc(phase / np.pi)

        return term1 * term2 * term3

    Q2, _ = nquad(
        lambda rho_i, rho_s, theta: (rho_s * rho_i)
        * _Q2_integrand(
            theta,
            rho_s,
            rho_i,
            xi,
            alpha_sq,
            phi_0,
            np_over_ns,
            np_over_ni,
        ),
        [rho_bounds, rho_bounds, theta_bounds],
    )
    K2 = (8 / (np.pi**5)) * xi * (alpha_sq**2) * np.abs(2 * np.pi * Q2) ** 2
    return K2


def single_photon_coupling_filter(
    xi: float,
    alpha_sq: float,
    phi_0: float,
    np_over_ns: float,
    np_over_ni: float,
    rho_bounds: Tuple[float, float] = (0, 2),
    theta_bounds: Tuple[float, float] = (-np.pi, np.pi),
) -> float:
    @njit()
    def Q1_integrand(
        theta: float,
        rho_s: float,
        rho_i: float,
        xi: float,
        alpha_sq: float,
        phi_0: float,
        np_over_ns: float,
        np_over_ni: float,
    ) -> float:
        rho_s_sq = rho_s**2
        rho_i_sq = rho_i**2
        rho_rho_cos = 2 * rho_s * rho_i * np.cos(theta)

        term1 = np.exp(-rho_i_sq - (1 + alpha_sq) * rho_s_sq - rho_rho_cos)

        # theta = theta_s - theta_i

        term2 = 1

        # Term 3: Single-mode coupling for signal only (Spatial filtering for one channel in Fig.1)
        # term3 = np.exp(-(alpha**2) * rho_s * rho_s)
        term3 = 1

        # Term 4: Phase matching (PPLN crystal in Fig. 1)
        # Using the general form from the paper:
        # sinc{φ₀/2 + ξ[(1-2np/ns)ρₛ² + (1-2np/ni)ρᵢ² + 2ρₛρᵢcos(θₛ-θᵢ)]}

        phase = (phi_0 / 2) + xi * (
            ((1 - (2 * np_over_ns)) * rho_s_sq)
            + ((1 - (2 * np_over_ni)) * rho_i_sq)
            + rho_rho_cos
        )
        term4 = np.sinc(phase / np.pi)

        return term1 * term2 * term3 * term4

    def Q1_inner(rho_i: float) -> float:
        result, _ = nquad(
            lambda rho_s, theta: rho_s
            * Q1_integrand(
                theta,
                rho_s,
                rho_i,
                xi,
                alpha_sq,
                phi_0,
                np_over_ns,
                np_over_ni,
            ),
            [rho_bounds, theta_bounds],
        )
        return result**2

    Q1, _ = quad(
        lambda rho_i: rho_i * Q1_inner(rho_i),
        *rho_bounds,
    )

    K1 = (4 / (np.pi**4)) * xi * (alpha_sq) * 2 * np.pi * Q1
    return K1


def brightness_and_heralding(
    xi: float,
    alpha: float,
    phi_0: float,
    n_p: float,
    n_i: float,
    n_s: float,
    rho_max: float = 2.0,
) -> Tuple[float, float, float]:
    np_over_ns = n_p / n_s
    np_over_ni = n_p / n_i

    alpha_sq = alpha**2

    K1 = single_photon_coupling_filter(
        xi, alpha_sq, phi_0, np_over_ns, np_over_ni
    )
    K2 = photon_pair_coupling_filter(
        xi, alpha_sq, phi_0, np_over_ns, np_over_ni
    )

    heralding = K2 / K1
    # Return heralding ratio - Γ₂|₁ = K₂/K₁ from Eq. 41
    return K1, K2, heralding


def calculate_optimisation_grid(
    xi: float,
    alpha_range: Union[float, NDArray[np.floating]],
    phi0_range: Union[float, NDArray[np.floating]],
    n_p: float,
    n_s: float,
    n_i: float,
    rho_max: float = 2.0,
    progress_bar: bool = True,
):
    """
    Calculate both K1, K2 and heralding ratio grids in a single pass.

    Parameters:
    -----------
    xi : float
        Focusing parameter ξ = L/(2zR)
    alpha_range : array
        Range of normalized target mode waist values (α)
    phi0_range : array
        Range of longitudinal phase mismatch values (φ0)

    n_p: float,
        refractive index of the pump
    n_s: float,
        refractive index of the signal
    n_i: float,
        refractive index of the idler
    rho_max : float, optional
        Maximum integration bound for rho

    Returns:
    --------
    K2_grid : 2D array
        K2/(kp0L) values for Figure 3 (brightness optimization)
    ratio_grid : 2D array
        Γ2|1 values for Figure 6 (heralding ratio optimization)
    """
    n_phi = len(phi0_range)
    n_alpha = len(alpha_range)
    K1_grid = np.zeros((n_phi, n_alpha))
    K2_grid = np.zeros((n_phi, n_alpha))
    ratio_grid = np.zeros((n_phi, n_alpha))

    total_iterations = n_phi * n_alpha
    if progress_bar and _HAVE_TQDM:
        pbar = tqdm(
            total=total_iterations, desc=f'Calculating grids for ξ = {xi}'
        )

    for i, phi0 in enumerate(phi0_range):
        for j, alpha in enumerate(alpha_range):
            # Get all values in a single calculation
            K1, K2, ratio = brightness_and_heralding(
                xi=xi,
                alpha=alpha,
                phi_0=phi0,
                n_p=n_p,
                n_i=n_i,
                n_s=n_s,
                rho_max=rho_max,
            )
            # Store values for both figures
            ratio_grid[i, j] = ratio  # Heralding ratio for Figure 6
            K1_grid[i, j] = K1
            K2_grid[i, j] = K2  # K2 for Figure 3
            if progress_bar and _HAVE_TQDM:
                pbar.update(1)

    if progress_bar and _HAVE_TQDM:
        pbar.close()
    return K1_grid, K2_grid, ratio_grid


def make_lookup_table(
    xi_values: Union[float, NDArray[np.floating]],
    alpha: Union[float, NDArray[np.floating]],
    phi_0: Union[float, NDArray[np.floating]],
    crystal: citrine.Crystal,
    wavelength_pump: citrine.Wavelength,
    wavelength_signal: citrine.Wavelength,
    wavelength_idler: citrine.Wavelength,
    name: Optional[str] = None,
    temp_dir: Optional[str] = None,
    rho_max: float = 2.0,
    temperature: float = None,
    overwrite: bool = False,
) -> Dict:
    if temp_dir is None:
        temp_dir = f'optimsiation_grids_{crystal.name}'

    if not os.path.exists(temp_dir):
        os.mkdir(temp_dir)

    (n_p, n_s, n_i) = crystal.refractive_indices(
        wavelength_pump,
        wavelength_signal,
        wavelength_idler,
        temperature=temperature,
    )

    if temperature is None:
        temperature = crystal.sellmeier_e.temperature

    k1_lut = []
    k2_lut = []
    heralding_lut = []

    def calculate(xi: float):
        k1, k2, heralding = calculate_optimisation_grid(
            xi,
            alpha,
            phi_0,
            n_p,
            n_s,
            n_i,
            rho_max,
        )

        np.savez(
            data_filename,
            K1_grid=k1,
            K2_grid=k2,
            Heralding=heralding,
        )

        return k2, k2, heralding

    for xi in xi_values:
        data_filename = f'{temp_dir}/xi_{xi}.npz'
        file_exists = os.path.exists(data_filename)

        if file_exists:
            if overwrite is True:
                k1, k2, heralding = calculate(xi)
            else:
                data = np.load(data_filename)

                (k1, k2, heralding) = (
                    data['K1_grid'],
                    data['K2_grid'],
                    data['Heralding'],
                )
        else:
            k1, k2, heralding = calculate(xi)

        k1_lut.append(k1)
        k2_lut.append(k2)
        heralding_lut.append(heralding)

    simulation_results = {
        'refractive_index': {
            'pump': n_p,
            'signal': n_s,
            'idler': n_i,
        },
        'wavelength': {
            'pump': str(wavelength_pump),
            'signal': str(wavelength_signal),
            'idler': str(wavelength_idler),
        },
        'temperature': temperature,
        'xi': np.array(xi_values),
        'alpha': alpha,
        'phi': phi_0,
        'K1': np.array(k1_lut),
        'K2': np.array(k2_lut),
        'Heralding': np.array(heralding_lut),
    }

    if name is None:
        datestamp = datetime.datetime.now().strftime('%Y_%m_%d:%H:%M')
        name = f'brightness-and-heralding-{crystal.name}-lut-{datestamp}.pkl'

    with open(name, 'wb') as f:
        pickle.dump(simulation_results, f)

    return BrightnessHeraldingLUT(**simulation_results), name


@dataclass(frozen=True)
class BrightnessHeraldingLUT:
    refractive_index: Dict[str, float]
    wavelength: Dict[str, citrine.Wavelength]
    temperature: float
    xi: Union[float, NDArray[np.floating]]
    alpha: Union[float, NDArray[np.floating]]
    phi: Union[float, NDArray[np.floating]]
    K1: NDArray[np.floating]
    K2: NDArray[np.floating]
    Heralding: NDArray[np.floating]

    @classmethod
    def from_file(cls, path: str):
        with open(path, 'rb') as f:
            results = pickle.load(f)

        return BrightnessHeraldingLUT(**results)

    def interpolate(
        self,
        target_xi_values: Union[float, List[float], NDArray[np.floating]],
    ) -> Tuple[
        NDArray[np.floating],
        NDArray[np.floating],
        NDArray[np.floating],
    ]:
        """
        Create interpolated K2 and heralding grids for specified xi values.

        Parameters:
        -----------
        target_xi_values : list or array
            Xi values for which to create interpolated grids

        Returns:
        --------
        interpolated_grids : dict
            Dictionary with keys as xi values, each containing K2_grid and heralding_grid
        alpha_range : array
            Alpha values used in the grids
        phi0_range : array
            Phi0 values used in the grids
        """

        # Extract data from lookup table
        xi_lut = self.xi
        alpha_range = self.alpha
        phi0_range = self.phi
        K2_lut = self.K2  # Shape: (n_xi, n_phi0, n_alpha)
        heralding_lut = self.Heralding  # Shape: (n_xi, n_phi0, n_alpha)

        # Create interpolators for K2 and heralding
        # The interpolator expects points in the order (xi, phi0, alpha)
        interpolator_K2 = RegularGridInterpolator(
            (xi_lut, phi0_range, alpha_range),
            K2_lut,
            bounds_error=False,
            fill_value=None,
        )

        interpolator_heralding = RegularGridInterpolator(
            (xi_lut, phi0_range, alpha_range),
            heralding_lut,
            bounds_error=False,
            fill_value=None,
        )

        interpolated_grids = {}

        for target_xi in target_xi_values:
            if target_xi < xi_lut.min() or target_xi > xi_lut.max():
                print(
                    f'Warning: ξ = {target_xi} is outside the lookup table range [{xi_lut.min():.3f}, {xi_lut.max():.3f}]'
                )

            phi_mesh, alpha_mesh = np.meshgrid(
                phi0_range, alpha_range, indexing='ij'
            )
            xi_mesh = np.full_like(phi_mesh, target_xi)

            coords = np.stack(
                [xi_mesh.ravel(), phi_mesh.ravel(), alpha_mesh.ravel()],
                axis=-1,
            )

            K2_interp = interpolator_K2(coords).reshape(phi_mesh.shape)
            heralding_interp = interpolator_heralding(coords).reshape(
                phi_mesh.shape
            )

            interpolated_grids[target_xi] = {
                'K2_grid': K2_interp,
                'heralding_grid': heralding_interp,
            }

            print(f'Created interpolated grids for ξ = {target_xi}')

        return interpolated_grids, alpha_range, phi0_range
