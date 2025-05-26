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
    'BrightnessHeraldingLUT',
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
    """Calculate the photon pair coupling efficiency K2 for SPDC into single spatial modes.

    This function implements Equation (26) from Smirr et al., which describes the spatial
    dependence of the two-photon coupling probability P2. The function K2 accounts for
    spatial interferences caused by both phase matching and coupling to the target mode.

    The integrand Q2 contains three main physical terms:
    1. Pump-target mode overlap (spatial filtering)
    2. Signal-idler correlation from SPDC process
    3. Phase matching with longitudinal and transverse mismatch

    Args:
        xi (float): Focusing parameter ξ = L/(2zR), where L is crystal length and zR is
            Rayleigh range.
        alpha_sq (float): Squared normalized target mode waist parameter α² = (a0/w0)²,
            where a0 is target mode waist and w0 is pump beam waist.
        phi_0 (float): Longitudinal phase mismatch φ0 = Δk0·L (Eq. 21).
        np_over_ns (float): Ratio of refractive indices np/ns (pump to signal).
        np_over_ni (float): Ratio of refractive indices np/ni (pump to idler).
        rho_bounds (Tuple[float, float], optional): Integration bounds for normalized
            radial coordinates ρs, ρi. Defaults to (0, 2).
        theta_bounds (Tuple[float, float], optional): Integration bounds for angular
            coordinate θ = θs - θi. Defaults to (0, 2π).

    Returns:
        float: Dimensionless photon pair coupling factor K2/(kp0·L).

    Note:
        Reference: Smirr et al., Eq. (26), page 5:
        K2 = |∫∫ d²κs/(2π)² ∫∫ d²κi/(2π)² S̃p(κs + κi, 0) × Õ0(κs, z0)Õ0(κi, z0)
             × e^(-iz0(np/n's |κs|²/kp0 + np/n'i |κi|²/kp0)) × L sinc(ΔK(κs, κi)L/2)|²

        The normalized coordinates are defined as:
        - ρs = w0·κs/2, ρi = w0·κi/2 (Eq. 50)
        - The phase mismatch ΔK includes both longitudinal (φ0) and transverse terms (Eq. 21)
    """

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
        """Integrand Q2 for two-photon coupling calculation.

        This implements the integrand from Eq. (26), representing the amplitude for
        simultaneous coupling of both signal and idler photons to target spatial modes.
        """
        rho_s_sq = rho_s**2
        rho_i_sq = rho_i**2
        rho_rho_cos = 2 * rho_s * rho_i * np.cos(theta)

        # Term 1: Pump-target mode overlap (Spatial filtering in Fig.1)
        # Represents the overlap between the pump beam spatial profile and the
        # target mode profiles for both signal and idler photons
        # From Gaussian beam S̃p and target modes Õ0 in Eq. (26)
        term1 = np.exp(-(1 + alpha_sq) * (rho_s_sq + rho_i_sq))

        # Term 2: Signal-idler correlation (SPDC process in crystal, Fig. 1)
        # Accounts for the intrinsic spatial correlation between down-converted photons
        # arising from momentum conservation in the SPDC process
        term2 = np.exp(-2 * rho_s * rho_i * np.cos(theta))

        # Term 3: Phase matching condition (PPLN crystal phase matching, Fig. 1)
        # Implements the sinc function from Eq. (26) with phase mismatch ΔK from Eq. (21)
        # The phase includes:
        # - φ0/2: longitudinal phase mismatch contribution
        # - ξ[(1-2np/ns)ρs² + (1-2np/ni)ρi² + 2ρsρi·cos(θ)]: transverse mismatch
        # When np_over_ns = np_over_ni = 1 (degenerate case):
        # (1-2np/ns) = (1-2) = -1 and (1-2np/ni) = (1-2) = -1
        phase = (phi_0 / 2) + xi * (
            ((1 - (2 * np_over_ns)) * rho_s_sq)
            + ((1 - (2 * np_over_ni)) * rho_i_sq)
            + rho_rho_cos
        )
        term3 = np.sinc(phase / np.pi)

        return term1 * term2 * term3

    # Perform the 3D integration over (ρi, ρs, θ) as in Eq. (26)
    # The Jacobian factor (ρs * ρi) converts from Cartesian to polar coordinates
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

    # Calculate K2 with normalization factors from Eq. (26)
    # The factor (8/π⁵) comes from the coordinate transformations and normalizations
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
    """Calculate the single photon coupling efficiency K1 for heralding ratio calculation.

    This function implements Equation (39) from Smirr et al., which describes the
    single-photon coupling probability P1. This is used to calculate the heralding
    ratio Γ2|1 = K2/K1 (Eq. 41).

    The calculation involves:
    1. Inner integral over signal photon modes (with spatial filtering)
    2. Outer integral over idler photon modes (free space)
    3. Phase matching for the combined signal-idler system

    Args:
        xi (float): Focusing parameter ξ = L/(2zR).
        alpha_sq (float): Squared normalized target mode waist parameter α².
        phi_0 (float): Longitudinal phase mismatch φ0 = Δk0·L.
        np_over_ns (float): Ratio of refractive indices np/ns.
        np_over_ni (float): Ratio of refractive indices np/ni.
        rho_bounds (Tuple[float, float], optional): Integration bounds for normalized
            radial coordinates. Defaults to (0, 2).
        theta_bounds (Tuple[float, float], optional): Integration bounds for angular
            coordinate. Defaults to (-π, π).

    Returns:
        float: Dimensionless single photon coupling factor K1/(kp0·L).

    Note:
        Reference: Smirr et al., Eq. (39), page 6:
        K1 = ∫∫ d²κi/(2π)² |∫∫ d²κs/(2π)² S̃p(κs + κi, 0)Õ0(κs, z0)
             × e^(-iz0 np/n's |κs|²/kp0) L sinc(ΔK L/2)|²

        K1 represents the probability of detecting at least one photon in the target
        spatial mode, which is needed to calculate the conditional probability (heralding
        ratio) of detecting the second photon given that the first was detected.
    """

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
        """Integrand Q1 for single-photon coupling calculation.

        This implements the integrand from Eq. (39), representing the amplitude for
        coupling one photon (signal) to the target mode while the idler remains in free space.
        """
        rho_s_sq = rho_s**2
        rho_i_sq = rho_i**2
        rho_rho_cos = 2 * rho_s * rho_i * np.cos(theta)

        # Term 1: Combined pump-target and SPDC correlation
        # For single photon coupling, we have:
        # - Signal photon: spatial filtering (factor α²) + pump overlap + SPDC correlation
        # - Idler photon: only pump overlap + SPDC correlation (no spatial filtering)
        term1 = np.exp(-rho_i_sq - (1 + alpha_sq) * rho_s_sq - rho_rho_cos)

        # Term 2: Additional correlations (set to 1 for this implementation)
        # Could include additional mode coupling effects if needed
        term2 = 1

        # Term 3: Single-mode coupling for signal only
        # Additional spatial filtering factor for the signal photon
        # (Currently set to 1, could be expanded for more complex target modes)
        term3 = 1

        # Term 4: Phase matching condition (same as in K2 calculation)
        # Uses the same phase mismatch formula from Eq. (21)
        phase = (phi_0 / 2) + xi * (
            ((1 - (2 * np_over_ns)) * rho_s_sq)
            + ((1 - (2 * np_over_ni)) * rho_i_sq)
            + rho_rho_cos
        )
        term4 = np.sinc(phase / np.pi)

        return term1 * term2 * term3 * term4

    def Q1_inner(rho_i: float) -> float:
        """Inner integral over signal photon modes for fixed idler mode.

        This corresponds to the inner integral in Eq. (39), which is then squared
        to get the probability amplitude.
        """
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

    # Outer integral over idler photon modes
    # This completes the calculation of Eq. (39)
    Q1, _ = quad(
        lambda rho_i: rho_i * Q1_inner(rho_i),
        *rho_bounds,
    )

    # Calculate K1 with normalization factors from Eq. (39)
    # The factor (4/π⁴) comes from the coordinate transformations and normalizations
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
    """Calculate brightness factors K1, K2 and heralding ratio for SPDC source optimization.

    This function combines the calculations from Equations (26), (39), and (41) in
    Smirr et al. to provide the key figures of merit for SPDC source performance.

    Args:
        xi (float): Focusing parameter ξ = L/(2zR), where L is crystal length and zR is
            pump beam Rayleigh range. Typical range: 0.1 to 10.
        alpha (float): Normalized target mode waist α = a0/w0, where a0 is target mode
            waist and w0 is pump beam waist. Optimal value around 1-2.
        phi_0 (float): Longitudinal phase mismatch φ0 = Δk0·L. Optimization parameter
            for temperature tuning.
        n_p (float): Pump refractive index at pump wavelength.
        n_i (float): Idler refractive index at idler wavelength.
        n_s (float): Signal refractive index at signal wavelength.
        rho_max (float, optional): Maximum integration bound for normalized radial
            coordinates. Defaults to 2.0, which provides good convergence.

    Returns:
        Tuple[float, float, float]: A tuple containing:
            - K1 (float): Single photon coupling factor K1/(kp0·L). Related to heralded
              single photon brightness.

            - K2 (float): Photon pair coupling factor K2/(kp0·L). Related to coincidence
              brightness.

            - heralding (float): Heralding ratio Γ2|1 = K2/K1. Measures the conditional
              probability of detecting the idler photon given detection of the signal photon.

    Note:
        References:
        - K2: Smirr et al., Eq. (26) - spatial filtering factor for pair collection
        - K1: Smirr et al., Eq. (39) - spatial filtering factor for single photon
        - Γ2|1: Smirr et al., Eq. (41) - heralding ratio for single photon sources

        The heralding ratio Γ2|1 is a key figure of merit for single photon sources,
        as it determines the quality of heralded single photons. Values closer to 1
        indicate better performance, with typical experimental values of 0.5-0.9.

        For entangled pair sources, both K1 and K2 should be maximized to optimize
        the coincidence detection rate while maintaining good spatial mode purity.
    """
    # Convert to ratios needed for the integration functions
    np_over_ns = n_p / n_s
    np_over_ni = n_p / n_i
    alpha_sq = alpha**2

    # Set integration bounds (symmetric around origin)
    rho_bounds = (0, rho_max)

    # Calculate single photon coupling factor K1 (Eq. 39)
    K1 = single_photon_coupling_filter(
        xi, alpha_sq, phi_0, np_over_ns, np_over_ni, rho_bounds
    )

    # Calculate photon pair coupling factor K2 (Eq. 26)
    K2 = photon_pair_coupling_filter(
        xi, alpha_sq, phi_0, np_over_ns, np_over_ni, rho_bounds
    )

    # Calculate heralding ratio Γ2|1 = K2/K1 (Eq. 41)
    # This represents the conditional probability of detecting the second photon
    # given that the first photon was detected in the target spatial mode
    heralding = K2 / K1

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
    """Calculate optimization grids for SPDC source design across parameter space.

    This function systematically evaluates the brightness factors K1, K2 and heralding
    ratio across ranges of the normalized target mode waist (α) and longitudinal phase
    mismatch (φ0) parameters. This enables optimization of SPDC sources as described
    in Section III.C of Smirr et al.

    The results can be used to generate contour plots like Figures 3, 4, and 6 in the
    paper, showing the optimal parameter combinations for different focusing conditions.

    Args:
        xi (float): Focusing parameter ξ = L/(2zR). Different values correspond to
            different pump beam focusing conditions (see Fig. 4 in paper).
        alpha_range (Union[float, NDArray[np.floating]]): Range of normalized target
            mode waist values α = a0/w0 to evaluate. Typical range: 0.5 to 3.5
            (see Fig. 3 in paper).
        phi0_range (Union[float, NDArray[np.floating]]): Range of longitudinal phase
            mismatch values φ0 = Δk0·L to evaluate. This parameter is typically
            optimized via crystal temperature control.
        n_p (float): Pump refractive index.
        n_s (float): Signal refractive index.
        n_i (float): Idler refractive index.
        rho_max (float, optional): Maximum integration bound for radial coordinates.
            Defaults to 2.0.
        progress_bar (bool, optional): Whether to display progress bar during calculation.
            Defaults to True.

    Returns:
        Tuple[NDArray, NDArray, NDArray]: A tuple containing:
            - K1_grid (NDArray): 2D array with shape (n_phi, n_alpha) of single photon
              coupling efficiency values for heralding calculations.
            - K2_grid (NDArray): 2D array with shape (n_phi, n_alpha) of photon pair
              coupling efficiency values for brightness optimization (Fig. 3).
            - ratio_grid (NDArray): 2D array with shape (n_phi, n_alpha) of heralding
              ratio Γ2|1 values for heralding optimization (Fig. 6).

    Note:
        This function enables the systematic optimization described in Section III.C:
        1. For brightness optimization: find (α, φ0) that maximizes K2_grid
        2. For heralding optimization: find (α, φ0) that maximizes ratio_grid
        3. For balanced optimization: find compromise between brightness and heralding

        The optimal parameters typically follow the Boyd-Kleinman conditions for
        second harmonic generation, adapted for the down-conversion geometry.
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

    # Systematic evaluation across the (φ0, α) parameter space
    for i, phi0 in enumerate(phi0_range):
        for j, alpha in enumerate(alpha_range):
            # Calculate all figures of merit for this parameter combination
            K1, K2, ratio = brightness_and_heralding(
                xi=xi,
                alpha=alpha,
                phi_0=phi0,
                n_p=n_p,
                n_i=n_i,
                n_s=n_s,
                rho_max=rho_max,
            )

            # Store results in grids for contour plotting and optimization
            K1_grid[i, j] = K1  # For heralding ratio calculations
            K2_grid[i, j] = K2  # For brightness optimization (Fig. 3)
            ratio_grid[i, j] = ratio  # For heralding optimization (Fig. 6)

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
    """Generate lookup tables for SPDC optimization across focusing parameter range.

    This function creates comprehensive lookup tables that enable fast interpolation
    of brightness and heralding calculations across different focusing conditions.
    This is essential for the systematic optimization described in Section III of
    Smirr et al., where multiple ξ values must be evaluated to find global optima.

    Args:
        xi_values (Union[float, NDArray[np.floating]]): Range of focusing parameter
            values ξ = L/(2zR) to calculate. Typical range: 0.1 to 10 (from collimated
            to tightly focused).
        alpha (Union[float, NDArray[np.floating]]): Range of normalized target mode
            waist values α = a0/w0.
        phi_0 (Union[float, NDArray[np.floating]]): Range of longitudinal phase
            mismatch values φ0 = Δk0·L.
        crystal (citrine.Crystal): Crystal object containing material properties and
            Sellmeier equations.
        wavelength_pump (citrine.Wavelength): Wavelength object for pump beam.
        wavelength_signal (citrine.Wavelength): Wavelength object for signal beam.
        wavelength_idler (citrine.Wavelength): Wavelength object for idler beam.
        name (Optional[str], optional): Output filename for the lookup table pickle file.
            Defaults to None (auto-generated).
        temp_dir (Optional[str], optional): Directory for temporary storage of
            intermediate results. Defaults to None.
        rho_max (float, optional): Maximum radial integration bound. Defaults to 2.0.
        temperature (float, optional): Crystal temperature for refractive index
            calculations. Defaults to None.
        overwrite (bool, optional): Whether to overwrite existing intermediate files.
            Defaults to False.

    Returns:
        Tuple[BrightnessHeraldingLUT, str]: A tuple containing:
            - lookup_table (BrightnessHeraldingLUT): Lookup table object containing all
              calculated grids and parameters.
            - filename (str): Path to the saved lookup table file.

    Note:
        The lookup table enables:
        1. Fast parameter optimization without recalculating integrals
        2. Interpolation between calculated points for smooth optimization
        3. Systematic comparison across different focusing regimes

        The lookup table structure enables the optimization workflow from Fig. 4:
        1. Calculate grids for multiple ξ values
        2. Find optimal (α, φ0) for each ξ
        3. Determine global optimum across all ξ values
        4. Use interpolation for fine-tuning around optima
    """
    if temp_dir is None:
        temp_dir = f'optimsiation_grids_{crystal.name}'

    if not os.path.exists(temp_dir):
        os.mkdir(temp_dir)

    # Calculate refractive indices for the crystal at specified wavelengths
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
        """Calculate and save optimization grids for a single ξ value."""
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

        return k1, k2, heralding

    # Process each focusing parameter value
    for xi in xi_values:
        data_filename = f'{temp_dir}/xi_{xi}.npz'
        file_exists = os.path.exists(data_filename)

        if file_exists:
            if overwrite is True:
                k1, k2, heralding = calculate(xi)
            else:
                # Load existing results to avoid recalculation
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

    # Compile complete simulation results
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

    # Generate filename with timestamp if not provided
    if name is None:
        datestamp = datetime.datetime.now().strftime('%Y_%m_%d:%H:%M')
        name = f'brightness-and-heralding-{crystal.name}-lut-{datestamp}.pkl'

    # Save lookup table to file
    with open(name, 'wb') as f:
        pickle.dump(simulation_results, f)

    return BrightnessHeraldingLUT(**simulation_results), name


@dataclass(frozen=True)
class BrightnessHeraldingLUT:
    """Lookup table for SPDC brightness and heralding optimization.

    This class encapsulates the complete parameter space calculations needed for
    SPDC source optimization as described in Smirr et al. It provides methods for
    interpolating between calculated points and extracting optimal parameters.

    The lookup table enables rapid optimization across the multidimensional parameter
    space (ξ, α, φ0) without recalculating expensive integrals.

    Attributes:
        refractive_index (Dict[str, float]): Refractive indices for pump, signal,
            and idler wavelengths.
        wavelength (Dict[str, citrine.Wavelength]): Wavelength specifications for
            the three interacting beams.
        temperature (float): Crystal temperature used for calculations.
        xi (Union[float, NDArray[np.floating]]): Focusing parameter values ξ = L/(2zR).
        alpha (Union[float, NDArray[np.floating]]): Normalized target mode waist
            values α = a0/w0.
        phi (Union[float, NDArray[np.floating]]): Longitudinal phase mismatch
            values φ0 = Δk0·L.
        K1 (NDArray[np.floating]): Single photon coupling efficiency grids with
            shape (n_xi, n_phi, n_alpha).
        K2 (NDArray[np.floating]): Photon pair coupling efficiency grids with
            shape (n_xi, n_phi, n_alpha).
        Heralding (NDArray[np.floating]): Heralding ratio grids Γ2|1 = K2/K1 with
            shape (n_xi, n_phi, n_alpha).
    """

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
        """Load lookup table from pickled file.

        Args:
            path (str): Path to the pickled lookup table file.

        Returns:
            BrightnessHeraldingLUT: Loaded lookup table object.
        """
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
        """Create interpolated grids for specified focusing parameter values.

        This method enables fine-grained optimization by interpolating between
        the calculated lookup table points. This is particularly useful for
        finding precise optimal parameters around the global maxima identified
        from the coarse grid calculations.

        Args:
            target_xi_values (Union[float, List[float], NDArray[np.floating]]):
                Focusing parameter values for which to create interpolated grids.
                These can be between the original calculated ξ values.

        Returns:
            Tuple[NDArray[np.floating], NDArray[np.floating], NDArray[np.floating]]:
                A tuple containing:
                - interpolated_grids (dict): Dictionary with ξ values as keys, each
                  containing 'K2_grid' and 'heralding_grid' for interpolated
                  brightness and heralding optimization grids.
                - alpha_range (NDArray): Alpha values used in the grids (from
                  original lookup table).
                - phi0_range (NDArray): Phi0 values used in the grids (from
                  original lookup table).

        Note:
            The interpolation uses RegularGridInterpolator for smooth, continuous
            parameter optimization. This enables finding optimal parameters with
            higher precision than the original grid spacing.

            Warning will be issued if target ξ values are outside the lookup table range.
        """
        # Extract data dimensions and ranges from lookup table
        xi_lut = self.xi
        alpha_range = self.alpha
        phi0_range = self.phi
        K2_lut = self.K2  # Shape: (n_xi, n_phi0, n_alpha)
        heralding_lut = self.Heralding  # Shape: (n_xi, n_phi0, n_alpha)

        # Create 3D interpolators over (ξ, φ0, α) parameter space
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

        # Generate interpolated grids for each target ξ value
        for target_xi in target_xi_values:
            # Check if target ξ is within lookup table bounds
            if target_xi < xi_lut.min() or target_xi > xi_lut.max():
                print(
                    f'Warning: ξ = {target_xi} is outside the lookup table range '
                    f'[{xi_lut.min():.3f}, {xi_lut.max():.3f}]'
                )

            # Create meshgrid for the (φ0, α) parameter space at fixed ξ
            phi_mesh, alpha_mesh = np.meshgrid(
                phi0_range, alpha_range, indexing='ij'
            )
            xi_mesh = np.full_like(phi_mesh, target_xi)

            # Prepare coordinates for interpolation
            coords = np.stack(
                [xi_mesh.ravel(), phi_mesh.ravel(), alpha_mesh.ravel()],
                axis=-1,
            )

            # Perform interpolation and reshape to grid format
            K2_interp = interpolator_K2(coords).reshape(phi_mesh.shape)
            heralding_interp = interpolator_heralding(coords).reshape(
                phi_mesh.shape
            )

            # Store interpolated results
            interpolated_grids[target_xi] = {
                'K2_grid': K2_interp,
                'heralding_grid': heralding_interp,
            }

            print(f'Created interpolated grids for ξ = {target_xi}')

        return interpolated_grids, alpha_range, phi0_range
