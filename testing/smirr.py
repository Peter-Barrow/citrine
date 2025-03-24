import numpy as np
import matplotlib.pyplot as plt
import matplotlib
from scipy import integrate
import numba as numba
from tqdm import tqdm

matplotlib.use('TkAgg')


@numba.njit()
def Q2_integrand(theta, rho_i, rho_s, xi, alpha, phi_0):
    """
    Implements the Q2 integrand from equation 58 of the paper, with each term
    corresponding to elements in Figure 1's experimental setup.

    Q₂' = exp{-(1+α²)(ρₛ²+ρᵢ²)}
        × exp{-2ρₛρᵢcos(θₛ-θᵢ)}
        × exp{i(-4ξζ)(np/nₛ'ρₛ² + np/nᵢ'ρᵢ²)}
        × sinc{φ₀/2 + ξ[(1-2np/ns)ρₛ² + (1-2np/ni)ρᵢ² + 2ρₛρᵢcos(θₛ-θᵢ)]}

    Each term corresponds to a physical component in Figure 1:

    Term 1: exp{-(1+α²)(ρₛ²+ρᵢ²)}
    - Represents overlap between pump beam and target spatial mode
    - Related to "Pump" (red) and "Spatial filtering" in Fig. 1
    - α = a₀/w₀ is ratio of fiber mode waist to pump beam waist

    Term 2: exp{-2ρₛρᵢcos(θₛ-θᵢ)}
    - Represents signal-idler photon correlations in transverse momentum
    - Related to SPDC process in the "crystal" in Fig. 1

    Term 3: exp{i(-4ξζ)(np/nₛ'ρₛ² + np/nᵢ'ρᵢ²)}
    - Phase term for spatial filtering position
    - Related to distance between crystal and coupling position in Fig. 1
    - ζ = z₀/L is the normalized position of the coupling

    Term 4: sinc{φ₀/2 + ξ[(1-2np/ns)ρₛ² + (1-2np/ni)ρᵢ² + 2ρₛρᵢcos(θₛ-θᵢ)]}
    - Phase matching function in the nonlinear crystal
    - Related to PPLN crystal structure shown in Fig. 1
    - φ₀ controlled by crystal temperature

    Parameters are rearranged for scipy integration compatibility.
    """

    rho_s_sq = rho_s**2
    rho_i_sq = rho_i**2
    rho_rho_cos = 2 * rho_s * rho_i * np.cos(theta)

    # Term 1: Pump-target mode overlap (Pump beam + Spatial filtering in Fig.1)
    term1 = np.exp(-(1 + alpha**2) * (rho_s_sq + rho_i_sq))

    # Term 2: Signal-idler correlation (SPDC in crystal in Fig. 1)
    term2 = np.exp(-rho_rho_cos)
    # term2 = 1

    # Term 3: Spatial filtering phase (Position of coupling in Fig. 1)
    # For this implementation, we use ζ = 0 (centered collection)
    # With ζ = 0, this term becomes exp(i0) = 1
    # In general form: exp{i(-4ξζ)(np/nₛ'ρₛ² + np/nᵢ'ρᵢ²)}
    zeta = 0  # centered collection
    n_p = 2.1779
    n_i = 2.1374
    n_s = 2.1374
    # np_over_ns_prime = n_p / n_s  # assuming equal refractive indices
    # np_over_ni_prime = n_p / n_i  # assuming equal refractive indices
    # term3 = np.exp(
    #     1j
    #     * (-4 * xi * zeta)
    #     * (np_over_ns_prime * rho_s**2 + np_over_ni_prime * rho_i**2)
    # )
    # term3 = np.abs(term3) ** 2
    term3 = 1

    # Term 4: Phase matching (PPLN crystal in Fig. 1)
    # Using the general form from the paper:
    # sinc{φ₀/2 + ξ[(1-2np/ns)ρₛ² + (1-2np/ni)ρᵢ² + 2ρₛρᵢcos(θₛ-θᵢ)]}
    np_over_ns = n_p / n_s  # For degenerate case where np=ns=1
    np_over_ni = n_p / n_i  # For degenerate case where np=ni=1

    # When np_over_ns = np_over_ni = 1:
    # (1-2np/ns) = (1-2) = -1 and (1-2np/ni) = (1-2) = -1
    phase = (phi_0 / 2) + xi * (
        ((1 - (2 * np_over_ns)) * rho_s_sq)
        + ((1 - (2 * np_over_ni)) * rho_i_sq)
        + rho_rho_cos
    )
    term4 = np.sinc(phase / np.pi)

    # Return squared absolute value with Jacobian for spherical coordinates
    # As required by Eq. 26 for K₂ calculation, we need |Q₂|²ρₛρᵢ
    # return np.abs(term1 * term2 * term3 * term4) ** 2  # * rho_s * rho_i
    return term1 * term2 * term3 * term4 * rho_s * rho_i  # ) ** 2


@numba.njit()
def Q1_integrand(theta, rho_i, rho_s, xi, alpha, phi_0):
    """
    Implements the Q1 integrand derived from equations in the paper for K1/(kp0*L) computation.
    This represents the case where only the signal photon is coupled to the target spatial mode.

    The terms correspond to elements in Figure 1, but with coupling for only one photon:

    Term 1: exp{-(ρₛ²+ρᵢ²)}
    - Represents pump beam profile without the α² factor that's in Q₂
    - Related to "Pump" (red) in Fig. 1

    Term 2: exp{-2ρₛρᵢcos(θₛ-θᵢ)}
    - Represents signal-idler photon correlations
    - Related to SPDC process in the "crystal" in Fig. 1

    Term 3: exp{-α²ρₛ²}
    - Single-mode coupling for signal photon only
    - Related to "Spatial filtering" in Fig. 1, but only for Channel A

    Term 4: Phase matching sinc function
    - Phase matching in the nonlinear crystal
    - Related to PPLN crystal structure shown in Fig. 1

    Parameters are rearranged for scipy integration compatibility.
    """

    # Q'₁ = exp{-(ρ²ₛ+ρ²ᵢ+2ρₛρᵢcos(θₛ-θᵢ))}
    #       × exp{-α²ρ²ₛ}
    #       × sinc{(φ₀/2) - ξ[(ρ²ₛ+ρ²ᵢ+2ρₛρᵢcos(θₛ-θᵢ)) - 2nₚ/nₛρ²ₛ - 2nₚ/nᵢρ²ᵢ]}

    rho_s_sq = rho_s**2
    rho_i_sq = rho_i**2
    rho_rho_cos = 2 * rho_s * rho_i * np.cos(theta)

    # Term 1: Pump beam profile (Pump beam in Fig. 1)
    # term1 = np.exp(-(x + y + (2 * rho_s * rho_i * np.cos(theta))))
    # term1 = np.exp(-(1 + alpha**2) * (x + y))
    # term1 = np.exp(-2 * rho_s * rho_i * np.cos(theta))
    term1 = np.exp(-rho_i_sq - (1 + alpha**2) * rho_s_sq - rho_rho_cos)

    # theta = theta_s - theta_i

    # Term 2: Signal-idler correlation (SPDC in crystal in Fig. 1)
    # term2 = np.exp(-2 * rho_s * rho_i * np.cos(theta))
    term2 = 1

    # Term 3: Single-mode coupling for signal only (Spatial filtering for one channel in Fig.1)
    # term3 = np.exp(-(alpha**2) * rho_s * rho_s)
    term3 = 1

    # Term 4: Phase matching (PPLN crystal in Fig. 1)
    # Using the general form from the paper:
    # sinc{φ₀/2 + ξ[(1-2np/ns)ρₛ² + (1-2np/ni)ρᵢ² + 2ρₛρᵢcos(θₛ-θᵢ)]}
    n_p = 2.1779
    n_i = 2.1374
    n_s = 2.1374
    np_over_ns = n_p / n_s  # assuming equal refractive indices
    np_over_ni = n_p / n_i  # assuming equal refractive indices

    phase = (phi_0 / 2) + xi * (
        ((1 - (2 * np_over_ns)) * rho_s_sq)
        + ((1 - (2 * np_over_ni)) * rho_i_sq)
        + rho_rho_cos
    )
    # phase = (phi_0 / 2) + xi * (
    #     ((rho_s**2) + (rho_i**2) + (2 * rho_s * rho_i * np.cos(theta)))
    #     - (2 * np_over_ns * (rho_s**2))
    #     - (2 * np_over_ni * (rho_i**2))
    # )
    term4 = np.sinc(phase / np.pi)

    # Return squared absolute value with Jacobian for spherical coordinates
    # return np.abs(term1 * term2 * term3 * term4) ** 2  # * rho_s * rho_i
    return term1 * term2 * term3 * term4


def compute_K2_K1_ratio_scipy(xi, alpha, phi_0, rho_max=2.0, epsrel=1.49e-8):
    """
    Compute the heralding ratio Γ₂|₁ = K₂/K₁ from Eq. 41 using scipy's tplquad integration.

    This represents the physical quantity shown in Figure 8 of the paper:
    - The probability of detecting the idler photon in the target spatial mode
      when the signal is already detected in that mode
    - Essentially measures how well the source can herald single photons

    In Figure 1, this corresponds to the conditional probability of detecting a photon
    in Channel B given a detection in Channel A.

    K₂ is computed according to Eq. 26:
    K₂ = ∫∫d²φₛ∫∫d²φᵢ|Q₂|²

    K₁ is computed according to Eq. 39 (similar form with different integrand).

    The prefactors come from:
    - For K₂: 8/(π⁵) × ξ × α⁴ from Eq. 55
    - For K₁: 4/(π⁴) × ξ × α² (follows same pattern with adjusted dimensions)
    """
    # Define integration bounds
    rho_bounds = [0, rho_max]  # Same bounds for both rho_s and rho_i
    theta_bounds = [0, 2 * np.pi]

    # Using scipy's tplquad for triple integration to compute K₂
    K2_result, K2_error = integrate.tplquad(
        Q2_integrand,
        theta_bounds[0],
        theta_bounds[1],  # theta bounds
        rho_bounds[0],  # rho_i lower bound
        rho_bounds[1],  # rho_i upper bound
        rho_bounds[0],  # rho_s lower bound
        rho_bounds[1],  # rho_s upper bound
        args=(xi, alpha, phi_0),
        epsrel=epsrel,
    )

    def Q1_inner(rho_i):
        result, _ = integrate.nquad(
            lambda rho_s, theta: rho_s
            * Q1_integrand(theta, rho_i, rho_s, xi, alpha, phi_0),
            [[0, rho_max], [-np.pi, np.pi]],
        )
        return result**2

    K1_result, _ = integrate.quad(
        lambda rho_i: rho_i * Q1_inner(rho_i),
        0,
        rho_max,
    )

    # Apply prefactors from Eq. 55 for K₂ and analogous equation for K₁
    K2_prefactor = (8 / (np.pi**5)) * xi * (alpha**4)  # From Eq. 55
    K1_prefactor = (4 / (np.pi**4)) * xi * alpha**2  # Analogous to K₂ prefactor
    # K2_prefactor = 1
    # K1_prefactor = 1

    K2 = K2_prefactor * ((2 * np.pi * K2_result) ** 2)
    K1 = K1_prefactor * K1_result * 2 * np.pi

    # Return heralding ratio - Γ₂|₁ = K₂/K₁ from Eq. 41
    return K2 / K1, K1, K2


def plot_figure8_scipy(phi0_values=None, rho_max=2.0, num_points=30):
    """
    Reproduce Figure 8 from the paper using scipy's tplquad integration.

    Figure 8 in the paper shows the comparison between theoretical predictions (solid lines)
    and experimental measurements (dots) for the heralding ratio Γ₂|₁ = K₂/K₁ (Eq. 41)
    as a function of the normalized target mode waist α = a₀/w₀ (Eq. 49)
    for two different focusing parameters: ξ = 0.76 and ξ = 2.7.

    This directly corresponds to the configuration in Figure 1:
    - Varying α means changing the size of the "Spatial filtering" mode
      relative to the pump beam size
    - ξ values represent different focusing strengths of the pump beam
    - Heralding ratio (Γ₂|₁) is the probability of detecting an idler photon
      in Channel B when a signal photon is detected in Channel A

    The phi₀ values (2.0 and 3.2) come from experimental optimization described
    in the paper's section IV (as mentioned on page 9).

    Parameters:
    -----------
    phi0_values : dict, optional
        Dictionary with optimal phi0 values for each xi, e.g., {0.76: 2.0, 2.7: 3.2}
        If None, default values from the paper will be used
    rho_max : float
        Maximum integration bound for rho (transverse wavevector magnitude)
    num_points : int
        Number of points to calculate along the alpha axis
    """
    if phi0_values is None:
        # Default phi0 values from the paper (Fig. 8)
        phi0_values = {0.76: 2.0, 2.7: 3.2}

    # Define parameters from the paper
    xi_values = [0.76, 2.7]  # Two focal lengths corresponding to ξ = L/(2zR)
    alpha_range = np.linspace(0.4, 4.0, num_points)  # α = a₀/w₀ from Eq. 49

    # Experimental data points from Figure 8 (estimated from the paper)
    # Format: [alpha, heralding_ratio]
    exp_data_xi076 = [
        [0.7, 0.7],
        [0.8, 0.8],
        [1.0, 0.95],
        [1.2, 0.9],
        [1.5, 0.7],
        [2.0, 0.5],
        [2.5, 0.45],
        [3.0, 0.25],
    ]

    exp_data_xi27 = [
        [0.8, 0.7],
        [1.0, 0.8],
        [1.25, 0.95],
        [1.5, 0.95],
        [1.8, 0.85],
        [2.0, 0.6],
        [2.5, 0.6],
    ]

    experimental_data = [exp_data_xi076, exp_data_xi27]

    # Create figure with two subplots
    fig, axes = plt.subplots(2, 1, figsize=(10, 12))

    # Loop through xi values
    for i, (xi, exp_data) in enumerate(zip(xi_values, experimental_data)):
        phi0 = phi0_values[xi]

        # Compute theoretical curve - implementing Γ₂|₁ from Eq. 41
        print(f'Computing theoretical curve for ξ = {xi}, φ₀ = {phi0}')
        heralding_ratios = []
        K1 = []
        K2 = []

        for alpha in tqdm(alpha_range, desc=f'Alpha values for ξ = {xi}'):
            ratio, k1, k2 = compute_K2_K1_ratio_scipy(
                xi=xi, alpha=alpha, phi_0=phi0, rho_max=rho_max
            )
            heralding_ratios.append(ratio)
            K1.append(k1)
            K2.append(k2)

        # Extract experimental data
        exp_alpha, exp_ratio = zip(*exp_data)

        # Plot experimental data
        axes[i].plot(exp_alpha, exp_ratio, 'o', markersize=8, color='orange')

        # # Find the maximum of the theoretical curve
        # max_theoretical = max(heralding_ratios)
        # max_experimental = max(exp_ratio)

        # # Scale the theoretical curve to match experimental maximum
        # # Note: The paper doesn't specifically mention this scaling, but it's common practice
        # # when comparing theoretical and experimental curves
        # scaling_factor = (
        #     max_experimental / max_theoretical if max_theoretical > 0 else 1
        # )
        # # heralding_ratios = np.array(heralding_ratios) * scaling_factor

        # Plot theoretical curve
        axes[i].plot(
            alpha_range,
            heralding_ratios / np.max(heralding_ratios),
            '-',
            linewidth=2,
        )

        # Add labels and title
        axes[i].set_xlabel(r'$\alpha$')
        axes[i].set_ylabel(r'$\Gamma_{2|1}$')
        axes[i].set_xlim(0.4, 4.0)
        axes[i].set_ylim(0, 1.0)
        axes[i].grid(True)

        # Add experiment parameters - these values are from the experimental setup in section IV
        if i == 0:
            axes[i].set_title(
                f'ξ = {xi}, φ₀ = {phi0}, w₀ = 37.9 μm, fᵢ = 100 mm'
            )
        else:
            axes[i].set_title(
                f'ξ = {xi}, φ₀ = {phi0}, w₀ = 20.1 μm, fᵢ = 50 mm'
            )
        # axes[i + 1].plot(alpha_range, K1, label='K1')
        # axes[i + 1].plot(alpha_range, K2, label='K2')
        # axes[i + 1].legend()
        # axes[i + 1].set_xlim(0.4, 4.0)
        # axes[i + 2].plot(alpha_range, heralding_ratios)
        # axes[i + 2].set_xlim(0.4, 4.0)

    plt.tight_layout()
    plt.suptitle(
        'Figure 8: Comparison of theoretical vs. experimental heralding ratio',
        fontsize=14,
        y=1.02,
    )
    return fig, axes


if __name__ == '__main__':
    # Set random seed for reproducibility
    np.random.seed(42)

    # Option 1: Full integration approach
    print('Generating Figure 8 using numerical integration...')
    fig8_int, axes8_int = plot_figure8_scipy(num_points=30, rho_max=2)

    plt.show()
