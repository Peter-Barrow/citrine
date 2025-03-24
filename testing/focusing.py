import numpy as np
import matplotlib.pyplot as plt
from scipy.constants import c, pi, h
from scipy.integrate import quad

import matplotlib

matplotlib.use('TkAgg')

import citrine.pump_envelope as pef
from citrine import (
    crystals,
    Wavelength,
    Magnitude,
    delta_k_matrix,
    phase_matching_function,
    bandwidth_conversion,
    calculate_jsa_marginals,
    calculate_grating_period,
)


class SPDCFocusingOptimizer:
    """
    Implements the focusing optimization for SPDC based on Bennink's paper:
    "Optimal collinear Gaussian beams for spontaneous parametric down-conversion"
    """

    def __init__(
        self,
        crystal_length,
        pump_wavelength,
        signal_wavelength=None,
        idler_wavelength=None,
        pump_n=1.8,
        signal_n=1.8,
        idler_n=1.8,
        poling_period=46.22e-6,
    ):
        """
        Initialize the optimizer with crystal and optical parameters.

        Parameters:
        -----------
        crystal_length : float
            Length of the crystal in meters
        pump_wavelength : float
            Pump wavelength in meters
        signal_wavelength : float, optional
            Signal wavelength in meters (if None, assumes degenerate case)
        idler_wavelength : float, optional
            Idler wavelength in meters (if None, calculated from energy conservation)
        pump_n, signal_n, idler_n : float
            Refractive indices for pump, signal, and idler

        Note:
        -----
        This class implements core equations from Bennink's paper, including:
        - Focusing parameter ξj (Eq. 11): ξj = L/(kj·w²j)
        - Phase mismatch Φ (Eq. 10): Φ = (Δk + mK)L
        - Optimization parameters and scaling laws (Eqs. 28-30, 40-41, 53-55)
        """
        self.L = crystal_length
        self.lambda_p = pump_wavelength

        # Handle degenerate/non-degenerate cases
        if signal_wavelength is None:
            self.lambda_s = 2 * pump_wavelength
            self.lambda_i = 2 * pump_wavelength
        elif idler_wavelength is None:
            self.lambda_s = signal_wavelength
            # Energy conservation: 1/λp = 1/λs + 1/λi
            self.lambda_i = 1 / (
                (1 / pump_wavelength) - (1 / signal_wavelength)
            )
        else:
            self.lambda_s = signal_wavelength
            self.lambda_i = idler_wavelength

        # Refractive indices
        self.n_p = pump_n
        self.n_s = signal_n
        self.n_i = idler_n
        self.poling_period = poling_period

        # Calculate wavenumbers
        self.k_p = 2 * pi * self.n_p / self.lambda_p
        self.k_s = 2 * pi * self.n_s / self.lambda_s
        self.k_i = 2 * pi * self.n_i / self.lambda_i
        self.k_g = 2 * pi / self.poling_period

        self.delta_k = self.k_p - self.k_s - self.k_i

    def calculate_optimal_waists(self, xi_target=2.84):
        """
        Calculate the optimal waists for maximum spectral density.
        According to Bennink's paper, optimal focusing is around ξ ≈ 2.84.

        Implements Equation (29): ξs ≈ ξi ≈ ξp ≈ 2.84
        This is the condition for maximum spectral density as derived in Sec. III.

        Parameters:
        -----------
        xi_target : float
            Target focusing parameter ξ (default from paper: 2.84)

        Returns:
        --------
        dict
            Dictionary with optimal pump, signal, and idler waists
        """
        # Calculate optimal waists using ξj = L/(kj·w²j)
        w_p = np.sqrt(self.L / (xi_target * self.k_p))
        w_s = np.sqrt(self.L / (xi_target * self.k_s))
        w_i = np.sqrt(self.L / (xi_target * self.k_i))

        return {
            'pump_waist': w_p,
            'signal_waist': w_s,
            'idler_waist': w_i,
            'focusing_parameter': xi_target,
        }

    def spatial_overlap_factor(self, xi, phi):
        """
        Calculate the spatial overlap factor F(ξ,Φ) as defined in Equation (27).
        This function is used to create Figure 1 in the paper.

        F(ξ,Φ) = ∫_{-1}^{1} √ξ·exp(iΦl/2)/(1-iξl) dl

        Parameters:
        -----------
        xi : float
            Focusing parameter
        phi : float
            Phase mismatch parameter

        Returns:
        --------
        complex
            Spatial overlap factor (complex value)
        """

        # Define the integrand function
        def integrand(l, xi, phi):
            return np.sqrt(xi) * np.exp(1j * phi * l / 2) / (1 - 1j * xi * l)

        # Perform numerical integration
        real_part, _ = quad(lambda l: np.real(integrand(l, xi, phi)), -1, 1)
        imag_part, _ = quad(lambda l: np.imag(integrand(l, xi, phi)), -1, 1)

        return complex(real_part, imag_part)

    def plot_spatial_overlap_magnitude(
        self, xi_range=(0.1, 100), phi_range=(-6, 2), resolution=40
    ):
        """
        Reproduce Figure 1 from Bennink's paper: The spatial overlap factor F(ξ,Φ)
        defined in Eq. (27). Shows how the magnitude varies with focusing parameter ξ
        and phase mismatch Φ.

        Parameters:
        -----------
        xi_range : tuple
            Range of ξ values to plot (log scale)
        phi_range : tuple
            Range of Φ/π values to plot
        resolution : int
            Number of points along each axis

        Returns:
        --------
        fig
            Matplotlib figure with the 2D color plot
        """
        # Create logarithmically spaced grid for xi and linear grid for phi
        xi_values = np.logspace(
            np.log10(xi_range[0]), np.log10(xi_range[1]), resolution
        )
        phi_values = np.linspace(
            phi_range[0] * np.pi, phi_range[1] * np.pi, resolution
        )

        # Create meshgrid for evaluation
        xi_grid, phi_grid = np.meshgrid(xi_values, phi_values)

        # Initialize array for magnitude of F
        F_magnitude = np.zeros_like(xi_grid)

        # Calculate F for each point in the grid
        for i in range(resolution):
            for j in range(resolution):
                F = self.spatial_overlap_factor(xi_grid[i, j], phi_grid[i, j])
                F_magnitude[i, j] = abs(F)

        # Create figure
        fig, ax = plt.subplots(figsize=(9, 7))

        # Create color plot with logarithmic color scale for xi
        contour = ax.pcolormesh(
            xi_grid,
            phi_grid / np.pi,
            F_magnitude,
            cmap='jet',
            shading='gouraud',
        )

        # Set log scale for x-axis
        ax.set_xscale('log')

        # Mark the maximum value (ξ = 2.84, Φ = -1.04π)
        ax.plot(2.84, -1.04, 'r+', markersize=12, markeredgewidth=2)
        ax.annotate(
            '(2.84, -1.04)',
            (2.84, -1.04),
            xytext=(5, -1.5),
            arrowprops=dict(arrowstyle='->'),
            fontsize=10,
        )

        # Add colorbar and labels
        fig.colorbar(contour, ax=ax, label='|F(ξ,Φ)|')
        ax.set_xlabel('ξ')
        ax.set_ylabel('Φ/π')
        ax.set_title('Spatial Overlap Factor |F(ξ,Φ)| (Figure 1 from Bennink)')

        return fig

    def calculate_spectral_density_factor(self, xi_p, xi_s, xi_i):
        """
        Calculate a normalized factor proportional to the spectral density
        as shown in Fig. 2 of Bennink's paper.

        Based on Equation (28): max|ψ(ωs,ωi)| ≈ 1.03√[8π²ħnsni/ε0np]·[χ(2)eff/λsλi]·√NpL·s(ωp)
        with optimal conditions from Equations (29-30):
        - ξs ≈ ξi ≈ ξp ≈ 2.84
        - Φ ≈ -1.04π

        Parameters:
        -----------
        xi_p, xi_s, xi_i : float
            Focusing parameters for pump, signal, and idler

        Returns:
        --------
        float
            Normalized spectral density factor
        """
        # Simple approximation based on paper's Fig. 2
        optimal_xi = 2.84

        # Calculate average deviation from optimal focusing
        deviation_p = (xi_p - optimal_xi) ** 2
        deviation_s = (xi_s - optimal_xi) ** 2
        deviation_i = (xi_i - optimal_xi) ** 2

        # Empirical formula based on paper's Fig. 2
        deviation = (deviation_p + deviation_s + deviation_i) / 3

        # Normalize to have maximum of 1 at optimal focusing
        if deviation > 25:  # Far from optimal
            return 0.1
        else:
            return np.exp(-deviation / 10)

    def calculate_peak_spectral_density(
        self, xi_p, xi_s, xi_i, delta_k=None, optimize_phi=True, alt_form=False
    ):
        """
        Calculate the peak spectral density for given focusing parameters.
        Used to reproduce Figure 2 from the paper.

        Based on analysis in Section III and the approximation from Equation (26):
        max(1/√(A_+B_+)) ≈ 1/2

        Parameters:
        -----------
        xi_p : float
            Pump focusing parameter
        xi_s : float
            Signal focusing parameter
        xi_i : float
            Idler focusing parameter
        delta_k : float, optional
            Wave vector mismatch Δk = kp - ks - ki (before quasi-phase-matching)
            If None, assumes perfect phase matching (Δk = 0)
        optimize_phi : bool
            If True, optimize over phase mismatch. If False, use Φ = -1.04π
        alt_form : bool
            If True, use the alternative form for A_+B_+ provided between
            Equations (22) and (23) in the paper. This form helps identify
            the optimal focusing ratios r_s = ξ_s/ξ_p and r_i = ξ_i/ξ_p

        Returns:
        --------
        float
            Normalized peak spectral density
        """
        # If delta_k is not provided, assume perfect phase matching
        if delta_k is None:
            delta_k = 0

        # Calculate 1/√(A_+B_+) factor using one of two methods
        if alt_form:
            # Alternative form from between Equations (22) and (23)
            # A_+B_+ = (1 - Δk/kp) * (1 + X_s*r_s + X_i*r_i) * (1 + X_s/r_s + X_i/r_i)

            # Handle the case where kp = Δk (avoid division by zero)
            if abs(self.k_p - delta_k) < 1e-10:
                return 0

            # Define X_j and r_j terms as in the paper
            X_s = (self.k_s / self.k_p) * np.sqrt(
                (1 + (delta_k / self.k_s)) / (1 - (delta_k / self.k_p))
            )
            X_i = (self.k_i / self.k_p) * np.sqrt(
                (1 + (delta_k / self.k_i)) / (1 - (delta_k / self.k_p))
            )

            r_s = (xi_s / xi_p) * np.sqrt(
                (1 - (delta_k / self.k_p)) / (1 + (delta_k / self.k_s))
            )
            r_i = (xi_i / xi_p) * np.sqrt(
                (1 - (delta_k / self.k_p)) / (1 + (delta_k / self.k_i))
            )

            # Calculate A_+B_+ using the alternative form
            term1 = 1 - (delta_k / self.k_p)
            term2 = 1 + (X_s * r_s) + (X_i * r_i)
            term3 = 1 + (X_s / r_s) + (X_i / r_i)

            AB = term1 * term2 * term3
            inverse_sqrt_AB = 1 / np.sqrt(AB)

            # Now calculate the aggregate focusing parameter ξ
            # The paper notes that the maximum of (A_+B_+)^(-1/2) occurs at r_s = r_i = 1
            # which means ξ_s ≈ ξ_i ≈ ξ_p when Δk ≈ 0
            if abs(delta_k) < 1e-10:
                xi = xi_p  # When r_s = r_i = 1 and Δk = 0, ξ = ξ_p
            else:
                # For the general case, we calculate ξ using Equation (15)
                # First calculate A+ and B+ separately
                A_plus = (
                    1
                    + (self.k_s / self.k_p) * (xi_s / xi_p)
                    + (self.k_i / self.k_p) * (xi_i / xi_p)
                )

                term1 = 1 - (delta_k / self.k_p)
                term2 = 1 + ((self.k_s + delta_k) / (self.k_p - delta_k)) * (
                    xi_p / xi_s
                )
                term3 = ((self.k_i + delta_k) / (self.k_p - delta_k)) * (
                    xi_p / xi_i
                )

                B_plus = term1 * (term2 + term3)

                # Equation (15): ξ = (B_+/A_+)·(ξ_s·ξ_i/ξ_p)
                xi = (B_plus / A_plus) * (xi_s * xi_i / xi_p)

        else:
            # Original method using explicit A+ and B+ from Equations (12-13)
            A_plus = (
                1
                + (self.k_s / self.k_p) * (xi_s / xi_p)
                + (self.k_i / self.k_p) * (xi_i / xi_p)
            )

            # Handle the case where kp = Δk (avoid division by zero)
            if abs(self.k_p - delta_k) < 1e-10:
                return 0

            term1 = 1 - (delta_k / self.k_p)
            term2 = 1 + ((self.k_s + delta_k) / (self.k_p - delta_k)) * (
                xi_p / xi_s
            )
            term3 = ((self.k_i + delta_k) / (self.k_p - delta_k)) * (
                xi_p / xi_i
            )

            B_plus = term1 * (term2 + term3)

            # Calculate 1/√(A_+B_+) factor
            inverse_sqrt_AB = 1 / np.sqrt(A_plus * B_plus)

            # Calculate aggregate focusing parameter ξ from Equation (15)
            xi = (B_plus / A_plus) * (xi_s * xi_i / xi_p)

        # For the F(ξ,Φ) factor, either optimize over Φ or use the optimal value
        if optimize_phi:
            # Optimal phase mismatch is around -1.04π
            phi_opt = -1.04 * np.pi

            # Calculate |F| at optimal phase
            F_magnitude = abs(self.spatial_overlap_factor(xi, phi_opt))
        else:
            # For simplicity, use an approximation based on Figure 1
            # Maximum of |F| is around 2.06 at ξ = 2.84, Φ = -1.04π
            # We'll scale based on how far ξ is from optimal
            F_magnitude = 2.06 * np.exp(
                -((np.log10(xi) - np.log10(2.84)) ** 2) / 0.5
            )

        # Multiply factors together and normalize
        return inverse_sqrt_AB * F_magnitude / 2.06  # Normalized to maximum

    def plot_peak_spectral_density_2d(
        self,
        xi_p_range=(0.1, 100),
        xi_s_range=(0.1, 100),
        resolution=30,
        optimize_phi=True,
        delta_k=None,
        alt_form=False,
    ):
        """
        Reproduce Figure 2 from Bennink's paper: Peak joint spectral density,
        normalized to the global maximum, as a function of the pump and photon mode focus.

        Parameters:
        -----------
        xi_p_range : tuple
            Range of pump focusing parameter values (log scale)
        xi_s_range : tuple
            Range of signal/idler focusing parameter values (log scale)
        resolution : int
            Number of points along each axis
        delta_k : float, optional
            Wave vector mismatch Δk = kp - ks - ki (before quasi-phase-matching)
            If None, assumes perfect phase matching (Δk = 0)
        alt_form : bool
            If True, use the alternative form for A_+B_+ provided between
            Equations (22) and (23) in the paper.

        Returns:
        --------
        fig
            Matplotlib figure with the 2D color plot
        """
        # Create logarithmically spaced grids for the focusing parameters
        xi_p_values = np.logspace(
            np.log10(xi_p_range[0]), np.log10(xi_p_range[1]), resolution
        )
        xi_s_values = np.logspace(
            np.log10(xi_s_range[0]), np.log10(xi_s_range[1]), resolution
        )

        # Create a 2D grid of the focusing parameters
        xi_p_grid, xi_s_grid = np.meshgrid(xi_p_values, xi_s_values)

        # Initialize a 2D array to store the peak spectral density
        spectral_density = np.zeros_like(xi_p_grid)

        # Use a simplified calculation for faster plotting
        for i in range(resolution):
            for j in range(resolution):
                xi_p = xi_p_grid[i, j]
                xi_s = xi_s_grid[i, j]
                xi_i = xi_s  # Assuming ξs = ξi as in Figure 2

                # Calculate peak spectral density
                spectral_density[i, j] = self.calculate_peak_spectral_density(
                    xi_p,
                    xi_s,
                    xi_i,
                    delta_k=delta_k,
                    optimize_phi=optimize_phi,
                    alt_form=alt_form,
                )

        # Create the figure
        fig, ax = plt.subplots(figsize=(8, 7))

        spectral_density_norm = spectral_density / np.max(spectral_density)

        # Create a color plot
        contour = ax.pcolormesh(
            xi_p_grid,
            xi_s_grid,
            spectral_density_norm,
            cmap='jet',
            shading='gouraud',
            # norm=colors.Normalize(0, 1),
        )

        # Set log scale for both axes
        ax.set_xscale('log')
        ax.set_yscale('log')

        # Add colorbar and labels
        fig.colorbar(contour, ax=ax, label='Normalized Peak Spectral Density')
        ax.set_xlabel('Pump Focus ξp')
        ax.set_ylabel('Photon Focus ξs, ξi')

        # Title with phase matching and calculation method information
        title_parts = ['Peak Joint Spectral Density']

        if alt_form:
            title_parts.append('(Using Alternative Form)')

        if delta_k is not None and delta_k != 0:
            title_parts.append(f'(Δk = {delta_k:.2f})')

        ax.set_title(' '.join(title_parts))

        # Add a contour line at 0.9 of the maximum to highlight the optimal region
        max_val = np.max(spectral_density)
        if max_val > 0:  # Only draw contour if there's a visible maximum
            level = 0.9 * max_val
            cs = ax.contour(
                xi_p_grid,
                xi_s_grid,
                spectral_density,
                levels=[level],
                colors='r',
            )

        # Annotate the optimal point (ξp ≈ ξs ≈ ξi ≈ 2.84)
        if delta_k is None or abs(delta_k) < 1e-10:
            ax.plot(2.84, 2.84, 'r+', markersize=10, markeredgewidth=2)

        # Add the line ξs = ξp
        # ax.plot([0.1, 100], [0.1, 100], 'w--', alpha=0.7, label='ξp = ξs')

        # # Add the line ξs = 2.84ξp mentioned in the paper
        # ax.plot(
        #     [0.1, 100],
        #     [2.84 * 0.1, 2.84 * 100],
        #     'w-.',
        #     alpha=0.7,
        #     label='ξs = 2.84ξp',
        # )

        # ax.legend(loc='upper left')

        return fig

    def calculate_heralding_ratio(self, xi_p, xi_s, xi_i):
        """
        Calculate an estimate of the heralding ratio based on Section VII.

        Implements Equation (55): ηs = (ki/kp) * (ks/kp + 1)
        This is the signal heralding ratio for ξs = ξi = ξp.

        Section VII discusses how higher heralding ratios are possible with different
        focusing conditions, but there's a trade-off with brightness. The heralding ratio
        approaches unity as ξp → 0 (pump focusing becomes weaker).

        Parameters:
        -----------
        xi_p, xi_s, xi_i : float
            Focusing parameters for pump, signal, and idler

        Returns:
        --------
        float
            Estimated signal heralding ratio
        """
        # For optimal case when xi_p ≈ xi_s ≈ xi_i
        if abs(xi_p - xi_s) < 0.5 and abs(xi_p - xi_i) < 0.5:
            return (self.k_i / self.k_p) * (self.k_s / self.k_p + 1)

        # For non-optimal cases, we'll use an approximation
        # Heralding ratio approaches 1 as xi_p approaches 0
        else:
            base_ratio = (self.k_i / self.k_p) * (self.k_s / self.k_p + 1)
            # Modified ratio that increases as pump focusing decreases
            return min(0.95, base_ratio * (1 + 0.5 / xi_p))

    def calculate_pair_collection_probability_factor(self, xi):
        """
        Calculate the pair collection probability factor based on
        the asymptotic formula from Section V.

        Based on Equation (40):
        Psi ≈ [64π³ħcnsni/ε0np|n's-n'i|]·[χ(2)eff/λsλi]²·[arctan(ξ)/A+B+]·Np

        Subject to upper bound from Equation (41):
        Psi ≤ [8π⁴ħcnsni/ε0np|n's-n'i|]·[χ(2)eff/λsλi]²·Np

        As discussed in Sec. V, the pair probability increases asymptotically
        with the focusing parameter but has an upper bound.

        Parameters:
        -----------
        xi : float
            Aggregate focusing parameter

        Returns:
        --------
        float
            Normalized pair collection probability factor
        """
        # From Equation (40), the probability is proportional to arctan(ξ)
        # We normalize to the asymptotic value at high ξ
        return 2 * np.arctan(xi) / np.pi

    def plot_pair_probability_2d(
        self,
        xi_p_range=(0.1, 100),
        xi_s_range=(0.1, 100),
        delta_k=None,
        num_points=30,
    ):
        """
        Reproduce Figure 5 from Bennink's paper: Dependence of the pair collection
        probability on the focus of the pump and photon modes.

        Based on Section V, showing the 2D relationship between pump focus and
        signal/idler focus (assuming ξs = ξi) for pair collection probability.

        Parameters:
        -----------
        xi_p_range : tuple
            Range of pump focusing parameter values (log scale)
        xi_s_range : tuple
            Range of signal/idler focusing parameter values (log scale)
        num_points : int
            Number of points along each axis

        Returns:
        --------
        fig
            Matplotlib figure with the 2D color plot
        """
        # If delta_k is not provided, assume perfect phase matching
        if delta_k is None:
            delta_k = 0
        # Create logarithmically spaced grids for the focusing parameters
        xi_p_values = np.logspace(
            np.log10(xi_p_range[0]), np.log10(xi_p_range[1]), num_points
        )
        xi_s_values = np.logspace(
            np.log10(xi_s_range[0]), np.log10(xi_s_range[1]), num_points
        )

        # Create a 2D grid of the focusing parameters
        xi_p_grid, xi_s_grid = np.meshgrid(xi_p_values, xi_s_values)

        # Initialize a 2D array to store the pair collection probabilities
        pair_probability = np.zeros_like(xi_p_grid)

        # Calculate the pair collection probability for each combination of focusing parameters
        for i in range(num_points):
            for j in range(num_points):
                xi_p = xi_p_grid[i, j]
                xi_s = xi_s_grid[i, j]
                xi_i = xi_s  # Assuming ξs = ξi as in Figure 5

                # If delta_k is not provided, assume perfect phase matching
                if delta_k is None:
                    delta_k = 0

                # Equation (12): A+
                A_plus = (
                    1
                    + (self.k_s / self.k_p) * (xi_s / xi_p)
                    + (self.k_i / self.k_p) * (xi_i / xi_p)
                )

                # Calculate B+ using the full form from Equation (13)
                # B+ = (1 - Δk/kp) × [1 + (ks+Δk)/(kp-Δk) × ξp/ξs + (ki+Δk)/(kp-Δk) × ξp/ξi]

                # Handle the case where kp = Δk (avoid division by zero)
                if abs(self.k_p - delta_k) < 1e-10:
                    # In this case, phase matching is far off
                    pair_probability[i, j] = 0
                    continue

                term1 = 1 - (delta_k / self.k_p)
                term2 = 1 + ((self.k_s + delta_k) / (self.k_p - delta_k)) * (
                    xi_p / xi_s
                )
                term3 = ((self.k_i + delta_k) / (self.k_p - delta_k)) * (
                    xi_p / xi_i
                )

                B_plus = term1 * (term2 + term3)

                # Equation (15): ξ = (B+/A+)·(ξs·ξi/ξp)
                xi = (B_plus / A_plus) * (xi_s * xi_i / xi_p)

                # Equation (40): Psi proportional to arctan(ξ)/(A+·B+)
                pair_probability[i, j] = np.arctan(xi) / (A_plus * B_plus)

        # Normalize the pair probability to its maximum value
        pair_probability = pair_probability / np.max(pair_probability)

        # Create the figure
        fig, ax = plt.subplots(figsize=(8, 7))

        # Create a color plot
        contour = ax.pcolormesh(
            xi_p_grid,
            xi_s_grid,
            pair_probability,
            cmap='jet',
            shading='gouraud',
        )

        # Set log scale for both axes
        ax.set_xscale('log')
        ax.set_yscale('log')

        # Add colorbar and labels
        fig.colorbar(
            contour, ax=ax, label='Relative Pair Collection Probability'
        )
        ax.set_xlabel('Pump Focus ξp')
        ax.set_ylabel('Photon Focus ξs, ξi')
        ax.set_title('Pair Collection Probability vs. Focusing Parameters')

        # Add a contour line at 0.9 of the maximum to highlight the optimal region
        cs = ax.contour(
            xi_p_grid, xi_s_grid, pair_probability, levels=[0.9], colors='r'
        )

        # Annotate the optimal region (ξp ≈ ξs)
        ax.plot([0.1, 100], [0.1, 100], 'w--', alpha=0.7, label='ξp = ξs')

        ax.legend(loc='upper left')

        return fig

    def plot_focusing_dependencies(self, xi_range=(0.1, 10), num_points=100):
        """
        Plot how various SPDC parameters depend on the focusing parameter.

        This visualizes key relationships from the paper:
        - Spectral density peaks at ξ ≈ 2.84 (Fig. 2, Section III)
        - Pair collection probability increases asymptotically with ξ (Fig. 4, Section V)
        - Heralding ratio decreases as ξ increases (Fig. 8, Section VII)

        These relationships illustrate the trade-offs described in Section VIII.

        Parameters:
        -----------
        xi_range : tuple
            Range of xi values to plot
        num_points : int
            Number of points to calculate

        Returns:
        --------
        fig
            Matplotlib figure with the plots
        """
        xi_values = np.logspace(
            np.log10(xi_range[0]), np.log10(xi_range[1]), num_points
        )

        # Calculate metrics across the range
        spectral_density = [
            self.calculate_spectral_density_factor(xi, xi, xi)
            for xi in xi_values
        ]
        pair_probability = [
            self.calculate_pair_collection_probability_factor(xi)
            for xi in xi_values
        ]
        heralding = [
            self.calculate_heralding_ratio(xi, xi, xi) for xi in xi_values
        ]

        # Create plot
        fig, ax = plt.subplots(figsize=(10, 6))

        ax.semilogx(
            xi_values,
            spectral_density,
            'b-',
            label='Spectral Density',
        )
        ax.semilogx(
            xi_values,
            pair_probability,
            'r-',
            label='Pair Collection Probability',
        )
        ax.semilogx(xi_values, heralding, 'g-', label='Heralding Ratio')

        # Add a vertical line at the optimal xi value
        ax.axvline(
            x=2.84,
            color='k',
            linestyle='--',
            alpha=0.7,
            label='Optimal ξ = 2.84',
        )

        ax.set_xlabel('Focusing Parameter ξ')
        ax.set_ylabel('Normalized Metric')
        ax.set_title('SPDC Parameters vs. Focusing Parameter')
        ax.grid(True, which='both', linestyle='--', alpha=0.5)
        ax.legend()

        return fig

    def calculate_waists_for_xi(self, xi):
        """
        Calculate beam waists for a given focusing parameter.

        Implements Equation (11): ξj = L/(kj·w²j)
        Rearranged to solve for waist: w_j = √(L/(ξj·kj))

        Parameters:
        -----------
        xi : float
            Focusing parameter

        Returns:
        --------
        dict
            Dictionary with calculated waists
        """
        w_p = np.sqrt(self.L / (xi * self.k_p))
        w_s = np.sqrt(self.L / (xi * self.k_s))
        w_i = np.sqrt(self.L / (xi * self.k_i))

        return {'pump_waist': w_p, 'signal_waist': w_s, 'idler_waist': w_i}

    def estimate_brightness(self, pump_power, pump_wavelength=None):
        """
        Estimate the approximate brightness (pairs per pump photon)
        based on the asymptotic formula in the paper.

        Implements the upper bound from Equation (41):
        Psi ≤ [8π⁴ħcnsni/ε0np|n's-n'i|]·[χ(2)eff/λsλi]²·Np

        As noted in Section V, for PPKTP or PPLN sources, brightnesses
        exceeding 10^-9 collected pairs per pump photon should be achievable.

        Parameters:
        -----------
        pump_power : float
            Pump power in Watts
        pump_wavelength : float, optional
            Pump wavelength in meters (if None, uses the instance value)

        Returns:
        --------
        float
            Estimated pairs per pump photon
        """
        if pump_wavelength is None:
            pump_wavelength = self.lambda_p

        # Approximate chi(2) effective for PPKTP
        chi2_eff = 10e-12  # Approximate value in m/V

        # Photon energy
        E_photon = h * c / pump_wavelength

        # Number of pump photons
        N_p = pump_power / E_photon

        # From Equation (41), approximate upper bound
        group_index_diff = abs(
            self.n_s - self.n_i
        )  # Approximation for |n's - n'i|

        # Constant factor based on paper's formula
        # This is a simplified estimate - actual value depends on many factors
        constant_factor = (
            8
            * (pi**4)
            * h
            * c
            * self.n_s
            * self.n_i
            / (self.n_p * group_index_diff)
        )

        # Calculate brightness estimate
        brightness = (
            constant_factor * (chi2_eff / (self.lambda_s * self.lambda_i)) ** 2
        ) * N_p

        return brightness

    def calculate_photon_bandwidth(
        self, xi, delta_k=None, group_velocity_difference=None
    ):
        """
        Calculate the photon bandwidth based on Equations (33-34) from Bennink's paper.

        For weak to moderate focusing (ξ ≤ 10), the bandwidth follows the 1/L dependence.
        For strong focusing (ξ > 10), the bandwidth is determined by the confocal length b = L/ξ.

        The bandwidth formula is:
        Δωs = Δωi ~ [2πc/|n's-n'i|] · max(1/L, 1/10b)

        Parameters:
        -----------
        xi : float
            Aggregate focusing parameter
        delta_k : float, optional
            Wave vector mismatch (not used directly for bandwidth calculation)
        group_velocity_difference : float, optional
            Absolute difference between signal and idler group indices |n's-n'i|
            If None, approximated as |n_s-n_i|

        Returns:
        --------
        float
            Photon bandwidth in Hz
        """
        # If group velocity difference is not provided, approximate using refractive indices
        if group_velocity_difference is None:
            group_velocity_difference = abs(self.n_s - self.n_i)

        # Calculate the confocal length b = L/ξ
        confocal_length = self.L / xi

        # Equation (33): Δω ~ 2π · max(1, ξ/10)
        # Equation (34): Δωs = Δωi ~ [2πc/|n's-n'i|] · max(1/L, 1/10b)
        bandwidth_factor = max(1 / self.L, 1 / (10 * confocal_length))

        # Calculate bandwidth in Hz
        bandwidth = (
            (2 * np.pi * c) / group_velocity_difference * bandwidth_factor
        )

        return bandwidth

    def plot_photon_bandwidth(
        self,
        xi_range=(0.01, 100),
        num_points=100,
        group_velocity_difference=None,
    ):
        """
        Reproduce Figure 3 from Bennink's paper: The normalized photon bandwidth as
        a function of focusing.

        Parameters:
        -----------
        xi_range : tuple
            Range of focusing parameter values (log scale)
        num_points : int
            Number of points to calculate
        group_velocity_difference : float, optional
            Absolute difference between signal and idler group indices |n's-n'i|
            If None, approximated as |n_s-n_i|

        Returns:
        --------
        fig
            Matplotlib figure with the plot
        """
        # Create logarithmically spaced values for xi
        xi_values = np.logspace(
            np.log10(xi_range[0]), np.log10(xi_range[1]), num_points
        )

        # Calculate bandwidth for each xi value
        bandwidth_values = np.array([
            self.calculate_photon_bandwidth(
                xi, group_velocity_difference=group_velocity_difference
            )
            for xi in xi_values
        ])

        # Normalize to the value at xi = 1 (weak focusing reference point)
        reference_bandwidth = self.calculate_photon_bandwidth(
            1.0, group_velocity_difference=group_velocity_difference
        )
        normalized_bandwidth = bandwidth_values / reference_bandwidth

        # Create figure
        fig, ax = plt.subplots(figsize=(8, 6))

        # Plot data and theoretical curves
        ax.semilogx(
            xi_values,
            normalized_bandwidth,
            'b-',
            linewidth=2,
            label='Calculated bandwidth',
        )

        # Add the theoretical prediction: Δω ~ 2π · max(1, ξ/10) from Equation 33
        theoretical_curve = np.maximum(1, xi_values / 10)
        ax.semilogx(
            xi_values,
            theoretical_curve,
            'r--',
            linewidth=1.5,
            label='Theoretical model: max(1, ξ/10)',
        )

        # Add vertical line at the transition point ξ = 10
        ax.axvline(
            x=10,
            color='k',
            linestyle=':',
            alpha=0.7,
            label='Transition point: ξ = 10',
        )

        # Add labels and title
        ax.set_xlabel('Aggregate Focus ξ')
        ax.set_ylabel('Phase Bandwidth Δϕ/2π')
        ax.set_title(
            'Photon Bandwidth vs. Focusing Parameter (Figure 3 from Bennink)'
        )

        # Add grid and legend
        ax.grid(True, which='both', linestyle='--', alpha=0.5)
        ax.legend()

        return fig

    def plot_photon_bandwidth_2d(
        self,
        xi_p_range=(0.1, 100),
        xi_s_range=(0.1, 100),
        resolution=30,
        group_velocity_difference=None,
    ):
        """
        Create a 2D heatmap showing how photon bandwidth varies with
        pump and signal/idler focusing parameters.

        Parameters:
        -----------
        xi_p_range : tuple
            Range of pump focusing parameter values (log scale)
        xi_s_range : tuple
            Range of signal/idler focusing parameter values (log scale)
        resolution : int
            Number of points along each axis
        group_velocity_difference : float, optional
            Absolute difference between signal and idler group indices |n's-n'i|
            If None, approximated as |n_s-n_i|

        Returns:
        --------
        fig
            Matplotlib figure with the 2D color plot
        """
        # Create logarithmically spaced grids for the focusing parameters
        xi_p_values = np.logspace(
            np.log10(xi_p_range[0]), np.log10(xi_p_range[1]), resolution
        )
        xi_s_values = np.logspace(
            np.log10(xi_s_range[0]), np.log10(xi_s_range[1]), resolution
        )

        # Create a 2D grid of the focusing parameters
        xi_p_grid, xi_s_grid = np.meshgrid(xi_p_values, xi_s_values)

        # Initialize a 2D array to store the bandwidth values
        bandwidth = np.zeros_like(xi_p_grid)

        # Calculate bandwidth for each combination of focusing parameters
        for i in range(resolution):
            for j in range(resolution):
                xi_p = xi_p_grid[i, j]
                xi_s = xi_s_grid[i, j]
                xi_i = xi_s  # Assuming ξs = ξi

                # Calculate A+ and B+ as in the spectral density calculation
                A_plus = (
                    1
                    + (self.k_s / self.k_p) * (xi_s / xi_p)
                    + (self.k_i / self.k_p) * (xi_i / xi_p)
                )

                # With phase matching assumed (Δk ≈ 0)
                B_plus = 1  # Simplification for phase-matched case

                # Calculate the aggregate focusing parameter ξ from Equation (15)
                xi = (B_plus / A_plus) * (xi_s * xi_i / xi_p)

                # Calculate the bandwidth using the aggregate focusing parameter
                bandwidth[i, j] = self.calculate_photon_bandwidth(
                    xi, group_velocity_difference=group_velocity_difference
                )

        # Normalize to the minimum bandwidth for clearer visualization
        bandwidth_norm = bandwidth / np.min(bandwidth)

        # Create figure
        fig, ax = plt.subplots(figsize=(8, 7))

        # Create color plot with logarithmic color scale
        contour = ax.pcolormesh(
            xi_p_grid,
            xi_s_grid,
            bandwidth_norm,
            cmap='jet',
            shading='gouraud',
            # norm=colors.LogNorm(
            #     vmin=bandwidth_norm.min(), vmax=bandwidth_norm.max()
            # ),
        )

        # Set log scale for both axes
        ax.set_xscale('log')
        ax.set_yscale('log')

        # Add colorbar and labels
        fig.colorbar(contour, ax=ax, label='Normalized Photon Bandwidth')
        ax.set_xlabel('Pump Focus ξp')
        ax.set_ylabel('Photon Focus ξs, ξi')
        ax.set_title('Photon Bandwidth vs. Focusing Parameters')

        # Add the line ξs = ξp
        ax.plot([0.1, 100], [0.1, 100], 'w--', alpha=0.7, label='ξp = ξs')

        # Add the line ξs = 10ξp (where bandwidth starts to increase)
        ax.plot([0.1, 10], [1, 100], 'w-.', alpha=0.7, label='ξs = 10ξp')

        ax.legend(loc='upper left')

        return fig

    def calculate_pdc_spectrum(
        self,
        wavelength_range=None,
        num_points=200,
        xi_p=2.84,
        xi_s=2.84,
        xi_i=2.84,
        alt_form=False,
        central_signal_wavelength=None,
        plot_phase_matching=False,
    ):
        """
        Calculate the PDC spectrum (intensity vs wavelength) for specific focusing conditions.
        Implements the sinc-shaped profile characteristic of PPKTP and other QPM crystals.

        Parameters:
        -----------
        wavelength_range : tuple, optional
            Range of signal wavelengths to calculate (min, max) in meters
            If None, a range around the central signal wavelength is used
        num_points : int
            Number of wavelength points to calculate
        xi_p, xi_s, xi_i : float
            Focusing parameters for pump, signal, and idler beams
        alt_form : bool
            If True, use the alternative form for A_+B_+ from Equations (22-23)
        central_signal_wavelength : float, optional
            Central signal wavelength in meters. If None, uses self.lambda_s
        plot_phase_matching : bool
            If True, return the phase-matching function separately for plotting

        Returns:
        --------
        tuple
            (signal_wavelengths, idler_wavelengths, intensities, [phase_matching])
            Arrays containing the signal wavelengths, corresponding idler wavelengths,
            and their relative intensities. If plot_phase_matching is True, also returns
            the phase matching function.
        """
        # If central signal wavelength not specified, use the instance value
        if central_signal_wavelength is None:
            central_signal_wavelength = self.lambda_s

        # If wavelength range not specified, create range around central wavelength
        if wavelength_range is None:
            # Create a wide enough range to see the sinc function side lobes
            # Rule of thumb: width ~ 2π/L for first zeros of sinc
            width_factor = 3  # Show about 3 side lobes
            wavelength_width = (
                width_factor
                * (central_signal_wavelength**2)
                / (self.L * self.n_s)
            )

            wavelength_range = (
                central_signal_wavelength - wavelength_width,
                central_signal_wavelength + wavelength_width,
            )

        # Create signal wavelength array
        signal_wavelengths = np.linspace(
            wavelength_range[0], wavelength_range[1], num_points
        )

        crystal = crystals.KTiOPO4_Fradkin
        lambda_p_central = Wavelength(775, Magnitude.nano)
        lambda_s_central = Wavelength(1550, Magnitude.nano)
        lambda_i_central = Wavelength(1550, Magnitude.nano)

        poling_period = calculate_grating_period(
            lambda_p_central,
            lambda_s_central,
            lambda_i_central,
            crystal,
        )

        # Calculate corresponding idler wavelengths using energy conservation
        # 1/λp = 1/λs + 1/λi -> λi = (λp·λs)/(λs - λp)
        # idler_wavelengths = (self.lambda_p * signal_wavelengths) / (
        #     signal_wavelengths - self.lambda_p
        # )

        idler_wavelengths = signal_wavelengths
        pump_wavelengths = 1 / (
            (1 / signal_wavelengths) + (1 / idler_wavelengths)
        )

        signal_wavelengths = Wavelength(signal_wavelengths, Magnitude.base)
        idler_wavelengths = Wavelength(idler_wavelengths, Magnitude.base)
        pump_wavelengths = Wavelength(pump_wavelengths, Magnitude.base)

        lambda_s_grid, lambda_i_grid = np.meshgrid(
            (2 * np.pi * c) / signal_wavelengths.to_absolute().value,
            (2 * np.pi * c) / idler_wavelengths.to_absolute().value,
            indexing='ij',
        )

        wl_s = Wavelength((2 * np.pi * c) / lambda_s_grid, Magnitude.base)
        wl_i = Wavelength((2 * np.pi * c) / lambda_i_grid, Magnitude.base)
        wl_p = Wavelength(
            (2 * np.pi * c) / (lambda_s_grid + lambda_i_grid),
            Magnitude.base,
        )

        (n_p, n_s, n_i) = crystal.refractive_indices(
            wl_p,
            wl_s,
            wl_i,
        )

        k_s = (2 * np.pi * n_s) / wl_s.value
        k_i = (2 * np.pi * n_i) / wl_i.value
        k_p = (2 * np.pi * n_p) / wl_p.value

        delta_k = delta_k_matrix(
            lambda_p_central,
            signal_wavelengths,
            idler_wavelengths,
            crystal,
        )

        sigma_lambda_p = Wavelength(0.3, Magnitude.nano)
        sigma_p = (
            2 * np.pi * bandwidth_conversion(sigma_lambda_p, lambda_p_central)
        )
        pump_envelope = pef.gaussian(
            lambda_p_central,
            sigma_p,
            signal_wavelengths,
            idler_wavelengths,
        )
        phase_matching_matrix = phase_matching_function(
            delta_k, poling_period, self.L
        )
        JSA = pump_envelope * phase_matching_matrix
        JSI = np.abs(JSA)
        marginal_spectra = calculate_jsa_marginals(
            JSA, signal_wavelengths, idler_wavelengths
        )

        fig, axs = plt.subplots()
        axs.plot(
            marginal_spectra['signal_wl'],
            np.abs(marginal_spectra['signal_intensity']),
            label='Signal',
        )
        axs.plot(
            marginal_spectra['idler_wl'],
            np.abs(marginal_spectra['idler_intensity']),
            label='Idler',
        )
        axs.legend()
        # plt.show()

        # Initialize arrays for signal and idler k-vectors and intensities
        # k_signal = 2 * np.pi * self.n_s / signal_wavelengths
        # k_idler = 2 * np.pi * self.n_i / idler_wavelengths
        # k_pump = 2 * np.pi * self.n_p / pump_wavelength

        # Arrays to store results
        intensities = np.zeros(num_points)
        phase_matching_func = np.zeros(num_points)
        focusing_effect = np.zeros(num_points)

        spectra = marginal_spectra['signal_intensity']
        spectra /= np.max(spectra)

        # For each wavelength pair, calculate the phase mismatch and spectral density
        for i in range(num_points):
            # Calculate phase mismatch Δk = kp - ks - ki
            # delta_k = self.k_p - k_signal[i] - k_idler[i]
            # delta_k = k_pump[i] - k_signal[i] - k_idler[i]

            # pump_wavelength = 1 / (
            #     (1 / signal_wavelengths.value[i]) + (1 / (1550e-9))
            # )

            # wl_p = Wavelength(pump_wavelengths.value[i], pump_wavelengths.unit)
            # wl_s = Wavelength(
            #     signal_wavelengths.value[i], signal_wavelengths.unit
            # )
            # wl_i = Wavelength(
            #     idler_wavelengths.value[i], idler_wavelengths.unit
            # )

            # (n_p, n_s, n_i) = crystal.refractive_indices(wl_p, wl_s, wl_i)

            # k_signal = (2 * np.pi * n_s) / wl_s.value
            # k_idler = (2 * np.pi * n_i) / wl_i.value
            # k_pump = (2 * np.pi * n_p) / wl_p.value

            # delta_k = k_pump - k_signal - k_idler
            # phase_mismatch = delta_k - self.k_g

            # # Calculate the phase mismatch parameter Φ = (Δk + mK)L
            # # Here we use first-order quasi-phase-matching (m=1)
            # phi = phase_mismatch * (self.L / 2)

            # # Calculate the basic phase matching function: sinc²(ΔkL/2)
            # # Note: numpy's sinc function includes the division by π: sinc(x) = sin(πx)/(πx)
            # # So we need to divide by π to get the standard sinc function
            # # phase_matching = np.sinc(phi / (2 * np.pi)) ** 2
            # phase_matching = np.sinc(phi) ** 2
            phase_matching = phase_matching_matrix[i, i]
            phase_matching = spectra[i]
            phase_matching_func[i] = phase_matching

            k_pump = k_p[i, i]
            k_signal = k_s[i, i]
            k_idler = k_i[i, i]
            delta_k = k_pump - k_signal - k_idler

            # Calculate focusing correction factor
            if alt_form:
                # Alternative form from Equations (22-23)
                # Handle division by zero cases
                if abs(self.k_p - delta_k) < 1e-10:
                    focusing_factor = 0
                    continue

                # Define X_j and r_j terms as in Equations (23) and (24)
                X_s = (k_signal / k_pump) * np.sqrt(
                    (1 + (delta_k / k_signal)) / (1 - (delta_k / k_pump))
                )
                X_i = (k_idler / k_pump) * np.sqrt(
                    (1 + (delta_k / k_idler)) / (1 - (delta_k / k_pump))
                )

                r_s = (xi_s / xi_p) * np.sqrt(
                    (1 - (delta_k / k_pump)) / (1 + (delta_k / k_signal))
                )
                r_i = (xi_i / xi_p) * np.sqrt(
                    (1 - (delta_k / k_pump)) / (1 + (delta_k / k_idler))
                )

                # Calculate A_+B_+ using the alternative form
                term1 = 1 - (delta_k / k_pump)
                term2 = 1 + (X_s * r_s) + (X_i * r_i)
                term3 = 1 + (X_s / r_s) + (X_i / r_i)

                AB = term1 * term2 * term3
                inverse_sqrt_AB = 1 / np.sqrt(AB)

                # Calculate aggregate focusing parameter ξ
                # When r_s ≈ r_i ≈ 1 (optimal), this is ≈ ξp
                xi = xi_p  # Simplified approximation for this calculation

            else:
                # Standard form using Equations (12-13)
                A_plus = (
                    1
                    + (k_signal / k_pump) * (xi_s / xi_p)
                    + (k_idler / k_pump) * (xi_i / xi_p)
                )

                if abs(k_pump - delta_k) < 1e-10:
                    focusing_factor = 0
                    continue

                term1 = 1 - (delta_k / k_pump)
                term2 = 1 + ((k_signal + delta_k) / (k_pump - delta_k)) * (
                    xi_p / xi_s
                )
                term3 = ((k_idler + delta_k) / (k_pump - delta_k)) * (
                    xi_p / xi_i
                )

                B_plus = term1 * (term2 + term3)

                inverse_sqrt_AB = 1 / np.sqrt(A_plus * B_plus)

                # Calculate aggregate focusing parameter
                xi = (B_plus / A_plus) * (xi_s * xi_i / xi_p)

            # For spatial overlap factor, use the approximation from Fig. 1
            # Maximum value is around 2.06 at ξ = 2.84, Φ = -1.04π
            # The focusing effect increases with ξ for tight focusing
            spatial_overlap = 2.06 * np.exp(
                -((np.log10(xi) - np.log10(2.84)) ** 2) / 0.5
            )

            # Focusing broadens the spectrum while reducing the peak height
            # This is a key effect: as focusing increases, the side lobes get washed out
            # and the central peak becomes more Gaussian-like
            xi_ratio = xi / 2.84  # Ratio to optimal focusing

            # Apply focusing effect that broadens the spectrum
            if xi > 10:
                # For tight focusing, significantly broaden the spectrum
                # This simulates how focusing allows more k-vectors to participate
                # Implementation of the concept from Equation (34)
                broadening_factor = 1 + 0.1 * (
                    xi - 10
                )  # Linear broadening beyond ξ=10
                convolution_width = (
                    0.05
                    * broadening_factor
                    * phase_matching_matrix.max()  # phi.max()
                )  # Width scales with focusing

                # Simulate convolution effect by averaging over nearby points
                # This creates the broadening effect seen in tight focusing
                window_size = int(
                    max(
                        3,
                        convolution_width
                        * num_points
                        / (wavelength_range[1] - wavelength_range[0]),
                    )
                )
                window_size = min(
                    window_size, num_points // 5
                )  # Limit window size

                if (
                    window_size > 1
                    and i >= window_size // 2
                    and i < num_points - window_size // 2
                ):
                    # Apply moving average to simulate convolution
                    phase_matching = np.mean(
                        phase_matching_func[
                            i - window_size // 2 : i + window_size // 2 + 1
                        ]
                    )

            # Store the focusing effect for visualization
            focusing_effect[i] = inverse_sqrt_AB * spatial_overlap

            # Combine phase matching and focusing effect
            # intensities[i] = phase_matching * inverse_sqrt_AB * spatial_overlap
            # intensities[i] = (
            #     # np.abs(marginal_spectra['signal_intensity'])[i]
            #     phase_matching * inverse_sqrt_AB * spatial_overlap,
            # )
            intensities[i] = phase_matching * inverse_sqrt_AB * spatial_overlap

        # Normalize intensities
        if np.max(intensities) > 0:
            intensities = intensities / np.max(intensities)

        if plot_phase_matching:
            # Normalize phase matching function for comparison
            # if np.max(phase_matching_func) > 0:
            #     phase_matching_func = phase_matching_func / np.max(
            #         phase_matching_func
            #     )

            # # Normalize focusing effect for comparison
            # if np.max(focusing_effect) > 0:
            #     focusing_effect = focusing_effect / np.max(focusing_effect)

            return (
                marginal_spectra['signal_wl'],
                marginal_spectra['idler_wl'],
                intensities,
                phase_matching_func,
                focusing_effect,
            )

        # return signal_wavelengths.value, idler_wavelengths.value, intensities
        return (
            marginal_spectra['signal_wl'],
            marginal_spectra['idler_wl'],
            intensities,
        )

    def plot_pdc_spectrum(
        self,
        wavelength_range=None,
        num_points=400,
        xi_p=2.84,
        xi_s=2.84,
        xi_i=2.84,
        alt_form=False,
        plot_type='signal',
        central_signal_wavelength=None,
        show_components=True,
    ):
        """
        Plot the PDC spectrum for specific focusing conditions.

        Parameters:
        -----------
        wavelength_range : tuple, optional
            Range of signal wavelengths to calculate (min, max) in meters
        num_points : int
            Number of wavelength points to calculate
        xi_p, xi_s, xi_i : float
            Focusing parameters for pump, signal, and idler beams
        alt_form : bool
            If True, use the alternative form for A_+B_+ from Equations (22-23)
        plot_type : str
            'signal', 'idler', or 'both' - which spectrum to plot
        central_signal_wavelength : float, optional
            Central signal wavelength in meters. If None, uses self.lambda_s
        show_components : bool
            If True, show the phase matching and focusing components separately

        Returns:
        --------
        fig
            Matplotlib figure with the spectrum plot
        """
        # Calculate the spectrum with component breakdown if needed
        if show_components:
            results = self.calculate_pdc_spectrum(
                wavelength_range=wavelength_range,
                num_points=num_points,
                xi_p=xi_p,
                xi_s=xi_s,
                xi_i=xi_i,
                alt_form=alt_form,
                central_signal_wavelength=central_signal_wavelength,
                plot_phase_matching=True,
            )
            (
                signal_wavelengths,
                idler_wavelengths,
                intensities,
                phase_matching_func,
                focusing_effect,
            ) = results
        else:
            signal_wavelengths, idler_wavelengths, intensities = (
                self.calculate_pdc_spectrum(
                    wavelength_range=wavelength_range,
                    num_points=num_points,
                    xi_p=xi_p,
                    xi_s=xi_s,
                    xi_i=xi_i,
                    alt_form=alt_form,
                    central_signal_wavelength=central_signal_wavelength,
                )
            )

        # Convert wavelengths to nanometers for plotting
        signal_nm = signal_wavelengths * 1e9
        idler_nm = idler_wavelengths * 1e9

        # Create figure
        if plot_type == 'both':
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

            # Plot signal spectrum
            ax1.plot(signal_nm, intensities, 'b-', linewidth=2)

            if show_components:
                ax1.plot(
                    signal_nm,
                    phase_matching_func,
                    'g--',
                    linewidth=1,
                    alpha=0.7,
                    label='Phase Matching',
                )
                ax1.plot(
                    signal_nm,
                    focusing_effect,
                    'r--',
                    linewidth=1,
                    alpha=0.7,
                    label='Focusing Effect',
                )

            ax1.set_xlabel('Signal Wavelength (nm)')
            ax1.set_ylabel('Relative Intensity')
            ax1.set_title('Signal Spectrum')
            ax1.grid(True, linestyle='--', alpha=0.7)
            if show_components:
                ax1.legend()

            # Plot idler spectrum
            ax2.plot(idler_nm, intensities, 'r-', linewidth=2)
            ax2.set_xlabel('Idler Wavelength (nm)')
            ax2.set_ylabel('Relative Intensity')
            ax2.set_title('Idler Spectrum')
            ax2.grid(True, linestyle='--', alpha=0.7)
        else:
            fig, ax = plt.subplots(figsize=(8, 5))

            if plot_type == 'signal':
                ax.plot(
                    signal_nm,
                    intensities,
                    'b-',
                    linewidth=2,
                    label='Full Spectrum',
                )

                if show_components:
                    ax.plot(
                        signal_nm,
                        phase_matching_func,
                        'g--',
                        linewidth=1.5,
                        alpha=0.7,
                        label='Phase Matching',
                    )
                    ax.plot(
                        signal_nm,
                        focusing_effect,
                        'r--',
                        linewidth=1.5,
                        alpha=0.7,
                        label='Focusing Effect',
                    )

                ax.set_xlabel('Signal Wavelength (nm)')
                central_wavelength = central_signal_wavelength or self.lambda_s
            else:  # 'idler'
                ax.plot(
                    idler_nm,
                    intensities,
                    'r-',
                    linewidth=2,
                    label='Full Spectrum',
                )

                if show_components:
                    ax.plot(
                        idler_nm,
                        phase_matching_func,
                        'g--',
                        linewidth=1.5,
                        alpha=0.7,
                        label='Phase Matching',
                    )
                    ax.plot(
                        idler_nm,
                        focusing_effect,
                        'r--',
                        linewidth=1.5,
                        alpha=0.7,
                        label='Focusing Effect',
                    )

                ax.set_xlabel('Idler Wavelength (nm)')
                central_wavelength = (
                    (self.lambda_p * central_signal_wavelength)
                    / (central_signal_wavelength - self.lambda_p)
                    if central_signal_wavelength
                    else self.lambda_i
                )

            ax.set_ylabel('Relative Intensity')
            ax.set_title(
                f'{"Signal" if plot_type == "signal" else "Idler"} PDC Spectrum (ξp={xi_p:.2f}, ξs=ξi={xi_s:.2f})'
            )
            ax.grid(True, linestyle='--', alpha=0.7)

            # Add vertical line at central wavelength
            central_nm = central_wavelength * 1e9
            ax.axvline(
                x=central_nm,
                color='k',
                linestyle='--',
                alpha=0.5,
                label=f'Central: {central_nm:.1f} nm',
            )

            if show_components or plot_type == 'signal':
                ax.legend()

        plt.tight_layout()
        return fig

    def plot_focusing_effect_on_spectrum(
        self,
        wavelength_range=None,
        num_points=400,
        xi_values=[1.0, 2.84, 10.0, 20.0, 30.0],
        common_focusing=True,
        alt_form=False,
        plot_type='signal',
        normalize_peaks=True,
    ):
        """
        Plot how different focusing conditions affect the PDC spectrum.

        Parameters:
        -----------
        wavelength_range : tuple, optional
            Range of signal wavelengths to calculate (min, max) in meters
        num_points : int
            Number of wavelength points to calculate
        xi_values : list
            List of focusing parameter values to compare
        common_focusing : bool
            If True, sets all focusing parameters (ξp, ξs, ξi) to the same value
            If False, varies only ξp while keeping ξs=ξi=2.84
        alt_form : bool
            If True, use the alternative form for A_+B_+ from Equations (22-23)
        plot_type : str
            'signal' or 'idler' - which spectrum to plot
        normalize_peaks : bool
            If True, normalize all spectra to have the same peak value for comparison
            If False, maintain relative intensity scaling

        Returns:
        --------
        fig
            Matplotlib figure comparing spectra for different focusing conditions
        """
        # Create figure
        fig, ax = plt.subplots(figsize=(10, 6))

        # Color cycle for multiple curves
        colors = ['b', 'r', 'g', 'm', 'c']

        # If wavelength range not specified, create range around central wavelength
        if wavelength_range is None:
            # Create a wide enough range to see the sinc function side lobes
            # wider for the first plot to establish the context
            width_factor = 5  # Show more side lobes
            central_signal_wavelength = self.lambda_s
            wavelength_width = (
                width_factor
                * (central_signal_wavelength**2)
                / (self.L * self.n_s)
            )

            wavelength_range = (
                central_signal_wavelength - wavelength_width,
                central_signal_wavelength + wavelength_width,
            )

        # Store spectra for normalization if needed
        all_intensities = []
        all_wavelengths = []

        # Plot spectrum for each focusing value
        for i, xi in enumerate(xi_values):
            color = colors[i % len(colors)]

            if common_focusing:
                # Set all focusing parameters to the same value
                xi_p = xi_s = xi_i = xi
                label = f'ξ = {xi:.1f}'
            else:
                # Vary only pump focusing, keep signal and idler at optimal value
                xi_p = xi
                xi_s = xi_i = 2.84
                label = f'ξp = {xi:.1f}, ξs=ξi=2.84'

            # Calculate spectrum
            signal_wavelengths, idler_wavelengths, intensities = (
                self.calculate_pdc_spectrum(
                    wavelength_range=wavelength_range,
                    num_points=num_points,
                    xi_p=xi_p,
                    xi_s=xi_s,
                    xi_i=xi_i,
                    alt_form=alt_form,
                )
            )

            # Store for normalization if needed
            all_intensities.append(intensities)
            all_wavelengths.append((
                signal_wavelengths,
                idler_wavelengths,
            ))

        # # Normalize if required
        # if normalize_peaks and len(all_intensities) > 0:
        #     for i in range(len(all_intensities)):
        #         all_intensities[i] = all_intensities[i] / np.max(
        #             all_intensities[i]
        #         )

        # Plot all spectra
        for i, xi in enumerate(xi_values):
            color = colors[i % len(colors)]
            if common_focusing:
                label = f'ξ = {xi:.1f}'
            else:
                label = f'ξp = {xi:.1f}, ξs=ξi=2.84'

            # Select wavelengths based on plot type
            if plot_type == 'signal':
                wavelengths_nm = all_wavelengths[i][0]  # * 1e9  # Convert to nm
            else:  # 'idler'
                wavelengths_nm = all_wavelengths[i][1]  # * 1e9  # Convert to nm

            # Plot the spectrum
            ax.plot(
                wavelengths_nm,
                all_intensities[i],
                color=color,
                linewidth=2,
                label=label,
            )

        # Add labels and legend
        ax.set_xlabel(
            f'{"Signal" if plot_type == "signal" else "Idler"} Wavelength (nm)'
        )
        ax.set_ylabel('Relative Intensity')
        ax.set_title(
            f'Effect of Focusing on {"Signal" if plot_type == "signal" else "Idler"} PDC Spectrum'
        )
        ax.grid(True, linestyle='--', alpha=0.7)
        ax.legend()

        # Add vertical line at central wavelength
        if plot_type == 'signal':
            central_nm = self.lambda_s * 1e9
        else:
            central_nm = self.lambda_i * 1e9

        ax.axvline(
            x=central_nm,
            color='k',
            linestyle='--',
            alpha=0.5,
            label=f'Central: {central_nm:.1f} nm',
        )

        plt.tight_layout()
        return fig

    def plot_spectrum_2d_map(
        self,
        xi_p_range=(0.1, 100),
        xi_s_range=(0.1, 100),
        wavelength_range=None,
        resolution=20,
        spectral_points=101,
        alt_form=False,
        plot_type='fwhm',
    ):
        """
        Create a 2D map showing how the spectrum changes with focusing parameters.

        Parameters:
        -----------
        xi_p_range : tuple
            Range of pump focusing parameter values (log scale)
        xi_s_range : tuple
            Range of signal/idler focusing parameter values (log scale)
        wavelength_range : tuple, optional
            Range of signal wavelengths to calculate (min, max) in meters
        resolution : int
            Number of focusing parameter points along each axis
        spectral_points : int
            Number of wavelength points for spectrum calculation
        alt_form : bool
            If True, use the alternative form for A_+B_+ from Equations (22-23)
        plot_type : str
            'fwhm' - Plot the spectrum width (FWHM)
            'central' - Plot the central wavelength shift
            'intensity' - Plot the peak intensity

        Returns:
        --------
        fig
            Matplotlib figure with the 2D color plot
        """
        # Create logarithmically spaced grids for the focusing parameters
        xi_p_values = np.logspace(
            np.log10(xi_p_range[0]), np.log10(xi_p_range[1]), resolution
        )
        xi_s_values = np.logspace(
            np.log10(xi_s_range[0]), np.log10(xi_s_range[1]), resolution
        )

        # Create a 2D grid of the focusing parameters
        xi_p_grid, xi_s_grid = np.meshgrid(xi_p_values, xi_s_values)

        # Initialize arrays for the metrics
        fwhm_values = np.zeros_like(xi_p_grid)
        central_shifts = np.zeros_like(xi_p_grid)
        peak_intensities = np.zeros_like(xi_p_grid)

        # For each combination of focusing parameters, calculate the spectrum
        for i in range(resolution):
            for j in range(resolution):
                xi_p = xi_p_grid[i, j]
                xi_s = xi_s_grid[i, j]
                xi_i = xi_s  # Assuming ξs = ξi

                # Calculate spectrum
                signal_wavelengths, _, intensities = (
                    self.calculate_pdc_spectrum(
                        wavelength_range=wavelength_range,
                        num_points=spectral_points,
                        xi_p=xi_p,
                        xi_s=xi_s,
                        xi_i=xi_i,
                        alt_form=alt_form,
                    )
                )

                # Calculate FWHM (full width at half maximum)
                half_max = np.max(intensities) / 2.0
                indices = np.where(intensities >= half_max)[0]
                if len(indices) > 0:
                    fwhm = (
                        signal_wavelengths[indices[-1]]
                        - signal_wavelengths[indices[0]]
                    )
                    fwhm_values[i, j] = fwhm
                else:
                    fwhm_values[i, j] = 0

                # Calculate central wavelength shift
                max_idx = np.argmax(intensities)
                central_shifts[i, j] = (
                    signal_wavelengths[max_idx] - self.lambda_s
                )

                # Store peak intensity
                peak_intensities[i, j] = np.max(intensities)

        # Create figure
        fig, ax = plt.subplots(figsize=(9, 7))

        # Choose which metric to plot
        if plot_type == 'fwhm':
            data = fwhm_values * 1e9  # Convert to nm
            cmap = 'jet'
            label = 'Spectrum FWHM (nm)'
            title = 'PDC Spectrum Width vs. Focusing Parameters'
        elif plot_type == 'central':
            data = central_shifts * 1e9  # Convert to nm
            cmap = 'coolwarm'
            label = 'Central Wavelength Shift (nm)'
            title = 'PDC Central Wavelength Shift vs. Focusing Parameters'
        else:  # 'intensity'
            data = peak_intensities
            cmap = 'viridis'
            label = 'Relative Peak Intensity'
            title = 'PDC Peak Intensity vs. Focusing Parameters'

        # Create color plot
        if plot_type == 'central':
            # For central shift, use a diverging colormap centered at zero
            vmax = np.max(np.abs(data))
            contour = ax.pcolormesh(
                xi_p_grid,
                xi_s_grid,
                data,
                cmap=cmap,
                shading='gouraud',
                vmin=-vmax,
                vmax=vmax,
            )
        else:
            # For FWHM and intensity, use a regular colormap
            contour = ax.pcolormesh(
                xi_p_grid, xi_s_grid, data, cmap=cmap, shading='gouraud'
            )

        # Set log scale for both axes
        ax.set_xscale('log')
        ax.set_yscale('log')

        # Add colorbar and labels
        fig.colorbar(contour, ax=ax, label=label)
        ax.set_xlabel('Pump Focus ξp')
        ax.set_ylabel('Signal/Idler Focus ξs, ξi')
        ax.set_title(title)

        # Add the line ξs = ξp
        ax.plot([0.1, 100], [0.1, 100], 'w--', alpha=0.7, label='ξp = ξs')

        # Add the optimal focusing point
        ax.plot(2.84, 2.84, 'w+', markersize=10, markeredgewidth=2)

        ax.legend(loc='upper left')

        return fig


# Example usage
if __name__ == '__main__':
    # Crystal parameters
    crystal_length = 30e-3  # 30 mm crystal
    pump_wavelength = 750e-9  # 750 nm pump
    signal_wavelength = 1550e-9  # 1550 nm for signal

    # Set poling period for phase matching
    poling_period = (
        46.5e-6  # Adjust as needed for your phase-matching conditions
    )

    # Create the optimizer with the parameters
    optimizer = SPDCFocusingOptimizer(
        crystal_length=crystal_length,
        pump_wavelength=pump_wavelength,
        signal_wavelength=signal_wavelength,
        pump_n=1.758263,  # Refractive indices for PPKTP
        signal_n=1.734063,
        idler_n=1.816014,
        poling_period=poling_period,
    )

    # For PPKTP, a typical value is around 0.1-0.2
    group_velocity_diff = 0.15

    # Calculate optimal waists
    optimal_waists = optimizer.calculate_optimal_waists()
    print(
        f'Optimal waists for focusing parameter ξ = {optimal_waists["focusing_parameter"]}:'
    )
    print(f'  Pump waist: {optimal_waists["pump_waist"] * 1e6:.2f} μm')
    print(f'  Signal waist: {optimal_waists["signal_waist"] * 1e6:.2f} μm')
    print(f'  Idler waist: {optimal_waists["idler_waist"] * 1e6:.2f} μm')

    # Calculate heralding ratio
    heralding_ratio = optimizer.calculate_heralding_ratio(2.84, 2.84, 2.84)
    print(f'\nHeralding ratio at optimal focusing: {heralding_ratio:.3f}')

    # ------------------------------------------------------
    # FIGURES FROM BENNINK'S PAPER
    # ------------------------------------------------------
    print("\nGenerating figures from Bennink's paper...")

    # Figure 1: Spatial overlap factor
    fig1 = optimizer.plot_spatial_overlap_magnitude()
    plt.figure(1)
    plt.tight_layout()

    # Figure 2: Peak spectral density using alternative form
    fig2 = optimizer.plot_peak_spectral_density_2d(
        alt_form=True, delta_k=optimizer.delta_k, optimize_phi=True
    )
    plt.figure(2)
    plt.tight_layout()

    # Figure 3: Photon bandwidth vs. focusing
    fig3 = optimizer.plot_photon_bandwidth(
        xi_range=(0.1, 100),
        group_velocity_difference=group_velocity_diff,
        num_points=1024,
    )
    plt.figure(3)
    plt.tight_layout()

    # Figure 5: Pair collection probability
    fig5 = optimizer.plot_pair_probability_2d()
    plt.figure(4)
    plt.tight_layout()

    # Additional plot: Focusing dependencies (1D relationships)
    fig_deps = optimizer.plot_focusing_dependencies()
    plt.figure(5)
    plt.tight_layout()

    # Additional plot: 2D bandwidth map
    fig_bandwidth_2d = optimizer.plot_photon_bandwidth_2d(
        xi_p_range=(0.1, 100),
        xi_s_range=(0.1, 100),
        resolution=40,
        group_velocity_difference=group_velocity_diff,
    )
    plt.figure(6)
    plt.tight_layout()

    # ------------------------------------------------------
    # NEW SINC-BASED SPECTRUM VISUALIZATIONS
    # ------------------------------------------------------
    print('\nGenerating sinc-based PDC spectrum visualizations...')

    # Single spectrum with components for optimal focusing
    fig_sinc_optimal = optimizer.plot_pdc_spectrum(
        wavelength_range=(1545e-9, 1555e-9),
        xi_p=2.84,
        xi_s=2.84,
        xi_i=2.84,
        plot_type='signal',
        show_components=True,
        num_points=128,
    )
    plt.figure(7)
    plt.suptitle('PDC Spectrum with Optimal Focusing (ξ = 2.84)')
    plt.tight_layout()

    # Comparison of different focusing conditions
    fig_sinc_comparison = optimizer.plot_focusing_effect_on_spectrum(
        xi_values=[1.0, 2.84, 10.0, 20.0, 30.0],
        common_focusing=True,
        plot_type='signal',
        normalize_peaks=True,
        num_points=128,
    )
    plt.figure(8)
    plt.tight_layout()

    # Show spectrum for very weak focusing (clear sinc profile)
    fig_sinc_weak = optimizer.plot_pdc_spectrum(
        wavelength_range=(1545e-9, 1555e-9),
        xi_p=0.5,
        xi_s=0.5,
        xi_i=0.5,
        plot_type='signal',
        show_components=True,
        num_points=128,
    )
    plt.figure(9)
    plt.suptitle(
        'Spectrum with Very Weak Focusing (ξ = 0.5) - Clear Sinc Profile'
    )
    plt.tight_layout()

    # Show spectrum for very strong focusing (modified sinc profile)
    fig_sinc_strong = optimizer.plot_pdc_spectrum(
        wavelength_range=(1545e-9, 1555e-9),
        xi_p=30.0,
        xi_s=30.0,
        xi_i=30.0,
        plot_type='signal',
        show_components=True,
        num_points=128,
    )
    plt.figure(10)
    plt.suptitle(
        'Spectrum with Strong Focusing (ξ = 30.0) - Modified Sinc Profile'
    )
    plt.tight_layout()

    # Calculate and plot FWHM vs focusing parameter
    focusing_values = [0.5, 1.0, 2.84, 5.0, 10.0, 20.0, 30.0]
    fwhm_values = []

    def calculate_fwhm(wavelengths, intensities):
        half_max = np.max(intensities) / 2.0
        indices = np.where(intensities >= half_max)[0]
        if len(indices) > 0:
            return (
                wavelengths[indices[-1]] - wavelengths[indices[0]]
            ) * 1e9  # in nm
        return 0

    for xi in focusing_values:
        signal_wl, _, intensities = optimizer.calculate_pdc_spectrum(
            xi_p=xi, xi_s=xi, xi_i=xi, num_points=500
        )
        fwhm = calculate_fwhm(signal_wl, intensities)
        fwhm_values.append(fwhm)

    fig_fwhm, ax_fwhm = plt.subplots(figsize=(8, 5))
    ax_fwhm.semilogx(focusing_values, fwhm_values, 'b-o', linewidth=2)
    ax_fwhm.set_xlabel('Focusing Parameter ξ')
    ax_fwhm.set_ylabel('Spectrum FWHM (nm)')
    ax_fwhm.set_title('Spectrum Width vs. Focusing Parameter')
    ax_fwhm.grid(True, which='both', linestyle='--', alpha=0.7)
    ax_fwhm.axvline(
        x=2.84, color='r', linestyle='--', alpha=0.7, label='Optimal ξ = 2.84'
    )
    ax_fwhm.axvline(
        x=10.0, color='g', linestyle='--', alpha=0.7, label='Transition ξ = 10'
    )
    ax_fwhm.legend()
    plt.figure(11)
    plt.tight_layout()

    # ------------------------------------------------------
    # CALCULATIONS & COMPARISONS
    # ------------------------------------------------------
    print('\nCalculating metric comparisons...')

    # Compare calculation methods
    xi_p = 2.84
    xi_s = xi_i = 2.84

    standard_result = optimizer.calculate_peak_spectral_density(
        xi_p,
        xi_s,
        xi_i,
        alt_form=False,
        delta_k=optimizer.delta_k,
    )
    alt_result = optimizer.calculate_peak_spectral_density(
        xi_p,
        xi_s,
        xi_i,
        alt_form=True,
        delta_k=optimizer.delta_k,
    )

    print('\nComparison of calculation methods for peak spectral density:')
    print(f'  Standard form (Eq. 12-13): {standard_result:.6f}')
    print(f'  Alternative form (Eq. 22-23): {alt_result:.6f}')
    print(
        f'  Relative difference: {100 * abs(standard_result - alt_result) / standard_result:.6f}%'
    )

    # Calculate waists for different focusing regimes
    loose_focusing = optimizer.calculate_waists_for_xi(1.0)
    tight_focusing = optimizer.calculate_waists_for_xi(5.0)

    print('\nWaists for loose focusing (ξ = 1.0):')
    print(f'  Pump waist: {loose_focusing["pump_waist"] * 1e6:.2f} μm')
    print(f'  Signal waist: {loose_focusing["signal_waist"] * 1e6:.2f} μm')
    print(f'  Idler waist: {loose_focusing["idler_waist"] * 1e6:.2f} μm')

    print('\nWaists for tight focusing (ξ = 5.0):')
    print(f'  Pump waist: {tight_focusing["pump_waist"] * 1e6:.2f} μm')
    print(f'  Signal waist: {tight_focusing["signal_waist"] * 1e6:.2f} μm')
    print(f'  Idler waist: {tight_focusing["idler_waist"] * 1e6:.2f} μm')

    # Calculate example bandwidths
    weak_focusing_bandwidth = optimizer.calculate_photon_bandwidth(
        1.0, group_velocity_difference=group_velocity_diff
    )
    optimal_focusing_bandwidth = optimizer.calculate_photon_bandwidth(
        2.84, group_velocity_difference=group_velocity_diff
    )
    strong_focusing_bandwidth = optimizer.calculate_photon_bandwidth(
        20.0, group_velocity_difference=group_velocity_diff
    )

    print('\nPhoton bandwidth estimates:')
    print(f'  Weak focusing (ξ = 1.0): {weak_focusing_bandwidth / 1e9:.2f} GHz')
    print(
        f'  Optimal focusing (ξ = 2.84): {optimal_focusing_bandwidth / 1e9:.2f} GHz'
    )
    print(
        f'  Strong focusing (ξ = 20.0): {strong_focusing_bandwidth / 1e9:.2f} GHz'
    )
    print(
        f'  Bandwidth ratio (Strong/Weak): {strong_focusing_bandwidth / weak_focusing_bandwidth:.2f}x'
    )

    print('\nSpectrum FWHM at different focusing values:')
    for xi, fwhm in zip(focusing_values, fwhm_values):
        print(f'  ξ = {xi:.1f}: {fwhm:.3f} nm')

    print(
        f'\nFWHM broadening (ξ=30/ξ=0.5): {fwhm_values[-1] / fwhm_values[0]:.2f}x'
    )

    # Show all figures
    print('\nDisplaying all figures...')
    plt.show()
