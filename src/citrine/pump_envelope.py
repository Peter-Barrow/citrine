import numpy as np
from scipy.special import erf
from .citrine import Wavelength

__all__ = [
    'gaussian',
    'sech2',
    'skewed_gaussian',
]


def gaussian(
    lambda_p: Wavelength,
    sigma_p: float,
    lambda_s: Wavelength,
    lambda_i: Wavelength,
    k: float = None,
) -> np.ndarray:
    """
    Generate a pump envelope matrix with a Gaussian profile.

    Implements a Gaussian pump envelope in frequency space according to:
    $$
    \\alpha(\\omega_s + \\omega_i) = \\exp\\left(-\\frac{(\\omega_s + \\omega_i - \\omega_p)^2}{\\sigma_p^2}\\right)
    $$

    Can be chirped according to:
    $$
    \\chi(\\omega_s + \\omega_i) = \\exp(-ik(\\omega_s + \\omega_i - \\omega_p)^2),
    $$

    returning $\\alpha\\omega_s + \\omega_i \\cdot \\chi\\omega_s + \\omega_i$

    Args:
        lambda_p: Central wavelength of the pump.
        sigma_p: Pump bandwidth.
        lambda_s: Signal wavelengths array.
        lambda_i: Idler wavelengths array.
        k: Chirp parameter, controls strength and direction of
            chirp. Positive values create up-chirp, negative values create
            down-chirp.

    Returns:
        A 2D pump envelope matrix.

    Notes:
        The equation uses the following variables:
            $\\omega_s$: signal frequency
            $\\omega_i$: idler frequency
            $\\omega_p$: pump central frequency
            $\\sigma_p$: pump bandwidth

    """
    # Create meshgrid for signal and idler wavelengths
    lambda_s_meshgrid, lambda_i_meshgrid = np.meshgrid(
        lambda_s.value, lambda_i.value, indexing='ij'
    )

    # Convert to Wavelength objects for both grids
    lambda_s_grid = Wavelength(lambda_s_meshgrid, lambda_s.unit)
    lambda_i_grid = Wavelength(lambda_i_meshgrid, lambda_i.unit)

    # Convert signal and idler wavelengths to angular frequencies
    omega_s = lambda_s_grid.as_angular_frequency().value
    omega_i = lambda_i_grid.as_angular_frequency().value
    omega_p = lambda_p.as_angular_frequency().value

    # Calculate the Gaussian pump envelope
    # pump_envelope = np.exp(-(((omega_s + omega_i - omega_p) / sigma_p) ** 2))
    delta_omega = (omega_s + omega_i) - omega_p
    pump_envelope = np.exp(-(delta_omega**2) / (2 * sigma_p**2))

    if k is not None:
        print('Applying chirp')
        pump_envelope = np.asarray(pump_envelope, dtype=np.complex128)
        pump_envelope *= chirp_factor(lambda_p, k, lambda_s, lambda_i)

    return pump_envelope


def sech2(
    lambda_p: Wavelength,
    sigma_p: float,  # TODO: pump width needs its own type
    lambda_s: Wavelength,
    lambda_i: Wavelength,
    k: float = None,
) -> np.ndarray:
    """Generate a pump envelope matrix with a hyperbolic secant squared profile.

    Implements a sech² pump envelope in frequency space according to:
    $$
    \\alpha(\\omega_s + \\omega_i) = \\text{sech}^2(\\pi\\sigma_p(\\omega_s + \\omega_i - \\omega_p))
    $$

    Can be chirped according to:
    $$
    \\chi(\\omega_s + \\omega_i) = \\exp(-ik(\\omega_s + \\omega_i - \\omega_p)^2),
    $$

    returning $\\alpha\\omega_s + \\omega_i \\cdot \\chi\\omega_s + \\omega_i$

    Args:
        lambda_p: Central wavelength of the pump.
        sigma_p: Pump bandwidth parameter.
        lambda_s: Signal wavelengths array.
        lambda_i: Idler wavelengths array.
        k: Chirp parameter, controls strength and direction of
            chirp. Positive values create up-chirp, negative values create
            down-chirp.

    Returns:
        A 2D pump envelope matrix.

    Notes:
        The equation uses the following variables:
            $\\omega_s$: signal frequency
            $\\omega_i$: idler frequency
            $\\omega_p$: pump central frequency
            $\\sigma_p$: pump bandwidth parameter
    """
    lambda_s_meshgrid, lambda_i_meshgrid = np.meshgrid(
        lambda_s.value, lambda_i.value, indexing='ij'
    )

    # Convert to Wavelength objects for both grids
    lambda_s_grid = Wavelength(lambda_s_meshgrid, lambda_s.unit)
    lambda_i_grid = Wavelength(lambda_i_meshgrid, lambda_i.unit)

    # Convert signal and idler wavelengths to angular frequencies
    omega_s = lambda_s_grid.as_angular_frequency().value
    omega_i = lambda_i_grid.as_angular_frequency().value
    omega_p = lambda_p.as_angular_frequency().value

    delta_omega = (omega_s + omega_i) - omega_p
    pump_envelope = (1 / np.cosh((np.pi * sigma_p * delta_omega) / 2)) ** 2

    if k is not None:
        print('Applying chirp')
        pump_envelope = np.asarray(pump_envelope, dtype=np.complex128)
        pump_envelope *= chirp_factor(lambda_p, k, lambda_s, lambda_i)

    return pump_envelope


def chirp_factor(
    lambda_p: Wavelength,
    k: float,
    lambda_s: Wavelength,
    lambda_i: Wavelength,
) -> np.ndarray:
    """Generate a chirp factor matrix with a quadratic phase profile.

    Implements a quadratic spectral phase (chirp) according to:
    $$
    \\chi(\\omega_s + \\omega_i) = \\exp(-ik(\\omega_s + \\omega_i - \\omega_p)^2)
    $$

    Args:
        lambda_p: Central wavelength of the pump.
        k: Chirp parameter, controls strength and direction of chirp.
        lambda_s: Signal wavelengths array.
        lambda_i: Idler wavelengths array.

    Returns:
        A 2D complex-valued chirp factor matrix.

    Notes:
        The equation uses the following variables:
            $\\omega_s$: signal frequency
            $\\omega_i$: idler frequency
            $\\omega_p$: pump central frequency
            $k$: chirp parameter controlling strength and direction
    """
    # Create meshgrid for signal and idler wavelengths
    lambda_s_meshgrid, lambda_i_meshgrid = np.meshgrid(
        lambda_s.value, lambda_i.value, indexing='ij'
    )

    # Convert to Wavelength objects for both grids
    lambda_s_grid = Wavelength(lambda_s_meshgrid, lambda_s.unit)
    lambda_i_grid = Wavelength(lambda_i_meshgrid, lambda_i.unit)

    # Convert signal and idler wavelengths to angular frequencies
    omega_s = lambda_s_grid.as_angular_frequency().value
    omega_i = lambda_i_grid.as_angular_frequency().value
    omega_p = lambda_p.as_angular_frequency().value

    # Calculate the frequency detuning
    delta_omega = (omega_s + omega_i) - omega_p

    # Calculate the chirp factor (the exponential term)
    chirp = np.exp(-1j * k * (delta_omega**2))

    return chirp


def asymmetric_chirped(
    lambda_p: Wavelength,
    sigma_p: float,
    lambda_s: Wavelength,
    lambda_i: Wavelength,
    k: float = None,
    tail_strength: float = 2.5,
    cutoff_steepness: float = 25,
    edge_asymmetry: float = 0.5,
) -> np.ndarray:
    """
    Generate a pump envelope matrix with an asymmetric chirped profile.

    Implements an asymmetric pump envelope in frequency space that models
    experimental observations with asymmetric tails and sharp cutoffs.

    Args:
        lambda_p: Central wavelength of the pump.
        sigma_p: Pump bandwidth.
        lambda_s: Signal wavelengths array.
        lambda_i: Idler wavelengths array.
        k: Chirp parameter, controls strength and direction of chirp.
           Positive values create up-chirp, negative values create down-chirp.
        tail_strength: Controls the prominence of the spectral tail (default 2.5).
           Positive for tail on lower frequency side, negative for higher frequency side.
        cutoff_steepness: Controls how sharp the cutoff edge is (default 25).
           Higher values create a sharper cutoff.
        edge_asymmetry: Controls the position of the sharp edge relative to center (default 0.5).
           Values from 0 to 1, with 0.5 being centered.

    Returns:
        A 2D complex-valued pump envelope matrix.
    """
    # Create meshgrid for signal and idler wavelengths
    lambda_s_meshgrid, lambda_i_meshgrid = np.meshgrid(
        lambda_s.value, lambda_i.value, indexing='ij'
    )

    # Convert to Wavelength objects for both grids
    lambda_s_grid = Wavelength(lambda_s_meshgrid, lambda_s.unit)
    lambda_i_grid = Wavelength(lambda_i_meshgrid, lambda_i.unit)

    # Convert signal and idler wavelengths to angular frequencies
    omega_s = lambda_s_grid.as_angular_frequency().value
    omega_i = lambda_i_grid.as_angular_frequency().value
    omega_p = lambda_p.as_angular_frequency().value

    # Calculate frequency detuning
    delta_omega = omega_s + omega_i - omega_p

    # Create asymmetric base profile using skewed Gaussian
    # This gives more control over the tail shape
    skew = (
        -tail_strength
    )  # Negative for low-frequency tail, positive for high-frequency tail
    skewed_delta = delta_omega + (skew * np.abs(delta_omega))

    base_amplitude = np.exp(-(skewed_delta**2) / (2 * sigma_p**2))

    # Apply sharp cutoff on one edge
    cutoff_position = (
        (edge_asymmetry - 0.5) * 2 * sigma_p
    )  # Scale based on edge_asymmetry
    high_cut = 1 / (
        1 + np.exp(cutoff_steepness * (delta_omega - cutoff_position))
    )

    # Apply gentle rise on other edge
    rise_position = -2.0 * sigma_p
    rise_steepness = 3
    low_cut = 1 - 1 / (
        1 + np.exp(-rise_steepness * (delta_omega - rise_position))
    )

    # Combine to get final amplitude profile
    pump_envelope = base_amplitude * high_cut * low_cut
    # pump_envelope = base_amplitude

    # Ensure complex data type for chirp application
    pump_envelope = np.asarray(pump_envelope, dtype=np.complex128)

    # Apply chirp if specified
    #     if k is not None:
    #         chirp = np.exp(-1j * k * delta_omega**2)
    #         pump_envelope *= chirp

    return pump_envelope


def skewed_gaussian(
    lambda_p: Wavelength,
    sigma_p: float,
    lambda_s: Wavelength,
    lambda_i: Wavelength,
    alpha: float = 3.0,
    k: float = None,
) -> np.ndarray:
    """
    Generate a pump envelope matrix with a skewed Gaussian profile.

    Implements a skewed Gaussian pump envelope in frequency space according to:
    $$
    \\alpha(\\omega_s + \\omega_i) = \\exp\\left(-\\frac{(\\omega_s + \\omega_i - \\omega_p)^2}{2\\sigma_p^2}\\right)
    \\cdot \\left(1 + \\text{erf}\\left(\\alpha \\cdot \\frac{\\omega_s + \\omega_i - \\omega_p}{\\sigma_p \\sqrt{2}}\\right)\\right)
    $$

    Can be chirped according to:
    $$
    \\chi(\\omega_s + \\omega_i) = \\exp(-ik(\\omega_s + \\omega_i - \\omega_p)^2),
    $$

    returning $\\alpha(\\omega_s + \\omega_i) \\cdot \\chi(\\omega_s + \\omega_i)$

    Args:
        lambda_p: Central wavelength of the pump.
        sigma_p: Pump bandwidth.
        lambda_s: Signal wavelengths array.
        lambda_i: Idler wavelengths array.
        alpha: Skewness parameter (default 3.0).
               Positive values skew toward lower frequencies (longer wavelengths),
               negative values skew toward higher frequencies (shorter wavelengths).
        k: Chirp parameter, controls strength and direction of
            chirp. Positive values create up-chirp, negative values create down-chirp.

    Returns:
        A 2D pump envelope matrix.

    Notes:
        The skewed Gaussian combines a standard Gaussian with the error function (erf)
        to create asymmetry while maintaining smoothness.
    """
    # Create meshgrid for signal and idler wavelengths
    lambda_s_meshgrid, lambda_i_meshgrid = np.meshgrid(
        lambda_s.value, lambda_i.value, indexing='ij'
    )

    # Convert to Wavelength objects for both grids
    lambda_s_grid = Wavelength(lambda_s_meshgrid, lambda_s.unit)
    lambda_i_grid = Wavelength(lambda_i_meshgrid, lambda_i.unit)

    # Convert signal and idler wavelengths to angular frequencies
    omega_s = lambda_s_grid.as_angular_frequency().value
    omega_i = lambda_i_grid.as_angular_frequency().value
    omega_p = lambda_p.as_angular_frequency().value

    # Calculate frequency detuning
    delta_omega = omega_s + omega_i - omega_p

    # Normalize detuning by sigma
    normalized_delta = delta_omega / sigma_p

    # Calculate the Gaussian part
    gaussian_part = np.exp(-(normalized_delta**2) / 2)

    # Calculate the skewness part using error function
    skewness_part = 1 + erf(alpha * normalized_delta / np.sqrt(2))

    # Combine to get skewed Gaussian
    pump_envelope = gaussian_part * skewness_part

    # Normalize to peak value of 1
    pump_envelope = pump_envelope / np.max(pump_envelope)

    # Apply chirp if specified
    if k is not None:
        # Convert to complex array for chirp
        pump_envelope = np.asarray(pump_envelope, dtype=np.complex128)
        chirp = np.exp(-1j * k * delta_omega**2)
        pump_envelope *= chirp

    return pump_envelope
