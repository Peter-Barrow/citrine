import numpy as np
import matplotlib.pyplot as plt

import matplotlib

matplotlib.use('TkAgg')


def realistic_pump_envelope(
    lambda_center: float,
    bandwidth: float,  # in nm
    chirp_param: float = 0.0,
    pulse_shape: str = 'gaussian',
    spectral_filter: str = None,
    filter_width: float = None,
    filter_center: float = None,
    num_points: int = 1000,
    wavelength_range: tuple = None,
    plot_details: bool = True,
):
    """
    Generate a realistic pump envelope function including chirp and optional spectral filtering.

    Args:
        lambda_center: Central wavelength of the pump in nm.
        bandwidth: Pump bandwidth in nm (FWHM).
        chirp_param: Chirp parameter in fs². 0 = no chirp.
        pulse_shape: Base pulse shape ('gaussian', 'sech2', or 'square').
        spectral_filter: Optional spectral filter type ('gaussian', 'square', 'edge', None).
        filter_width: Width of the spectral filter in nm.
        filter_center: Center of the spectral filter in nm (if None, uses lambda_center).
        num_points: Number of points in the spectrum.
        wavelength_range: Optional (min_wavelength, max_wavelength) in nm.
        plot_details: If True, shows additional details in the plot.

    Returns:
        tuple: (wavelengths, complex_amplitude, intensity)
    """
    # Speed of light
    c = 299792458  # m/s

    # Create wavelength array
    if wavelength_range is None:
        # Default range based on pulse shape (sech2 needs wider range)
        if pulse_shape == 'sech2':
            range_factor = 4.0
        else:
            range_factor = 3.0
        min_lambda = lambda_center - range_factor * bandwidth
        max_lambda = lambda_center + range_factor * bandwidth
    else:
        min_lambda, max_lambda = wavelength_range

    wavelengths = np.linspace(min_lambda, max_lambda, num_points)

    # Convert to angular frequency
    omega_center = 2 * np.pi * c / (lambda_center * 1e-9)  # rad/s
    omega = 2 * np.pi * c / (wavelengths * 1e-9)  # rad/s
    delta_omega = omega - omega_center

    # Convert to appropriate units for calculations
    delta_omega_fs = delta_omega * 1e-15  # rad/fs

    # Calculate FWHM to sigma conversion factor based on pulse shape
    if pulse_shape == 'gaussian':
        fwhm_to_sigma = 2 * np.sqrt(2 * np.log(2))
    elif pulse_shape == 'sech2':
        fwhm_to_sigma = 2 * np.arccosh(np.sqrt(2))
    else:  # square
        fwhm_to_sigma = (
            2.0  # Not technically sigma, but helps with width matching
        )

    # Convert bandwidth from wavelength to frequency domain
    bandwidth_freq = (
        bandwidth * 1e-9 * omega_center**2 / (2 * np.pi * c)
    )  # rad/s
    sigma = bandwidth_freq / fwhm_to_sigma * 1e-15  # rad/fs

    # Create base amplitude based on pulse shape
    if pulse_shape == 'gaussian':
        base_amplitude = np.exp(-(delta_omega_fs**2) / (2 * sigma**2))
    elif pulse_shape == 'sech2':
        base_amplitude = 1 / np.cosh(delta_omega_fs / sigma)
    elif pulse_shape == 'square':
        base_amplitude = np.zeros_like(delta_omega_fs)
        idx = np.abs(delta_omega_fs) <= sigma
        base_amplitude[idx] = 1.0
    else:
        raise ValueError(f'Unsupported pulse shape: {pulse_shape}')

    # Apply spectral filter if specified
    if spectral_filter:
        if filter_center is None:
            filter_center = lambda_center

        filter_omega_center = 2 * np.pi * c / (filter_center * 1e-9)
        filter_delta = omega - filter_omega_center
        filter_delta_fs = filter_delta * 1e-15

        if filter_width is None:
            filter_width = bandwidth * 1.5

        filter_width_freq = (
            filter_width * 1e-9 * omega_center**2 / (2 * np.pi * c)
        )
        filter_sigma = filter_width_freq / fwhm_to_sigma * 1e-15

        if spectral_filter == 'gaussian':
            filter_profile = np.exp(
                -(filter_delta_fs**2) / (2 * filter_sigma**2)
            )
        elif spectral_filter == 'square':
            filter_profile = np.zeros_like(filter_delta_fs)
            idx = np.abs(filter_delta_fs) <= filter_sigma
            filter_profile[idx] = 1.0
        elif spectral_filter == 'edge':
            # Long-pass filter
            filter_profile = 1 / (
                1 + np.exp((filter_delta_fs - filter_sigma / 2) * 10)
            )
        else:
            raise ValueError(f'Unsupported filter type: {spectral_filter}')

        # Apply filter to base amplitude
        base_amplitude = base_amplitude * filter_profile

    # Apply chirp (complex phase)
    if chirp_param != 0:
        chirp_factor = np.exp(-1j * chirp_param * delta_omega_fs**2)
        complex_amplitude = base_amplitude * chirp_factor
    else:
        complex_amplitude = base_amplitude

    # Calculate intensity
    intensity = np.abs(complex_amplitude) ** 2

    # Normalize intensity
    if np.max(intensity) > 0:
        intensity = intensity / np.max(intensity)

    # Plot the results
    plt.figure(figsize=(12, 8))

    # Main plot - Intensity spectrum
    plt.subplot(2, 1, 1)
    plt.plot(wavelengths, intensity, 'b-', linewidth=2)
    plt.title(f'Pump Envelope Intensity (Chirp = {chirp_param} fs²)')
    plt.xlabel('Wavelength (nm)')
    plt.ylabel('Normalized Intensity')
    plt.grid(True, alpha=0.3)

    # Phase plot
    plt.subplot(2, 1, 2)
    plt.plot(wavelengths, np.angle(complex_amplitude), 'r-', linewidth=2)
    plt.title('Spectral Phase')
    plt.xlabel('Wavelength (nm)')
    plt.ylabel('Phase (radians)')
    plt.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.show()

    # If requested, show more detailed plots
    if plot_details and chirp_param != 0:
        plt.figure(figsize=(12, 8))

        # Plot amplitude and phase separately
        plt.subplot(2, 2, 1)
        plt.plot(
            wavelengths, np.abs(base_amplitude), 'g-', label='Base amplitude'
        )
        plt.plot(
            wavelengths, np.abs(complex_amplitude), 'b--', label='With chirp'
        )
        plt.title('Amplitude Components')
        plt.xlabel('Wavelength (nm)')
        plt.ylabel('Amplitude')
        plt.legend()
        plt.grid(True, alpha=0.3)

        # Plot intensity components
        plt.subplot(2, 2, 2)
        plt.plot(
            wavelengths,
            np.abs(base_amplitude) ** 2,
            'g-',
            label='Base intensity',
        )
        plt.plot(wavelengths, intensity, 'b--', label='With chirp')
        plt.title('Intensity Components')
        plt.xlabel('Wavelength (nm)')
        plt.ylabel('Normalized Intensity')
        plt.legend()
        plt.grid(True, alpha=0.3)

        # Plot real and imaginary parts
        plt.subplot(2, 2, 3)
        plt.plot(
            wavelengths, np.real(complex_amplitude), 'b-', label='Real part'
        )
        plt.plot(
            wavelengths,
            np.imag(complex_amplitude),
            'r-',
            label='Imaginary part',
        )
        plt.title('Real and Imaginary Components')
        plt.xlabel('Wavelength (nm)')
        plt.ylabel('Amplitude')
        plt.legend()
        plt.grid(True, alpha=0.3)

        # Plot the spectral chirp effect
        plt.subplot(2, 2, 4)
        plt.plot(wavelengths, chirp_param * delta_omega_fs**2, 'r-')
        plt.title('Quadratic Spectral Phase')
        plt.xlabel('Wavelength (nm)')
        plt.ylabel('Phase (radians)')
        plt.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.show()

    return wavelengths, complex_amplitude, intensity


def model_observed_spectrum(
    lambda_center=800,  # Adjust to match your central wavelength
    bandwidth=15,  # Adjust based on your spectrum width
    chirp_param=0.02,  # Substantial chirp
    tail_strength=2.5,  # Controls the prominence of the tail
    plot=True,
):
    """Model the asymmetric chirped spectrum with pronounced tail as observed in the experiment."""
    # Setup wavelength range
    num_points = 1000
    wavelength_range = 50  # Adjust based on your spectrum width
    wavelengths = np.linspace(
        lambda_center - wavelength_range,
        lambda_center + wavelength_range,
        num_points,
    )

    # Convert to angular frequency
    c = 299792458  # m/s
    omega_center = 2 * np.pi * c / (lambda_center * 1e-9)
    omega = 2 * np.pi * c / (wavelengths * 1e-9)
    delta_omega = omega - omega_center
    delta_omega_fs = delta_omega * 1e-15  # rad/fs

    # Parameters for asymmetric shape
    sigma = bandwidth * 1e-9 * omega_center**2 / (2 * np.pi * c) * 1e-15

    # Create asymmetric base profile using skewed Gaussian
    # This gives more control over the tail shape
    skew = -tail_strength  # Negative for left tail, positive for right tail
    skewed_x = delta_omega_fs + skew * np.abs(delta_omega_fs)
    base_amplitude = np.exp(-(skewed_x**2) / (2 * sigma**2))

    # Apply sharp cutoff on high end (right side in wavelength domain)
    cutoff_position = (
        0.5 * sigma
    )  # Adjust this to control where the sharp drop occurs
    cutoff_steepness = 25  # Higher value = sharper cutoff
    high_cut = 1 / (
        1 + np.exp(cutoff_steepness * (delta_omega_fs - cutoff_position))
    )

    # Apply gentle rise on low end (left side in wavelength domain)
    rise_position = (
        -2.0 * sigma
    )  # Adjust to control where the gentle rise starts
    rise_steepness = 3  # Lower value = more gradual rise
    low_cut = 1 - 1 / (
        1 + np.exp(-rise_steepness * (delta_omega_fs - rise_position))
    )

    # # Apply fine structure/noise
    # noise_amp = 0.04
    # np.random.seed(42)  # For reproducibility
    # noise = 1 + noise_amp * np.random.randn(num_points)
    # # Smooth the noise
    # from scipy.ndimage import gaussian_filter1d

    # noise = gaussian_filter1d(noise, sigma=2)

    # Apply chirp (phase)
    chirp_factor = np.exp(-1j * chirp_param * delta_omega_fs**2)

    # Combine all effects
    spectrum = base_amplitude * high_cut * low_cut * chirp_factor  # * noise
    intensity = np.abs(spectrum) ** 2
    intensity = intensity / np.max(intensity)

    if plot:
        plt.figure(figsize=(10, 6))
        plt.plot(wavelengths, intensity)
        plt.title('Chirped Pump Spectrum with Asymmetric Tail')
        plt.xlabel('Wavelength (nm)')
        plt.ylabel('Normalized Intensity')
        plt.grid(alpha=0.3)
        plt.show()

    return wavelengths, intensity
    return wavelengths, intensity


if __name__ == '__main__':
    # # Basic Gaussian pump with no chirp
    # realistic_pump_envelope(800, 15, chirp_param=0, pulse_shape='gaussian')

    # # Gaussian pump with moderate chirp
    # realistic_pump_envelope(800, 15, chirp_param=0.01, pulse_shape='gaussian')

    # # Sech² pulse with strong chirp
    # realistic_pump_envelope(800, 15, chirp_param=0.05, pulse_shape='sech2')

    # # Gaussian with spectral filtering (narrower bandwidth)
    # realistic_pump_envelope(
    #     800,
    #     15,
    #     chirp_param=0.01,
    #     pulse_shape='gaussian',
    #     spectral_filter='gaussian',
    #     filter_width=10,
    # )

    # # Square pulse with edge filter and chirp
    # realistic_pump_envelope(
    #     800,
    #     15,
    #     chirp_param=0.02,
    #     pulse_shape='square',
    #     spectral_filter='edge',
    #     filter_center=795,
    # )

    model_observed_spectrum()
