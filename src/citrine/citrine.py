from dataclasses import dataclass
from enum import Enum
from typing import Union, Literal, List, Optional, Tuple, Callable
import numpy as np
from numpy.typing import NDArray

__all__ = [
    'Magnitude',
    'AngularFrequency',
    'Wavelength',
    'spectral_window',
    'SellmeierCoefficients',
    '_permittivity',
    '_n_0',
    '_n_i',
    'refractive_index',
    'Orientation',
    'Crystal',
    'calculate_grating_period',
    'delta_k_matrix',
    'phase_matching_function',
    'pump_envelope_gaussian',
    'pump_envelope_sech2',
    'bandwidth_conversion',
    'Time',
    'hong_ou_mandel_interference',
]


# Constants
c = 299792458  # Speed of light in m/s


class Magnitude(Enum):
    """Enum to define the magnitude prefixes for units such as pico, nano, micro, etc."""

    pico = -12
    nano = -9
    micro = -6
    milli = -3
    base = 0
    kilo = 3
    mega = 6
    giga = 9


@dataclass
class AngularFrequency:
    """
    Class to represent angular frequency.

    Attributes:
        value (Union[float, np.ndarray]): The angular frequency.
    """

    value: Union[float, np.ndarray]


@dataclass
class Wavelength:
    """
    Class to represent wavelength with unit conversion.

    Attributes:
        value (Union[float, np.ndarray]): The wavelength value.
        unit (Magnitude): The unit of the wavelength (default: nano).
    """

    value: Union[float, np.ndarray]
    unit: Magnitude = Magnitude.nano

    def to_unit(self, new_unit: Magnitude) -> 'Wavelength':
        """
        Convert the wavelength to a new unit.

        Args:
            new_unit (Magnitude): The desired unit for the wavelength.

        Returns:
            Wavelength: New Wavelength object with converted units.
        """
        ratio: float = self.unit.value - new_unit.value
        return Wavelength(self.value * (10**ratio), new_unit)

    def to_absolute(self) -> 'Wavelength':
        """
        Convert the wavelength to base units (meters).

        Returns:
            Wavelength: Wavelength in meters.
        """
        return Wavelength(self.value * (10**self.unit.value), Magnitude.base)

    def as_angular_frequency(self) -> AngularFrequency:
        """
        Convert the wavelength to angular frequency.

        Returns:
            AngularFrequency: Angular frequency corresponding to the wavelength.
        """
        return AngularFrequency((2 * np.pi * c) / self.to_absolute().value)

    def as_wavevector(
        self, refractive_index: float = 1.0
    ) -> float | NDArray[np.floating]:
        """
        Convert the wavelength to a wavevector.

        Returns:
            float: The wavevector.
        """
        return (2 * np.pi * refractive_index) / self.to_absolute().value


def spectral_window(
    central_wavelength: Wavelength, spectral_width: Wavelength, steps: int
) -> Wavelength:
    """
    Generate an array of wavelengths within a specified spectral window.

    Args:
        central_wavelength (Wavelength): The central wavelength.
        spectral_width (Wavelength): The total spectral width.
        steps (int): The number of steps for the wavelength range.

    Returns:
        Wavelength: An array of wavelengths within the specified window.
    """
    width = spectral_width.to_unit(central_wavelength.unit).value / 2
    centre = central_wavelength.value
    return Wavelength(
        np.linspace(centre - width, centre + width, steps),
        central_wavelength.unit,
    )


@dataclass(frozen=True)
class SellmeierCoefficients:
    """
    Sellmeier coefficients for calculating refractive indices.

    Attributes:
        zeroth_order (Union[List[float], NDArray]): Zeroth-order Sellmeier coefficients.
        first_order (Union[List[float], NDArray]): First-order Sellmeier coefficients.
        second_order (Union[List[float], NDArray]): Second-order Sellmeier coefficients.
        temperature (float): The reference temperature for the coefficients (in Celsius).
    """

    first_order: Optional[Union[List[float], NDArray]]
    second_order: Optional[Union[List[float], NDArray]]
    temperature: float
    zeroth_order: Optional[Union[List[float], NDArray]] = None


def _permittivity(
    sellmeier: Union[List[float], NDArray],
    wavelength_um: Union[float, NDArray[np.floating]],
) -> Union[float, NDArray[np.floating]]:
    """
    Compute the permittivity using the Sellmeier equation.

    Args:
        sellmeier (Union[List[float], NDArray]): Sellmeier coefficients.
        wavelength_um (float): Wavelength in micrometers.

    Returns:
        float: The permittivity value.
    """

    first, *coeffs, last = sellmeier
    assert len(coeffs) % 2 == 0

    wl = wavelength_um**2
    p = first
    # TODO: comment whats going on here with the slice
    for numer, denom in zip(coeffs[0::2], coeffs[1::2]):
        p += numer / (1 - denom / wl)

    p -= last * wl
    return p


def _n_0(
    sellmeier: Union[List[float], NDArray],
    wavelength_um: Union[float, NDArray[np.floating]],
) -> Union[float, NDArray[np.floating]]:
    """
    Calculate the refractive index (n_0) using the zeroth-order Sellmeier coefficients.

    Args:
        sellmeier (Union[List[float], NDArray]): Zeroth-order Sellmeier coefficients.
        wavelength_um (float): Wavelength in micrometers.

    Returns:
        float: The refractive index (n_0).
    """
    return np.sqrt(_permittivity(sellmeier, wavelength_um))


def _n_i(
    sellmeier: Union[List[float], NDArray[np.floating]],
    wavelength_um: Union[float, NDArray[np.floating]],
) -> Union[float, NDArray[np.floating]]:
    """
    Calculate the refractive index (n_i) using the higher-order Sellmeier coefficients.

    Args:
        sellmeier (Union[List[float], NDArray]): Higher-order Sellmeier coefficients.
        wavelength_um (float): Wavelength in micrometers.

    Returns:
        float: The refractive index (n_i).
    """
    n = 0
    for i, s in enumerate(sellmeier):
        n += s / (wavelength_um**i)
    return n


def refractive_index(
    sellmeier: SellmeierCoefficients,
    wavelength: Wavelength,
    temperature: Optional[float] = None,
) -> Union[float, NDArray[np.floating]]:
    """
    Calculate the refractive index for a given wavelength and temperature using the Sellmeier equation.

    Args:
        sellmeier (SellmeierCoefficients): The Sellmeier coefficients for the material.
        wavelength (Wavelength): The wavelength at which to calculate the refractive index.
        temperature (Optional[float]): The temperature (in Celsius), defaults to the reference temperature.

    Returns:
        float: The refractive index at the given wavelength and temperature.
    """
    if temperature is None:
        temperature = sellmeier.temperature

    wavelength_um = wavelength.to_unit(Magnitude.micro).value
    n0: Union[float, NDArray[np.floating]] = 0.0
    if sellmeier.zeroth_order is not None:
        n0 = _n_0(sellmeier.zeroth_order, wavelength_um)

    n1 = None
    if sellmeier.first_order is not None:
        n1 = _n_i(sellmeier.first_order, wavelength_um)

    n2 = None
    if sellmeier.second_order is not None:
        n2 = _n_i(sellmeier.second_order, wavelength_um)

    t_offset = temperature - sellmeier.temperature

    if n1 is None:
        n1 = 0.0

    if n2 is None:
        n2 = 0.0

    n = n0 + (n1 * t_offset) + (n2 * t_offset**2)
    return n


class Orientation(Enum):
    """Enum to represent the orientation: ordinary or extraordinary."""

    ordinary = 0
    extraordinary = 1


@dataclass
class Crystal:
    """
    Class to represent a nonlinear crystal for refractive index and phase matching calculations.

    Attributes:
        name (str): Name of the crystal.
        sellmeier_o (SellmeierCoefficients): Ordinary Sellmeier coefficients.
        sellmeier_e (SellmeierCoefficients): Extraordinary Sellmeier coefficients.
        pump_orientation (Orientation): Pump photon orientation.
        signal_orientation (Orientation): Signal photon orientation.
        idler_orientation (Orientation): Idler photon orientation.
    """

    name: str
    sellmeier_o: SellmeierCoefficients
    sellmeier_e: SellmeierCoefficients
    pump_orientation: Orientation
    signal_orientation: Orientation
    idler_orientation: Orientation

    def refractive_index(
        self,
        wavelength: Wavelength,
        polarization: Orientation,
        photon: Literal['pump', 'signal', 'idler'],
        temperature: Optional[float] = None,
    ) -> Union[float, NDArray[np.floating]]:
        """
        Calculate the refractive index for a given wavelength and polarization using the Sellmeier equation.

        Args:
            wavelength (Wavelength): Wavelength.
            polarization (Literal['ordinary', 'extraordinary']): Polarization ('ordinary' or 'extraordinary').
            photon (Literal['pump', 'signal', 'idler']): Photon type.
            temperature (Optional[float]): Temperature in Celsius (defaults to reference temperature).

        Returns:
            float: Refractive index.
        """

        if polarization == Orientation.ordinary:
            sellmeier = self.sellmeier_o
        elif polarization == Orientation.extraordinary:
            sellmeier = self.sellmeier_e
        else:
            raise ValueError(
                "Polarization must be either 'ordinary' or 'extraordinary'."
            )

        return refractive_index(sellmeier, wavelength, temperature)

    def refractive_indices(
        self,
        pump_wavelength: Wavelength,
        signal_wavelength: Wavelength,
        idler_wavelength: Wavelength,
        temperature: Optional[float] = None,
    ) -> Tuple[
        Union[float, NDArray[np.floating]],
        Union[float, NDArray[np.floating]],
        Union[float, NDArray[np.floating]],
    ]:
        """
        Calculate the refractive indices for the pump, signal, and idler photons.

        Args:
            pump_wavelength (Wavelength): Pump photon wavelength.
            signal_wavelength (Wavelength): Signal photon wavelength.
            idler_wavelength (Wavelength): Idler photon wavelength.
            temperature (Optional[float]): Temperature in Celsius (defaults to reference temperature).

        Returns:
            Tuple[float, float, float]: Refractive indices for the pump, signal, and idler photons.
        """

        # TODO: refactor, there should be a convenient way to remove the "if" check on each call self.refractive_index

        n_pump = self.refractive_index(
            pump_wavelength, self.pump_orientation, 'pump', temperature
        )
        n_signal = self.refractive_index(
            signal_wavelength,
            self.signal_orientation,
            'signal',
            temperature,
        )
        n_idler = self.refractive_index(
            idler_wavelength, self.idler_orientation, 'idler', temperature
        )
        return n_pump, n_signal, n_idler


def calculate_grating_period(
    lambda_p_central: Wavelength,
    lambda_s_central: Wavelength,
    lambda_i_central: Wavelength,
    crystal: Crystal,
) -> Union[float, NDArray[np.floating]]:
    """
    Calculate the grating period (Λ) for the phase matching condition.

    Args:
        lambda_p_central (Wavelength): Central wavelength of the pump.
        lambda_s_central (Wavelength): Central wavelength of the signal.
        lambda_i_central (Wavelength): Central wavelength of the idler.

    Returns:
        float: Grating period in microns.
    """

    # n_s = crystal.refractive_index(lambda_s_central, polarization)
    # n_i = crystal.refractive_index(lambda_i_central, polarization)
    # n_p = crystal.refractive_index(lambda_p_central, polarization)

    (n_p, n_s, n_i) = crystal.refractive_indices(
        lambda_p_central, lambda_s_central, lambda_i_central
    )

    k_s = (2 * np.pi * n_s) / lambda_s_central.to_absolute().value
    k_i = (2 * np.pi * n_i) / lambda_i_central.to_absolute().value
    k_p = (2 * np.pi * n_p) / lambda_p_central.to_absolute().value

    return 2 * np.pi / (k_p - k_s - k_i)


def delta_k_matrix(
    lambda_p: Wavelength,
    lambda_s: Wavelength,
    lambda_i: Wavelength,
    crystal: Crystal,
) -> Union[float, NDArray[np.floating]]:
    """
    Calculate the Δk matrix for the phase matching function using the wavevector.

    Args:
        lambda_p (Wavelength): Central wavelength of the pump.
        lambda_s (Wavelength): Signal wavelengths (Wavelength type).
        lambda_i (Wavelength): Idler wavelengths (Wavelength type).
        grating_period (float): Grating period in microns.

    Returns:
        np.ndarray: A 2D matrix of Δk values.
    """
    # Generate a grid of signal and idler wavelengths
    lambda_s_grid, lambda_i_grid = np.meshgrid(
        (2 * np.pi * c) / lambda_s.to_absolute().value,
        (2 * np.pi * c) / lambda_i.to_absolute().value,
        indexing='ij',
    )

    wl_s = Wavelength((2 * np.pi * c) / lambda_s_grid, Magnitude.base)
    wl_i = Wavelength((2 * np.pi * c) / lambda_i_grid, Magnitude.base)
    wl_p = Wavelength(
        (2 * np.pi * c) / (lambda_s_grid + lambda_i_grid),
        Magnitude.base,
    )

    (n_p, n_s, n_i) = crystal.refractive_indices(wl_p, wl_s, wl_i)

    k_s = (2 * np.pi * n_s) / wl_s.value
    k_i = (2 * np.pi * n_i) / wl_i.value
    k_p = (2 * np.pi * n_p) / wl_p.value

    # k_s = wl_s.as_wavevector(n_s)
    # k_i = wl_s.as_wavevector(n_i)
    # k_p = wl_s.as_wavevector(n_p)

    # Δk calculation
    delta_k = k_p - k_s - k_i

    return delta_k


def phase_matching_function(
    delta_k: np.ndarray, grating_period, crystal_length
) -> np.ndarray:
    """
    Compute the phase matching function as a sinc function of Δk.

    Args:
        delta_k (np.ndarray): The Δk matrix from the phase matching condition.

    Returns:
        np.ndarray: The phase matching function.
    """
    x = (delta_k - (2 * np.pi / grating_period)) * (crystal_length / 2)

    return np.sinc(x)


def pump_envelope_gaussian(
    lambda_p: Wavelength,
    sigma_p: float,
    lambda_s: Wavelength,
    lambda_i: Wavelength,
) -> np.ndarray:
    """
    Generate a pump envelope matrix with a Gaussian profile.

    Args:
        lambda_p (Wavelength): Central wavelength of the pump.
        sigma_p (float): Pump bandwidth.
        lambda_s (Wavelength): Signal wavelengths array.
        lambda_i (Wavelength): Idler wavelengths array.

    Returns:
        np.ndarray: A 2D pump envelope matrix.
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
    pump_envelope = np.exp(-(((omega_s + omega_i - omega_p) / sigma_p) ** 2))

    return pump_envelope


def pump_envelope_sech2(
    lambda_p: Wavelength,
    sigma_p: float,  # TODO: pump width needs its own type
    lambda_s: Wavelength,
    lambda_i: Wavelength,
) -> np.ndarray:
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

    pump = (1 / np.cosh(np.pi * sigma_p * (omega_s + omega_i - omega_p))) ** 2
    return pump


def bandwidth_conversion(
    delta_lambda_FWHM: Wavelength, pump_wl: Wavelength
) -> float:
    """
    Convert pump bandwidth from FWHM in wavelength to frequency.

    Args:
        delta_lambda_FWHM (Wavelength): Bandwidth in wavelength units.
        pump_wl (Wavelength): Central pump wavelength.

    Returns:
        float: Converted bandwidth.
    """
    delta_nu_FWHM = (
        c
        * delta_lambda_FWHM.to_absolute().value
        / pump_wl.to_absolute().value ** 2
    )
    delta_nu = delta_nu_FWHM / np.sqrt(2 * np.log(2))
    return delta_nu


@dataclass
class Time:
    """
    Class to represent time in chosen units

    Attributes:
        value (Union[float, np.ndarray]): The time in chosen units
        unit (Magnitude): The unit of time

    """

    value: Union[float, NDArray[np.floating]]
    unit: Magnitude

    def count(self) -> int:
        n: int = 1

        try:
            n = len(self.value)
        except TypeError:
            n = 1

        return n

    def as_array(self) -> NDArray[np.floating]:
        data: NDArray[np.floating]
        if self.count() > 1:
            data = self.value
        else:
            data = np.asarray([self.value])
        return data

    def to_unit(self, new_unit: Magnitude) -> 'Time':
        """
        Convert the time to a new unit.

        Args:
            new_unit (Magnitude): The desired unit for the Time.

        Returns:
            time: New Time object with converted units.
        """
        ratio: float = self.unit.value - new_unit.value
        return Time(self.as_array() * (10**ratio), new_unit)

    def to_absolute(self) -> 'Time':
        """
        Convert the time to base units (seconds).

        Returns:
            Time: Time in seconds.
        """
        return Time(self.as_array() * (10**self.unit.value), Magnitude.base)


def hong_ou_mandel_interference(
    joint_spectral_amplitude,
    lambda_s: Wavelength,
    lambda_i: Wavelength,
    delays: Time,
) -> Tuple[NDArray, float, float, float]:
    """
    Calculates HOM interference between two single-photon spectra

    Arguments:

    Returns:

    """

    u, d, v = np.linalg.svd(joint_spectral_amplitude)
    d_diag = np.diag(d)
    u_conj = np.conj(u)

    probabilities: NDArray[np.floating] = np.zeros(
        delays.count(), dtype=np.float64
    )

    freq_s = lambda_s.as_angular_frequency().value
    freq_i = lambda_i.as_angular_frequency().value

    delays_absolute = delays.to_absolute().as_array()

    for i, tau in enumerate(delays_absolute):
        intereference_s = np.dot(v * np.exp(-1j * freq_s * tau), u_conj)

        intereference_i = np.dot(u.T, np.conj(v * np.exp(-1j * freq_i * tau)).T)

        interference = intereference_s * intereference_i
        interference = np.dot(d_diag, np.dot(interference, d_diag))

        probabilities[i] = np.abs(interference.sum())

    probabilities = probabilities / probabilities.max()

    probabilities = 0.5 - (0.5 * probabilities)

    d = d / np.sqrt(np.sum(d * np.conj(d)))
    purity = np.sum((d * np.conj(d)) ** 2)
    schmidt_number = 1 / np.sum(d**4)
    entropy = -np.sum((d**2) * np.log2(d**2))

    return probabilities, purity, schmidt_number, entropy


def _apodisation_amplitude(
    domain_width: float,
    poling_period: float,
    index: int,
    orientation: Union[List[int], NDArray],
):
    total = 0.0
    for i in range(index):
        total += (
            np.exp((2 * 1j * (i + 1) * np.pi * domain_width) / poling_period)
            * orientation[i]
        )

    amplitude = (
        (poling_period / (2 * np.pi))
        * (np.exp(-1j * ((2 * np.pi) / poling_period) * domain_width) - 1)
        * total
    )

    return amplitude


def _apodisation_error(
    domain_width: float,
    poling_period: float,
    index: int,
    orientation: Union[List[int], NDArray],
    target: complex,
):
    amplitude = _apodisation_amplitude(
        domain_width, poling_period, index, orientation
    )
    error = np.abs(amplitude - target)
    return error


def apodisation(
    crystal_length: float,
    poling_period: float,
    divisions: float,
    target_function: Callable[[float], complex],
):
    domain_width = poling_period / (2 * divisions)
    domains = round(crystal_length / domain_width)

    orientation = []
    orientation_up = []
    orientation_down = []

    error_up = []
    error_down = []

    for i in range(domains):
        orientation_up = orientation + [1]
        orientation_down = orientation + [-1]

        j = i + 1
        target = target_function(j * domain_width)

        error_up = _apodisation_error(
            domain_width, poling_period, j, orientation_up, target
        )
        error_down = _apodisation_error(
            domain_width, poling_period, j, orientation_down, target
        )

        if error_up <= error_down:
            orientation = orientation_up
        else:
            orientation = orientation_down

    # NOTE: this is essentially just run-length encoding, find someway to simplify
    domain_lengths = []  # TODO: should probably be called crystal_wall_position?
    run_length: int = 1
    for i in range(1, len(orientation)):
        if orientation[i] == orientation[i - 1]:
            run_length += 1
        else:
            domain_lengths += [run_length * domain_width]
            run_length = 1

    return (domain_lengths, orientation)
