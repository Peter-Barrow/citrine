from dataclasses import dataclass
from enum import Enum
from typing import Union, List, Optional, Tuple, Callable
import numpy as np
from numpy.typing import NDArray
from scipy.optimize import root

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
    'Photon',
    'Crystal',
    'calculate_grating_period',
    'delta_k_matrix',
    'phase_matching_function',
    'phase_mismatch',
    'joint_spectral_amplitude',
    'PhotonType',
    'calculate_marginal_spectrum',
    'calculate_jsa_marginals',
    'bandwidth_conversion',
    'Time',
    'Bunching',
    'hom_interference_from_jsa',
    'hong_ou_mandel_interference',
    'spectral_purity',
    'wavelength_temperature_tuning',
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

    def downconversion_result(
        self, downconversion: 'Wavelength'
    ) -> 'Wavelength':
        wvl = 1 / (
            (1 / self.to_absolute().value)
            - (1 / downconversion.to_absolute().value)
        )
        return Wavelength(abs(wvl), Magnitude.base)


def spectral_window(
    central_wavelength: Wavelength,
    spectral_width: Wavelength,
    steps: int,
    reverse: bool = False,
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

    if reverse is False:
        wvls = np.linspace(centre - width, centre + width, steps)
    else:
        wvls = np.linspace(centre + width, centre - width, steps)

    return Wavelength(wvls, central_wavelength.unit)


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

    This modification adds comments to explain each step of the Sellmeier
    equation calculation. The code is implementing the Sellmeier equation,
    which is used to determine the refractive index of a material as a
    function of wavelength.

    The equation typically takes the form:

    $$
    n^2 = A + \\frac{B * λ^2}{λ^2 - C} + \\frac{D * λ^2}{λ^2 - E} + \\ldots
    $$

    Where n is the refractive index, λ is the wavelength, and A, B, C, D, E,
    etc. are the Sellmeier coefficients specific to the material.

    Args:
        sellmeier (Union[List[float], NDArray]): Sellmeier coefficients.
        wavelength_um (float): Wavelength in micrometers.

    Returns:
        float: The permittivity value.
    """

    # Take array of sellemeier coefficients of the form [A, B, C, D, E, ...]
    # and split into A, [B, C, ...], E
    first, *coeffs, last = sellmeier
    assert len(coeffs) % 2 == 0

    # Square the wavelength (in micrometers)
    wl = wavelength_um**2

    # Initialize the permittivity with the first coefficient
    p = first

    # Iterate through the Sellmeier coefficients in pairs
    # coeffs[0::2] correspond to B, D, ... -> numerator
    # coeffs[1::2] correspond to C, E, ... -> denominator
    for number, denom in zip(coeffs[0::2], coeffs[1::2]):
        # Apply the Sellmeier equation term:
        # Add (numerator) / (1 - denominator / wavelength^2)
        p += number / (1 - denom / wl)

    # Apply the final term of the Sellmeier equation
    # Subtract the last coefficient multiplied by the squared wavelength
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


class PhaseMatchingCondition(Enum):
    """Phase-matching configurations for nonlinear optical processes.

    Each configuration defines the polarization orientations for (pump, signal, idler)
    where:
        - o/O: ordinary polarization (horizontal, H)
        - e/E: extraordinary polarization (vertical, V)

    The configurations are:
        Type 0 (o): All waves ordinary polarized (H,H,H)
        Type 0 (e): All waves extraordinary polarized (V,V,V)
        Type 1: Pump extraordinary, signal/idler ordinary (V,H,H)
        Type 2 (o): Mixed polarizations (V,H,V)
        Type 2 (e): Mixed polarizations (V,V,H)

    Note:
        - Ordinary (o) corresponds to horizontal (H) polarization
        - Extraordinary (e) corresponds to vertical (V) polarization
        - The tuple order is always (pump, signal, idler)
    """

    type0_o = (
        Orientation.ordinary,  # pump: H
        Orientation.ordinary,  # signal: H
        Orientation.ordinary,  # idler: H
    )
    type0_e = (
        Orientation.extraordinary,  # pump: V
        Orientation.extraordinary,  # signal: V
        Orientation.extraordinary,  # idler: V
    )
    type1 = (
        Orientation.extraordinary,  # pump: V
        Orientation.ordinary,  # signal: H
        Orientation.ordinary,  # idler: H
    )
    type2_o = (
        Orientation.extraordinary,  # pump: V
        Orientation.ordinary,  # signal: H
        Orientation.extraordinary,  # idler: V
    )
    type2_e = (
        Orientation.extraordinary,  # pump: V
        Orientation.extraordinary,  # signal: V
        Orientation.ordinary,  # idler: H
    )


class Photon(Enum):
    pump = 0
    signal = 1
    idler = 2


class Crystal:
    def __init__(
        self,
        name: str,
        sellmeier_o: SellmeierCoefficients,
        sellmeier_e: SellmeierCoefficients,
        phase_matching: PhaseMatchingCondition = PhaseMatchingCondition.type0_e,
        doi: str = None,
    ):
        """
        Class to represent a nonlinear crystal for refractive index and phase matching calculations.

        Attributes:
            name (str): Name of the crystal.
            sellmeier_o (SellmeierCoefficients): Ordinary Sellmeier coefficients.
            sellmeier_e (SellmeierCoefficients): Extraordinary Sellmeier coefficients.
            phase_matching (PhaseMatchingCondition): Phase matching conditions.
        """
        self.name = name
        self.sellmeier_o = sellmeier_o
        self.sellmeier_e = sellmeier_e
        self.phase_matching = phase_matching
        self.doi = doi

    @property
    def phase_matching(self) -> PhaseMatchingCondition:
        """Get the current phase matching condition.

        Returns:
            Current phase matching configuration.
        """
        return self._phase_matching

    @phase_matching.setter
    def phase_matching(self, condition: PhaseMatchingCondition):
        """Set the phase matching condition.

        Args:
            condition: New phase matching configuration.

        Raises:
            ValueError: If condition is not a valid PhaseMatchingCondition.
        """
        if not isinstance(condition, PhaseMatchingCondition):
            raise ValueError(
                f'Phase matching condition must be a PhaseMatchingCondition, not {type(condition)}'
            )
        self._phase_matching = condition

    def _sellmeier(self, polarisation: Orientation) -> SellmeierCoefficients:
        if not isinstance(polarisation, Orientation):
            raise ValueError(
                f'Polarisation must be an Orientation, not {type(polarisation)}'
            )
        if polarisation == Orientation.ordinary:
            return self.sellmeier_o
        return self.sellmeier_e

    def refractive_index(
        self,
        wavelength: Wavelength,
        photon: Photon,
        temperature: Optional[float] = None,
    ) -> Union[float, NDArray[np.floating]]:
        """
        Calculate the refractive index for a given wavelength and polarization using the Sellmeier equation.

        Args:
            wavelength (Wavelength): Wavelength.
            photon (Photon): Photon type.
            temperature (Optional[float]): Temperature in Celsius (defaults to reference temperature).

        Returns:
            float: Refractive index.
        """

        sellmeier = self._sellmeier(self.phase_matching.value[photon.value])
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
            pump_wavelength, Photon.pump, temperature
        )
        n_signal = self.refractive_index(
            signal_wavelength, Photon.signal, temperature
        )
        n_idler = self.refractive_index(
            idler_wavelength, Photon.idler, temperature
        )
        return n_pump, n_signal, n_idler


def calculate_grating_period(
    lambda_p_central: Wavelength,
    lambda_s_central: Wavelength,
    lambda_i_central: Wavelength,
    crystal: Crystal,
    temperature: float = None,
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
        lambda_p_central,
        lambda_s_central,
        lambda_i_central,
        temperature=temperature,
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
    temperature: float = None,
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

    (n_p, n_s, n_i) = crystal.refractive_indices(
        wl_p,
        wl_s,
        wl_i,
        temperature=temperature,
    )

    k_s = (2 * np.pi * n_s) / wl_s.value
    k_i = (2 * np.pi * n_i) / wl_i.value
    k_p = (2 * np.pi * n_p) / wl_p.value

    # k_s = wl_s.as_wavevector(n_s)
    # k_i = wl_s.as_wavevector(n_i)
    # k_p = wl_s.as_wavevector(n_p)

    # Δk calculation
    delta_k = k_p - k_s - k_i

    return delta_k


# Phase matching condition function
def phase_mismatch(
    lambda_p: Wavelength,
    lambda_s: Wavelength,
    crystal: Crystal,
    temperature: float,
    grating_period: float,
):
    _wl_p = lambda_p.to_absolute().value
    _wl_s = lambda_s.to_absolute().value
    _wl_i = 1.0 / ((1.0 / _wl_p) - (1.0 / _wl_s))

    wl_s = Wavelength((2 * np.pi * c) / _wl_s, Magnitude.base)
    wl_i = Wavelength((2 * np.pi * c) / _wl_i, Magnitude.base)
    wl_p = Wavelength((2 * np.pi * c) / _wl_p, Magnitude.base)

    (n_p, n_s, n_i) = crystal.refractive_indices(
        wl_p,
        wl_s,
        wl_i,
        temperature=temperature,
    )

    # Calculate k vectors
    k_p = (2 * np.pi * n_p) / wl_p.value
    k_s = (2 * np.pi * n_s) / wl_s.value
    k_i = (2 * np.pi * n_i) / wl_i.value
    k_g = (2 * np.pi) / (grating_period)  # Grating vector

    # Phase mismatch
    return k_p - k_s - k_i - k_g


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


def joint_spectral_amplitude(
    phase_mismatch_matrix: np.ndarray,
    pump_envelope_matrix: np.ndarray,
    normalisation: bool = True,
) -> np.ndarray:
    jsa_raw = pump_envelope_matrix * phase_mismatch_matrix

    norm = 1
    if normalisation:
        norm = np.sqrt(np.sum(np.abs(jsa_raw) ** 2))

    jsa = jsa_raw / norm
    return jsa


class PhotonType(Enum):
    IDLER = 0
    SIGNAL = 1


def calculate_marginal_spectrum(
    jsa_matrix: np.ndarray,
    lambda_s: Wavelength,
    lambda_i: Wavelength,
    photon_type: PhotonType,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Calculate the marginal spectrum for either signal or idler photons from a JSA matrix.

    Args:
        jsa_matrix: A 2D complex matrix representing the joint spectral amplitude.
                   Shape (len(lambda_s), len(lambda_i)) with indexing='ij'.
        lambda_s: Signal wavelengths (Wavelength type).
        lambda_i: Idler wavelengths (Wavelength type).
        photon_type: Enum specifying whether to calculate for SIGNAL or IDLER photon.

    Returns:
        Tuple containing:
            - Array of wavelengths for the selected photon
            - Array of corresponding intensity values (complex)
    """

    pt = photon_type.value
    assert (pt == 0) or (pt == 1), 'photon_type must be SIGNAL OR IDLER'

    intensities = np.sum(jsa_matrix.T, axis=pt)
    wavelengths = lambda_i.value

    # if photon_type == PhotonType.IDLER:
    #     intensities = np.flipud(intensities)

    return wavelengths, intensities


def calculate_jsa_marginals(
    jsa_matrix: np.ndarray,
    lambda_s: Wavelength,
    lambda_i: Wavelength,
) -> dict:
    """
    Calculate both signal and idler marginal spectra from a JSA matrix.

    Args:
        jsa_matrix: A 2D complex matrix representing the joint spectral amplitude.
                   Shape (len(lambda_s), len(lambda_i)) with indexing='ij'.
        lambda_s: Signal wavelengths (Wavelength type).
        lambda_i: Idler wavelengths (Wavelength type).

    Returns:
        Dictionary containing:
            - 'signal_wl': Array of signal wavelengths
            - 'signal_intensity': Array of signal intensity values
            - 'idler_wl': Array of idler wavelengths
            - 'idler_intensity': Array of idler intensity values
    """
    # Calculate signal marginal
    signal_wl, signal_intensity = calculate_marginal_spectrum(
        jsa_matrix, lambda_s, lambda_i, PhotonType.SIGNAL
    )

    # Calculate idler marginal
    idler_wl, idler_intensity = calculate_marginal_spectrum(
        jsa_matrix, lambda_s, lambda_i, PhotonType.IDLER
    )

    return {
        'signal_wl': signal_wl,
        'signal_intensity': signal_intensity,
        'idler_wl': idler_wl,
        'idler_intensity': np.flipud(idler_intensity),
    }


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


class Bunching(Enum):
    Bunching = -1
    AntiBunching = 1


def hom_interference_from_jsa(
    joint_spectral_amplitude: np.ndarray,
    wavelengths_signal: Wavelength,
    wavelengths_idler: Wavelength,
    time_delay: float = 0.0,
    bunching: Bunching = Bunching.Bunching,
) -> Tuple[float, np.ndarray]:
    """Calculates the Hong-Ou-Mandel (HOM) interference pattern and total
        coincidence rate.

    This function computes the quantum interference that occurs when two
        photons enter a 50:50 beam splitter from different inputs. The
        interference pattern depends on the joint spectral amplitude (JSA) of
        the photon pair and the time delay between their arrivals at the beam
        splitter.

    The calculation follows these steps:
    1. Create the second JSA with swapped signal and idler (anti-transpose)
    2. Apply a phase shift based on the time delay and wavelength difference
    3. Calculate the interference between the two quantum pathways
    4. Compute the joint spectral intensity (JSI) by taking the squared
        magnitude

    Args:
        jsa (np.ndarray): Complex 2D numpy array of shape (M, N) representing
            the Joint Spectral Amplitude. The first dimension corresponds to
            signal wavelengths and the second dimension to idler wavelengths.
        wavelengths_signal (Wavelength): 1D array of signal wavelengths in
            meters, must match the first dimension of jsa.
        wavelengths_idler (Wavelength): 1D array of idler wavelengths in
            meters, must match the second dimension of jsa. Note: should be in
                descending order to match standard convention.
        time_delay (float = 0.0): Time delay between signal and idler photons
            in seconds. At zero delay, maximum quantum interference occurs for
            indistinguishable photons.
        bunching (Bunching = Bunching.Bunching): Type of interference to be
            simulated

    Returns:
        Tuple containing:
            - float: Total coincidence probability (sum of all JSI values).
            - np.ndarray: 2D array representing the joint spectral intensity
                after interference.

    Notes:
        Hong-Ou-Mandel interference results from the destructive interference
            between two indistinguishable two-photon quantum states. For
            perfectly indistinguishable photons at zero time delay, they will
            always exit the beam splitter together from the same port,
            resulting in zero coincidence count rate (the HOM dip).

        The time delay introduces a wavelength-dependent phase shift to one
            pathway, reducing the interference and increasing the coincidence
            probability.
    """

    # Verify inputs
    check = joint_spectral_amplitude.shape != (
        len(wavelengths_signal.value),
        len(wavelengths_idler.value),
    )

    if check:
        raise ValueError(
            'Joint Spectral Amplitude dimensions must match wavelengths'
        )

    Signal, Idler = np.meshgrid(
        1 / wavelengths_signal.to_absolute().value,
        1 / wavelengths_idler.to_absolute().value,
        indexing='ij',
    )

    frequency_difference = Signal - Idler

    # Calculate phase shift from time delay
    # The factor 2π·c·(1/λ_s - 1/λ_i)·Δt represents the phase accumulated due
    #   to the different arrival times of different wavelength components
    phase = 2 * np.pi * c * frequency_difference * time_delay

    phase_factor = np.cos(phase) + 1j * np.sin(phase)

    joint_spectral_amplitude_delayed = (
        np.flipud(np.fliplr(joint_spectral_amplitude.T)) * phase_factor
    )

    # The choice of bunching or antibunching sets the type of interference
    #   being simulated, positive corresponds to constructive interference
    #   (bunching) and negative corresponds to destructive interference
    #   (antibunching)
    sign = float(bunching.value)

    # Calculate the interference between the signal and idler photons
    joint_spectral_amplitude_interference = (
        joint_spectral_amplitude + (sign * joint_spectral_amplitude_delayed)
    ) / 2

    # Calculate Joint Spectral Intensity by taking squared magnitude of
    # amplitude -> |JSA|² = |Re(JSA)|² + |Im(JSA)|²
    joint_spectral_intensity = (
        np.abs(joint_spectral_amplitude_interference) ** 2
    )

    # Calculate total coincidence probability (area under the JSI)
    coincidence_probability = np.sum(joint_spectral_intensity)

    return coincidence_probability, joint_spectral_intensity


def hong_ou_mandel_interference(
    jsa: np.ndarray,
    signal_wavelengths: np.ndarray,
    idler_wavelengths: np.ndarray,
    time_delays: np.ndarray,
    bunching: Bunching = Bunching.Bunching,
) -> np.ndarray:
    """Calculates the complete Hong-Ou-Mandel dip by scanning the time delays.

    This function generates the characteristic HOM dip by calculating the
        coincidence rate at each time delay value in the provided array.

    Args:
        jsa: Complex 2D numpy array representing the Joint Spectral Amplitude.
        signal_wavelengths: 1D array of signal wavelengths in meters.
        idler_wavelengths: 1D array of idler wavelengths in meters.
        time_delays: 1D array of time delay values in seconds to scan across.

    Returns:
        np.ndarray: 1D array of coincidence rates corresponding to each time
            delay, showing the characteristic HOM dip.

    Physics Notes:
        The width of the HOM dip is inversely proportional to the spectral
            bandwidth of the photons. Spectrally narrower photons produce wider
            dips, while broader bandwidth photons produce narrower dips,
            demonstrating the time-frequency uncertainty principle.

        The shape and visibility of the dip reveals information about the
        spectral entanglement and distinguishability of the photon pairs.
    """
    # Initialize array to store coincidence rates
    coincidence_rates = np.zeros(len(time_delays))

    # Calculate rate for each time delay
    for i, delay in enumerate(time_delays):
        rate, _ = hom_interference_from_jsa(
            jsa,
            signal_wavelengths,
            idler_wavelengths,
            delay,
            bunching,
        )
        coincidence_rates[i] = rate

    return coincidence_rates


def spectral_purity(jsa: np.ndarray) -> Tuple[float, float, float]:
    """Calculate the spectral purity of a biphoton state

    This function calculate the spectral purity, schmidt number and entropy of
        a biphoton state. This is achieved by performing a singular value
        decomposition (SVD) a discrete analogue to the Schmidt decomposition
        (continuous space).

    Args:
        joint_spectral_amplitude: 2D array representing the joint spectral
            amplitude.

    Returns:
        tuple: A tuple containing:
            - probabilities (NDArray): Normalized HOM interference pattern
            - purity (float): Spectral purity of the state (0 to 1)
            - schmidt_number (float): Schmidt number indicating mode
                entanglement

    Notes:
        The function performs these steps:
        1. SVD decomposition of the JSA
        2. Computes quantum metrics (purity, Schmidt number, entropy)

    """
    _, d, _ = np.linalg.svd(jsa)
    d = d / np.sqrt(np.sum(d * np.conj(d)))
    purity = np.sum((d * np.conj(d)) ** 2)

    schmidt_number = 1 / np.sum(d**4)
    entropy = -np.sum((d**2) * np.log2(d**2))

    return purity, schmidt_number, entropy


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


def wavelength_temperature_tuning(
    lambda_p: Wavelength,  # Pump wavelength (nm)
    lambda_s: Wavelength,  # Pump wavelength (nm)
    lambda_i: Wavelength,  # Pump wavelength (nm)
    poling_period: float,  # Crystal poling period (μm)
    crystal: Crystal,
    temp_range: tuple,  # Temperature range to evaluate (°C)
    num_points: int = 50,  # Number of calculation points
) -> tuple:
    """Calculate wavelength tuning curves as a function of temperature for a nonlinear crystal.

    This function calculates how the signal and idler wavelengths vary with temperature
    in a quasi-phase-matched nonlinear crystal. It uses the phase-matching condition:
    $$ \\Delta k = k_p - k_s - k_i - k_g = 0 $$
    where $k_p$, $k_s$, $k_i$ are the wave vectors for pump, signal, and idler,
    and $k_g$ is the grating vector from the crystal's poling period.

    Args:
        lambda_p: Pump wavelength object.
        lambda_s: Initial signal wavelength object.
        lambda_i: Initial idler wavelength object.
        poling_period: Crystal poling period in micrometers.
        crystal: Crystal object containing material properties and orientations.
        temp_range: Tuple of (min_temperature, max_temperature) in degrees Celsius.
        num_points: Number of temperature points to evaluate (default: 50).

    Returns:
        tuple: A tuple containing:
            - temperature array (ndarray): Temperature points in degrees Celsius
            - signal_wavelengths (ndarray): Calculated signal wavelengths
            - idler_wavelengths (ndarray): Calculated idler wavelengths

    Notes:
        The function solves the phase-matching equation at each temperature point
        to find the signal and idler wavelengths that satisfy energy conservation:
        $$ \\frac{1}{\\lambda_p} = \\frac{1}{\\lambda_s} + \\frac{1}{\\lambda_i} $$
        If no solution is found at a particular temperature, NaN values are returned
        for that point.
    """
    wl_p = lambda_p.to_absolute().value
    wl_s = lambda_s.to_absolute().value
    wl_i = lambda_i.to_absolute().value

    (n_p, n_s, _) = crystal.refractive_indices(lambda_p, lambda_s, lambda_i)

    k_p = (2 * np.pi * n_p) / wl_p
    k_s = (2 * np.pi * n_s) / wl_s
    # k_i = (2 * np.pi * n_i) / wl_i
    k_g = (2 * np.pi) / (poling_period)  # Grating vector

    signal_wavelengths = np.zeros(num_points)
    idler_wavelengths = np.zeros(num_points)

    temperature = np.linspace(temp_range[0], temp_range[1], num_points)

    for i, T in enumerate(temperature):

        def phase_mismatch_at_T(wl_s):
            # _lambda_s = Wavelength(wl_s, Magnitude.base)
            lambda_i_value = 1.0 / (
                (1.0 / lambda_p.to_absolute().value) - (1.0 / wl_s)
            )
            lambda_i_obj = Wavelength(lambda_i_value, Magnitude.base)
            _n_i = crystal.refractive_index(lambda_i_obj, Photon.idler, T)
            k_i = (2 * np.pi * _n_i) / wl_i
            mismatch = k_p - k_s - k_i - k_g
            return mismatch

        lambda_s_guess = lambda_s.to_absolute().value

        # Use root_scalar instead of root for a single equation
        result = root(
            phase_mismatch_at_T,
            lambda_s_guess,
        )
        if result.success:
            lambda_s_res = result.x[0]
            lambda_i = 1.0 / (
                (1.0 / lambda_p.to_absolute().value) - (1.0 / lambda_s_res)
            )

            signal_wavelengths[i] = lambda_s_res
            idler_wavelengths[i] = lambda_i

        else:
            signal_wavelengths[i] = np.nan
            idler_wavelengths[i] = np.nan

    return (temperature, signal_wavelengths, idler_wavelengths)
