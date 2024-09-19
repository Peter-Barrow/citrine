from dataclasses import dataclass
from enum import Enum
from typing import Union, Literal, List, Optional, Tuple
import numpy as np
from numpy.typing import NDArray

__all__ = [
    'Magnitude',
    'AngularFrequency',
    'Wavelength',
    'spectral_window',
    'SellmeierCoefficients',
    'refractive_index',
    'Orientation',
    'Crystal',
    'calculate_grating_period',
    'delta_k_matrix',
    'phase_matching_function',
    'pump_envelope_gaussian',
    'bandwidth_conversion',
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

    def as_wavevector(self) -> float:
        """
        Convert the wavelength to a wavevector.

        Returns:
            float: The wavevector.
        """
        return (2 * np.pi) / self.to_absolute().value


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
    sellmeier: Union[List[float], NDArray], wavelength_um: float
) -> float:
    """
    Compute the permittivity using the Sellmeier equation.

    Args:
        sellmeier (Union[List[float], NDArray]): Sellmeier coefficients.
        wavelength_um (float): Wavelength in micrometers.

    Returns:
        float: The permittivity value.
    """
    wl = wavelength_um**2
    a = 1
    b = a + 1
    p = sellmeier[0]
    lim: int = int(np.ceil((len(sellmeier) - 2) / 2))
    for i in range(0, lim):
        p += sellmeier[a] / (1 - sellmeier[b] / wl)
        a += 2
        b += 2

    p -= sellmeier[-1] * wl
    return p


def _n_0(sellmeier: Union[List[float], NDArray], wavelength_um: float) -> float:
    """
    Calculate the refractive index (n_0) using the zeroth-order Sellmeier coefficients.

    Args:
        sellmeier (Union[List[float], NDArray]): Zeroth-order Sellmeier coefficients.
        wavelength_um (float): Wavelength in micrometers.

    Returns:
        float: The refractive index (n_0).
    """
    return np.sqrt(_permittivity(sellmeier, wavelength_um))


def _n_i(sellmeier: Union[List[float], NDArray], wavelength_um: float) -> float:
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
) -> float:
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
    n0 = 0
    if sellmeier.zeroth_order is not None:
        n0 = _n_0(sellmeier.zeroth_order, wavelength_um)
    n1 = _n_i(sellmeier.first_order, wavelength_um)
    n2 = _n_i(sellmeier.second_order, wavelength_um)

    t_offset = temperature - sellmeier.temperature

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
        polarization: Literal['ordinary', 'extraordinary'],
        photon: Literal['pump', 'signal', 'idler'],
        temperature: Optional[float] = None,
    ) -> float:
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
        if polarization == 'ordinary':
            sellmeier = self.sellmeier_o
        elif polarization == 'extraordinary':
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
    ) -> Tuple[float, float, float]:
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
            pump_wavelength, self.pump_orientation.name, 'pump', temperature
        )
        n_signal = self.refractive_index(
            signal_wavelength,
            self.signal_orientation.name,
            'signal',
            temperature,
        )
        n_idler = self.refractive_index(
            idler_wavelength, self.idler_orientation.name, 'idler', temperature
        )
        return n_pump, n_signal, n_idler


def calculate_grating_period(
    lambda_p_central: Wavelength,
    lambda_s_central: Wavelength,
    lambda_i_central: Wavelength,
    crystal: Crystal,
) -> float:
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
) -> np.ndarray:
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
    lambda_s_grid, lambda_i_grid = np.meshgrid(
        lambda_s.value, lambda_i.value, indexing='ij'
    )

    # Convert to Wavelength objects for both grids
    lambda_s_grid = Wavelength(lambda_s_grid, lambda_s.unit)
    lambda_i_grid = Wavelength(lambda_i_grid, lambda_i.unit)

    # Convert signal and idler wavelengths to angular frequencies
    omega_s = lambda_s_grid.as_angular_frequency().value
    omega_i = lambda_i_grid.as_angular_frequency().value
    omega_p = lambda_p.as_angular_frequency().value

    # Calculate the Gaussian pump envelope
    pump_envelope = np.exp(-(((omega_s + omega_i - omega_p) / sigma_p) ** 2))

    return pump_envelope


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
