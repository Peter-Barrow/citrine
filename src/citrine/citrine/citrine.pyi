import numpy as np
from dataclasses import dataclass
from enum import Enum
from numpy.typing import NDArray
from typing import Literal

__all__ = ['Magnitude', 'AngularFrequency', 'Wavelength', 'spectral_window', 'SellmeierCoefficients', 'refractive_index', 'Orientation', 'Crystal', 'calculate_grating_period', 'delta_k_matrix', 'phase_matching_function', 'pump_envelope_gaussian', 'bandwidth_conversion']

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
    value: float | np.ndarray
    def __init__(self, value) -> None: ...

@dataclass
class Wavelength:
    """
    Class to represent wavelength with unit conversion.

    Attributes:
        value (Union[float, np.ndarray]): The wavelength value.
        unit (Magnitude): The unit of the wavelength (default: nano).
    """
    value: float | np.ndarray
    unit: Magnitude = ...
    def to_unit(self, new_unit: Magnitude) -> Wavelength:
        """
        Convert the wavelength to a new unit.

        Args:
            new_unit (Magnitude): The desired unit for the wavelength.

        Returns:
            Wavelength: New Wavelength object with converted units.
        """
    def to_absolute(self) -> Wavelength:
        """
        Convert the wavelength to base units (meters).

        Returns:
            Wavelength: Wavelength in meters.
        """
    def as_angular_frequency(self) -> AngularFrequency:
        """
        Convert the wavelength to angular frequency.

        Returns:
            AngularFrequency: Angular frequency corresponding to the wavelength.
        """
    def as_wavevector(self) -> float:
        """
        Convert the wavelength to a wavevector.

        Returns:
            float: The wavevector.
        """
    def __init__(self, value, unit=...) -> None: ...

def spectral_window(central_wavelength: Wavelength, spectral_width: Wavelength, steps: int) -> Wavelength:
    """
    Generate an array of wavelengths within a specified spectral window.

    Args:
        central_wavelength (Wavelength): The central wavelength.
        spectral_width (Wavelength): The total spectral width.
        steps (int): The number of steps for the wavelength range.

    Returns:
        Wavelength: An array of wavelengths within the specified window.
    """

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
    first_order: list[float] | NDArray | None
    second_order: list[float] | NDArray | None
    temperature: float
    zeroth_order: list[float] | NDArray | None = ...
    def __init__(self, first_order, second_order, temperature, zeroth_order=...) -> None: ...

def refractive_index(sellmeier: SellmeierCoefficients, wavelength: Wavelength, temperature: float | None = None) -> float:
    """
    Calculate the refractive index for a given wavelength and temperature using the Sellmeier equation.

    Args:
        sellmeier (SellmeierCoefficients): The Sellmeier coefficients for the material.
        wavelength (Wavelength): The wavelength at which to calculate the refractive index.
        temperature (Optional[float]): The temperature (in Celsius), defaults to the reference temperature.

    Returns:
        float: The refractive index at the given wavelength and temperature.
    """

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
    def refractive_index(self, wavelength: Wavelength, polarization: Literal['ordinary', 'extraordinary'], photon: Literal['pump', 'signal', 'idler'], temperature: float | None = None) -> float:
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
    def refractive_indices(self, pump_wavelength: Wavelength, signal_wavelength: Wavelength, idler_wavelength: Wavelength, temperature: float | None = None) -> tuple[float, float, float]:
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
    def __init__(self, name, sellmeier_o, sellmeier_e, pump_orientation, signal_orientation, idler_orientation) -> None: ...

def calculate_grating_period(lambda_p_central: Wavelength, lambda_s_central: Wavelength, lambda_i_central: Wavelength, crystal: Crystal) -> float:
    """
    Calculate the grating period (Λ) for the phase matching condition.

    Args:
        lambda_p_central (Wavelength): Central wavelength of the pump.
        lambda_s_central (Wavelength): Central wavelength of the signal.
        lambda_i_central (Wavelength): Central wavelength of the idler.

    Returns:
        float: Grating period in microns.
    """
def delta_k_matrix(lambda_p: Wavelength, lambda_s: Wavelength, lambda_i: Wavelength, crystal: Crystal) -> np.ndarray:
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
def phase_matching_function(delta_k: np.ndarray, grating_period, crystal_length) -> np.ndarray:
    """
    Compute the phase matching function as a sinc function of Δk.

    Args:
        delta_k (np.ndarray): The Δk matrix from the phase matching condition.

    Returns:
        np.ndarray: The phase matching function.
    """
def pump_envelope_gaussian(lambda_p: Wavelength, sigma_p: float, lambda_s: Wavelength, lambda_i: Wavelength) -> np.ndarray:
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
def bandwidth_conversion(delta_lambda_FWHM: Wavelength, pump_wl: Wavelength) -> float:
    """
    Convert pump bandwidth from FWHM in wavelength to frequency.

    Args:
        delta_lambda_FWHM (Wavelength): Bandwidth in wavelength units.
        pump_wl (Wavelength): Central pump wavelength.

    Returns:
        float: Converted bandwidth.
    """
