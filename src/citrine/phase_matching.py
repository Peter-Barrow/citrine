import numpy as np

__all__ = [
    'pmf_gaussian',
    'pmf_antisymmetric',
]


def _adjusted_delta_k(delta_k: np.ndarray, poling_period) -> np.ndarray:
    """
    Calculate the adjusted delta k value.

    Args:
        delta_k (np.ndarray): The original delta k values.
        poling_period (float): The poling period of the crystal.

    Returns:
        np.ndarray: The adjusted delta k values.
    """
    return delta_k - ((2 * np.pi) / poling_period)


def pmf_gaussian(
    delta_k: np.ndarray, poling_period, crystal_length
) -> np.ndarray:
    """
    Calculate the Gaussian phase-matching function.

    Args:
        delta_k (np.ndarray): The delta k values.
        poling_period (float): The poling period of the crystal.
        crystal_length (float): The length of the crystal.

    Returns:
        np.ndarray: The Gaussian phase-matching function values.
    """
    _delta_k = _adjusted_delta_k(delta_k, poling_period)
    return np.exp((1j * _delta_k * (crystal_length / 2)) ** 2)


# TODO: update naming/docs here, this is really an HG mode
def pmf_antisymmetric(
    delta_k: np.ndarray, poling_period, crystal_length
) -> np.ndarray:
    """
    Calculate the antisymmetric phase-matching function.

    Args:
        delta_k (np.ndarray): The delta k values.
        poling_period (float): The poling period of the crystal.
        crystal_length (float): The length of the crystal.

    Returns:
        np.ndarray: The antisymmetric phase-matching function values.
    """
    _delta_k = _adjusted_delta_k(delta_k, poling_period)
    return np.exp((1j * _delta_k * (crystal_length / 2)) ** 2) * _delta_k
