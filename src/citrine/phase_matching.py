import numpy as np

__all__ = [
    'pmf_gaussian',
    'pmf_antisymmetric',
]


def _adjusted_delta_k(delta_k: np.ndarray, poling_period) -> np.ndarray:
    return delta_k - ((2 * np.pi) / poling_period)


def pmf_gaussian(
    delta_k: np.ndarray, poling_period, crystal_length
) -> np.ndarray:
    _delta_k = _adjusted_delta_k(delta_k, poling_period)
    return np.exp((1j * _delta_k * (crystal_length / 2)) ** 2)


def pmf_antisymmetric(
    delta_k: np.ndarray, poling_period, crystal_length
) -> np.ndarray:
    _delta_k = _adjusted_delta_k(delta_k, poling_period)
    return np.exp((1j * _delta_k * (crystal_length / 2)) ** 2) * _delta_k
