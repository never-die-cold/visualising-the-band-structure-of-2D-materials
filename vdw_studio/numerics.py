"""Small shared validators for public numerical inputs."""
import numpy as np


def finite_scalar(value, name, *, positive=False):
    try:
        array = np.asarray(value, dtype=float)
    except (ValueError, TypeError) as exc:
        raise ValueError(f'{name} must be a finite scalar') from exc
    if array.shape != () or isinstance(value, (bool, np.bool_)) or not np.isfinite(array):
        raise ValueError(f'{name} must be a finite scalar')
    result = float(array)
    if positive and result <= 0:
        raise ValueError(f'{name} must be finite and positive')
    return result


def integer(value, name, *, minimum=1):
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)) or value < minimum:
        raise ValueError(f'{name} must be an integer >= {minimum}')
    return int(value)


def finite_vector(value, size, name):
    array = np.asarray(value, dtype=float)
    if array.shape != (size,) or not np.isfinite(array).all():
        raise ValueError(f'{name} must contain {size} finite coordinates')
    return array


def hermitian_matrix(value):
    matrix = np.asarray(value, dtype=complex)
    if (matrix.ndim != 2 or matrix.shape[0] != matrix.shape[1] or matrix.shape[0] < 1 or
            not np.isfinite(matrix).all() or not np.allclose(matrix, matrix.conj().T, atol=1e-10, rtol=1e-10)):
        raise ValueError('Hamiltonian must be a finite square Hermitian matrix')
    return matrix
