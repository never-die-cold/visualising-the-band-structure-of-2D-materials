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


def resolve_n_valence(model, n_bands, n_valence=None):
    """Resolve the occupied-band count used as the Fermi reference.

    Explicit argument, then a ``n_valence`` declared by the model, then half
    filling (``n_bands // 2``). Models that are not half filled - the sp³d⁵
    SK model occupies 18 of 54 states - must declare ``n_valence`` so plots,
    DOS and gap analysis share one reference instead of each deriving its own.

    ``model.n_sites`` is deliberately not consulted: a site count need not
    equal the Hilbert-space dimension (a model can report four states with
    ``n_sites`` standing for something else entirely), so halving it silently
    produces a wrong reference.
    """
    n_bands = integer(n_bands, 'n_bands')
    if n_valence is None:
        n_valence = getattr(model, 'n_valence', None)
    if n_valence is None:
        n_valence = n_bands // 2
    value = integer(n_valence, 'n_valence')
    if value >= n_bands:
        raise ValueError(f'n_valence must be an integer between 1 and {n_bands - 1}')
    return value


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
