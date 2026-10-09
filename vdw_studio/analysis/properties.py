"""Band gaps on periodic BZ meshes and on display paths, plus derivatives.

Site TB models use a spinless basis and half filling by default. Callers can
supply n_valence. Local k·p models report their own valley-local properties.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional, Sequence, Tuple

import numpy as np
from scipy.optimize import minimize

from ..engine.kpath import DISPLAY_SYMBOLS, KPath
from ..engine.solver import sample_bands
from ..structure.lattice import Lattice

HBAR2_OVER_2M0 = 3.809982
SLOPE_TO_VELOCITY = 1.51927e5


@dataclass
class GapResult:
    """gap=None retains compatibility for metal/zero-gap; status distinguishes them."""

    gap: Optional[float]
    direct: Optional[bool]
    vbm: float
    cbm: float
    vbm_k: np.ndarray
    cbm_k: np.ndarray
    vbm_label: str
    cbm_label: str
    n_valence: int
    scope: str = "unspecified"
    status: str = "unknown"
    raw_gap: Optional[float] = None
    search_metadata: dict = field(default_factory=dict)


def _periodic_distance(left, right, lattice: Lattice) -> float:
    delta = np.asarray(left) - np.asarray(right)
    delta -= np.rint(delta)
    shifts = np.array([(i, j) for i in (-1, 0, 1) for j in (-1, 0, 1)])
    return float(np.linalg.norm((delta + shifts) @ lattice.reciprocal_matrix[:2], axis=1).min())


def _nearest_hsym_label(kfrac, kpath: Optional[KPath], lattice: Lattice) -> str:
    if kpath is not None:
        for name, point in kpath.points.items():
            if _periodic_distance(kfrac, point, lattice) <= 1e-5:
                return DISPLAY_SYMBOLS.get(name, name)
    return "off-path"


def _validate_samples(model, points, energies, n_valence):
    points, energies = np.asarray(points, dtype=float), np.asarray(energies, dtype=float)
    if (points.ndim != 2 or points.shape[1] != 2 or energies.ndim != 2 or
            len(points) != len(energies) or not len(points) or energies.shape[1] < 2 or
            not np.isfinite(points).all() or not np.isfinite(energies).all()):
        raise ValueError("Expected finite (nk,2) k points and (nk,nbands>=2) energies")
    if n_valence is None:
        n_valence = getattr(model, "n_sites", energies.shape[1]) // 2
    if (isinstance(n_valence, (bool, np.bool_)) or not isinstance(n_valence, (int, np.integer)) or
            not 1 <= n_valence < energies.shape[1]):
        raise ValueError("n_valence must be an integer between 1 and nbands-1")
    if np.any(np.diff(energies, axis=1) < -1e-8):
        raise ValueError("Model energies must be sorted by band at each k point")
    return points, energies, int(n_valence)


def _make_gap(model, points, energies, n_valence, kpath, scope, zero_tol, metadata):
    points, energies, n_valence = _validate_samples(model, points, energies, n_valence)
    valence, conduction = energies[:, n_valence - 1], energies[:, n_valence]
    vi, ci = int(np.argmax(valence)), int(np.argmin(conduction))
    vbm, cbm = float(valence[vi]), float(conduction[ci])
    raw_gap = cbm - vbm
    status = "metal" if raw_gap < -zero_tol else "zero-gap" if raw_gap <= zero_tol else "insulator"
    direct = None
    if status == "insulator":
        # Include degenerate extrema: arbitrary argmax/argmin representatives
        # can otherwise report an indirect gap even when a common valley exists.
        shared = np.flatnonzero((valence >= vbm - 1e-6) & (conduction <= cbm + 1e-6))
        direct = bool(len(shared))
        if direct:
            vi = ci = int(shared[0])
    vk, ck = points[vi] % 1.0, points[ci] % 1.0
    return GapResult(
        gap=raw_gap if status == "insulator" else None, direct=direct,
        vbm=vbm, cbm=cbm, vbm_k=vk, cbm_k=ck,
        vbm_label=_nearest_hsym_label(vk, kpath, model.lattice),
        cbm_label=_nearest_hsym_label(ck, kpath, model.lattice),
        n_valence=n_valence, scope=scope, status=status, raw_gap=raw_gap,
        search_metadata=metadata,
    )


def _path_points(kpath, n_per_segment):
    if isinstance(n_per_segment, bool) or not isinstance(n_per_segment, (int, np.integer)) or n_per_segment < 2:
        raise ValueError("n_per_segment must be an integer >= 2")
    points = []
    for p0, p1 in kpath.segments():
        points.extend(np.linspace(p0, p1, n_per_segment, endpoint=False))
    points.append(kpath.points[kpath.path[-1]])
    return np.asarray(points)


def _validate_tol(zero_tol):
    if not np.isfinite(zero_tol) or zero_tol <= 0:
        raise ValueError("zero_tol must be finite and positive")


def analyze_path_gap(model, kpath: Optional[KPath] = None, n_per_segment: int = 40,
                     n_valence: Optional[int] = None, *, band=None, zero_tol: float = 1e-3) -> GapResult:
    """Analyze only the display path, optionally reusing solve_bands output."""
    _validate_tol(zero_tol)
    kpath = kpath or KPath.for_lattice(model.lattice)
    points = _path_points(kpath, n_per_segment)
    if band is not None:
        expected = np.column_stack((points, np.zeros(len(points)))) @ model.lattice.reciprocal_matrix
        if band.kpoints.shape != expected.shape or not np.allclose(band.kpoints, expected, atol=1e-8, rtol=0):
            raise ValueError("Reused band data does not match the requested k path")
    energies = model.bands(points) if band is None else band.energies
    return _make_gap(model, points, energies, n_valence, kpath, "path", zero_tol,
                     {"method": "path-sampling", "n_per_segment": n_per_segment, "zero_tol_eV": zero_tol})


def _search_edges(model, mesh, n_valence, seeds, max_starts, cancel_check=None):
    n1, n2 = mesh
    points = np.array([(i / n1, j / n2) for i in range(n1) for j in range(n2)])
    points, energies, n_valence = _validate_samples(model, points, sample_bands(model, points, cancel_check), n_valence)
    refined_points, converged = [], True
    for band, sign in ((n_valence - 1, -1), (n_valence, 1)):
        values = sign * energies[:, band].reshape(mesh)
        if np.ptp(values) < 1e-12:
            continue  # A flat band has no isolated local extremum.
        mask = np.ones(mesh, dtype=bool)
        for di, dj in ((1, 0), (-1, 0), (0, 1), (0, -1), (1, 1), (1, -1), (-1, 1), (-1, -1)):
            mask &= values <= np.roll(values, (di, dj), axis=(0, 1))
        indices = np.flatnonzero(mask.ravel())
        indices = indices[np.argsort(values.ravel()[indices], kind="stable")[:max_starts]]
        starts = list(points[indices])
        # Known path vertices supplement the periodic grid, not restrict it.
        starts.extend(seeds)
        unique = []
        for start in starts:
            if not any(np.linalg.norm((start - other + .5) % 1 - .5) < 1e-8 for other in unique):
                unique.append(np.asarray(start))
        def objective(k):
            if cancel_check is not None:
                cancel_check()
            result = np.asarray(model.energies_at(np.asarray(k) % 1), dtype=float)
            if result.shape != (energies.shape[1],) or not np.isfinite(result).all():
                raise ValueError("Model returned invalid local-search energies")
            return float(sign * result[band])
        for start in unique:
            result = minimize(objective, start, method="Nelder-Mead",
                              options={"xatol": 1e-10, "fatol": 1e-10, "maxiter": 600})
            refined_points.append(result.x % 1)
            converged &= bool(result.success)
    if refined_points:
        local_points = np.asarray(refined_points)
        local_energies = sample_bands(model, local_points, cancel_check)
        points, energies = np.vstack((points, local_points)), np.vstack((energies, local_energies))
    return points, energies, n_valence, converged


def analyze_gap(model, kpath: Optional[KPath] = None, n_per_segment: int = 40,
                n_valence: Optional[int] = None, *, mesh: Tuple[int, int] = (24, 24),
                zero_tol: float = 1e-3, convergence_tol: float = 1e-5,
                max_starts: int = 8, cancel_check=None) -> GapResult:
    """Search a full periodic cell and refine valence/conduction extrema.

    Two grids (mesh and 2*mesh) independently seed local searches. Agreement of
    both band edges is a numerical convergence check, not a proof of a global
    minimum for an arbitrary model. n_per_segment affects the separate path
    API only and remains accepted here for compatibility.
    """
    _validate_tol(zero_tol)
    if (len(mesh) != 2 or any(isinstance(n, bool) or not isinstance(n, (int, np.integer)) or n < 4 for n in mesh)):
        raise ValueError("gap mesh must contain two integers >= 4")
    if not np.isfinite(convergence_tol) or convergence_tol <= 0:
        raise ValueError("convergence_tol must be finite and positive")
    if isinstance(max_starts, bool) or not isinstance(max_starts, (int, np.integer)) or max_starts < 1:
        raise ValueError("max_starts must be a positive integer")
    kpath = kpath or KPath.for_lattice(model.lattice)
    seeds = [np.asarray(point) for point in kpath.points.values()]
    coarse = _search_edges(model, tuple(mesh), n_valence, seeds, max_starts, cancel_check)
    fine_mesh = tuple(2 * n for n in mesh)
    fine = _search_edges(model, fine_mesh, coarse[2], seeds, max_starts, cancel_check)
    v = coarse[2] - 1
    differences = [abs(coarse[1][:, v].max() - fine[1][:, v].max()),
                   abs(coarse[1][:, v+1].min() - fine[1][:, v+1].min())]
    converged = coarse[3] and fine[3] and max(differences) <= convergence_tol
    points, energies = np.vstack((coarse[0], fine[0])), np.vstack((coarse[1], fine[1]))
    metadata = {"method": "periodic-grid+Nelder-Mead", "initial_mesh": list(mesh),
                "refined_mesh": list(fine_mesh), "max_starts": max_starts,
                "converged": bool(converged), "edge_difference_eV": list(map(float, differences)),
                "convergence_tol_eV": convergence_tol, "zero_tol_eV": zero_tol}
    return _make_gap(model, points, energies, coarse[2], kpath, "brillouin-zone", zero_tol, metadata)



def _derivative_inputs(model, k0, band, dk):
    k0 = np.asarray(k0, dtype=float)
    if k0.shape != (2,) or not np.isfinite(k0).all():
        raise ValueError("k0 must contain two finite fractional coordinates")
    if not np.isfinite(dk) or dk <= 0:
        raise ValueError("dk must be finite and positive")
    if isinstance(band, bool) or not isinstance(band, (int, np.integer)) or band < 0:
        raise ValueError("band must be a nonnegative integer")
    basis = np.asarray(model.lattice.reciprocal_matrix, dtype=float)[:2]
    if basis.shape != (2, 3) or not np.isfinite(basis).all() or np.linalg.matrix_rank(basis) != 2:
        raise ValueError("reciprocal plane must have two independent finite row vectors")
    def energy(k):
        values = np.asarray(model.energies_at(k), dtype=float)
        if values.ndim != 1 or band >= len(values) or not np.isfinite(values).all():
            raise ValueError("band is out of range or model energies are invalid")
        return float(values[band])
    return k0, basis, energy


def _plane_frame(basis):
    """Orthonormal rows: projected global x, then normal cross x.

    Fall back to projected global y when x is normal to the reciprocal plane.
    This keeps the usual xy convention for slabs parallel to xy.
    """
    normal = np.cross(*basis)
    normal /= np.linalg.norm(normal)
    axis = np.array([1., 0., 0.])
    x = axis - (axis @ normal) * normal
    if np.linalg.norm(x) < 1e-8:
        axis = np.array([0., 1., 0.])
        x = axis - (axis @ normal) * normal
    x /= np.linalg.norm(x)
    return np.array([x, np.cross(normal, x)])


def _direction_step(basis, direction, dk, direction_space):
    d = np.asarray(direction, dtype=float)
    if not np.isfinite(d).all() or np.linalg.norm(d) == 0:
        raise ValueError("direction must be finite and nonzero")
    if direction_space == "fractional":
        if d.shape != (2,):
            raise ValueError("fractional direction must contain two components")
        df = dk * d / np.linalg.norm(d)
    elif direction_space == "cartesian":
        if d.shape == (2,):
            d = np.r_[d, 0.]
        if d.shape != (3,):
            raise ValueError("Cartesian direction must contain two or three components")
        dc = dk * d / np.linalg.norm(d)
        df = dc @ np.linalg.pinv(basis)
        if not np.allclose(df @ basis, dc, rtol=1e-8, atol=dk * 1e-8):
            raise ValueError("Cartesian direction must lie in the reciprocal plane")
    else:
        raise ValueError("direction_space must be fractional or cartesian")
    return df, float(np.linalg.norm(df @ basis))


def _curvature_mass(curvature):
    return float("inf") if abs(curvature) < 1e-12 else abs(2 * HBAR2_OVER_2M0 / curvature)


def effective_mass(model, k0: Sequence[float], band: int,
                   direction: Sequence[float] = (1.0, 0.0),
                   dk: float = 1e-4, *, direction_space: str = "fractional") -> float:
    """Directional |m*| in m₀, using the full reciprocal-row metric.

    k0 is fractional. By default direction and dk are fractional; for
    direction_space='cartesian', direction is a global Cartesian vector
    in the reciprocal plane and dk is in Å⁻¹. Flat curvature returns infinity.
    """
    k0, basis, energy = _derivative_inputs(model, k0, band, dk)
    df, h = _direction_step(basis, direction, dk, direction_space)
    e0 = energy(k0)
    curvature = ((energy(k0 + df) - e0) + (energy(k0 - df) - e0)) / h ** 2
    return _curvature_mass(curvature)


def principal_masses(model, k0: Sequence[float], band: int,
                     dk: float = 1e-4) -> Tuple[float, float, float]:
    """Principal |m₁|, |m₂| (m₀), and first-axis angle (radians).

    dk is a physical Å⁻¹ step. The angle is in an orthonormal reciprocal-plane
    frame (projected global x, normal cross x; projected y fallback). Eigenvalues
    are sorted by signed curvature, rather than by mass magnitude.
    """
    k0, basis, energy = _derivative_inputs(model, k0, band, dk)
    frame = _plane_frame(basis)
    inverse = np.linalg.inv(basis @ frame.T)
    e00 = energy(k0)
    def energy_at_cart(dc):
        # Row-vector convention: dc = df @ B_plane, df = dc @ inv(B_plane).
        return energy(k0 + np.asarray(dc) @ inverse) - e00
    h = dk
    hxx = (energy_at_cart([h, 0]) + energy_at_cart([-h, 0])) / h ** 2
    hyy = (energy_at_cart([0, h]) + energy_at_cart([0, -h])) / h ** 2
    hxy = (energy_at_cart([h, h]) - energy_at_cart([h, -h]) -
           energy_at_cart([-h, h]) + energy_at_cart([-h, -h])) / (4 * h ** 2)
    vals, vecs = np.linalg.eigh([[hxx, hxy], [hxy, hyy]])
    v = vecs[:, 0]
    theta = float(np.arctan2(v[1], v[0]))
    return _curvature_mass(vals[0]), _curvature_mass(vals[1]), theta


def fermi_velocity(model, k0: Sequence[float],
                   band: int, direction: Sequence[float] = (1.0, 0.0),
                   dk: float = 1e-5, *, direction_space: str = "fractional") -> float:
    """Directional one-sided speed |ΔE/Δk|/ħ in m/s, including Dirac cusps.

    Subtract E(k0), so a constant energy shift does not change the result.
    Direction and step conventions match effective_mass().
    """
    k0, basis, energy = _derivative_inputs(model, k0, band, dk)
    df, h = _direction_step(basis, direction, dk, direction_space)
    return abs((energy(k0 + df) - energy(k0)) / h) * SLOPE_TO_VELOCITY
