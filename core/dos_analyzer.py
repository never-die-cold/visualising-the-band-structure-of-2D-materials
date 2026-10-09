"""Normalized k-point spectra and VASP DOSCAR total DOS."""
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Tuple

import numpy as np

from .parser import _float


@dataclass
class DosData:
    energies: np.ndarray
    total_dos: np.ndarray
    vb_dos: Optional[np.ndarray] = None
    cb_dos: Optional[np.ndarray] = None
    fermi_level: float = 0.0
    energy_range: Tuple[float, float] = (-10.0, 10.0)
    sigma: float = 0.05
    scope: str = "sampled-spectrum"
    source: str = "eigenvalues"
    state_capacity: Optional[float] = 1.0
    expected_states: Optional[float] = None
    normalized_weights: Optional[np.ndarray] = None
    component_labels: tuple = ("E ≤ EF states", "E > EF states")
    note: str = ""
    spin_dos: Optional[np.ndarray] = None
    integrated_dos: Optional[np.ndarray] = None

    @property
    def integral(self) -> float:
        return float(np.trapezoid(self.total_dos, self.energies))


def _positive(value, name):
    if not np.isfinite(value) or value <= 0:
        raise ValueError(f"{name} must be finite and positive")


class DosAnalyzer:
    """Gaussian spectra; only a declared integration mesh is a BZ DOS.

    Array callers default to one state per column. from_band_data uses VASP's
    spin capacity (2 for ISPIN=1, 1 for ISPIN=2); SOC callers must specify 1.
    All three components use exactly the same normalized k-point weights.
    """
    def __init__(self, energies, fermi_level=0.0, *, weights=None,
                 state_capacity=1.0, occupied_fraction=None,
                 scope="sampled-spectrum", note="", max_workspace_mb=32., cancel_check=None):
        self.energies = np.asarray(energies, dtype=float)
        if self.energies.ndim != 2 or min(self.energies.shape) < 1 or not np.isfinite(self.energies).all():
            raise ValueError("energies must be a nonempty finite (nk, bands) array")
        if not np.isfinite(fermi_level):
            raise ValueError("fermi_level must be finite")
        if state_capacity not in (1., 2.):
            raise ValueError("state_capacity must be 1 or 2")
        if scope not in ("sampled-spectrum", "path-spectrum", "brillouin-zone"):
            raise ValueError("invalid DOS scope")
        _positive(max_workspace_mb, 'max_workspace_mb')
        self.fermi_level = float(fermi_level)
        self.state_capacity = float(state_capacity)
        self.weights = weights
        self.scope, self.note = scope, note
        self.max_workspace_mb = max_workspace_mb
        self.cancel_check = cancel_check
        self.occupied_fraction = None
        if occupied_fraction is not None:
            fraction = np.asarray(occupied_fraction, dtype=float)
            if (fraction.shape != self.energies.shape or not np.isfinite(fraction).all() or
                    np.any(fraction < -1e-3) or np.any(fraction > 1 + 1e-3)):
                raise ValueError("occupation fractions must match energies and lie in [0, 1]")
            self.occupied_fraction = np.clip(fraction, 0., 1.)
        self._dos_data = None

    @classmethod
    def from_band_data(cls, data, fermi_level=0., *, state_capacity=None, **kwargs):
        capacity = (2. if data.ispin == 1 else 1.) if state_capacity is None else state_capacity
        kind = getattr(data, 'sampling_kind', 'unknown')
        scope = 'brillouin-zone' if kind == 'mesh' else 'path-spectrum' if kind == 'path' else 'sampled-spectrum'
        weights = data.weights
        note = ''
        if weights is not None and np.all(np.asarray(weights) == 0):
            if scope == 'brillouin-zone':
                raise ValueError('Integration mesh has no positive k-point weights')
            weights = None
            note = 'All input k weights are zero; uniform normalized spectrum, not a BZ DOS'
        fraction = None if data.occupations is None else data.occupations / capacity
        if fraction is not None and np.isfinite(fraction).all() and (np.any(fraction < -1e-3) or np.any(fraction > 1 + 1e-3)):
            fraction = None
            note += '; Occupations outside state capacity: components split by energy, not occupancy'
        return cls(data.energies, fermi_level, weights=weights, state_capacity=capacity,
                   occupied_fraction=fraction, scope=scope, note=note, **kwargs)

    def set_fermi_level(self, efermi):
        if not np.isfinite(efermi):
            raise ValueError('fermi_level must be finite')
        self.fermi_level = float(efermi)
        self._dos_data = None

    def calculate_dos(self, energy_range=(-10., 10.), num_points=1000, sigma=.05,
                      per_kpoint_weight=None) -> DosData:
        if self.cancel_check is not None:
            self.cancel_check()
        _positive(sigma, 'sigma')
        bounds = np.asarray(energy_range, dtype=float)
        if bounds.shape != (2,) or not np.isfinite(bounds).all() or bounds[0] >= bounds[1]:
            raise ValueError('energy_range must contain finite increasing bounds')
        if isinstance(num_points, bool) or not isinstance(num_points, (int, np.integer)) or num_points < 2:
            raise ValueError('num_points must be an integer >= 2')
        weights = self.weights if per_kpoint_weight is None else per_kpoint_weight
        nk, nb = self.energies.shape
        if weights is None:
            weights = np.full(nk, 1. / nk)
        else:
            weights = np.asarray(weights, dtype=float)
            if (weights.shape != (nk,) or not np.isfinite(weights).all() or
                    np.any(weights < 0) or not np.any(weights > 0)):
                raise ValueError('weights must be finite nonnegative k weights with positive sum')
            weights = weights / np.max(weights)  # avoid overflow for large raw weights
            weights = weights / weights.sum()
        grid = np.linspace(*bounds, num_points)
        values = (self.energies - self.fermi_level).ravel()
        state_weights = np.repeat(weights * self.state_capacity, nb)
        fraction = ((values <= 0).astype(float) if self.occupied_fraction is None
                    else self.occupied_fraction.ravel())
        total, occupied = np.zeros(num_points), np.zeros(num_points)
        # One in-place Gaussian block, bounded independently of nk*nb.
        chunk = max(1, int(self.max_workspace_mb * 1024 ** 2 / (8 * num_points)))
        workspace = np.empty((num_points, min(chunk, len(values))))
        for start in range(0, len(values), chunk):
            if self.cancel_check is not None:
                self.cancel_check()
            end = min(start + chunk, len(values))
            gaussian = workspace[:, :end - start]
            np.subtract(grid[:, None], values[None, start:end], out=gaussian)
            gaussian /= sigma
            np.square(gaussian, out=gaussian)
            gaussian *= -.5
            np.exp(gaussian, out=gaussian)
            gaussian /= sigma * np.sqrt(2 * np.pi)
            w = state_weights[start:end]
            total += gaussian @ w
            occupied += gaussian @ (w * fraction[start:end])
        unoccupied = np.maximum(0., total - occupied)
        labels = ('Occupied states', 'Unoccupied states') if self.occupied_fraction is not None else ('E ≤ EF states', 'E > EF states')
        self._dos_data = DosData(grid, total, occupied, unoccupied, self.fermi_level,
                                 tuple(map(float, bounds)), float(sigma), self.scope,
                                 state_capacity=self.state_capacity, expected_states=nb * self.state_capacity,
                                 normalized_weights=weights.copy(), component_labels=labels, note=self.note)
        return self._dos_data

    def get_dos_at_fermi(self):
        if self._dos_data is None:
            self.calculate_dos()
        return float(np.interp(0., self._dos_data.energies, self._dos_data.total_dos))

    def get_band_edges_from_dos(self, threshold=.01):
        """Smearing-dependent visualization heuristic, not a material gap."""
        if not np.isfinite(threshold) or not 0 < threshold < 1:
            raise ValueError('threshold must lie in (0, 1)')
        if self._dos_data is None:
            self.calculate_dos()
        data = self._dos_data
        mask = data.total_dos > threshold * np.max(data.total_dos)
        vb = np.flatnonzero(mask & (data.energies <= 0))
        cb = np.flatnonzero(mask & (data.energies > 0))
        vbm = float(data.energies[vb[-1]]) if len(vb) else None
        cbm = float(data.energies[cb[0]]) if len(cb) else None
        return {'vbm': vbm, 'cbm': cbm, 'gap': None if vbm is None or cbm is None else cbm-vbm,
                'scope': 'broadened-spectrum', 'note': 'Threshold heuristic depends on sigma and energy grid'}


class DoscarParser:
    """Strict total DOSCAR reader (3/5 columns); projected blocks are ignored.

    Header order is EMAX EMIN NEDOS EFERMI. Already-normalized VASP total DOS
    includes its spin count and must not be multiplied by two again.
    """
    def __init__(self, filepath, cancel_check=None):
        self.filepath = Path(filepath)
        self.cancel_check = cancel_check

    def parse(self):
        if self.cancel_check is not None:
            self.cancel_check()
        if not self.filepath.exists():
            return None
        def error(line, message):
            return ValueError(f'{self.filepath}:{line}: {message}')
        with self.filepath.open(encoding='utf-8-sig') as stream:
            header = [stream.readline() for _ in range(6)]
            if any(not v.strip() for v in header):
                raise error(6, 'incomplete DOSCAR header')
            try:
                fields = header[5].split()
                if len(fields) != 5:
                    raise ValueError()
                emax, emin, nedos, ef, weight = _float(fields[0]), _float(fields[1]), int(fields[2]), _float(fields[3]), _float(fields[4])
            except (ValueError, IndexError) as exc:
                raise error(6, 'expected EMAX EMIN integer NEDOS EFERMI weight') from exc
            if not np.isfinite([emax, emin, ef, weight]).all() or emin >= emax or nedos < 2:
                raise error(6, 'invalid DOSCAR energy range or NEDOS')
            rows, columns = [], None
            for i in range(nedos):
                if self.cancel_check is not None:
                    self.cancel_check()
                fields = stream.readline().split()
                if len(fields) not in (3, 5) or (columns is not None and len(fields) != columns):
                    raise error(i+7, 'expected a complete 3/5-column total DOS row')
                columns = len(fields)
                try:
                    row = [_float(v) for v in fields]
                except ValueError as exc:
                    raise error(i+7, 'invalid DOS value') from exc
                if not np.isfinite(row).all():
                    raise error(i+7, 'DOS values must be finite')
                rows.append(row)
        rows = np.asarray(rows)
        energy = rows[:, 0]
        if np.any(np.diff(energy) <= 0) or not np.allclose(energy[[0, -1]], [emin, emax], atol=1e-5, rtol=1e-6):
            raise error(7, 'DOS energy grid must increase and match header bounds')
        spin = rows[:, 1:3].copy() if columns == 5 else None
        total = spin.sum(axis=1) if spin is not None else rows[:, 1].copy()
        integrated = rows[:, 3:5].sum(axis=1) if spin is not None else rows[:, 2].copy()
        relative = energy - ef
        return DosData(relative, total, np.where(relative <= 0, total, 0.),
                       np.where(relative > 0, total, 0.), ef,
                       (float(relative[0]), float(relative[-1])), 0., 'brillouin-zone',
                       source=str(self.filepath.resolve()), state_capacity=None,
                       component_labels=('E ≤ EF', 'E > EF'), spin_dos=spin,
                       integrated_dos=integrated,
                       note='Imported VASP DOSCAR total DOS; projected blocks are not read')


def load_spectrum(data, eigenval_path, *, fermi_level=0., energy_range=(-5., 5.),
                  num_points=800, sigma=.05, state_capacity=None, cancel_check=None):
    """Prefer a same-directory DOSCAR; otherwise broaden declared k samples.

    The UI's energy reference is absolute in EIGENVAL/DOSCAR units. Imported
    DOSCAR preserves its own EF metadata while shifting its energy axis to the
    requested display zero. No re-broadening of already-integrated DOSCAR.
    """
    doscar = DoscarParser(Path(eigenval_path).parent / 'DOSCAR', cancel_check).parse() if eigenval_path else None
    if doscar is None:
        return DosAnalyzer.from_band_data(data, fermi_level, state_capacity=state_capacity, cancel_check=cancel_check).calculate_dos(
            energy_range, num_points, sigma)
    absolute = doscar.energies + doscar.fermi_level
    doscar.energies = absolute - fermi_level
    doscar.vb_dos = np.where(doscar.energies <= 0, doscar.total_dos, 0.)
    doscar.cb_dos = np.where(doscar.energies > 0, doscar.total_dos, 0.)
    doscar.energy_range = (float(doscar.energies[0]), float(doscar.energies[-1]))
    doscar.note += f'; file EF={doscar.fermi_level:g} eV; display zero={fermi_level:g} eV'
    doscar.fermi_level = float(fermi_level)
    return doscar
