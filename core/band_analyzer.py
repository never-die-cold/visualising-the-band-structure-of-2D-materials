import numpy as np
from typing import Optional
from .parser import BandData, _equivalent_kpoints


class BandAnalyzer:
    """能带分析工具"""

    # ℏ² / (2m₀)  ≈ 3.80998 eV·Å²
    HBAR2_OVER_2M0 = 3.80998

    def __init__(
        self,
        band_data: BandData,
        lattice_constant_angstrom: Optional[float] = None, *,
        zero_gap_tol: float = 1e-3, occupation_tol: float = 1e-3,
        state_capacity: Optional[float] = None
    ):
        """
        参数:
            band_data: 解析后的能带数据
            lattice_constant_angstrom: deprecated compatibility argument; a scalar
                                     cannot calibrate a general reciprocal lattice.
            Physical masses require band_data.lattice_matrix (row vectors, Å).
        """
        self.data = band_data
        self.fermi_level = 0.0
        self.lattice_constant_angstrom = lattice_constant_angstrom
        self.zero_gap_tol = zero_gap_tol
        self.occupation_tol = occupation_tol
        self.state_capacity = (2.0 if band_data.ispin == 1 else 1.0) if state_capacity is None else state_capacity
        if not np.isfinite(zero_gap_tol) or zero_gap_tol <= 0:
            raise ValueError('zero_gap_tol must be finite and positive')
        if not np.isfinite(occupation_tol) or occupation_tol <= 0:
            raise ValueError('occupation_tol must be finite and positive')
        if self.state_capacity not in (1.0, 2.0):
            raise ValueError('state_capacity must be 1 or 2')

    def set_fermi_level(self, efermi: float):
        """设置费米能级"""
        if not np.isfinite(efermi):
            raise ValueError('fermi_level must be finite')
        self.fermi_level = float(efermi)

    def get_band_gap(self) -> dict:
        """Classify sampled k points using occupations, filling and crossings.

        Occupations fix the physical filling; changing the display energy zero
        does not change them. Fractional occupations without a resolved crossing
        are indeterminate because smearing/fixed occupations can cause them.
        """
        energy = np.asarray(self.data.energies, dtype=float) - self.fermi_level
        base = {'gap': None, 'direct': None, 'vbm': None, 'cbm': None,
                'status': 'unknown', 'scope': 'sampled-kpoints', 'reason': '',
                'raw_gap': None, 'zero_tol_eV': self.zero_gap_tol}
        if energy.ndim != 2 or not energy.size:
            return dict(base, reason='No band energies')
        if not np.isfinite(energy).all():
            raise ValueError('Band energies must be finite')
        capacity = self.state_capacity
        tol = self.occupation_tol
        occupations = self.data.occupations
        if occupations is not None:
            occupations = np.asarray(occupations, dtype=float)
            if occupations.shape != energy.shape or not np.isfinite(occupations).all():
                raise ValueError('Occupations must be finite and match the energy array')
            if np.any(occupations < -tol) or np.any(occupations > capacity + tol):
                return dict(base, reason='Occupations exceed the state capacity; check smearing/spin convention')
            filled = occupations >= capacity - tol
            empty = occupations <= tol
            changing = (np.any(occupations > capacity / 2 + tol, axis=0) &
                        np.any(occupations < capacity / 2 - tol, axis=0))
            if changing.any():
                return dict(base, gap=0.0, status='metal', reason='Band occupation changes across k points')
            if not np.all(filled | empty):
                return dict(base, reason='Partial occupations need smearing or filling information')
            occupied_columns = np.flatnonzero(np.all(filled, axis=0))
            empty_columns = np.flatnonzero(np.all(empty, axis=0))
            if abs(len(occupied_columns) * capacity - self.data.num_electrons) > tol:
                return dict(base, reason='Fully occupied band count disagrees with NELECT')
            crossing = False
        else:
            crossing = bool(np.any((energy.min(axis=0) < -self.zero_gap_tol) &
                                   (energy.max(axis=0) > self.zero_gap_tol)))
            if self.data.ispin == 1:
                filling = self.data.num_electrons / capacity
                if not np.isfinite(filling) or abs(filling - round(filling)) > tol:
                    return dict(base, gap=0.0 if crossing else None,
                                status='metal' if crossing else 'unknown',
                                reason='Fermi crossing' if crossing else 'Fractional filling requires occupations')
                count = int(round(filling))
                if not 0 < count < energy.shape[1]:
                    return dict(base, reason='Occupied and empty bands are both required')
                occupied_columns, empty_columns = np.arange(count), np.arange(count, energy.shape[1])
            else:
                # Total NELECT alone cannot assign spin-up/down populations.
                occupied_columns = np.flatnonzero(np.all(energy <= self.zero_gap_tol, axis=0))
                empty_columns = np.flatnonzero(np.all(energy >= -self.zero_gap_tol, axis=0))
                empty_columns = np.setdiff1d(empty_columns, occupied_columns)
        if not len(occupied_columns) or not len(empty_columns):
            return dict(base, gap=0.0 if crossing else None, status='metal' if crossing else 'unknown',
                        reason='Fermi crossing' if crossing else 'Occupied and empty bands are both required')
        valence = energy[:, occupied_columns]
        conduction = energy[:, empty_columns]
        vi, vb_local = np.unravel_index(np.argmax(valence), valence.shape)
        ci, cb_local = np.unravel_index(np.argmin(conduction), conduction.shape)
        vb, cb = int(occupied_columns[vb_local]), int(empty_columns[cb_local])
        vbm, cbm = float(energy[vi, vb]), float(energy[ci, cb])
        raw_gap = cbm - vbm
        status = ('metal' if crossing or raw_gap < -self.zero_gap_tol else
                  'zero-gap' if raw_gap <= self.zero_gap_tol else 'insulator')
        direct = None
        if status == 'insulator':
            v_candidates = np.flatnonzero(np.max(valence, axis=1) >= vbm - 1e-6)
            c_candidates = np.flatnonzero(np.min(conduction, axis=1) <= cbm + 1e-6)
            direct = False
            for v_index in v_candidates:
                equivalent = _equivalent_kpoints(self.data.kpoints[v_index], self.data.kpoints[c_candidates])
                if equivalent.any():
                    vi, ci = int(v_index), int(c_candidates[np.flatnonzero(equivalent)[0]])
                    vb = int(occupied_columns[np.argmax(valence[vi])])
                    cb = int(empty_columns[np.argmin(conduction[ci])])
                    direct = True
                    break
        return dict(base, gap=raw_gap if status == 'insulator' else 0.0,
                    direct=direct, vbm=vbm, cbm=cbm,
                    vbm_k=int(vi), cbm_k=int(ci), vbm_band=vb, cbm_band=cb,
                    raw_gap=raw_gap, status=status,
                    reason='Fermi crossing' if crossing else 'Band overlap' if status == 'metal' else '')

    def fit_effective_mass(self, band_index: int, k_index: int,
                           num_points: int = 5, *, side: Optional[str] = None) -> dict:
        """Fit signed directional mass on one straight physical path branch.

        At least five distinct samples are required. Reports include the actual
        window, direction, residual and curvature stability. These are numerical
        diagnostics, not proof of a continuum band or an experimental mass.
        side='left'/'right' explicitly chooses a branch at an unsegmented corner.
        """
        report = {'status': 'unavailable', 'reason': '', 'mass_m0': None,
                  'scope': 'path-direction', 'direction_cartesian': None,
                  'indices': [], 'span_angstrom_inv': None, 'rms_eV': None,
                  'curvature_eV_angstrom2': None, 'curvature_relative_error': None,
                  'window_curvature_change': None, 'lattice_source': self.data.lattice_source}
        def unavailable(reason):
            return dict(report, reason=reason)
        if self.data.lattice_matrix is None:
            return unavailable('Missing lattice: load the matching POSCAR to calibrate k distances')
        if (isinstance(band_index, bool) or not isinstance(band_index, (int, np.integer)) or
                not 0 <= band_index < self.data.num_bands or isinstance(k_index, bool) or
                not isinstance(k_index, (int, np.integer)) or not 0 <= k_index < self.data.nkpoints):
            return unavailable('Band or k-point index is out of range')
        if (isinstance(num_points, bool) or not isinstance(num_points, (int, np.integer)) or
                num_points < 2 or side not in (None, 'left', 'right')):
            return unavailable('Window must have at least two neighbors; side must be left or right')
        try:
            cart = self.data.kcartesian
        except (ValueError, np.linalg.LinAlgError):
            return unavailable('Invalid lattice matrix')
        if not np.isfinite(cart).all() or not np.isfinite(self.data.energies).all():
            return unavailable('Non-finite coordinates or energies')
        lower, upper = 0, self.data.nkpoints - 1
        if self.data.segments:
            containing = [(lo, hi) for lo, hi in self.data.segments if lo <= k_index <= hi]
            if len(containing) != 1:
                return unavailable('Reference point does not belong to one path segment')
            lower, upper = containing[0]
        left = next((i for i in range(k_index - 1, lower - 1, -1)
                     if np.linalg.norm(cart[k_index] - cart[i]) > 1e-10), None)
        right = next((i for i in range(k_index + 1, upper + 1)
                      if np.linalg.norm(cart[i] - cart[k_index]) > 1e-10), None)
        def unit(v):
            return v / np.linalg.norm(v)
        dl = None if left is None else unit(cart[k_index] - cart[left])
        dr = None if right is None else unit(cart[right] - cart[k_index])
        if side is None and dl is not None and dr is not None and np.linalg.norm(dl - dr) > 1e-4:
            return unavailable('Path turn at reference point: choose a left or right direction')
        direction = dl if side == 'left' else dr if side == 'right' else dr if dr is not None else dl
        if direction is None:
            return unavailable('No distinct neighbor in the selected direction')
        indices = [k_index]
        # Stop at the first turn, reversal or segment boundary. Repeated points
        # carry no additional fit information and are not counted as samples.
        for sign in (-1, 1):
            if (side == 'left' and sign == 1) or (side == 'right' and sign == -1):
                continue
            previous = k_index
            count = 0
            for i in range(k_index + sign, lower - 1 if sign < 0 else upper + 1, sign):
                step = sign * (cart[i] - cart[previous])
                if np.linalg.norm(step) <= 1e-10:
                    continue
                if np.linalg.norm(unit(step) - direction) > 1e-4:
                    break
                indices.append(i)
                previous = i
                count += 1
                if count >= num_points:
                    break
        indices = np.array(sorted(indices))
        x = (cart[indices] - cart[k_index]) @ direction
        report.update(indices=indices.tolist(), direction_cartesian=direction.tolist(),
                      span_angstrom_inv=float(np.ptp(x)))
        if len(indices) < 5 or np.ptp(x) <= 1e-8:
            return unavailable('Fewer than five distinct samples on one straight path branch')
        # Compare only adjacent bands in the same spin channel, avoiding false
        # rejection of independent spin degeneracies.
        local_band = band_index % self.data.nbands
        neighbors = []
        if local_band > 0:
            neighbors.append(band_index - 1)
        if local_band + 1 < self.data.nbands:
            neighbors.append(band_index + 1)
        y = self.data.energies[indices, band_index]
        for neighbor in neighbors:
            separation = self.data.energies[indices, neighbor] - y
            if np.min(np.abs(separation)) <= 1e-4 or separation.min() * separation.max() < 0:
                return unavailable('Band crossing or near-degeneracy in the fit window')
        y = y - self.data.energies[k_index, band_index]
        def fit(xx, yy):
            span = np.ptp(xx)
            z = xx / span
            design = np.column_stack((z ** 2, z, np.ones(len(z))))
            coefficients, _, rank, _ = np.linalg.lstsq(design, yy, rcond=None)
            residual = yy - design @ coefficients
            rms = float(np.sqrt(np.mean(residual ** 2)))
            variance = float(residual @ residual / (len(xx) - 3))
            covariance = variance * np.linalg.inv(design.T @ design)
            return coefficients[0] / span ** 2, rms, np.sqrt(covariance[0, 0]) / span ** 2, rank
        try:
            a, rms, error, rank = fit(x, y)
            report.update(rms_eV=rms, curvature_eV_angstrom2=float(2 * a),
                          curvature_relative_error=float(error / abs(a)) if a else None)
            if rank != 3 or abs(a) < 1e-10:
                return unavailable('Curvature is flat or unresolved')
            if rms > max(1e-5, .02 * np.ptp(y)) or error / abs(a) > .2:
                return unavailable('Parabolic residual or curvature uncertainty is too large')
            if len(indices) >= 7:
                inner = np.argsort(np.abs(x))[:max(5, len(x) // 2)]
                inner_a, _, _, _ = fit(x[inner], y[inner])
                change = float(abs(inner_a - a) / abs(a))
                report['window_curvature_change'] = change
                if change > .1:
                    return unavailable('Curvature changes by more than 10% when the window shrinks')
        except np.linalg.LinAlgError:
            return unavailable('Fit matrix is singular')
        return dict(report, status='available', reason='Straight branch; residual and window checks passed',
                    mass_m0=float(self.HBAR2_OVER_2M0 / a))

    def effective_mass(self, band_index: int, k_index: int,
                       num_points: int = 5) -> Optional[float]:
        """Signed directional mass in m₀, or None; see fit_effective_mass()."""
        return self.fit_effective_mass(band_index, k_index, num_points)['mass_m0']

    def mass_fit_reports_at_gap(self) -> dict:
        """VBM/CBM path-fit diagnostics, including reasons when unavailable."""
        gap = self.get_band_gap()
        if gap['status'] != 'insulator':
            report = {'status': 'unavailable', 'mass_m0': None,
                      'reason': 'An insulating sampled gap is required'}
            return {'vbm': dict(report), 'cbm': dict(report)}
        return {edge: self.fit_effective_mass(gap[f'{edge}_band'], gap[f'{edge}_k'])
                for edge in ('vbm', 'cbm')}

    def estimate_effective_masses_at_gap(self) -> dict:
        """
        在 VBM 和 CBM 附近分别估算有效质量。
        返回 {vbm_mass, cbm_mass}，单位 m₀。
        """
        gap_info = self.get_band_gap()
        if gap_info['status'] != 'insulator':
            return {'vbm_mass': None, 'cbm_mass': None}

        vbm_k = gap_info['vbm_k']
        cbm_k = gap_info['cbm_k']

        # Use the actual band columns, including either spin channel.
        vbm_mass = self.effective_mass(gap_info['vbm_band'], vbm_k)
        cbm_mass = self.effective_mass(gap_info['cbm_band'], cbm_k)

        return {
            'vbm_mass': vbm_mass,
            'cbm_mass': cbm_mass,
        }
