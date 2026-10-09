"""Independent state counting, component additivity and DOSCAR regressions."""
from pathlib import Path

import numpy as np
import pytest

from core.dos_analyzer import DosAnalyzer, DoscarParser, load_spectrum
from core.parser import BandData, VASPEigenvalParser
from vdw_studio.engine.solver import solve_dos


def gaussian(grid, center, sigma):
    return np.exp(-.5 * ((grid - center) / sigma) ** 2) / (sigma * np.sqrt(2 * np.pi))


def data(ispin=1, occupations=True):
    energy = np.array([[-1., 1.], [-2., 2.]])
    if ispin == 2:
        energy = np.column_stack((energy, energy + .1))
    occ = np.tile([2 / ispin, 0], (2, ispin)) if occupations else None
    return BandData(np.array([[0., 0, 0], [.25, 0, 0]]), np.array([0., .25]),
                    energy, 2 * ispin, 2, ispin=ispin, occupations=occ,
                    weights=np.array([1., 3.]), sampling_kind='path')


def test_weighted_components_equal_total_and_match_independent_gaussians():
    spectrum = DosAnalyzer(np.array([[-1., 1.], [-2., 2.]])).calculate_dos(
        (-5, 5), 2001, .1, np.array([1., 3.]))
    expected = sum(w * gaussian(spectrum.energies, e, .1)
                   for w, row in zip([.25, .75], [[-1, 1], [-2, 2]]) for e in row)
    np.testing.assert_allclose(spectrum.total_dos, expected, atol=1e-14)
    np.testing.assert_allclose(spectrum.total_dos, spectrum.vb_dos + spectrum.cb_dos, atol=1e-14)
    assert spectrum.integral == pytest.approx(2., abs=1e-12)
    assert np.trapezoid(spectrum.vb_dos, spectrum.energies) == pytest.approx(1.)


@pytest.mark.parametrize('ispin', [1, 2])
def test_vasp_spin_capacity_counts_same_number_of_states(ispin):
    spectrum = DosAnalyzer.from_band_data(data(ispin)).calculate_dos((-5, 5), 2001, .1)
    assert spectrum.integral == pytest.approx(4., abs=1e-12)
    assert spectrum.expected_states == 4
    assert spectrum.state_capacity == 2 / ispin
    assert spectrum.component_labels == ('Occupied states', 'Unoccupied states')
    assert spectrum.scope == 'path-spectrum'


def test_soc_explicit_capacity_and_partial_occupations_split_state_weights():
    d = data()
    d.occupations = np.array([[.75, .25], [.5, 0.]])
    d.num_electrons = 1
    spectrum = DosAnalyzer.from_band_data(d, state_capacity=1).calculate_dos((-5, 5), 2001, .1)
    assert spectrum.integral == pytest.approx(2.)
    assert np.trapezoid(spectrum.vb_dos, spectrum.energies) == pytest.approx(.25 * 1 + .75 * .5)
    np.testing.assert_allclose(spectrum.vb_dos + spectrum.cb_dos, spectrum.total_dos, atol=1e-14)


def test_duplicate_sampling_and_raw_weight_scaling_preserve_normalized_spectrum():
    energy = np.array([[-1., 1.], [-2., 2.]])
    original = DosAnalyzer(energy, weights=[1, 3]).calculate_dos((-5, 5), 1001, .1)
    doubled = DosAnalyzer(np.tile(energy, (2, 1)), weights=[1, 3, 1, 3]).calculate_dos((-5, 5), 1001, .1)
    duplicated = DosAnalyzer(energy[[0, 0, 1]], weights=[.5, .5, 3]).calculate_dos((-5, 5), 1001, .1)
    large = DosAnalyzer(energy, weights=[1e300, 3e300]).calculate_dos((-5, 5), 1001, .1)
    for result in (doubled, duplicated, large):
        np.testing.assert_allclose(result.total_dos, original.total_dos, atol=1e-14)


def test_occupation_partition_survives_display_energy_shift():
    analyzer = DosAnalyzer.from_band_data(data())
    first = analyzer.calculate_dos((-5, 5), 2001, .1)
    analyzer.set_fermi_level(3.)
    shifted = analyzer.calculate_dos((-8, 2), 2001, .1)
    np.testing.assert_allclose(shifted.vb_dos, first.vb_dos, atol=1e-13)
    np.testing.assert_allclose(shifted.cb_dos, first.cb_dos, atol=1e-13)


def test_zero_weight_path_fallback_is_explicit_and_cannot_be_a_mesh_dos():
    d = data()
    d.weights[:] = 0
    result = DosAnalyzer.from_band_data(d).calculate_dos()
    assert 'uniform' in result.note and result.scope == 'path-spectrum'
    np.testing.assert_allclose(result.normalized_weights, [.5, .5])
    d.sampling_kind = 'mesh'
    with pytest.raises(ValueError, match='no positive'):
        DosAnalyzer.from_band_data(d)


def test_smearing_occupations_outside_capacity_do_not_block_total_spectrum():
    d = data()
    d.occupations[0] = [2.05, -.05]
    result = DosAnalyzer.from_band_data(d).calculate_dos()
    assert 'outside state capacity' in result.note
    assert result.component_labels == ('E ≤ EF states', 'E > EF states')


def test_automatic_mesh_is_the_only_inferred_eigenvalue_bz_dos(tmp_path):
    from test_bandviz_mass import write_eigenval
    file = write_eigenval(tmp_path, [[0, 0, 0], [.25, 0, 0]])
    (tmp_path / 'KPOINTS').write_text('Grid\n0\nGamma\n4 4 1\n0 0 0\n')
    parsed = VASPEigenvalParser(str(file)).parse()
    assert parsed.sampling_kind == 'mesh'
    assert DosAnalyzer.from_band_data(parsed).calculate_dos().scope == 'brillouin-zone'
    (tmp_path / 'KPOINTS').unlink()
    parsed = VASPEigenvalParser(str(file)).parse()
    assert DosAnalyzer.from_band_data(parsed).calculate_dos().scope == 'sampled-spectrum'


def test_memory_chunk_size_does_not_change_spectrum():
    rng = np.random.default_rng(14)
    energy = rng.normal(size=(73, 7))
    small = DosAnalyzer(energy, max_workspace_mb=.01).calculate_dos((-5, 5), 1401, .09)
    large = DosAnalyzer(energy, max_workspace_mb=8.).calculate_dos((-5, 5), 1401, .09)
    np.testing.assert_allclose(small.total_dos, large.total_dos, atol=1e-13)
    assert small.integral == pytest.approx(7., abs=1e-11)


@pytest.mark.parametrize('kwargs', [{'sigma': 0}, {'sigma': -.1}, {'sigma': np.nan},
    {'num_points': 1}, {'num_points': 2.5}, {'energy_range': (2, -2)},
    {'energy_range': (0, np.inf)}, {'per_kpoint_weight': [0, 0]},
    {'per_kpoint_weight': [-1, 2]}, {'per_kpoint_weight': [1]}, {'per_kpoint_weight': [1, np.nan]}])
def test_invalid_broadening_inputs_have_explanatory_errors(kwargs):
    with pytest.raises(ValueError):
        DosAnalyzer(data().energies).calculate_dos(**kwargs)


def write_doscar(tmp_path, spin=False):
    path = tmp_path / 'DOSCAR'
    lines = ['1 1 0 0', '1 1 1 1 0', '0', 'CAR', 'DOS regression', '2 -2 5 0.3 1']
    for i, e in enumerate([-2, -1, 0, 1, 2]):
        value = [0., .5, 1., .5, 0.][i]
        lines.append(f'{e} {value} {value / 2} {i / 2} {i / 4}' if spin else f'{e} {value} {i / 2}')
    lines.extend(['Projected block outside total reader', 'ignored'])
    path.write_text('\n'.join(lines) + '\n')
    return path


@pytest.mark.parametrize('spin', [False, True])
def test_doscar_header_bounds_channels_and_original_integral(tmp_path, spin):
    path = write_doscar(tmp_path, spin)
    parsed = DoscarParser(path).parse()
    np.testing.assert_allclose(parsed.energies, np.arange(-2, 3) - .3)
    np.testing.assert_allclose(parsed.total_dos, np.array([0., .5, 1, .5, 0]) * (1.5 if spin else 1))
    assert parsed.integral == pytest.approx(3. if spin else 2.)
    assert parsed.scope == 'brillouin-zone' and parsed.sigma == 0
    assert parsed.source == str(path.resolve())
    assert parsed.expected_states is None  # A finite DOSCAR window is not the full Hilbert space.
    if spin:
        np.testing.assert_allclose(parsed.spin_dos.sum(axis=1), parsed.total_dos)
    assert parsed.integrated_dos[-1] == pytest.approx(3. if spin else 2.)


def test_doscar_is_preferred_and_display_shift_does_not_rebroaden(tmp_path):
    write_doscar(tmp_path, True)
    imported = load_spectrum(data(), tmp_path / 'EIGENVAL', fermi_level=1., sigma=.4)
    assert len(imported.energies) == 5 and imported.sigma == 0
    np.testing.assert_allclose(imported.energies, np.arange(-2, 3) - 1.)
    assert imported.integral == pytest.approx(3.)
    assert 'file EF=0.3' in imported.note
    np.testing.assert_allclose(imported.vb_dos + imported.cb_dos, imported.total_dos)


@pytest.mark.parametrize('mutation, message', [
    (lambda lines: lines[:5], 'incomplete'),
    (lambda lines: lines[:9], 'complete'),
    (lambda lines: lines[:6] + ['-2 nan 0'] + lines[7:], 'finite'),
    (lambda lines: lines[:6] + ['-2 0'] + lines[7:], '3/5-column'),
    (lambda lines: lines[:5] + ['-2 2 5 0.3 1'] + lines[6:], 'invalid'),
    (lambda lines: lines[:7] + ['-3 .5 .5'] + lines[8:], 'increase'),
])
def test_invalid_present_doscar_is_not_silently_replaced(tmp_path, mutation, message):
    path = write_doscar(tmp_path)
    path.write_text('\n'.join(mutation(path.read_text().splitlines())) + '\n')
    with pytest.raises(ValueError, match=message) as error:
        load_spectrum(data(), tmp_path / 'EIGENVAL')
    assert str(path) in str(error.value)


class FlatModel:
    name = 'Four explicit states'
    n_sites = 999  # Deliberately not the Hilbert-space dimension.
    def bands(self, k):
        return np.tile([-2., -1., 1., 2.], (len(k), 1))


@pytest.mark.parametrize('mesh, degeneracy', [((2, 3), 1), ((5, 7), 1), ((4, 3), 2)])
def test_model_dos_counts_actual_bands_and_explicit_spin(mesh, degeneracy):
    result = solve_dos(FlatModel(), mesh, .1, n_points=2001, spin_degeneracy=degeneracy)
    assert np.trapezoid(result.dos, result.energies) == pytest.approx(4 * degeneracy, abs=1e-12)
    assert result.expected_states == 4 * degeneracy
    assert result.n_bands == 4 and result.spin_degeneracy == degeneracy


@pytest.mark.parametrize('kwargs', [{'sigma': 0}, {'sigma': -.1}, {'sigma': np.nan},
    {'mesh': (2.5, 3)}, {'mesh': (2, 0)}, {'n_points': 1}, {'e_min': 2, 'e_max': 1},
    {'e_max': float('inf')}])
def test_invalid_model_dos_inputs_are_rejected(kwargs):
    with pytest.raises(ValueError):
        solve_dos(FlatModel(), **kwargs)
