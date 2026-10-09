"""Reject invalid scientific inputs before warnings, NaNs or large work."""
import warnings
import numpy as np
import pytest

from core.band_analyzer import BandAnalyzer
from vdw_studio.analysis import solve_exciton, keldysh_potential, coulomb_potential
from vdw_studio.analysis.berry import berry_curvature, valley_report, valley_chern, optical_circular_dichroism
from vdw_studio.analysis.orbital_moment import orbital_moment, orbital_moment_map, valley_zeeman_splitting
from vdw_studio.engine.kpath import KPath
from vdw_studio.engine.solver import solve_bands
from vdw_studio.engine.models import HoneycombModel, TightBindingModel, Hopping
from vdw_studio.structure.lattice import Lattice


@pytest.mark.parametrize('kwargs', [
    {'mu_over_m0': 0}, {'mu_over_m0': np.nan}, {'mu_over_m0': np.inf},
    {'eps_env': 0}, {'eps_env': -1}, {'eps_env': np.nan},
    {'r0': 0}, {'r0': -.1}, {'r0': np.nan}, {'r0': np.inf},
    {'n_levels': 0}, {'n_levels': 2.5}, {'n_levels': True}, {'n_levels': 11, 'n_basis': 10},
    {'n_basis': 0}, {'n_basis': 2.5}, {'n_quad': 1}, {'n_quad': 2.5},
    {'n_quad': 20, 'n_basis': 30}, {'r_max': 0}, {'r_max': -1}, {'r_max': np.nan},
])
def test_exciton_invalid_inputs_raise_without_numerical_warnings(kwargs):
    with warnings.catch_warnings():
        warnings.simplefilter('error', RuntimeWarning)
        with pytest.raises(ValueError):
            solve_exciton(**kwargs)


def test_no_bound_state_is_a_finite_basis_status_not_an_index_error():
    # A very small hard-wall radial box has huge positive kinetic energy.
    result = solve_exciton(.25, 1., r_max=.01, n_basis=10, n_quad=300, n_levels=3)
    assert result.status == 'no-bound-state' and result.binding_1s is None
    assert result.energies.size == 0 and result.bound_states_in_basis == 0
    assert result.requested_levels == 3


def test_partial_bound_spectrum_reports_actual_number_of_levels():
    result = solve_exciton(.25, 1., r_max=2., n_basis=30, n_quad=1000, n_levels=5)
    assert 0 < len(result.energies) < 5
    assert result.status == 'partial-bound-spectrum'
    assert result.bound_states_in_basis == len(result.energies)
    assert result.binding_1s == -result.energies[0]


@pytest.mark.parametrize('function, args', [(keldysh_potential, ([0, 1], 1)),
    (keldysh_potential, ([1, 2], 0)), (keldysh_potential, ([1, np.nan], 1)),
    (coulomb_potential, ([0, 1],)), (coulomb_potential, ([1, 2], 0))])
def test_invalid_potential_inputs_do_not_divide_by_zero(function, args):
    with warnings.catch_warnings():
        warnings.simplefilter('error', RuntimeWarning)
        with pytest.raises(ValueError):
            function(*args)


def dirac(q):
    return np.array([[1., q[0] - 1j * q[1]], [q[0] + 1j * q[1], -1.]])


@pytest.mark.parametrize('function', [berry_curvature, orbital_moment])
@pytest.mark.parametrize('kwargs', [{'band': -1}, {'band': 2}, {'band': .5},
    {'band': True}, {'band': 0, 'dq': 0}, {'band': 0, 'dq': -.1},
    {'band': 0, 'dq': np.nan}, {'band': 0, 'q': [0, np.nan]}, {'band': 0, 'q': [0]}])
def test_local_observables_require_finite_coordinates_steps_and_valid_band(function, kwargs):
    arguments = {'q': [0, 0]}
    arguments.update(kwargs)
    with pytest.raises(ValueError):
        function(dirac, **arguments)


@pytest.mark.parametrize('function', [berry_curvature, orbital_moment])
def test_degenerate_single_band_observable_has_explicit_undefined_error(function):
    with pytest.raises(ValueError, match='degeneracy'):
        function(lambda q: np.eye(2), [0, 0], 0)


@pytest.mark.parametrize('matrix', [np.zeros((2, 3)), np.array([[1, 2], [0, -1]]),
    np.array([[np.nan, 0], [0, -1]])])
def test_nonhermitian_and_invalid_hamiltonians_are_rejected(matrix):
    with pytest.raises(ValueError, match='Hamiltonian'):
        berry_curvature(lambda q: matrix, [0, 0], 0)


@pytest.mark.parametrize('kwargs', [{'qmax': 0}, {'qmax': np.nan}, {'n_grid': 0},
    {'n_grid': 4}, {'n_grid': 3.5}])
def test_valley_maps_require_odd_valid_grid_and_finite_radius(kwargs):
    with pytest.raises(ValueError):
        valley_report(dirac, dirac, 0, 1, **kwargs)
    arguments = {'qmax': .1}
    arguments.update(kwargs)
    with pytest.raises(ValueError):
        orbital_moment_map(dirac, 0, **arguments)


@pytest.mark.parametrize('kwargs', [{'n_theta': 0}, {'n_rad': 0}, {'n_rad': 3.5},
    {'qmax': -.1}, {'valley': 0}])
def test_valley_integral_inputs_are_validated(kwargs):
    arguments = {'qmax': .1, 'n_theta': 3, 'n_rad': 1}
    arguments.update(kwargs)
    with pytest.raises(ValueError):
        valley_chern(dirac, 0, **arguments)


def test_optical_band_order_and_nonfinite_zeeman_inputs_are_rejected():
    with pytest.raises(ValueError, match='below'):
        optical_circular_dichroism(dirac, [0, 0], 1, 0)
    with pytest.raises(ValueError):
        valley_zeeman_splitting(1., np.nan)


@pytest.mark.parametrize('path', [[], ['G'], ['G', 'unknown']])
def test_invalid_named_paths_have_explanatory_error(path):
    with pytest.raises(ValueError):
        KPath({'G': (0, 0)}, path)


@pytest.mark.parametrize('count', [0, 1, 2.5, True])
def test_path_sampling_requires_integer_at_least_two(count):
    model = HoneycombModel(2.46)
    with pytest.raises(ValueError):
        solve_bands(model, KPath.for_lattice(model.lattice), count)


def test_tilted_path_preserves_full_cartesian_k_and_distance():
    rotation = np.array([[1, 0, 0], [0, .6, -.8], [0, .8, .6]])
    flat = Lattice.hexagonal(2.46)
    tilted = Lattice(flat.matrix @ rotation)
    path = KPath.for_lattice(flat)
    k0, x0, _ = path.generate(flat, 10)
    k1, x1, _ = path.generate(tilted, 10)
    np.testing.assert_allclose(k1, k0 @ rotation, atol=1e-14)
    np.testing.assert_allclose(x1, x0, atol=1e-14)
    assert np.max(np.abs(k1[:, 2])) > 0


def test_nonfinite_lattice_and_bandviz_energy_reference_are_rejected():
    from test_dos_contract import data
    with pytest.raises(ValueError, match='finite'):
        Lattice(np.eye(3) * np.nan)
    with pytest.raises(ValueError, match='finite'):
        BandAnalyzer(data()).set_fermi_level(np.nan)


@pytest.mark.parametrize('changes', [
    {'n_sites': 1.5}, {'onsite': [np.nan]}, {'site_symbols': []},
    {'hoppings': [Hopping(0.5, 0, (0, 0, 0), 1)]},
    {'hoppings': [Hopping(0, 0, (np.inf, 0, 0), 1)]},
    {'hoppings': [Hopping(0, 0, (0, 0, 0), np.nan)]},
])
def test_invalid_tb_model_inputs_fail_before_diagonalization(changes):
    arguments = dict(lattice=Lattice.hexagonal(2.46), n_sites=1,
                     onsite=[0], site_symbols=['C'], hoppings=[])
    arguments.update(changes)
    with pytest.raises(ValueError):
        TightBindingModel(**arguments)


@pytest.mark.parametrize('point', [[0], [0, 0, 0, 0], [[0, 0]], [0, np.nan]])
def test_tb_kpoint_shape_is_not_silently_truncated(point):
    with pytest.raises(ValueError, match='kfrac'):
        HoneycombModel(2.46).energies_at(point)


def test_unpaired_same_site_hopping_is_not_silently_diagonalized():
    lattice = Lattice.hexagonal(2.46)
    invalid = TightBindingModel(lattice, 1, ['C'], [0],
                                [Hopping(0, 0, (1, 0, 0), 1)])
    with pytest.raises(ValueError, match='Hermitian'):
        invalid.energies_at([.25, 0])
    paired = TightBindingModel(lattice, 1, ['C'], [0],
        [Hopping(0, 0, (1, 0, 0), 1), Hopping(0, 0, (-1, 0, 0), 1)])
    np.testing.assert_allclose(paired.energies_at([.2, 0]), [2 * np.cos(.4 * np.pi)])


def test_single_variational_exciton_basis_is_valid():
    result = solve_exciton(.25, 1., r_max=.01, n_basis=1, n_quad=3, n_levels=1)
    assert result.status == 'no-bound-state'
