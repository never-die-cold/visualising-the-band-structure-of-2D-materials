"""Independent periodic extrema and sampled-data classification benchmarks."""

import numpy as np
import pytest
from scipy.optimize import root

from core.band_analyzer import BandAnalyzer
from core.parser import BandData
from vdw_studio.analysis import analyze_gap, analyze_path_gap
from vdw_studio.engine.models import HoneycombModel
from vdw_studio.engine.strain import apply_strain
from vdw_studio.simulation import simulate
from vdw_studio.structure.lattice import Lattice


class AnalyticPeriodicModel:
    """Two separated periodic cosine bands with prescribed global extrema."""
    n_sites = 2
    name = "analytic periodic test"
    lattice = Lattice.square(2.0)

    def __init__(self, vk=(.173, .287), ck=(.719, .613), gap=.9, shift=0.):
        self.vk, self.ck = np.array(vk), np.array(ck)
        self.gap, self.shift = gap, shift

    def bands(self, points):
        points = np.asarray(points)
        fv = (1 - np.cos(2 * np.pi * (points - self.vk))).sum(axis=1)
        fc = (1 - np.cos(2 * np.pi * (points - self.ck))).sum(axis=1)
        return np.column_stack((-.2 - fv + self.shift, -.2 + self.gap + fc + self.shift))

    def energies_at(self, point):
        return self.bands(np.asarray(point)[None, :])[0]


def assert_same_periodic_point(left, right, atol=2e-7):
    delta = np.asarray(left) - np.asarray(right)
    np.testing.assert_allclose(delta - np.rint(delta), 0, atol=atol)


@pytest.mark.parametrize("ex", [.02, .05, -.03])
def test_uniaxial_graphene_finds_shifted_dirac_with_independent_root(ex):
    model = apply_strain(HoneycombModel(a=2.46), ex=ex, ey=0.)
    def off_diagonal(point):
        value = model.hamiltonian(point)[0, 1]
        return [value.real, value.imag]
    reference = root(off_diagonal, [1/3, 1/3], tol=1e-11)
    assert reference.success
    assert np.linalg.norm(off_diagonal(reference.x)) < 1e-9
    gap = analyze_gap(model, mesh=(12, 12))
    path = analyze_path_gap(model, n_per_segment=40)
    assert gap.scope == "brillouin-zone"
    assert gap.status == "zero-gap" and gap.gap is None
    assert gap.raw_gap < 1e-7
    # The search may select either time-reversed Dirac valley.
    assert min(np.linalg.norm((gap.cbm_k - reference.x + .5) % 1 - .5),
               np.linalg.norm((gap.cbm_k + reference.x + .5) % 1 - .5)) < 2e-7
    assert gap.cbm_label == "off-path"
    assert gap.search_metadata["converged"] is True
    assert path.scope == "path" and path.gap > .1


@pytest.mark.parametrize("mesh", [(8, 8), (12, 15), (24, 24)])
def test_off_path_indirect_extrema_and_gap_match_closed_form(mesh):
    model = AnalyticPeriodicModel()
    gap = analyze_gap(model, mesh=mesh)
    assert gap.gap == pytest.approx(.9, abs=1e-8)
    assert gap.vbm == pytest.approx(-.2, abs=1e-8)
    assert gap.cbm == pytest.approx(.7, abs=1e-8)
    assert gap.direct is False
    assert_same_periodic_point(gap.vbm_k, model.vk)
    assert_same_periodic_point(gap.cbm_k, model.ck)
    assert gap.search_metadata["converged"] is True
    assert analyze_path_gap(model).gap > gap.gap + .05


@pytest.mark.parametrize("center", [(.993, .017), (.173, .287)])
def test_direct_extremum_and_periodic_boundary(center):
    model = AnalyticPeriodicModel(vk=center, ck=center)
    gap = analyze_gap(model, mesh=(8, 8))
    assert gap.gap == pytest.approx(.9, abs=1e-8)
    assert gap.direct is True
    assert_same_periodic_point(gap.vbm_k, center)
    assert_same_periodic_point(gap.cbm_k, center)


def test_bz_gap_is_independent_of_energy_zero_and_path_density():
    original = analyze_gap(AnalyticPeriodicModel(), n_per_segment=2, mesh=(8, 8))
    shifted = analyze_gap(AnalyticPeriodicModel(shift=5), n_per_segment=83, mesh=(8, 8))
    assert shifted.gap == pytest.approx(original.gap, abs=1e-8)
    assert shifted.direct == original.direct
    assert shifted.vbm - original.vbm == pytest.approx(5, abs=1e-8)
    assert_same_periodic_point(shifted.vbm_k, original.vbm_k)


@pytest.mark.parametrize("raw_gap, status", [(-.1, "metal"), (.0002, "zero-gap"), (.01, "insulator")])
def test_overlap_zero_tolerance_and_small_gap(raw_gap, status):
    vk = np.array([.173, .287])
    ck = (vk + .5) % 1 if raw_gap < 0 else vk
    gap = analyze_gap(AnalyticPeriodicModel(vk=vk, ck=ck, gap=raw_gap), mesh=(8, 8))
    assert gap.raw_gap == pytest.approx(raw_gap, abs=1e-8)
    assert gap.status == status
    assert (gap.gap is None) == (status != "insulator")
    assert gap.direct is None if status != "insulator" else gap.direct is True


def test_failed_local_optimizer_marks_result_unconverged(monkeypatch):
    from vdw_studio.analysis import properties
    original = properties.minimize
    def fail(*args, **kwargs):
        result = original(*args, **kwargs)
        result.success = False
        return result
    monkeypatch.setattr(properties, "minimize", fail)
    gap = analyze_gap(AnalyticPeriodicModel(), mesh=(8, 8))
    assert gap.search_metadata["converged"] is False


@pytest.mark.parametrize("kwargs", [
    {"mesh": (3, 8)}, {"mesh": (8.5, 8)}, {"n_valence": 0},
    {"n_valence": 2}, {"n_valence": 1.5}, {"zero_tol": 0},
    {"zero_tol": float("nan")}, {"max_starts": 0}, {"convergence_tol": -1},
])
def test_invalid_gap_parameters_are_rejected(kwargs):
    with pytest.raises(ValueError):
        analyze_gap(AnalyticPeriodicModel(), **kwargs)


def test_simulation_separates_path_and_bz_results():
    model = apply_strain(HoneycombModel(a=2.46), ex=.02)
    result = simulate(model, n_per_segment=10, dos_mesh=(8, 8), gap_mesh=(12, 12))
    assert result["gap"].status == "zero-gap"
    assert result["path_gap"].gap > .1
    assert result["gap"].scope == "brillouin-zone"
    assert result["path_gap"].scope == "path"


def band_data(energies, occupations=None, nelect=2, ispin=1, points=None):
    energies = np.asarray(energies, dtype=float)
    nk, nb = energies.shape
    if points is None:
        points = np.column_stack((np.linspace(0, .5, nk), np.zeros((nk, 2))))
    return BandData(np.asarray(points, dtype=float), np.arange(nk, dtype=float), energies, nb, nelect,
                    occupations=None if occupations is None else np.asarray(occupations, dtype=float), ispin=ispin)


@pytest.mark.parametrize("occupations", [None, [[2, 0], [0, 0]]])
def test_band_crossing_never_returns_false_positive_gap(occupations):
    data = band_data([[-1, .1], [.2, 1]], occupations)
    result = BandAnalyzer(data).get_band_gap()
    assert result["status"] == "metal"
    assert result["gap"] == 0.
    assert result["direct"] is None
    assert BandAnalyzer(data).estimate_effective_masses_at_gap() == {"vbm_mass": None, "cbm_mass": None}


def test_occupations_determine_edges_when_default_zero_is_outside_gap():
    data = band_data([[9, 12], [10, 11]], [[2, 0], [2, 0]])
    analyzer = BandAnalyzer(data)
    result = analyzer.get_band_gap()
    assert result["gap"] == 1
    assert result["status"] == "insulator" and result["direct"] is True
    analyzer.set_fermi_level(10.5)
    shifted = analyzer.get_band_gap()
    assert shifted["gap"] == result["gap"]
    assert shifted["vbm"] == -.5 and shifted["cbm"] == .5
    assert shifted["scope"] == "sampled-kpoints"


@pytest.mark.parametrize("partial", [.05, .5, 1., 1.5, 1.95])
def test_partial_occupations_without_crossing_remain_undetermined(partial):
    data = band_data([[-1, 1], [-.8, 1.2]], [[partial, 0], [partial, 0]])
    result = BandAnalyzer(data).get_band_gap()
    assert result["status"] == "unknown" and result["gap"] is None
    assert "Partial occupations" in result["reason"]


def test_small_smearing_tails_do_not_create_false_metal():
    data = band_data([[-1, 1], [-.8, 1.2]], [[1.9999, .0001], [1.9998, .0002]])
    result = BandAnalyzer(data).get_band_gap()
    assert result["status"] == "insulator"
    assert result["gap"] == 1.8


@pytest.mark.parametrize("gap, status", [(0., "zero-gap"), (.0002, "zero-gap"), (.01, "insulator")])
def test_sampled_zero_and_small_positive_gap(gap, status):
    data = band_data([[-1, 2], [0, gap]], [[2, 0], [2, 0]])
    result = BandAnalyzer(data).get_band_gap()
    assert result["status"] == status
    assert result["gap"] == pytest.approx(gap if status == "insulator" else 0)


def test_sampled_overlap_with_fixed_occupations_is_metal():
    data = band_data([[.2, 1], [-1, .1]], [[2, 0], [2, 0]])
    result = BandAnalyzer(data).get_band_gap()
    assert result["status"] == "metal" and result["gap"] == 0
    assert result["raw_gap"] == pytest.approx(-.1)


def test_spin_populations_can_have_different_occupied_band_counts():
    data = band_data([[-1, -.5, 2, -.8, 1, 3], [-.9, -.4, 2.1, -.7, .9, 3.1]],
                     [[1, 1, 0, 1, 0, 0], [1, 1, 0, 1, 0, 0]], nelect=3, ispin=2)
    result = BandAnalyzer(data).get_band_gap()
    assert result["gap"] == 1.3
    assert (result["vbm_band"], result["cbm_band"]) == (1, 4)


def test_spinor_capacity_must_be_explicit():
    data = band_data([[-1, 1], [-.8, 1.2]], [[1, 0], [1, 0]], nelect=1)
    assert BandAnalyzer(data).get_band_gap()["status"] == "unknown"
    result = BandAnalyzer(data, state_capacity=1).get_band_gap()
    assert result["status"] == "insulator" and result["gap"] == 1.8


def test_degenerate_extrema_choose_common_valley_for_directness():
    # argmax VBM is k0 while argmin CBM is k1; k2 realizes both edges.
    data = band_data([[-.5, 2], [-1, .8], [-.5, .8]], [[2, 0]] * 3)
    result = BandAnalyzer(data).get_band_gap()
    assert result["direct"] is True
    assert result["vbm_k"] == result["cbm_k"] == 2


def test_missing_conduction_data_remains_undetermined():
    data = band_data([[-1], [-.8]], [[2], [2]])
    result = BandAnalyzer(data).get_band_gap()
    assert result["status"] == "unknown" and result["gap"] is None


def test_occupation_and_electron_count_disagreement_is_not_insulating():
    data = band_data([[-1, 1], [-.8, 1.2]], [[2, 0], [2, 0]], nelect=2.5)
    result = BandAnalyzer(data).get_band_gap()
    assert result["status"] == "unknown" and result["gap"] is None
    assert "NELECT" in result["reason"]


def test_reused_path_data_must_match_actual_coordinates():
    from vdw_studio.engine.kpath import KPath
    from vdw_studio.engine.solver import solve_bands
    model = HoneycombModel(a=2.46)
    path = KPath.for_lattice(model.lattice)
    band = solve_bands(model, path, 10)
    result = analyze_path_gap(model, path, n_per_segment=10, band=band)
    assert result.status == "zero-gap"
    band.kpoints[1, 0] += .01
    with pytest.raises(ValueError, match="does not match"):
        analyze_path_gap(model, path, n_per_segment=10, band=band)


def test_custom_zero_tolerance_preserves_small_positive_gap_precision():
    data = band_data([[-1, 2], [0, 1e-5]], [[2, 0], [2, 0]])
    result = BandAnalyzer(data, zero_gap_tol=1e-7).get_band_gap()
    assert result["status"] == "insulator"
    assert result["gap"] == pytest.approx(1e-5)


def test_flat_bands_have_converged_direct_gap():
    model = AnalyticPeriodicModel()
    model.bands = lambda points: np.tile([-1., 1.], (len(points), 1))
    result = analyze_gap(model, mesh=(8, 8))
    assert result.gap == 2.
    assert result.direct is True
    assert result.search_metadata["converged"] is True
