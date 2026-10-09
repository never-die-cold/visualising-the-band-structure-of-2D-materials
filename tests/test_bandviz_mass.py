"""Physical path calibration and independently specified parabolic mass fits."""

from pathlib import Path

import numpy as np
import pytest

from core.parser import BandData, VASPEigenvalParser, VASPLatticeParser
from core.band_analyzer import BandAnalyzer


RAW = np.array([[2., 0, 0], [-1, np.sqrt(3), 0], [0, 0, 12.]])


def write_poscar(tmp_path, scale="1", matrix=RAW):
    path = tmp_path / "POSCAR"
    path.write_text("Lattice fixture\n" + scale + "\n" +
                    "\n".join(" ".join(map(str, row)) for row in matrix) + "\n", encoding="utf-8")
    return path


def write_eigenval(tmp_path, points, energies=None):
    lines = ["1 1 1 1", "1 1 1 1 0", "0", "CAR", "Physical regression", f"2 {len(points)} 2"]
    for i, point in enumerate(points):
        ev, ec = (-1, 2) if energies is None else energies[i]
        lines.extend(["", " ".join(map(str, point)) + " 1", f"1 {ev} 2", f"2 {ec} 0"])
    path = tmp_path / "EIGENVAL"
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


@pytest.mark.parametrize("scale, factor", [("2", 2), (str(-abs(np.linalg.det(RAW)) * 8), 2)])
def test_scalar_and_negative_volume_poscar_calibrate_hexagonal_distances(tmp_path, scale, factor):
    poscar = write_poscar(tmp_path, scale)
    file = write_eigenval(tmp_path, [[0, 0, 0], [.25, 0, 0], [.25, .25, 0]])
    data = VASPEigenvalParser(str(file)).parse()
    np.testing.assert_allclose(data.lattice_matrix, factor * RAW)
    # |b1| = |b2| = 2π/√3 for a=2, divided by the scale.
    step = .25 * 2 * np.pi / np.sqrt(3) / factor
    np.testing.assert_allclose(data.kdistances, [0, step, 2 * step], rtol=1e-12)
    assert data.distance_unit == "angstrom^-1"
    assert data.lattice_source == str(poscar.resolve())
    np.testing.assert_allclose(data.lattice_matrix @ data.reciprocal_matrix.T, 2 * np.pi * np.eye(3), atol=1e-14)


def test_three_poscar_scales_multiply_cartesian_components(tmp_path):
    matrix, conversion = VASPLatticeParser(str(write_poscar(tmp_path, "2 3 4"))).parse()
    np.testing.assert_allclose(matrix, [[4, 0, 0], [-2, 3 * np.sqrt(3), 0], [0, 0, 48]])
    assert conversion is None


@pytest.mark.parametrize("scale", ["2", str(-abs(np.linalg.det(RAW)) * 8)])
def test_cartesian_kpoints_are_matched_automatically_with_poscar(tmp_path, scale):
    write_poscar(tmp_path, scale)
    points = np.array([[0, 0, 0], [.25, 0, 0], [.25, .25, 0]])
    file = write_eigenval(tmp_path, points)
    # For unscaled a=2, k/(2π/s) has components (fx/2, (fx/2+fy)/√3).
    coords = [[p[0] / 2, (p[0] / 2 + p[1]) / np.sqrt(3), 0] for p in points]
    (tmp_path / "KPOINTS").write_text("Path\n3\nCartesian\n" +
        "\n".join(" ".join(map(str, c)) + " 1 ! X" for c in coords) + "\n")
    data = VASPEigenvalParser(str(file)).parse()
    assert data.kpoint_labels == [(0, "X"), (1, "X"), (2, "X")]
    assert data.distance_unit == "angstrom^-1"


def test_three_scales_require_explicit_cartesian_kpoint_conversion(tmp_path):
    write_poscar(tmp_path, "2 3 4")
    file = write_eigenval(tmp_path, [[0, 0, 0]])
    (tmp_path / "KPOINTS").write_text("Path\n1\nCartesian\n0 0 0 1\n")
    with pytest.raises(ValueError, match="Cartesian.*matrix"):
        VASPEigenvalParser(str(file)).parse()
    assert VASPEigenvalParser(str(file), cartesian_to_fractional=RAW.T).parse().nkpoints == 1


def test_explicit_lattice_and_poscar_path_interfaces(tmp_path):
    file = write_eigenval(tmp_path, [[0, 0, 0], [.25, 0, 0]])
    data = VASPEigenvalParser(str(file), lattice_matrix=RAW).parse()
    assert data.kdistances[-1] == pytest.approx(np.pi / (2 * np.sqrt(3)))
    assert data.lattice_source == "explicit matrix"
    poscar = write_poscar(tmp_path)
    renamed = poscar.with_name("chosen-CONTCAR")
    poscar.rename(renamed)
    selected = VASPEigenvalParser(str(file), poscar_path=str(renamed)).parse()
    np.testing.assert_allclose(selected.kdistances, data.kdistances)
    with pytest.raises(FileNotFoundError, match="POSCAR"):
        VASPEigenvalParser(str(file), poscar_path=str(poscar)).parse()


def test_physical_distance_does_not_include_disconnected_segment_jump(tmp_path):
    write_poscar(tmp_path)
    points = [[0, 0, 0], [.25, 0, 0], [.5, 0, 0], [0, .5, 0], [.25, .5, 0], [.5, .5, 0]]
    file = write_eigenval(tmp_path, points)
    (tmp_path / "KPOINTS").write_text("Path\n3\nLine-mode\nReciprocal\n"
        "0 0 0 ! G\n.5 0 0 ! X\n0 .5 0 ! M\n.5 .5 0 ! K\n")
    data = VASPEigenvalParser(str(file)).parse()
    step = np.pi / (2 * np.sqrt(3))
    np.testing.assert_allclose(data.kdistances, np.array([0, 1, 2, 2, 3, 4]) * step)


@pytest.mark.parametrize("scale, matrix, message", [("0", RAW, "nonzero"),
    ("nan", RAW, "nonzero"), ("1 2", RAW, "one nonzero"), ("1 -2 3", RAW, "positive"),
    ("1", np.zeros((3, 3)), "singular"), ("1", RAW * float('nan'), "finite")])
def test_invalid_poscar_lattice_has_file_and_line_error(tmp_path, scale, matrix, message):
    file = write_poscar(tmp_path, scale, matrix)
    with pytest.raises(ValueError, match=message) as error:
        VASPLatticeParser(str(file)).parse()
    assert f"{file}:" in str(error.value)


def path_data(cart, energies, *, lattice=RAW, segments=(), ispin=1):
    cart, energies = np.asarray(cart, dtype=float), np.asarray(energies, dtype=float)
    fractional = cart @ RAW.T / (2 * np.pi)
    return BandData(fractional, np.r_[0, np.cumsum(np.linalg.norm(np.diff(cart, axis=0), axis=1))],
                    energies, energies.shape[1], 2, occupations=np.tile([2, 0], (len(cart), ispin)),
                    lattice_matrix=lattice, lattice_source="test lattice", segments=list(segments), ispin=ispin)


def parabola_data():
    x = np.linspace(-.12, .12, 13)
    cart = np.column_stack((x, x * 0, x * 0))
    return x, path_data(cart, np.column_stack((-1 - 3 * x ** 2, 2 + 5 * x ** 2)))


@pytest.mark.parametrize("band, expected", [(0, -3.80998 / 3), (1, 3.80998 / 5)])
def test_nonorthogonal_bandviz_mass_matches_cartesian_parabola(band, expected):
    _, data = parabola_data()
    report = BandAnalyzer(data).fit_effective_mass(band, 6)
    assert report['status'] == 'available'
    assert report['mass_m0'] == pytest.approx(expected, rel=1e-10)
    assert report['indices'] == list(range(1, 12))
    assert report['span_angstrom_inv'] == pytest.approx(.2)
    assert report['rms_eV'] < 1e-14
    assert report['window_curvature_change'] < 1e-10
    np.testing.assert_allclose(report['direction_cartesian'], [1, 0, 0], atol=1e-12)


def test_bandviz_mass_does_not_depend_on_fermi_zero_or_forced_vertex():
    _, data = parabola_data()
    analyzer = BandAnalyzer(data)
    analyzer.set_fermi_level(17.)
    # Off-grid reference still fits a general quadratic rather than forcing its vertex.
    assert analyzer.effective_mass(1, 5) == pytest.approx(3.80998 / 5, rel=1e-10)
    reports = analyzer.mass_fit_reports_at_gap()
    assert reports['cbm']['mass_m0'] == pytest.approx(3.80998 / 5)
    assert reports['vbm']['mass_m0'] < 0


def test_no_lattice_scalar_does_not_silently_calibrate_mass():
    _, data = parabola_data()
    data.lattice_matrix = None
    analyzer = BandAnalyzer(data, lattice_constant_angstrom=2.)
    assert analyzer.effective_mass(1, 6) is None
    assert 'Missing lattice' in analyzer.fit_effective_mass(1, 6)['reason']


def test_unsegmented_corner_requires_selected_direction_and_fits_only_that_branch():
    x = np.arange(7) * .02
    cart = np.vstack((np.column_stack((x - x[-1], x * 0, x * 0)),
                     np.column_stack((x[1:] * 0, x[1:], x[1:] * 0))))
    energy = 2 + 5 * cart[:, 0] ** 2 + 9 * cart[:, 1] ** 2
    data = path_data(cart, np.column_stack((-1 - .1 * (cart ** 2).sum(axis=1), energy)))
    analyzer = BandAnalyzer(data)
    assert 'Path turn' in analyzer.fit_effective_mass(1, 6)['reason']
    left = analyzer.fit_effective_mass(1, 6, side='left')
    right = analyzer.fit_effective_mass(1, 6, side='right')
    assert left['mass_m0'] == pytest.approx(3.80998 / 5)
    assert right['mass_m0'] == pytest.approx(3.80998 / 9)
    assert max(left['indices']) == 6 and min(right['indices']) == 6
    # A neighboring fit truncates at the turn rather than including both branches.
    nearby = analyzer.fit_effective_mass(1, 5)
    assert nearby['mass_m0'] == pytest.approx(3.80998 / 5)
    assert max(nearby['indices']) == 6


def test_segment_boundary_mass_uses_only_its_own_samples():
    _, data = parabola_data()
    data.segments = [(0, 6), (7, 12)]
    report = BandAnalyzer(data).fit_effective_mass(1, 6)
    assert report['status'] == 'available'
    assert max(report['indices']) == 6
    assert report['mass_m0'] == pytest.approx(3.80998 / 5)


@pytest.mark.parametrize("shape", ['quartic', 'noise', 'flat', 'linear'])
def test_unresolved_and_nonparabolic_curvature_has_no_quantitative_mass(shape):
    x, data = parabola_data()
    data.energies[:, 1] = (2 + 5 * x ** 2 + 1200 * x ** 4 if shape == 'quartic' else
                          2 + 5 * x ** 2 + .04 * np.cos(np.arange(len(x)) * np.pi) if shape == 'noise' else
                          2 + 3 * x if shape == 'linear' else np.full(len(x), 2.))
    report = BandAnalyzer(data).fit_effective_mass(1, 6)
    assert report['status'] == 'unavailable'
    assert report['mass_m0'] is None and report['reason']


@pytest.mark.parametrize("touch", [False, True])
def test_band_crossing_or_touching_window_is_rejected(touch):
    x, data = parabola_data()
    data.energies = np.column_stack((data.energies, data.energies[:, 1] + (x ** 2 if touch else x + .005)))
    data.num_bands = 3
    report = BandAnalyzer(data).fit_effective_mass(1, 6)
    assert 'crossing or near-degeneracy' in report['reason']
    assert report['mass_m0'] is None


def test_spin_degeneracy_is_not_a_same_channel_crossing():
    _, data = parabola_data()
    data.energies = np.tile(data.energies, (1, 2))
    data.ispin, data.num_bands = 2, 4
    assert BandAnalyzer(data).effective_mass(3, 6) == pytest.approx(3.80998 / 5)


def test_duplicate_points_do_not_supply_missing_fit_samples():
    _, data = parabola_data()
    data.kpoints[:] = data.kpoints[6]
    assert BandAnalyzer(data).effective_mass(1, 6) is None
    data.kpoints[:] = np.array([[i // 4 / 100, 0, 0] for i in range(13)])
    assert 'five distinct' in BandAnalyzer(data).fit_effective_mass(1, 6)['reason']
