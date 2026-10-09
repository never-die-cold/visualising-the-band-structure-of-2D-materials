"""BandViz regressions: real VASP data, strict errors and path coordinates."""

import hashlib
import json
from pathlib import Path

import numpy as np
import pytest

from core.band_analyzer import BandAnalyzer
from core.parser import BandData, VASPEigenvalParser, VASPKpointsParser


FIXTURES = Path(__file__).parent / "fixtures" / "vasp"
ROOT = Path(__file__).resolve().parents[1]


def write_eigenval(tmp_path, points=None, ispin=1, nelect=2):
    if points is None:
        points = [[0, 0, 0], [.25, 0, 0], [.5, 0, 0]]
    lines = [f"1 1 1 {ispin}", "1 1 1 1 0", "0", "CAR", "Regression fixture",
             f"{nelect} {len(points)} 2"]
    for point in points:
        lines.extend(["", " ".join(map(str, point)) + " 0.25"])
        lines.extend(["1 -1 2", "2 2 0"] if ispin == 1 else
                     ["1 -1 -0.5 1 0.75", "2 2 0.8 0 0.25"])
    path = tmp_path / "EIGENVAL"
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


def write_kpoints(tmp_path, text):
    path = tmp_path / "KPOINTS"
    path.write_text(text, encoding="utf-8")
    return path


@pytest.mark.parametrize("name", ["EIGENVAL.nonspin", "EIGENVAL.spin"])
def test_real_files_match_pymatgen_reference(name):
    reference = json.loads((FIXTURES / "reference.json").read_text())[name]
    path = FIXTURES / name
    digest = hashlib.sha256(path.read_text(encoding="utf-8").encode()).hexdigest()
    assert digest == reference["fixture_lf_sha256"]
    data = VASPEigenvalParser(str(path)).parse()
    assert (data.nelect, data.nkpoints, data.nbands, data.ispin) == (
        reference["nelect"], reference["nkpoints"], reference["nbands"], reference["ispin"])
    np.testing.assert_allclose(data.kpoints, reference["kpoints"], atol=0, rtol=0)
    np.testing.assert_allclose(data.weights, reference["weights"], atol=0, rtol=0)
    for index, key in enumerate(["1"] if data.ispin == 1 else ["1", "-1"]):
        eigenvalues = np.asarray(reference["channels"][key])
        np.testing.assert_allclose(data.spin_energies[:, :, index], eigenvalues[:, :, 0], atol=0, rtol=0)
        np.testing.assert_allclose(data.spin_occupations[:, :, index], eigenvalues[:, :, 1], atol=0, rtol=0)


@pytest.mark.parametrize("ispin", [1, 2])
def test_standard_header_spin_and_fractional_electrons(tmp_path, ispin):
    path = write_eigenval(tmp_path, ispin=ispin, nelect=2.5)
    data = VASPEigenvalParser(str(path)).parse()
    assert (data.nelect, data.nkpoints, data.nbands, data.ispin) == (2.5, 3, 2, ispin)
    assert data.energies.shape == data.occupations.shape == (3, 2 * ispin)
    np.testing.assert_allclose(data.weights, [.25, .25, .25])
    np.testing.assert_allclose(data.kdistances, [0, .25, .5])
    if ispin == 2:
        np.testing.assert_allclose(data.energies[0], [-1, 2, -.5, .8])
        np.testing.assert_allclose(data.occupations[0], [1, 0, .75, .25])


@pytest.mark.parametrize("mutation, expected", [
    (lambda lines: lines[:5], "header is incomplete"),
    (lambda lines: lines[:6], "expected 3 k points, got 0"),
    (lambda lines: lines[:-1], "expected band 2/2"),
    (lambda lines: lines[:-4], "expected 3 k points, got 2"),
    (lambda lines: lines + lines[6:10], "unexpected data after 3"),
])
def test_truncation_and_extra_blocks(tmp_path, mutation, expected):
    path = write_eigenval(tmp_path)
    path.write_text("\n".join(mutation(path.read_text().splitlines())) + "\n")
    with pytest.raises(ValueError, match=expected):
        VASPEigenvalParser(str(path)).parse()


@pytest.mark.parametrize("line, text, expected", [
    (0, "1 1 1 3", "unsupported ISPIN"),
    (0, "Generated file", "invalid EIGENVAL header"),
    (5, "2 3", "expected NELECT NKPTS NBANDS"),
    (5, "invalid 3 2", "expected numeric NELECT"),
    (5, "2 3.5 2", "integer NKPTS/NBANDS"),
    (5, "nan 3 2", "NELECT must be"),
    (5, "-2 3 2", "NELECT must be"),
    (5, "2 0 2", "NKPTS/NBANDS positive"),
    (5, "2 3 0", "NKPTS/NBANDS positive"),
    (7, "0 0 0", "nonnegative finite weight"),
    (7, "0 0 0 -1", "nonnegative finite weight"),
    (7, "0 nan 0 .25", "finite coordinates"),
    (8, "2 -1 2", "expected band index 1"),
    (8, "1 nan 2", "finite energies/occupations"),
    (8, "1 -1", "expected 3 columns"),
    (8, "1 -1 2 0 0", "expected 3 columns"),
    (8, "", "expected 3 columns"),
    (8, "1 invalid 2", "invalid band row"),
])
def test_invalid_rows_locate_file_and_line(tmp_path, line, text, expected):
    path = write_eigenval(tmp_path)
    lines = path.read_text().splitlines()
    lines[line] = text
    path.write_text("\n".join(lines) + "\n")
    with pytest.raises(ValueError, match=expected) as error:
        VASPEigenvalParser(str(path)).parse()
    assert str(path) in str(error.value)
    assert f":{line + 1}:" in str(error.value)


def test_spin_header_rejects_single_channel_rows(tmp_path):
    path = write_eigenval(tmp_path)
    lines = path.read_text().splitlines()
    lines[0] = "1 1 1 2"
    path.write_text("\n".join(lines))
    with pytest.raises(ValueError, match="expected 5 columns"):
        VASPEigenvalParser(str(path)).parse()


def test_fortran_notation_and_bom(tmp_path):
    path = write_eigenval(tmp_path)
    lines = path.read_text().splitlines()
    lines[8] = "1 -1D+00 2d0"
    path.write_text("\n".join(lines), encoding="utf-8-sig")
    data = VASPEigenvalParser(str(path)).parse()
    assert data.energies[0, 0] == -1
    assert data.occupations[0, 0] == 2


def test_legacy_format_requires_explicit_opt_in(tmp_path):
    path = write_eigenval(tmp_path)
    lines = path.read_text().splitlines()
    lines[0], lines[5] = "Generated example", "1 1 2 2"
    path.write_text("\n".join(lines))
    # An old line path with deduplicated ends is deliberately not guessed.
    write_kpoints(tmp_path, "old\n3\nLine-Mode\nReciprocal\n0 0 0 G\n.5 0 0 X\n")
    with pytest.raises(ValueError, match="allow_legacy=True"):
        VASPEigenvalParser(str(path)).parse()
    with pytest.warns(UserWarning, match="legacy teaching"):
        data = VASPEigenvalParser(str(path), allow_legacy=True).parse()
    assert data.input_format == "legacy-teaching"
    assert data.nkpoints == 3
    assert data.kpoint_labels == []


@pytest.mark.parametrize("separator", ["\n", "\n\n", "\n# next segment\n"])
@pytest.mark.parametrize("coordinates", ["Reciprocal", "Fractional", "R"])
def test_line_mode_inclusive_endpoints_labels_and_optional_blanks(tmp_path, separator, coordinates):
    text = f"Path\n3 ! points per segment\nLine-Mode\n{coordinates}\n"
    text += "0 0 0 GAMMA\n.5 0 0 ! X" + separator + ".5 0 0 # X\n.5 .5 0 M\n"
    path = write_kpoints(tmp_path, text)
    parsed = VASPKpointsParser(str(path)).parse_path()
    assert parsed.labels == [(0, "Γ"), (2, "X"), (3, "X"), (5, "M")]
    assert parsed.segments == [(0, 2), (3, 5)]
    expected = [[0, 0, 0], [.25, 0, 0], [.5, 0, 0], [.5, 0, 0], [.5, .25, 0], [.5, .5, 0]]
    np.testing.assert_allclose(parsed.points, expected)
    eigenval = write_eigenval(tmp_path, expected)
    data = VASPEigenvalParser(str(eigenval)).parse()
    assert data.kpoint_labels == parsed.labels
    np.testing.assert_allclose(data.kdistances, [0, .25, .5, .5, .75, 1])


def test_explicit_list_comments_blanks_and_tetrahedra(tmp_path):
    path = write_kpoints(tmp_path, "Explicit\n3\nReciprocal\n0 0 0 1 Γ\n\n# comment\n.25 0 0 2\n.5 0 0 1 ! x\nTetrahedra\n0 1\n")
    parsed = VASPKpointsParser(str(path)).parse_path()
    assert parsed.labels == [(0, "Γ"), (2, "X")]
    assert parsed.points.shape == (3, 3)
    assert parsed.segments == []


@pytest.mark.parametrize("mode", ["Cartesian", "cartesian", "K", "k"])
def test_cartesian_coordinates_need_explicit_conversion(tmp_path, mode):
    path = write_kpoints(tmp_path, f"Path\n3\nL\n{mode}\n0 0 0 Γ\n.5 0 0 X\n")
    parser = VASPKpointsParser(str(path))
    assert parser.parse() == [(0, "Γ"), (2, "X")]
    # Nonorthogonal row-vector conversion, in VASP's Cartesian 2pi/a units.
    matrix = np.array([[2., 1., 0.], [0., 3., 0.], [0., 0., 1.]])
    expected = np.array([[0, 0, 0], [.5, .25, 0], [1, .5, 0]])
    eigenval = write_eigenval(tmp_path, expected)
    with pytest.raises(ValueError, match="cartesian_to_fractional"):
        VASPEigenvalParser(str(eigenval)).parse()
    data = VASPEigenvalParser(str(eigenval), cartesian_to_fractional=matrix).parse()
    assert data.kpoint_labels == [(0, "Γ"), (2, "X")]


@pytest.mark.parametrize("text, expected", [
    ("Path\n3\nL\nR\n0 0 0 G\n", "paired endpoints"),
    ("Path\n1\nL\nR\n0 0 0 G\n.5 0 0 X\n", "N >= 2"),
    ("Path\n3\nR\n0 0 0 1 G\n", "expected 3 explicit"),
    ("Path\n3\nL\nR\n0 nan 0 G\n.5 0 0 X\n", "finite k-point"),
])
def test_invalid_kpoints(tmp_path, text, expected):
    path = write_kpoints(tmp_path, text)
    with pytest.raises(ValueError, match=expected):
        VASPKpointsParser(str(path)).parse()


def test_kpoints_count_and_coordinate_mismatch_are_rejected(tmp_path):
    eigenval = write_eigenval(tmp_path)
    path = write_kpoints(tmp_path, "Path\n4\nL\nR\n0 0 0 G\n.5 0 0 X\n")
    with pytest.raises(ValueError, match="expands to 4 points but EIGENVAL has 3"):
        VASPEigenvalParser(str(eigenval)).parse()
    path.write_text("Path\n3\nL\nR\n0 0 0 G\n.6 0 0 X\n")
    with pytest.raises(ValueError, match="coordinate mismatch at k-point 2"):
        VASPEigenvalParser(str(eigenval)).parse()


def test_missing_optional_and_explicit_kpoints(tmp_path):
    path = write_eigenval(tmp_path)
    assert VASPEigenvalParser(str(path)).parse().kpoint_labels == []
    assert VASPKpointsParser(str(tmp_path / "missing")).parse() == []
    with pytest.raises(FileNotFoundError, match="KPOINTS file not found"):
        VASPEigenvalParser(str(path), str(tmp_path / "missing")).parse()


def test_automatic_mesh_has_no_path_labels(tmp_path):
    eigenval = write_eigenval(tmp_path)
    write_kpoints(tmp_path, "Mesh\n0\nGamma\n4 4 1\n")
    data = VASPEigenvalParser(str(eigenval)).parse()
    assert data.kpoint_labels == []
    assert data.segments == []


def test_disconnected_segments_do_not_add_jump_distance(tmp_path):
    points = [[0, 0, 0], [.25, 0, 0], [.5, 0, 0], [0, .5, 0], [.25, .5, 0], [.5, .5, 0]]
    eigenval = write_eigenval(tmp_path, points)
    write_kpoints(tmp_path, "Path\n3\nL\nR\n0 0 0 G\n.5 0 0 X\n0 .5 0 M\n.5 .5 0 K\n")
    data = VASPEigenvalParser(str(eigenval)).parse()
    assert data.segments == [(0, 2), (3, 5)]
    np.testing.assert_allclose(data.kdistances, [0, .25, .5, .5, .75, 1])


@pytest.mark.parametrize("second", [[0, 0, 0], [1, 0, 0]])
def test_repeated_or_reciprocal_equivalent_endpoint_is_direct(second):
    data = BandData(np.array([[0, 0, 0], second]), np.array([0., 1.]),
                    np.array([[-.5, 2], [-1, .8]]), 2, 2)
    gap = BandAnalyzer(data).get_band_gap()
    assert gap["gap"] == 1.3
    assert gap["vbm_k"] == 0 and gap["cbm_k"] == 1
    assert gap["direct"] is True


def test_spin_down_participates_in_gap_and_mass_band_selection(tmp_path, monkeypatch):
    data = VASPEigenvalParser(str(write_eigenval(tmp_path, ispin=2))).parse()
    # Fully occupied/empty spin channels establish an insulating filling.
    # The parser fixture separately verifies preservation of partial occupations.
    data.occupations[:] = [1, 0, 1, 0]
    analyzer = BandAnalyzer(data)
    gap = analyzer.get_band_gap()
    assert gap["gap"] == 1.3
    assert (gap["vbm_band"], gap["cbm_band"]) == (2, 3)
    calls = []
    monkeypatch.setattr(analyzer, "effective_mass", lambda band, point: calls.append((band, point)))
    analyzer.estimate_effective_masses_at_gap()
    assert [band for band, _ in calls] == [2, 3]


@pytest.mark.parametrize("name, nk, nb, ne", [("graphene/EIGENVAL", 58, 8, 8),
    ("mos2/EIGENVAL", 58, 12, 18), ("mos2_x4_EIGENVAL", 232, 12, 18)])
def test_teaching_examples_are_standard_and_explicit(name, nk, nb, ne):
    data = VASPEigenvalParser(str(ROOT / "data" / "example" / name)).parse()
    assert (data.nkpoints, data.nbands, data.nelect, data.ispin) == (nk, nb, ne, 1)
    assert data.input_format == "vasp"
    assert data.weights.sum() == pytest.approx(1)
    np.testing.assert_allclose(data.occupations.sum(axis=1), ne)
    if nk == 58:
        assert data.kpoint_labels == [(0, "Γ"), (19, "M"), (38, "K"), (57, "Γ")]
