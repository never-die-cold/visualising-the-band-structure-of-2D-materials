"""Real BandViz worker completion and rendered spin/segment regressions."""

import os
from pathlib import Path

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import numpy as np
import pytest

pytest.importorskip("PyQt5.QtWidgets")

from PyQt5.QtCore import QEventLoop, QTimer
from PyQt5.QtWidgets import QApplication

from core.parser import BandData
from core.band_analyzer import BandAnalyzer
from gui.band_widget import BandStructureWidget
from gui.control_panel import ControlPanel
from gui.dos_widget import DosWidget
from gui.workers.parse_worker import ParseWorker


@pytest.fixture(scope="module")
def app():
    return QApplication.instance() or QApplication([])


@pytest.mark.parametrize('doscar', [False, True])
def test_worker_prefers_doscar_and_distinguishes_sampled_spectrum(app, tmp_path, doscar):
    from test_bandviz_mass import write_eigenval
    from test_dos_contract import write_doscar
    file = write_eigenval(tmp_path, [[0, 0, 0], [.25, 0, 0]])
    if doscar:
        write_doscar(tmp_path, True)
    worker = ParseWorker(str(file))
    result, errors = [], []
    loop = QEventLoop()
    worker.finished.connect(lambda *values: (result.extend(values), loop.quit()))
    worker.error.connect(lambda message: (errors.append(message), loop.quit()))
    timer = QTimer()
    timer.setSingleShot(True)
    timer.timeout.connect(loop.quit)
    timer.start(15000)
    worker.start()
    loop.exec_()
    timer.stop()
    assert worker.wait(2000)
    assert not errors and len(result) == 3
    spectrum = result[2]
    widget = DosWidget()
    try:
        widget.set_data(spectrum)
        if doscar:
            assert spectrum.integral == pytest.approx(3.)
            assert spectrum.sigma == 0 and spectrum.scope == 'brillouin-zone'
            assert 'states/eV/cell' in widget.ax.get_xlabel()
            assert widget.ax.get_title() == 'Density of States'
        else:
            assert spectrum.expected_states == 4
            assert 'not BZ DOS' in widget.ax.get_title()
            assert 'Occupied states' in [text.get_text() for text in widget.ax.get_legend().get_texts()]
    finally:
        widget.close()


@pytest.mark.parametrize("calibrated", [True, False])
def test_physical_lattice_worker_and_visible_mass_diagnostics(app, tmp_path, calibrated):
    from test_bandviz_mass import write_poscar, write_eigenval, RAW
    x = np.linspace(-.12, .12, 13)
    cart = np.column_stack((x, x * 0, x * 0))
    frac = cart @ RAW.T / (2 * np.pi)
    energy = np.column_stack((-1 - 3 * x ** 2, 2 + 5 * x ** 2))
    file = write_eigenval(tmp_path, frac, energy)
    if calibrated:
        write_poscar(tmp_path)
        mode, coords = "Cartesian", cart / (2 * np.pi)
    else:
        mode, coords = "Reciprocal", frac
    (tmp_path / "KPOINTS").write_text("Path\n13\n" + mode + "\n" +
        "\n".join(" ".join(map(str, c)) + " 1" for c in coords) + "\n")
    worker = ParseWorker(str(file))
    result, errors = [], []
    loop = QEventLoop()
    worker.finished.connect(lambda *values: (result.extend(values), loop.quit()))
    worker.error.connect(lambda message: (errors.append(message), loop.quit()))
    timeout = QTimer()
    timeout.setSingleShot(True)
    timeout.timeout.connect(loop.quit)
    timeout.start(15000)
    worker.start()
    loop.exec_()
    timeout.stop()
    assert worker.wait(2000)
    assert not errors and len(result) == 3
    data, analyzer, _ = result
    panel, widget = ControlPanel(), BandStructureWidget()
    try:
        panel.update_analysis(analyzer.get_band_gap(), analyzer.initial_mass_reports)
        widget.set_data(data, analyzer)
        if calibrated:
            np.testing.assert_allclose(data.kdistances, x - x[0], atol=1e-14)
            assert 'Å' in widget.ax.get_xlabel()
            assert '0.762' in panel.label_cbm_mass.text()
            assert 'direction' in panel.label_cbm_mass.toolTip()
            assert 'Fit RMS' in panel.label_cbm_mass.toolTip()
            assert 'Samples [1, 2' in panel.label_cbm_mass.toolTip()
        else:
            assert 'lattice unavailable' in widget.ax.get_xlabel()
            assert 'unavailable' in panel.label_cbm_mass.text()
            assert 'Missing lattice' in panel.label_cbm_mass.toolTip()
    finally:
        panel.close()
        widget.close()


@pytest.mark.parametrize("name, spin, bands", [("EIGENVAL.nonspin", 1, 12), ("EIGENVAL.spin", 2, 840)])
def test_real_vasp_worker_completes_and_plots_all_channels(app, name, spin, bands):
    path = Path(__file__).parent / "fixtures" / "vasp" / name
    worker = ParseWorker(str(path), emin=-5, emax=5)
    loop = QEventLoop()
    result, errors = [], []
    def finished(*values):
        result.extend(values)
        loop.quit()
    def failed(message):
        errors.append(message)
        loop.quit()
    worker.finished.connect(finished)
    worker.error.connect(failed)
    timeout = QTimer()
    timeout.setSingleShot(True)
    timeout.timeout.connect(loop.quit)
    timeout.start(15000)
    worker.start()
    loop.exec_()
    timeout.stop()
    assert worker.wait(2000), "worker did not stop"
    assert errors == []
    assert len(result) == 3
    data, analyzer, dos = result
    assert data.ispin == spin
    assert data.energies.shape == (data.nkpoints, bands * spin)
    assert np.isfinite(dos.total_dos).all()
    widget = BandStructureWidget()
    try:
        widget.set_data(data, analyzer)
        curves = [line for line in widget.ax.lines if line.get_color() in ("blue", "darkorange")]
        assert len(curves) == bands * spin
        if data.nkpoints == 1:
            assert all(line.get_marker() == "." for line in curves)
            assert widget.ax.get_xlim()[0] < 0 < widget.ax.get_xlim()[1]
        if spin == 2:
            assert sum(line.get_color() == "darkorange" for line in curves) == bands
            assert {text.get_text() for text in widget.ax.get_legend().get_texts()} >= {"Spin up", "Spin down"}
    finally:
        widget.close()


@pytest.mark.parametrize("continuous", [False, True])
def test_disconnected_and_shared_segment_boundaries_render_separately(app, continuous):
    second = [.5, 0, 0] if continuous else [0, .5, 0]
    points = np.array([[0, 0, 0], [.25, 0, 0], [.5, 0, 0],
                       second, [.5, .25, 0], [.5, .5, 0]])
    data = BandData(points, np.array([0., .25, .5, .5, .75, 1.]),
                    np.tile([-1., 2., -.5, .8], (6, 1)), 4, 2,
                    [(0, "Γ"), (2, "X"), (3, "X" if continuous else "M"), (5, "K")],
                    ispin=2, segments=[(0, 2), (3, 5)])
    widget = BandStructureWidget()
    try:
        widget.set_data(data)
        curves = [line for line in widget.ax.lines if line.get_color() in ("blue", "darkorange")]
        assert len(curves) == 8
        assert all(len(line.get_xdata()) == 3 for line in curves)
        assert all(np.array_equal(line.get_xdata(), [0, .25, .5]) or
                   np.array_equal(line.get_xdata(), [.5, .75, 1]) for line in curves)
        labels = [label.get_text() for label in widget.ax.get_xticklabels()]
        assert labels == ["Γ", "X" if continuous else "X|M", "K"]
        np.testing.assert_allclose(widget.ax.get_xticks(), [0, .5, 1])
    finally:
        widget.close()


@pytest.mark.parametrize("energies, occupations, state, text", [
    ([[-1, .1], [.2, 1]], None, "metal", "metal"),
    ([[-1, 2], [0, 0]], [[2, 0], [2, 0]], "zero-gap", "zero gap"),
    ([[-1, 1], [-.8, 1.2]], [[.5, 0], [.5, 0]], "unknown", "undetermined"),
    ([[-1, 1], [-.8, 1.2]], [[2, 0], [2, 0]], "insulator", "Sampled Eg"),
])
def test_sampled_status_is_visible_without_false_indirect_annotation(app, energies, occupations, state, text):
    data = BandData(np.array([[0., 0, 0], [.5, 0, 0]]), np.array([0., .5]),
                    np.array(energies), 2, 2,
                    occupations=None if occupations is None else np.array(occupations))
    analyzer = BandAnalyzer(data)
    result = analyzer.get_band_gap()
    assert result["status"] == state
    widget, panel = BandStructureWidget(), ControlPanel()
    try:
        widget.set_data(data, analyzer)
        assert text in widget.ax.texts[0].get_text()
        if state != "insulator":
            assert "indirect" not in widget.ax.texts[0].get_text()
        panel.update_analysis(result)
        assert ("undetermined" if state == "unknown" else state) in panel.label_gap.text()
    finally:
        widget.close()
        panel.close()
