"""Task identity, isolated runtime inputs and consistent strained exports."""
from dataclasses import FrozenInstanceError
import json
import os
from threading import Event

os.environ.setdefault('QT_QPA_PLATFORM', 'offscreen')

import numpy as np
import pytest

from vdw_studio.engine.strain import apply_strain
from vdw_studio.gui.workers import SimulationWorker
from vdw_studio.presets import get_preset
from vdw_studio.task_state import SimulationSnapshot, align_structure


def test_snapshot_values_are_immutable_and_runtime_does_not_alias_inputs():
    p = get_preset('graphene')
    original = p.make_model()
    base = p.make_structure(p.structure_key)
    model = apply_strain(original, .02, .02)
    snapshot = SimulationSnapshot.capture(p, model, base, strain=(.02, .02),
        n_per_segment=10, dos_mesh=(6, 6), gap_mesh=(8, 8))
    worker = SimulationWorker(model, snapshot=snapshot)
    model.onsite[:] = 99
    model.lattice.matrix[:] = np.eye(3)
    base.atoms[0].frac[:] = 1
    with pytest.raises(FrozenInstanceError):
        snapshot.material_key = 'other'
    with pytest.raises(TypeError):
        snapshot.lattice_rows[0][0] = 99
    result = worker.compute()
    assert result['task_id'] == snapshot.task_id
    assert result['gap'].status == 'zero-gap'
    np.testing.assert_allclose(result['lattice'].matrix, snapshot.lattice_rows)
    assert result['structure'].lattice.parameters()[0] == pytest.approx(2.5092)
    assert worker.model.onsite[0] == 0
    assert snapshot.atoms[0][1] != (1., 1., 1.)
    state = snapshot.to_dict()['model_state']['state']
    assert state['onsite'] == [0., 0.]
    assert 'hoppings' in state
    json.dumps(snapshot.to_dict(), allow_nan=False)


def test_structure_alignment_preserves_base_and_slab_heights():
    p = get_preset('bilayer_graphene')
    base = p.make_structure(p.structure_key)
    original = base.cart_coords.copy()
    model = apply_strain(p.make_model(), .02, -.01)
    aligned = align_structure(base, model)
    np.testing.assert_allclose(aligned.cart_coords[:, :2], original[:, :2] * [1.02, .99], atol=1e-12)
    np.testing.assert_allclose(aligned.cart_coords[:, 2], original[:, 2])
    np.testing.assert_allclose(base.cart_coords, original)


def test_local_valley_snapshot_records_actual_local_endpoints():
    from vdw_studio.simulation import simulate, KP_QMAX
    p = get_preset('mos2_kp')
    model, st = p.make_model(), p.make_structure(p.structure_key)
    snapshot = SimulationSnapshot.capture(p, model, st, n_per_segment=10, gap_mesh=(8, 8))
    result = simulate(model, n_per_segment=10, lattice=st.lattice)
    points = np.array([point for _, point in snapshot.path[1]])
    cart = points @ st.lattice.reciprocal_matrix[:2]
    np.testing.assert_allclose(cart, result['band'].kpoints[[0, 10, -1]], atol=1e-12)
    assert np.linalg.norm(cart[0] - cart[1]) == pytest.approx(KP_QMAX)


def test_snapshot_rejects_mismatched_runtime_model():
    p = get_preset('graphene')
    model, structure = p.make_model(), p.make_structure(p.structure_key)
    snapshot = SimulationSnapshot.capture(p, model, structure)
    model.onsite[0] = 1.
    with pytest.raises(ValueError, match='does not match'):
        SimulationWorker(model, snapshot=snapshot)


@pytest.fixture(scope='module')
def app():
    from PyQt5.QtWidgets import QApplication
    return QApplication.instance() or QApplication([])


def wait_until(app, predicate, milliseconds=15000):
    from PyQt5.QtCore import QEventLoop, QTimer
    loop, poll, timeout = QEventLoop(), QTimer(), QTimer()
    timeout.setSingleShot(True)
    timeout.timeout.connect(loop.quit)
    poll.timeout.connect(lambda: loop.quit() if predicate() else None)
    poll.start(20)
    timeout.start(milliseconds)
    loop.exec_()
    poll.stop()
    timeout.stop()
    assert predicate(), 'Timed out waiting for GUI state'


def test_strained_geometry_tree_and_export_follow_exact_model_without_accumulation(app, tmp_path, monkeypatch):
    from PyQt5.QtWidgets import QFileDialog
    from core.parser import VASPLatticeParser
    from vdw_studio.gui.main_window import MainWindow
    win = MainWindow()
    try:
        win.set_material('graphene')
        win.strain_x_spin.setValue(2.)
        win.strain_y_spin.setValue(2.)
        win._rebuild_model()
        win._rebuild_model()
        np.testing.assert_allclose(win.structure.lattice.matrix, win.model.lattice.matrix)
        assert win.structure.lattice.parameters()[0] == pytest.approx(2.5092)
        rows = {win.tree.topLevelItem(0).child(i).text(0): win.tree.topLevelItem(0).child(i).text(1)
                for i in range(win.tree.topLevelItem(0).childCount())}
        assert float(rows['a (Å)']) == pytest.approx(2.5092)
        file = tmp_path / 'strained.POSCAR'
        monkeypatch.setattr(QFileDialog, 'getSaveFileName', lambda *args, **kwargs: (str(file), ''))
        win.export_structure()
        lattice, _ = VASPLatticeParser(file).parse()
        np.testing.assert_allclose(lattice, win.model.lattice.matrix, atol=1e-9)
        win.strain_x_spin.setValue(0.)
        win.strain_y_spin.setValue(0.)
        assert win.structure.lattice.parameters()[0] == pytest.approx(2.46)
    finally:
        win.close()


def test_material_switch_while_running_cannot_mix_task_and_results(app, monkeypatch):
    from PyQt5.QtWidgets import QMessageBox
    from vdw_studio.gui.main_window import MainWindow
    gate, started = Event(), Event()
    compute = SimulationWorker.compute
    def held_compute(self):
        started.set()
        if not gate.wait(10):
            raise RuntimeError('test gate timed out')
        return compute(self)
    monkeypatch.setattr(SimulationWorker, 'compute', held_compute)
    errors = []
    monkeypatch.setattr(QMessageBox, 'critical', lambda *args: errors.append(args[-1]))
    win = MainWindow()
    try:
        win.set_material('graphene')
        win.kpoint_spin.setValue(10)
        win.mesh_spin.setValue(6)
        win.run_button.click()
        wait_until(app, started.is_set)
        snapshot = win._active_snapshot
        assert not win.material_combo.isEnabled() and not win.strain_x_spin.isEnabled()
        win.set_material('hbn')
        # Programmatic combo changes bypass disabled controls, so guard the callback too.
        win.material_combo.setCurrentIndex(win.material_combo.findData('hbn'))
        assert win.preset.key == 'graphene' and win.material_combo.currentData() == 'graphene'
        win.model.onsite[:] = 8  # worker owns an independent model
        gate.set()
        wait_until(app, lambda: bool(win._results) or bool(errors))
        assert not errors
        assert win._results['snapshot'] is snapshot
        assert win._results['gap'].status == 'zero-gap'
        assert snapshot.material_key == 'graphene'
        stale = dict(win._results)
        win.set_material('hbn')
        assert win._results == {}
        win._on_results(stale)
        assert win._results == {} and win.preset.key == 'hbn'
        assert '过期' in win.log_box.toPlainText()
    finally:
        gate.set()
        if win.worker is not None:
            win.worker.wait(10000)
        win.close()
