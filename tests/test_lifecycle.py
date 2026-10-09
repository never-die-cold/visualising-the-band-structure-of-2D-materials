"""Real event loops verify cancellation, close deferral and stale-result guards."""
import os
from threading import Event, get_ident

os.environ.setdefault('QT_QPA_PLATFORM', 'offscreen')

import numpy as np
import pytest
from PyQt5.QtWidgets import QApplication, QMessageBox

from core.dos_analyzer import DosAnalyzer
from core.parser import VASPEigenvalParser
from gui.workers.parse_worker import ParseWorker
from vdw_studio.gui.workers import SimulationWorker
from vdw_studio.engine.solver import solve_dos
from vdw_studio.analysis.properties import analyze_gap
from vdw_studio.analysis.berry import valley_report
from vdw_studio.engine.models import HoneycombModel
from test_task_state import wait_until


@pytest.fixture(scope='module')
def app():
    app = QApplication.instance() or QApplication([])
    previous = app.quitOnLastWindowClosed()
    app.setQuitOnLastWindowClosed(False)
    yield app
    app.setQuitOnLastWindowClosed(previous)


def cancel_after(count):
    state = {'calls': 0}
    def check():
        state['calls'] += 1
        if state['calls'] >= count:
            raise InterruptedError('cancel checkpoint')
    return check, state


def test_numerical_work_cancels_inside_sampling_optimization_and_broadening(tmp_path):
    from test_bandviz_mass import write_eigenval
    points = [[i / 100, 0, 0] for i in range(80)]
    file = write_eigenval(tmp_path, points)
    check, state = cancel_after(4)
    with pytest.raises(InterruptedError):
        VASPEigenvalParser(file, cancel_check=check).parse()
    assert state['calls'] == 4
    check, _ = cancel_after(3)
    with pytest.raises(InterruptedError):
        DosAnalyzer(np.zeros((100, 4)), max_workspace_mb=.01,
                    cancel_check=check).calculate_dos()
    model = HoneycombModel(2.46)
    check, _ = cancel_after(3)
    with pytest.raises(InterruptedError):
        solve_dos(model, mesh=(16, 16), cancel_check=check)
    check, _ = cancel_after(10)
    with pytest.raises(InterruptedError):
        analyze_gap(model, mesh=(8, 8), cancel_check=check)
    check, _ = cancel_after(2)
    with pytest.raises(InterruptedError):
        valley_report(lambda q: np.diag([-1., 1.]), lambda q: np.diag([-1., 1.]),
                      0, 1, n_grid=9, cancel_check=check)


@pytest.mark.parametrize('close', [False, True])
def test_vdw_cancel_and_close_wait_for_thread_without_blocking_ui(app, monkeypatch, close):
    from vdw_studio.gui.main_window import MainWindow
    gate, started = Event(), Event()
    compute = SimulationWorker.compute
    def held(self):
        started.set()
        if not gate.wait(10):
            raise RuntimeError('gate timeout')
        return compute(self)
    monkeypatch.setattr(SimulationWorker, 'compute', held)
    errors = []
    monkeypatch.setattr(QMessageBox, 'critical', lambda *args: errors.append(args[-1]))
    win = MainWindow()
    win.show()
    try:
        win.set_material('graphene')
        win.kpoint_spin.setValue(10)
        win.mesh_spin.setValue(6)
        win.run_button.click()
        wait_until(app, started.is_set)
        worker = win.worker
        if close:
            win.close()
            assert win.isVisible() and worker.isRunning()
            assert win._closing
        else:
            win.cancel_button.click()
            assert not win.cancel_button.isEnabled()
        # The event loop remains able to run timers while the worker has not stopped.
        from PyQt5.QtCore import QTimer
        ticks = []
        QTimer.singleShot(10, lambda: ticks.append(True))
        wait_until(app, lambda: bool(ticks))
        gate.set()
        wait_until(app, lambda: worker.done and not worker.isRunning() and
                   (not win.isVisible() if close else win.run_button.isEnabled()))
        assert not errors and worker.cancelled and not win._results
        assert not win._poll_timer.isActive()
    finally:
        gate.set()
        if win.worker is not None:
            win.worker.wait(10000)
        win.close()


def root_window(tmp_path, monkeypatch):
    import gui.main_window as module
    from storage.config_manager import ConfigManager
    from storage.database import Database
    monkeypatch.setattr(module, 'ConfigManager', lambda *args: ConfigManager(str(tmp_path / 'settings.json')))
    monkeypatch.setattr(module, 'Database', lambda *args: Database(str(tmp_path / 'test.db')))
    monkeypatch.setattr(module.MainWindow, '_init_example_data', lambda self: None)
    return module.MainWindow()


def test_bandviz_close_defers_destruction_until_import_thread_exits(app, tmp_path, monkeypatch):
    from test_bandviz_mass import write_eigenval
    file = write_eigenval(tmp_path, [[0, 0, 0], [.25, 0, 0]])
    gate, started = Event(), Event()
    run = ParseWorker.run
    def held(self):
        started.set()
        gate.wait(10)
        run(self)
    monkeypatch.setattr(ParseWorker, 'run', held)
    errors = []
    monkeypatch.setattr(QMessageBox, 'critical', lambda *args: errors.append(args[-1]))
    win = root_window(tmp_path, monkeypatch)
    win.show()
    try:
        win._load_eigenval(str(file))
        wait_until(app, started.is_set)
        worker = win._worker
        win.close()
        assert win.isVisible() and worker.isRunning()
        gate.set()
        wait_until(app, lambda: not win.isVisible() and not win._workers)
        assert win.current_data is None and not errors
        assert not win._cleanup_timer.isActive()
    finally:
        gate.set()
        for worker in tuple(win._workers):
            worker.cancel()
            worker.wait(10000)
        win.close()


def test_continuous_imports_do_not_overwrite_latest_file(app, tmp_path, monkeypatch):
    from test_bandviz_mass import write_eigenval
    first_dir, second_dir = tmp_path / 'first', tmp_path / 'second'
    first_dir.mkdir()
    second_dir.mkdir()
    first = write_eigenval(first_dir, [[0, 0, 0]])
    second = write_eigenval(second_dir, [[0, 0, 0], [.25, 0, 0]])
    gate, started = Event(), Event()
    run = ParseWorker.run
    def held(self):
        if self.eigenval_path == str(first):
            started.set()
            gate.wait(10)
        run(self)
    monkeypatch.setattr(ParseWorker, 'run', held)
    win = root_window(tmp_path, monkeypatch)
    try:
        win._load_eigenval(str(first))
        wait_until(app, started.is_set)
        win._load_eigenval(str(second))
        wait_until(app, lambda: win.current_file == second)
        gate.set()
        wait_until(app, lambda: not win._workers)
        assert win.current_file == second and win.current_data.nkpoints == 2
        assert win.control_panel.btn_load.isEnabled()
    finally:
        gate.set()
        for worker in tuple(win._workers):
            worker.cancel()
            worker.wait(10000)
        win.close()


def test_dos_refresh_debounces_runs_off_main_thread_and_rejects_stale_result(app, tmp_path, monkeypatch):
    from test_bandviz_mass import write_eigenval
    import gui.workers.dos_worker as module
    original = module.load_spectrum
    file = write_eigenval(tmp_path, [[0, 0, 0], [.25, 0, 0]])
    win = root_window(tmp_path, monkeypatch)
    gate, started = Event(), Event()
    threads, sigmas = [], []
    def tracked(*args, **kwargs):
        threads.append(get_ident())
        sigmas.append(kwargs['sigma'])
        if kwargs['sigma'] == .12:
            started.set()
            gate.wait(10)
        return original(*args, **kwargs)
    monkeypatch.setattr(module, 'load_spectrum', tracked)
    try:
        win._load_eigenval(str(file))
        wait_until(app, lambda: win.current_data is not None and not win._workers)
        for value in (.06, .07, .08):
            win.control_panel.spin_dos_sigma.setValue(value)
        wait_until(app, lambda: win.current_dos_data.sigma == .08 and not win._workers)
        assert sigmas == [.08] and threads[0] != get_ident()
        win.control_panel.spin_dos_sigma.setValue(.12)
        wait_until(app, started.is_set)
        win.control_panel.spin_dos_sigma.setValue(.16)
        wait_until(app, lambda: win.current_dos_data.sigma == .16)
        gate.set()
        wait_until(app, lambda: not win._workers)
        assert win.current_dos_data.sigma == .16
        win.control_panel.spin_emin.setValue(-7.)
        wait_until(app, lambda: win.current_dos_data.energy_range[0] == -7.)
    finally:
        gate.set()
        for worker in tuple(win._workers):
            worker.cancel()
            worker.wait(10000)
        win.close()


def test_background_failure_restores_vdw_controls(app, monkeypatch):
    from vdw_studio.gui.main_window import MainWindow
    def fail(self):
        raise ValueError('intentional calculation failure')
    monkeypatch.setattr(SimulationWorker, 'compute', fail)
    messages = []
    monkeypatch.setattr(QMessageBox, 'critical', lambda *args: messages.append(args[-1]))
    win = MainWindow()
    try:
        win.run_button.click()
        wait_until(app, lambda: bool(messages))
        assert 'intentional' in messages[0]
        assert win.run_button.isEnabled() and win.material_combo.isEnabled()
        assert not win._poll_timer.isActive()
    finally:
        win.close()


def test_bandviz_bad_import_restores_controls_without_changing_loaded_file(app, tmp_path, monkeypatch):
    from test_bandviz_mass import write_eigenval
    valid = write_eigenval(tmp_path, [[0, 0, 0], [.25, 0, 0]])
    invalid = tmp_path / 'bad-EIGENVAL'
    invalid.write_text('truncated input\n')
    messages = []
    monkeypatch.setattr(QMessageBox, 'critical', lambda *args: messages.append(args[-1]))
    win = root_window(tmp_path, monkeypatch)
    try:
        win._load_eigenval(str(valid))
        wait_until(app, lambda: win.current_data is not None and not win._workers)
        win._load_eigenval(str(invalid))
        wait_until(app, lambda: bool(messages) and not win._workers)
        assert win.current_file == valid and win.current_data.nkpoints == 2
        assert win.control_panel.btn_load.isEnabled() and win.control_panel.spin_fermi.isEnabled()
        assert 'header is incomplete' in messages[0]
    finally:
        for worker in tuple(win._workers):
            worker.cancel()
            worker.wait(10000)
        win.close()
