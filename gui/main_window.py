from PyQt5.QtWidgets import (
    QMainWindow, QWidget, QHBoxLayout,
    QTabWidget, QFileDialog, QMessageBox
)
from PyQt5.QtCore import Qt, QTimer
from pathlib import Path

from .band_widget import BandStructureWidget
from .dos_widget import DosWidget
from .control_panel import ControlPanel
from .file_tree import FileTreeWidget
from .workers.parse_worker import ParseWorker
from .workers.dos_worker import DosWorker
from core.parser import VASPEigenvalParser
from core.band_analyzer import BandAnalyzer
from core.dos_analyzer import load_spectrum
from storage.database import Database
from storage.config_manager import ConfigManager
from utils.logger import setup_logger


class MainWindow(QMainWindow):
    """主窗口：整合文件树、能带图、DOS 图、控制面板、持久化、日志、后台线程"""

    def __init__(self):
        super().__init__()

        # --- 日志 ---
        self.logger = setup_logger("bandviz")
        self.logger.info("Application starting...")

        # --- 持久化 ---
        self.db = Database()
        self.config = ConfigManager()
        for diagnostic in self.config.diagnostics:
            self.logger.warning(f'Configuration recovery: {diagnostic}')
        self.logger.info("Database and config loaded")

        # --- 窗口设置 ---
        self.setWindowTitle("2D Material Band Structure Visualizer")
        geo = self.config.get('window_geometry', {})
        self.setGeometry(
            geo.get('x', 100), geo.get('y', 100),
            geo.get('width', 1400), geo.get('height', 900)
        )

        # --- 中心布局 ---
        central = QWidget()
        self.setCentralWidget(central)
        main_layout = QHBoxLayout(central)
        main_layout.setContentsMargins(4, 4, 4, 4)

        # 左侧：文件树
        self.file_tree = FileTreeWidget()
        self.file_tree.file_selected.connect(self._on_file_tree_selected)
        main_layout.addWidget(self.file_tree, stretch=1)

        # 中间：图表 Tab
        self.tab_widget = QTabWidget()
        self.band_widget = BandStructureWidget()
        self.tab_widget.addTab(self.band_widget, "Band Structure")
        self.dos_widget = DosWidget()
        self.tab_widget.addTab(self.dos_widget, "DOS")
        main_layout.addWidget(self.tab_widget, stretch=3)

        # 右侧：控制面板
        self.control_panel = ControlPanel()
        self.control_panel.set_main_window(self)
        main_layout.addWidget(self.control_panel, stretch=1)

        # 底部状态栏
        self.statusBar().showMessage("Ready")

        # --- 状态变量 ---
        self.current_data = None
        self.current_analyzer = None
        self.current_dos_data = None
        self.current_file = None
        self._worker = None
        self._workers = set()
        self._active_load_id = None
        self._dos_request = 0
        self._closing = False
        self._dos_timer = QTimer(self)
        self._dos_timer.setSingleShot(True)
        self._dos_timer.timeout.connect(self._start_dos_refresh)
        self._cleanup_timer = QTimer(self)
        self._cleanup_timer.timeout.connect(self._cleanup_workers)

        # --- 恢复上次配置 ---
        self._restore_config()

        # --- 初始化示例数据 ---
        self._init_example_data()

        self.logger.info("MainWindow initialized")

    def closeEvent(self, event):
        """关闭前保存窗口几何和配置"""
        self._closing = True
        self.cancel_tasks()
        if any(w.isRunning() for w in self._workers):
            event.ignore()
            self._cleanup_timer.start(50)
            return
        self._persist_config(self.config.set, 'window_geometry', {
            'x': self.x(), 'y': self.y(),
            'width': self.width(), 'height': self.height()
        })
        self.logger.info("Application closing")
        event.accept()

    # ------------------------------------------------------------------
    # 配置恢复
    # ------------------------------------------------------------------
    def _restore_config(self):
        """从 JSON 恢复用户上次的配置到控制面板"""
        emin, emax = self.config.get_energy_range()
        self.control_panel.spin_emin.setValue(emin)
        self.control_panel.spin_emax.setValue(emax)
        self.control_panel.spin_fermi.setValue(self.config.get('fermi_level', 0.0))
        self.control_panel.spin_dos_sigma.setValue(self.config.get('dos_sigma', 0.05))
        self.control_panel.chk_dos_total.setChecked(self.config.get('dos_show_total', True))
        self.control_panel.chk_dos_vb.setChecked(self.config.get('dos_show_vb', True))
        self.control_panel.chk_dos_cb.setChecked(self.config.get('dos_show_cb', True))

        # 恢复最近文件
        for path in self.config.get_recent_files():
            self.file_tree.add_recent_file(path)

    def _persist_config(self, method, *args):
        try:
            method(*args)
            return True
        except (OSError, ValueError) as exc:
            self.logger.warning(f"Could not save settings: {exc}")
            self.statusBar().showMessage('Settings could not be saved; previous file retained')
            return False

    def _save_config(self):
        self._persist_config(self.config.update, {
            'energy_range': {'min': self.control_panel.spin_emin.value(),
                             'max': self.control_panel.spin_emax.value()},
            'fermi_level': self.control_panel.spin_fermi.value(),
            'dos_sigma': self.control_panel.spin_dos_sigma.value(),
            'dos_show_total': self.control_panel.chk_dos_total.isChecked(),
            'dos_show_vb': self.control_panel.chk_dos_vb.isChecked(),
            'dos_show_cb': self.control_panel.chk_dos_cb.isChecked(),
        })

    # ------------------------------------------------------------------
    # 示例数据初始化
    # ------------------------------------------------------------------
    def _init_example_data(self):
        example_dir = Path('data/example')
        if example_dir.exists():
            self.file_tree.set_project_directory(str(example_dir))
            materials = self.file_tree.scan_for_materials(str(example_dir))
            for name, path in materials:
                if path not in self.file_tree.get_recent_files():
                    self.file_tree.add_recent_file(path)

    # ------------------------------------------------------------------
    # 文件加载
    # ------------------------------------------------------------------
    def _on_file_tree_selected(self, filepath: str):
        self._load_eigenval(filepath)

    def load_file(self):
        start_dir = self.config.get('last_open_dir', '.')
        filepath, _ = QFileDialog.getOpenFileName(
            self, "Open EIGENVAL", start_dir,
            "EIGENVAL (EIGENVAL);;All Files (*)"
        )
        if filepath:
            self._persist_config(self.config.set, 'last_open_dir', str(Path(filepath).parent))
            self._load_eigenval(filepath)

    def _load_eigenval(self, filepath: str):
        """启动后台线程解析 EIGENVAL，避免阻塞 GUI"""
        file_path = Path(filepath)
        if self._closing:
            return
        kpoints_path = file_path.parent / "KPOINTS"

        if not file_path.exists():
            QMessageBox.critical(self, "Error", f"File not found:\n{filepath}")
            return

        self.cancel_tasks()
        self._set_parse_busy(True)
        self.control_panel.set_progress(True, "Starting parse...")
        self.statusBar().showMessage(f"Loading: {file_path.name}")
        self.logger.info(f"Loading file: {filepath}")

        emin, emax = self.control_panel.get_energy_range()
        self._worker = ParseWorker(
            eigenval_path=str(file_path),
            kpoints_path=str(kpoints_path) if kpoints_path.exists() else None,
            efermi=self.control_panel.spin_fermi.value(),
            emin=emin, emax=emax,
            dos_sigma=self.control_panel.spin_dos_sigma.value(),
            parent=self
        )
        worker = self._worker
        self._active_load_id = worker.task_id
        self._workers.add(worker)
        worker.progress.connect(lambda message: self._on_worker_progress(message) if self._accepts_import(worker) else None)
        worker.finished.connect(lambda *values: self._accept_import(worker, *values))
        worker.error.connect(lambda message: self._on_worker_error(message) if self._accepts_import(worker) else None)
        worker.cancelled.connect(lambda: self._on_import_cancelled(worker))
        self._worker.start()
        self._cleanup_timer.start(50)

    def _accepts_import(self, worker):
        return not self._closing and worker.task_id == self._active_load_id

    def _accept_import(self, worker, *values):
        if not self._accepts_import(worker):
            return
        if worker._cancel_requested.is_set():
            self._on_import_cancelled(worker)
            return
        self.current_file = Path(worker.eigenval_path)
        self._dos_request += 1
        self._on_worker_finished(*values)
        self._set_parse_busy(False)

    def _on_import_cancelled(self, worker):
        if self._accepts_import(worker):
            self.control_panel.set_progress(False, 'Cancelled')
            self._set_parse_busy(False)

    def _set_parse_busy(self, busy):
        for widget in (self.control_panel.spin_fermi, self.control_panel.spin_emin,
                       self.control_panel.spin_emax, self.control_panel.spin_dos_sigma):
            widget.setEnabled(not busy)
        if not busy and self.current_dos_data is not None:
            self.control_panel.spin_dos_sigma.setEnabled(self.current_dos_data.source == 'eigenvalues')

    def cancel_tasks(self):
        self._dos_timer.stop()
        self._dos_request += 1
        for worker in self._workers:
            worker.cancel()
        self.control_panel.btn_cancel.setEnabled(False)
        if self._workers:
            self.control_panel.label_status.setText('Cancelling…')

    def _cleanup_workers(self):
        for worker in tuple(self._workers):
            if not worker.isRunning():
                # Give queued result/error notifications an event-loop turn
                # before deleting their sender QObject.
                if not getattr(worker, '_retired', False):
                    worker._retired = True
                    continue
                self._workers.remove(worker)
                if self._worker is worker:
                    self._worker = None
                worker.deleteLater()
        if not self._workers:
            self._cleanup_timer.stop()
            if self._closing:
                self.close()
            elif not self._dos_timer.isActive():
                self.control_panel.set_progress(False, self.control_panel.label_status.text())

    def _on_worker_progress(self, message: str):
        self.control_panel.set_progress(True, message)
        self.statusBar().showMessage(message)

    def _on_worker_finished(self, band_data, analyzer, dos_data):
        self.current_data = band_data
        self.current_analyzer = analyzer
        self.current_dos_data = dos_data

        # 更新能带图
        self.band_widget.set_data(band_data, analyzer)

        # 更新 DOS
        self.dos_widget.set_data(dos_data)
        self.tab_widget.setTabText(1, "DOS" if dos_data.scope == 'brillouin-zone' else "K-point spectrum")
        imported = dos_data.source != 'eigenvalues'
        self.control_panel.spin_dos_sigma.setEnabled(not imported)
        self.control_panel.spin_dos_sigma.setToolTip('DOSCAR already contains the VASP DOS' if imported else '')

        # 更新分析面板
        gap_info = analyzer.get_band_gap()
        self.control_panel.update_analysis(gap_info, analyzer.initial_mass_reports)

        # 更新文件树
        self.file_tree.add_recent_file(str(self.current_file))

        # 保存到数据库
        self._save_to_database()

        # 保存配置
        self._persist_config(self.config.add_recent_file, str(self.current_file))
        self._save_config()

        # 恢复 UI
        self.control_panel.set_progress(False, "Done")
        self.statusBar().showMessage(f"Loaded: {self.current_file.name}")
        self.setWindowTitle(f"Band Viz - {self.current_file.name}")
        self.logger.info(f"Successfully loaded: {self.current_file.name}")


    def _on_worker_error(self, message: str):
        self.control_panel.set_progress(False, "Error")
        self.statusBar().showMessage("Error loading file")
        QMessageBox.critical(self, "Error", f"Failed to parse file:\n{message}")
        self.logger.error(f"Parse error: {message}")
        self._set_parse_busy(False)

    def _save_to_database(self):
        """将当前计算结果保存到 SQLite"""
        if not self.current_data or not self.current_analyzer:
            return

        try:
            gap_info = self.current_analyzer.get_band_gap()
            self.db.add_task(
                name=self.current_file.stem,
                path=str(self.current_file.resolve()),
                system=self.current_file.parent.name,
                band_gap=gap_info.get('gap'),
                is_direct=gap_info.get('direct'),
                vbm=gap_info.get('vbm'),
                cbm=gap_info.get('cbm'),
                num_bands=self.current_data.num_bands,
                num_electrons=self.current_data.num_electrons,
                num_kpoints=len(self.current_data.kpoints),
                kpoint_labels=self.current_data.kpoint_labels,
            )
            self.db.add_history(
                task_id=None,
                action="load",
                details=str(self.current_file)
            )
        except Exception as e:
            self.logger.warning(f"Failed to save to database: {e}")

    # ------------------------------------------------------------------
    # 控制面板回调
    # ------------------------------------------------------------------
    def set_fermi_level(self, efermi: float):
        self.band_widget.set_fermi_level(efermi)
        if self.current_analyzer:
            self.current_analyzer.set_fermi_level(efermi)
            gap_info = self.current_analyzer.get_band_gap()
            self.control_panel.update_analysis(gap_info, self.current_analyzer.mass_fit_reports_at_gap())
        if self.current_dos_data is not None:
            # 重新计算 DOS（费米能级改变）
            self._update_dos()
        self._persist_config(self.config.set, 'fermi_level', efermi)

    def set_energy_range(self, emin: float, emax: float):
        if emin >= emax:
            self.statusBar().showMessage('Energy minimum must be below maximum')
            return
        self.band_widget.set_energy_range(emin, emax)
        self.dos_widget.set_energy_range(emin, emax)
        self._persist_config(self.config.set_energy_range, emin, emax)
        self._update_dos()

    def set_dos_sigma(self, sigma: float):
        self._update_dos(sigma=sigma)
        self._persist_config(self.config.set, 'dos_sigma', sigma)

    def set_dos_visibility(self, total: bool, vb: bool, cb: bool):
        self.dos_widget.set_visibility(total=total, vb=vb, cb=cb)
        self._persist_config(self.config.set, 'dos_show_total', total)
        self._persist_config(self.config.set, 'dos_show_vb', vb)
        self._persist_config(self.config.set, 'dos_show_cb', cb)

    def _update_dos(self, sigma: float = None):
        if self.current_data is None or self._closing:
            return
        self._dos_request += 1
        for worker in self._workers:
            if isinstance(worker, DosWorker):
                worker.cancel()
        self._dos_timer.start(180)

    def _start_dos_refresh(self):
        if self.current_data is None or self._closing:
            return

        emin, emax = self.control_panel.get_energy_range()
        worker = DosWorker(self.current_data, self.current_file, self._dos_request,
            fermi_level=self.control_panel.spin_fermi.value(),
            state_capacity=self.current_analyzer.state_capacity,
            energy_range=(emin, emax),
            sigma=self.control_panel.spin_dos_sigma.value(), parent=self)
        self._workers.add(worker)
        worker.completed.connect(lambda result: self._accept_dos(worker, result))
        worker.error.connect(lambda message: self._on_dos_error(worker, message))
        self.control_panel.set_progress(True, 'Updating spectrum…')
        worker.start()
        self._cleanup_timer.start(50)

    def _accept_dos(self, worker, result):
        if not self._closing and worker.request_id == self._dos_request and worker.data is self.current_data:
            self.current_dos_data = result
            self.dos_widget.set_data(result)
            self.control_panel.set_progress(False, 'Spectrum updated')

    def _on_dos_error(self, worker, message):
        if not self._closing and worker.request_id == self._dos_request:
            self.control_panel.set_progress(False, message)
            self.logger.warning('Spectrum refresh failed: %s', message)

    def export_figure(self, fmt: str):
        if not self.current_file:
            QMessageBox.warning(self, "Warning", "No data loaded!")
            return

        current_tab = self.tab_widget.currentIndex()
        if current_tab == 0:
            default_name = self.current_file.stem + f"_band.{fmt}"
            widget = self.band_widget
        else:
            default_name = self.current_file.stem + f"_dos.{fmt}"
            widget = self.dos_widget

        filepath, _ = QFileDialog.getSaveFileName(
            self, f"Save {fmt.upper()}", default_name,
            f"{fmt.upper()} (*.{fmt})"
        )
        if filepath:
            dpi = 300 if fmt == 'png' else 150
            widget.save_figure(filepath, dpi)
            self.statusBar().showMessage(f"Exported: {filepath}")
            self.logger.info(f"Exported figure: {filepath}")
