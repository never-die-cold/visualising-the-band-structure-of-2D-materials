from PyQt5.QtWidgets import (
    QMainWindow, QWidget, QHBoxLayout,
    QTabWidget, QFileDialog, QMessageBox
)
from PyQt5.QtCore import Qt
from pathlib import Path

from .band_widget import BandStructureWidget
from .dos_widget import DosWidget
from .control_panel import ControlPanel
from .file_tree import FileTreeWidget
from .workers.parse_worker import ParseWorker
from core.parser import VASPEigenvalParser
from core.band_analyzer import BandAnalyzer
from core.dos_analyzer import DosAnalyzer
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
        self.db = Database("data/bandviz.db")
        self.config = ConfigManager("config/settings.json")
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

        # --- 恢复上次配置 ---
        self._restore_config()

        # --- 初始化示例数据 ---
        self._init_example_data()

        self.logger.info("MainWindow initialized")

    def closeEvent(self, event):
        """关闭前保存窗口几何和配置"""
        self.config.set('window_geometry', {
            'x': self.x(), 'y': self.y(),
            'width': self.width(), 'height': self.height()
        })
        self.logger.info("Application closing, config saved")
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

    def _save_config(self):
        """将当前控制面板状态保存到 JSON"""
        self.config.set_energy_range(
            self.control_panel.spin_emin.value(),
            self.control_panel.spin_emax.value()
        )
        self.config.set('fermi_level', self.control_panel.spin_fermi.value())
        self.config.set('dos_sigma', self.control_panel.spin_dos_sigma.value())
        self.config.set('dos_show_total', self.control_panel.chk_dos_total.isChecked())
        self.config.set('dos_show_vb', self.control_panel.chk_dos_vb.isChecked())
        self.config.set('dos_show_cb', self.control_panel.chk_dos_cb.isChecked())

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
            self.config.set('last_open_dir', str(Path(filepath).parent))
            self._load_eigenval(filepath)

    def _load_eigenval(self, filepath: str):
        """启动后台线程解析 EIGENVAL，避免阻塞 GUI"""
        file_path = Path(filepath)
        kpoints_path = file_path.parent / "KPOINTS"

        if not file_path.exists():
            QMessageBox.critical(self, "Error", f"File not found:\n{filepath}")
            return

        # 如果有正在运行的 worker，先终止
        if self._worker is not None and self._worker.isRunning():
            self._worker.terminate()
            self._worker.wait()

        self.current_file = file_path
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
        self._worker.progress.connect(self._on_worker_progress)
        self._worker.finished.connect(self._on_worker_finished)
        self._worker.error.connect(self._on_worker_error)
        self._worker.start()

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

        # 更新分析面板
        gap_info = analyzer.get_band_gap()
        self.control_panel.update_analysis(gap_info)

        # 更新文件树
        self.file_tree.add_recent_file(str(self.current_file))

        # 保存到数据库
        self._save_to_database()

        # 保存配置
        self.config.add_recent_file(str(self.current_file))
        self._save_config()

        # 恢复 UI
        self.control_panel.set_progress(False, "Done")
        self.statusBar().showMessage(f"Loaded: {self.current_file.name}")
        self.setWindowTitle(f"Band Viz - {self.current_file.name}")
        self.logger.info(f"Successfully loaded: {self.current_file.name}")

        self._worker = None

    def _on_worker_error(self, message: str):
        self.control_panel.set_progress(False, "Error")
        self.statusBar().showMessage("Error loading file")
        QMessageBox.critical(self, "Error", f"Failed to parse file:\n{message}")
        self.logger.error(f"Parse error: {message}")
        self._worker = None

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
            self.control_panel.update_analysis(gap_info)
        if self.current_dos_data is not None:
            # 重新计算 DOS（费米能级改变）
            self._update_dos()
        self.config.set('fermi_level', efermi)

    def set_energy_range(self, emin: float, emax: float):
        self.band_widget.set_energy_range(emin, emax)
        self.dos_widget.set_energy_range(emin, emax)
        self.config.set_energy_range(emin, emax)

    def set_dos_sigma(self, sigma: float):
        self._update_dos(sigma=sigma)
        self.config.set('dos_sigma', sigma)

    def set_dos_visibility(self, total: bool, vb: bool, cb: bool):
        self.dos_widget.set_visibility(total=total, vb=vb, cb=cb)
        self.config.set('dos_show_total', total)
        self.config.set('dos_show_vb', vb)
        self.config.set('dos_show_cb', cb)

    def _update_dos(self, sigma: float = None):
        if self.current_data is None:
            return
        if sigma is None:
            sigma = self.control_panel.spin_dos_sigma.value()

        emin, emax = self.control_panel.get_energy_range()
        dos_analyzer = DosAnalyzer(
            self.current_data.energies,
            fermi_level=self.control_panel.spin_fermi.value()
        )
        dos_data = dos_analyzer.calculate_dos(
            energy_range=(emin, emax),
            num_points=800,
            sigma=sigma
        )
        self.current_dos_data = dos_data
        self.dos_widget.set_data(dos_data)

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
