"""vdW Studio 主窗口。

布局仿主流仿真建模软件（Materials Studio / QuantumATK）：

┌──────────────────────────────────────────────────────────────┐
│ 菜单栏（文件 / 仿真 / 帮助）  工具栏（材料选择 · 运行 · 导出）   │
├──────────┬───────────────────────────────────┬───────────────┤
│ 工程树    │  标签页: 结构(3D) | 能带 | DOS |  │  参数面板      │
│ (QDock)  │  布里渊区                          │  (QDock)      │
├──────────┴───────────────────────────────────┴───────────────┤
│ 日志 (QDock)                        状态栏 + 进度条            │
└──────────────────────────────────────────────────────────────┘

以 ``python -m vdw_studio.gui`` 启动。
"""

from __future__ import annotations

from typing import Optional

import numpy as np
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg
from matplotlib.figure import Figure
from PyQt5.QtCore import QTimer, Qt
from PyQt5.QtGui import QFont
from PyQt5.QtWidgets import (
    QComboBox, QDockWidget, QDoubleSpinBox, QFileDialog, QFormLayout,
    QLabel, QMainWindow, QMessageBox, QPlainTextEdit, QProgressBar,
    QPushButton, QSpinBox, QStatusBar, QTabWidget, QTreeWidget,
    QTreeWidgetItem, QWidget,
)

from ..analysis.properties import effective_mass
from ..engine.solver import BandStructure, DOSResult
from ..io import write_poscar, write_xyz
from ..presets import PRESETS, get_preset, list_presets
from ..visualization import (
    plot_band_structure,
    plot_bz_path,
    plot_dos,
    plot_structure,
)
from .workers import SimulationWorker


class MplCanvas(FigureCanvasQTAgg):
    """matplotlib 画布（嵌入 Qt）。"""

    def __init__(self, width=6.4, height=4.8, dpi=100):
        self.fig = Figure(figsize=(width, height), dpi=dpi)
        super().__init__(self.fig)
        self.ax = self.fig.add_subplot(111)


class MainWindow(QMainWindow):
    """vdW Studio 主窗口。"""

    def __init__(self) -> None:
        super().__init__()
        self.setWindowTitle("vdW Studio — 二维半导体仿真平台")
        self.resize(1280, 820)

        self.preset = None
        self.structure = None
        self.model = None
        self.worker: Optional[SimulationWorker] = None
        self._results: dict = {}

        self._build_ui()
        self._connect()
        self.log("vdW Studio 就绪。选择材料后点击“运行仿真”。")
        self.set_material("mos2_kp")

    # ------------------------------------------------------------------
    # 界面构建
    # ------------------------------------------------------------------
    def _build_ui(self) -> None:
        # --- 中央标签页 ---
        self.tabs = QTabWidget()
        self.structure_canvas = MplCanvas(6.0, 5.2)
        self.band_canvas = MplCanvas(6.4, 4.8)
        self.dos_canvas = MplCanvas(4.4, 4.8)
        self.bz_canvas = MplCanvas(4.6, 4.4)
        for name, canvas in (("结构 (3D)", self.structure_canvas),
                             ("能带", self.band_canvas),
                             ("态密度", self.dos_canvas),
                             ("布里渊区", self.bz_canvas)):
            self.tabs.addTab(canvas, name)
        self.setCentralWidget(self.tabs)

        # --- 左侧：工程树 ---
        self.tree = QTreeWidget()
        self.tree.setHeaderLabel("工程树")
        self.tree.setColumnCount(2)
        self.tree.setColumnWidth(0, 150)
        self.addDockWidget(Qt.LeftDockWidgetArea, self._wrap(self.tree,
                                                              "工程树", 240))

        # --- 右侧：参数面板 ---
        panel = QWidget()
        form = QFormLayout(panel)
        self.material_combo = QComboBox()
        for key in list_presets():
            self.material_combo.addItem(get_preset(key).name, key)
        form.addRow("材料预设", self.material_combo)

        self.kpoint_spin = QSpinBox()
        self.kpoint_spin.setRange(10, 200)
        self.kpoint_spin.setValue(40)
        form.addRow("k 路径每段点数", self.kpoint_spin)

        self.sigma_spin = QDoubleSpinBox()
        self.sigma_spin.setRange(0.005, 0.5)
        self.sigma_spin.setSingleStep(0.005)
        self.sigma_spin.setValue(0.05)
        form.addRow("DOS 展宽 σ (eV)", self.sigma_spin)

        self.mesh_spin = QSpinBox()
        self.mesh_spin.setRange(8, 96)
        self.mesh_spin.setValue(48)
        form.addRow("DOS 网格", self.mesh_spin)

        self.field_spin = QDoubleSpinBox()
        self.field_spin.setRange(0.0, 3.0)
        self.field_spin.setSingleStep(0.1)
        form.addRow("垂直电场 (V/Å)", self.field_spin)
        self.field_label = QLabel("（仅硅烯等翘曲体系）")
        self.field_label.setStyleSheet("color: #888;")
        form.addRow("", self.field_label)

        self.run_button = QPushButton("▶ 运行仿真")
        self.run_button.setStyleSheet(
            "QPushButton{background:#1f4e79;color:white;padding:6px;"
            "font-weight:bold;border-radius:4px}"
            "QPushButton:disabled{background:#9aa7b5}")
        form.addRow(self.run_button)

        self.source_label = QLabel()
        self.source_label.setWordWrap(True)
        self.source_label.setStyleSheet("color:#555;font-size:11px;")
        form.addRow("参数出处", self.source_label)
        self.addDockWidget(Qt.RightDockWidgetArea,
                           self._wrap(panel, "仿真参数", 260))

        # --- 底部：日志 ---
        self.log_box = QPlainTextEdit()
        self.log_box.setReadOnly(True)
        self.log_box.setFont(QFont("Consolas", 9))
        self.log_box.setMaximumHeight(140)
        self.addDockWidget(Qt.BottomDockWidgetArea,
                           self._wrap(self.log_box, "日志", 140))

        # --- 状态栏 ---
        self.progress = QProgressBar()
        self.progress.setMaximumWidth(260)
        self.progress.setTextVisible(False)
        status = QStatusBar()
        self.status_label = QLabel("就绪")
        status.addWidget(self.status_label)
        status.addPermanentWidget(self.progress)
        self.setStatusBar(status)

        # --- 菜单栏 ---
        menu_file = self.menuBar().addMenu("文件(&F)")
        menu_file.addAction("导出结构(&E)…", self.export_structure)
        menu_file.addAction("退出(&Q)", self.close)
        menu_sim = self.menuBar().addMenu("仿真(&S)")
        menu_sim.addAction("运行仿真(&R)", self.run_simulation)
        menu_help = self.menuBar().addMenu("帮助(&H)")
        menu_help.addAction("关于(&A)", self._about)

    def _wrap(self, widget: QWidget, title: str, width: int) -> QDockWidget:
        dock = QDockWidget(title, self)
        dock.setWidget(widget)
        dock.setFeatures(QDockWidget.DockWidgetMovable)
        dock.setFixedWidth(width)
        return dock

    # ------------------------------------------------------------------
    # 信号
    # ------------------------------------------------------------------
    def _connect(self) -> None:
        self.material_combo.currentIndexChanged.connect(self._on_material)
        self.run_button.clicked.connect(self.run_simulation)

    # ------------------------------------------------------------------
    # 材料切换
    # ------------------------------------------------------------------
    def set_material(self, key: str) -> None:
        """切换材料预设（供测试与外部调用）。"""
        idx = next(i for i in range(self.material_combo.count())
                   if self.material_combo.itemData(i) == key)
        self.material_combo.setCurrentIndex(idx)

    def _on_material(self) -> None:
        key = self.material_combo.currentData()
        if not key:
            return
        self.preset = get_preset(key)
        try:
            self.structure = self.preset.make_structure(
                self.preset.structure_key)
        except Exception as exc:  # noqa: BLE001
            self.log(f"结构构建失败: {exc}")
            return
        self._rebuild_model()
        self._draw_structure()
        self._fill_tree()
        self.source_label.setText(self.preset.source)
        self.field_spin.setVisible(self.preset.key == "silicene")
        self.field_label.setVisible(self.preset.key == "silicene")
        self.log(f"已加载材料: {self.preset.name} "
                 f"({self.structure.formula_str}, "
                 f"{self.structure.n_atoms} 原子/胞, 引擎 {self.preset.engine})")

    def _rebuild_model(self) -> None:
        if self.preset.key == "silicene":
            self.model = self.preset.make_model(
                electric_field=self.field_spin.value())
        else:
            self.model = self.preset.make_model()

    # ------------------------------------------------------------------
    # 结构视图与工程树
    # ------------------------------------------------------------------
    def _draw_structure(self) -> None:
        self.structure_canvas.fig.clf()
        ax = self.structure_canvas.fig.add_subplot(111, projection="3d")
        try:
            plot_structure(self.structure, ax=ax,
                           title=f"{self.preset.name} — "
                                 f"{self.structure.formula_str}")
        except Exception as exc:  # noqa: BLE001
            self.log(f"结构绘制失败: {exc}")
        self.structure_canvas.draw_idle()

    def _fill_tree(self) -> None:
        self.tree.clear()
        st = self.structure
        lat_item = QTreeWidgetItem(["结构", st.formula_str])
        lat_item.addChild(QTreeWidgetItem(["原子数", str(st.n_atoms)]))
        a, b, c, _, _, gamma = st.lattice.parameters()
        lat_item.addChild(QTreeWidgetItem(["a (Å)", f"{a:.4f}"]))
        lat_item.addChild(QTreeWidgetItem(["b (Å)", f"{b:.4f}"]))
        lat_item.addChild(QTreeWidgetItem(["γ (°)", f"{gamma:.2f}"]))
        self.tree.addTopLevelItem(lat_item)

        eng = QTreeWidgetItem(["仿真", self.preset.engine])
        eng.addChild(QTreeWidgetItem(["模型", type(self.model).__name__]))
        self.tree.addTopLevelItem(eng)

        self.result_item = QTreeWidgetItem(["结果", "（未运行）"])
        self.tree.addTopLevelItem(self.result_item)
        self.tree.expandAll()

    # ------------------------------------------------------------------
    # 运行仿真
    # ------------------------------------------------------------------
    def run_simulation(self) -> None:
        if self.worker is not None and self.worker.isRunning():
            self.log("仿真已在运行中…")
            return
        self._rebuild_model()
        self.run_button.setEnabled(False)
        self.progress.setRange(0, 0)     # 忙碌指示
        self.log(f"开始仿真: {self.preset.name} "
                 f"(k点/段={self.kpoint_spin.value()}, "
                 f"σ={self.sigma_spin.value():.3f} eV)")
        self.worker = SimulationWorker(
            self.model, n_per_segment=self.kpoint_spin.value(),
            dos_mesh=(self.mesh_spin.value(), self.mesh_spin.value()),
            dos_sigma=self.sigma_spin.value(),
            n_valence=self.preset.n_valence if self.preset.engine != "kp"
            else 2)
        self.worker.start()
        # 轮询模式：QTimer 主线程周期检查进度/完成（不依赖 Qt 信号,
        # 规避部分 PyQt5+Python3.12 组合的 emit 兼容性问题）
        self._progress_seen = 0
        self._poll_timer = QTimer(self)
        self._poll_timer.timeout.connect(self._poll_worker)
        self._poll_timer.start(80)

    def _poll_worker(self) -> None:
        """主线程轮询后台线程的进度与完成状态。"""
        w = self.worker
        if w is None:
            return
        msgs = w.progress_messages
        while self._progress_seen < len(msgs):
            self.log(msgs[self._progress_seen])
            self._progress_seen += 1
        if w.done:
            self._poll_timer.stop()
            if w.error:
                self._on_error(w.error)
            else:
                self._on_results(w.result)

    def _on_results(self, results: dict) -> None:
        self._results = results
        band: BandStructure = results["band"]
        dos: DOSResult = results["dos"]
        gap = results["gap"]

        # 能带页
        self.band_canvas.fig.clf()
        ax = self.band_canvas.fig.add_subplot(111)
        plot_band_structure(band, ax=ax)
        self.band_canvas.draw_idle()

        # DOS 页
        self.dos_canvas.fig.clf()
        ax = self.dos_canvas.fig.add_subplot(111)
        plot_dos(dos, ax=ax)
        self.dos_canvas.draw_idle()

        # 布里渊区页
        self.bz_canvas.fig.clf()
        ax = self.bz_canvas.fig.add_subplot(111)
        plot_bz_path(self.model.lattice, results["kpath"], ax=ax)
        self.bz_canvas.draw_idle()

        # 工程树结果
        def item(name, value):
            return QTreeWidgetItem([name, value])
        self.result_item.takeChildren()
        if gap.gap is None:
            self.result_item.addChild(item("带隙", "无（金属/半金属）"))
        else:
            kind = "直接" if gap.direct else "间接"
            self.result_item.addChild(
                item("带隙", f"{gap.gap:.3f} eV ({kind}, {gap.cbm_label})"))
        if self.preset.engine == "tb" and gap.gap:
            try:
                k0 = gap.cbm_k
                me = effective_mass(self.model, k0, band=gap.n_valence,
                                    direction=(1, 0))
                self.result_item.addChild(
                    item("m_e* (≈CBM)", f"{me:.3f} m₀"))
            except Exception:  # noqa: BLE001 — 质量仅为附加信息
                pass
        if self.preset.gap_ref is not None:
            ref, tol = self.preset.gap_ref
            ok = gap.gap is not None and abs(gap.gap - ref) <= tol
            self.result_item.addChild(item(
                "文献参考", f"{ref:.3f} eV {'✓' if ok else '✗ 偏差>容差'}"))
        else:
            self.result_item.addChild(item("文献参考", self.preset.gap_note))
        self.tree.expandAll()

        self.tabs.setCurrentIndex(1)
        self.progress.setRange(0, 1)
        self.progress.setValue(1)
        self.run_button.setEnabled(True)
        self.status_label.setText(f"完成: {self.preset.name}")
        self.log("仿真完成。")

    def _on_error(self, msg: str) -> None:
        self.progress.setRange(0, 1)
        self.progress.setValue(0)
        self.run_button.setEnabled(True)
        self.log(f"[错误] {msg}")
        QMessageBox.critical(self, "仿真失败", msg.splitlines()[0])

    # ------------------------------------------------------------------
    # 导出
    # ------------------------------------------------------------------
    def export_structure(self) -> None:
        if self.structure is None:
            return
        path, _ = QFileDialog.getSaveFileName(
            self, "导出结构", self.preset.structure_key + ".POSCAR",
            "POSCAR (*.POSCAR);;XYZ (*.xyz)")
        if not path:
            return
        if path.lower().endswith(".xyz"):
            write_xyz(self.structure, path)
        else:
            write_poscar(self.structure, path)
        self.log(f"结构已导出: {path}")

    # ------------------------------------------------------------------
    def _about(self) -> None:
        from .. import __version__
        QMessageBox.about(
            self, "关于 vdW Studio",
            f"vdW Studio v{__version__}\n\n"
            "二维半导体材料仿真平台：结构建模 → 仿真 → 分析 → 可视化。\n"
            "物理参数均标注文献出处（见 docs/REFERENCES.md）。")

    def log(self, msg: str) -> None:
        self.log_box.appendPlainText(msg)

    def closeEvent(self, event):  # noqa: N802 (Qt 命名)
        if self.worker is not None and self.worker.isRunning():
            self.worker.wait(2000)
        super().closeEvent(event)
