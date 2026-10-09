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
from ..presets import get_preset, list_presets
from ..simulation import KP_DOS_REASON
from ..visualization import (
    plot_band_structure,
    plot_bz_path,
    plot_dos,
    plot_structure,
    setup_cjk_fonts,
)
from .workers import SimulationWorker
from ..task_state import SimulationSnapshot, align_structure
from ..visualization.fonts import plot_text


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
        setup_cjk_fonts()
        self.setWindowTitle("vdW Studio — 二维半导体仿真平台")
        self.resize(1280, 820)

        self.preset = None
        self.structure = None
        self.model = None
        self.worker: Optional[SimulationWorker] = None
        self._results: dict = {}
        self._active_snapshot = None
        self._base_structure = None
        self._closing = False

        self._build_ui()
        self._connect()
        self._poll_timer = QTimer(self)
        self._poll_timer.timeout.connect(self._poll_worker)
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
        self.valley_canvas = MplCanvas(6.2, 4.6)
        for name, canvas in (("结构 (3D)", self.structure_canvas),
                             ("能带", self.band_canvas),
                             ("态密度", self.dos_canvas),
                             ("布里渊区", self.bz_canvas),
                             ("谷物理", self.valley_canvas)):
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

        self.strain_x_spin = QDoubleSpinBox()
        self.strain_x_spin.setRange(-10.0, 10.0)
        self.strain_x_spin.setSingleStep(0.5)
        self.strain_x_spin.setDecimals(1)
        form.addRow("应变 εx (%)", self.strain_x_spin)
        self.strain_y_spin = QDoubleSpinBox()
        self.strain_y_spin.setRange(-10.0, 10.0)
        self.strain_y_spin.setSingleStep(0.5)
        self.strain_y_spin.setDecimals(1)
        form.addRow("应变 εy (%)", self.strain_y_spin)
        self.strain_label = QLabel("（仅紧束缚引擎；键长标度律 t∝d⁻ⁿ）")
        self.strain_label.setStyleSheet("color: #888;")
        form.addRow("", self.strain_label)

        self.run_button = QPushButton("▶ 运行仿真")
        self.run_button.setStyleSheet(
            "QPushButton{background:#1f4e79;color:white;padding:6px;"
            "font-weight:bold;border-radius:4px}"
            "QPushButton:disabled{background:#9aa7b5}")
        form.addRow(self.run_button)
        self.cancel_button = QPushButton('取消仿真')
        self.cancel_button.setEnabled(False)
        form.addRow(self.cancel_button)

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
        self.run_action = menu_sim.addAction("运行仿真(&R)", self.run_simulation)
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
        self.cancel_button.clicked.connect(self.cancel_simulation)
        for widget in (self.strain_x_spin, self.strain_y_spin, self.field_spin,
                       self.kpoint_spin, self.mesh_spin, self.sigma_spin):
            widget.valueChanged.connect(self._on_parameters_changed)

    def _on_parameters_changed(self):
        if self.preset is None or self._poll_timer.isActive():
            return
        self._active_snapshot = None
        self._results = {}
        self._rebuild_model()
        self._draw_structure()
        self._fill_tree()
        self._clear_result_plots()

    def _clear_result_plots(self):
        for canvas in (self.band_canvas, self.dos_canvas, self.bz_canvas, self.valley_canvas):
            canvas.fig.clf()
            canvas.ax = canvas.fig.add_subplot(111)
            canvas.ax.text(.5, .5, plot_text('参数已更新，请运行仿真', 'Parameters changed; run simulation'),
                           ha='center', va='center', transform=canvas.ax.transAxes)
            canvas.ax.set_axis_off()
            canvas.draw_idle()

    def _set_busy(self, busy: bool) -> None:
        """运行期间保持材料与参数一致，按模型能力恢复控件。"""
        self.run_button.setEnabled(not busy)
        self.run_action.setEnabled(not busy)
        self.cancel_button.setEnabled(busy and not self._closing)
        for widget in (self.material_combo, self.kpoint_spin, self.field_spin):
            widget.setEnabled(not busy)
        is_tb = self.preset.engine == "tb"
        for widget in (self.strain_x_spin, self.strain_y_spin, self.strain_label):
            widget.setEnabled(not busy and is_tb)
        for widget in (self.sigma_spin, self.mesh_spin):
            widget.setEnabled(not busy and self.preset.engine != "kp")

    def _show_dos_unavailable(self, reason: str) -> None:
        self.dos_canvas.fig.clf()
        ax = self.dos_canvas.fig.add_subplot(111)
        ax.text(0.5, 0.5, plot_text(reason, 'BZ DOS unavailable for the local valley model'), ha="center", va="center",
                wrap=True, transform=ax.transAxes)
        ax.set_axis_off()
        self.dos_canvas.draw_idle()

    # ------------------------------------------------------------------
    # 材料切换
    # ------------------------------------------------------------------
    def set_material(self, key: str) -> None:
        """切换材料预设（供测试与外部调用）。"""
        if self._poll_timer.isActive():
            self.log("请等待当前仿真完成后切换材料。")
            return
        idx = next(i for i in range(self.material_combo.count())
                   if self.material_combo.itemData(i) == key)
        self.material_combo.setCurrentIndex(idx)

    def _on_material(self) -> None:
        if self._poll_timer.isActive():
            self.material_combo.blockSignals(True)
            self.material_combo.setCurrentIndex(self.material_combo.findData(self.preset.key))
            self.material_combo.blockSignals(False)
            return
        key = self.material_combo.currentData()
        if not key:
            return
        self.preset = get_preset(key)
        try:
            self._base_structure = self.preset.make_structure(
                self.preset.structure_key)
        except Exception as exc:  # noqa: BLE001
            self.log(f"结构构建失败: {exc}")
            return
        self._rebuild_model()
        self._active_snapshot = None
        self._results = {}
        self._clear_result_plots()
        self._draw_structure()
        self._fill_tree()
        self.source_label.setText(self.preset.source)
        field_ok = self.preset.key in ("silicene", "bilayer_graphene")
        self.field_spin.setVisible(field_ok)
        self.field_label.setText("（仅硅烯/双层石墨烯等体系）")
        self.field_label.setVisible(field_ok)
        is_tb = self.preset.engine == "tb"
        for w in (self.strain_x_spin, self.strain_y_spin,
                  self.strain_label):
            w.setEnabled(is_tb)
        # 谷物理标签页仅 k·p 模型有意义
        vidx = self.tabs.indexOf(self.valley_canvas)
        self.tabs.setTabVisible(vidx, self.preset.engine == "kp")
        self._set_busy(False)
        if self.preset.engine == "kp":
            self._show_dos_unavailable(KP_DOS_REASON)
        self.log(f"已加载材料: {self.preset.name} "
                 f"({self.structure.formula_str}, "
                 f"{self.structure.n_atoms} 原子/胞, 引擎 {self.preset.engine})")

    def _rebuild_model(self) -> None:
        if self.preset.key in ("silicene", "bilayer_graphene"):
            self.model = self.preset.make_model(
                electric_field=self.field_spin.value())
        else:
            self.model = self.preset.make_model()
        # 应变工程（仅位点型紧束缚引擎；k·p/sp3d5 参数无应变依赖）
        if self.preset.engine == "tb":
            from ..engine.strain import apply_strain
            self.model = apply_strain(
                self.model,
                ex=self.strain_x_spin.value() / 100.0,
                ey=self.strain_y_spin.value() / 100.0)
        self.structure = align_structure(self._base_structure, self.model)

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
        if self._closing:
            return
        if self._poll_timer.isActive():
            self.log("仿真已在运行中…")
            return
        self._rebuild_model()
        self._results = {}
        self._draw_structure()
        self._fill_tree()
        self._active_snapshot = SimulationSnapshot.capture(
            self.preset, self.model, self.structure,
            strain=(self.strain_x_spin.value() / 100., self.strain_y_spin.value() / 100.) if self.preset.engine == 'tb' else (0., 0.),
            electric_field=self.field_spin.value() if self.preset.key in ('silicene', 'bilayer_graphene') else 0.,
            n_per_segment=self.kpoint_spin.value(), dos_mesh=(self.mesh_spin.value(),) * 2,
            dos_sigma=self.sigma_spin.value(), include_valley=self.preset.engine == 'kp')
        self._set_busy(True)
        self.progress.setRange(0, 0)     # 忙碌指示
        self.log(f"开始仿真: {self.preset.name} "
                 f"(k点/段={self.kpoint_spin.value()}, "
                 f"σ={self.sigma_spin.value():.3f} eV)")
        self.worker = SimulationWorker(
            self.model, n_per_segment=self.kpoint_spin.value(),
            dos_mesh=(self.mesh_spin.value(), self.mesh_spin.value()),
            dos_sigma=self.sigma_spin.value(),
            n_valence=self.preset.n_valence,
            lattice=self.structure.lattice,
            include_valley=self.preset.engine == "kp", snapshot=self._active_snapshot)
        self.worker.start()
        # 轮询模式：QTimer 主线程周期检查进度/完成（不依赖 Qt 信号,
        # 规避部分 PyQt5+Python3.12 组合的 emit 兼容性问题）
        self._progress_seen = 0
        self._poll_timer.start(80)

    def cancel_simulation(self):
        if self.worker is not None and self.worker.isRunning():
            self.worker.cancel()
            self.cancel_button.setEnabled(False)
            self.status_label.setText('正在取消…')

    def _poll_worker(self) -> None:
        """主线程轮询后台线程的进度与完成状态。"""
        w = self.worker
        if w is None:
            return
        msgs = w.progress_messages
        while self._progress_seen < len(msgs):
            self.log(msgs[self._progress_seen])
            self._progress_seen += 1
        if w.done and not w.isRunning():
            self._poll_timer.stop()
            if self._closing:
                self.close()
            elif w.cancelled:
                self._active_snapshot = None
                self.progress.setRange(0, 1)
                self.progress.setValue(0)
                self._set_busy(False)
                self.status_label.setText('已取消')
                self.log('仿真已取消。')
            elif w.error:
                self._on_error(w.error)
            else:
                self._on_results(w.result)

    def _on_results(self, results: dict) -> None:
        snapshot = results.get('snapshot')
        if snapshot is not None and (self._active_snapshot is None or snapshot.task_id != self._active_snapshot.task_id):
            self.log('忽略过期任务结果。')
            return
        self._results = results
        band: BandStructure = results["band"]
        dos: Optional[DOSResult] = results["dos"]
        gap = results["gap"]

        # 能带页
        self.band_canvas.fig.clf()
        ax = self.band_canvas.fig.add_subplot(111)
        title = (f"{band.model_name} — K 谷附近 (|q| ≤ {results['qmax']} Å$^{{-1}}$)"
                 if results["domain"] == "valley-local" else None)
        plot_band_structure(band, ax=ax, title=title)
        self.band_canvas.draw_idle()

        # DOS 页
        if dos is None:
            self._show_dos_unavailable(results["dos_reason"])
        else:
            self.dos_canvas.fig.clf()
            ax = self.dos_canvas.fig.add_subplot(111)
            plot_dos(dos, ax=ax)
            self.dos_canvas.draw_idle()

        # 布里渊区页
        self.bz_canvas.fig.clf()
        ax = self.bz_canvas.fig.add_subplot(111)
        plot_bz_path(results["lattice"], results["kpath"], ax=ax)
        self.bz_canvas.draw_idle()

        # 谷物理页 (仅 k·p 模型)
        rep = results.get("valley")
        if rep is not None:
            self.valley_canvas.fig.clf()
            ax = self.valley_canvas.fig.add_subplot(121)
            om = rep["omega"].T
            im = ax.imshow(om, origin="lower",
                           extent=[-0.25, 0.25, -0.25, 0.25],
                           cmap="RdBu_r",
                           vmin=-np.abs(om).max(), vmax=np.abs(om).max())
            ax.scatter([0], [0], marker="x", color="black", s=50)
            ax.set_xlabel("q$_x$ (Å$^{-1}$)")
            ax.set_ylabel("q$_y$ (Å$^{-1}$)")
            ax.set_title("Ω$_{vb}$ (Å²) near K", fontsize=10)
            self.valley_canvas.fig.colorbar(im, ax=ax, fraction=0.046)
            ax2 = self.valley_canvas.fig.add_subplot(122)
            dk, dkm = rep["dichroism_K"], rep["dichroism_Kminus"]
            ax2.bar([0, 1], [dk.f_plus, dkm.f_plus], 0.4,
                    color="#c0392b", label="σ$^+$")
            ax2.bar([0, 1], [dk.f_minus, dkm.f_minus], 0.4,
                    color="#1f4e79", label="σ$^-$")
            ax2.set_xticks([0, 1])
            ax2.set_xticklabels(["K", "−K"])
            ax2.set_ylabel("光学矩阵元² (eV²Å²)")
            ax2.set_title("valley-selective excitation", fontsize=10)
            ax2.legend(fontsize=8)
            self.valley_canvas.fig.tight_layout()
            self.valley_canvas.draw_idle()
            self.log(f"谷物理: Ω_vb(K) = {rep['omega_vb_K']:.2f} Å²; "
                     f"K 谷主导 {dk.dominant} (×{dk.ratio:.0f}), "
                     f"−K 谷主导 {dkm.dominant} (×{dkm.ratio:.0f})")

        # 工程树结果
        def item(name, value):
            return QTreeWidgetItem([name, value])
        self.result_item.takeChildren()
        scope_label = {"brillouin-zone": "全布里渊区", "valley-local": "K 谷附近", "path": "路径（实验性）"}.get(gap.scope, "未指定")
        self.result_item.addChild(item("带隙计算域", scope_label))
        if gap.gap is None:
            self.result_item.addChild(item("带隙", "零隙" if gap.status == "zero-gap" else "金属/能带重叠"))
        else:
            kind = "直接" if gap.direct else "间接"
            self.result_item.addChild(
                item("带隙", f"{gap.gap:.3f} eV ({kind}, {gap.cbm_label})"))
        if results.get("path_gap") is not None:
            path_gap = results["path_gap"]
            self.result_item.addChild(item("路径带隙", f"{path_gap.raw_gap:.6f} eV"))
            self.result_item.addChild(item("带边坐标", f"VBM {gap.vbm_k.round(6)} / CBM {gap.cbm_k.round(6)}"))
            if gap.scope == "brillouin-zone":
                converged = gap.search_metadata.get("converged", False)
                self.result_item.addChild(item("网格收敛检查", "通过" if converged else "未通过，请加密搜索"))
        engine = self.preset.engine if snapshot is None else snapshot.engine
        reference = self.preset.gap_ref if snapshot is None else snapshot.gap_reference
        if engine == "tb" and gap.gap:
            try:
                k0 = gap.cbm_k
                result_model = self.model if self.worker is None or snapshot is None else self.worker.model
                me = effective_mass(result_model, k0, band=gap.n_valence,
                                    direction=(1, 0))
                self.result_item.addChild(
                    item("m_e* (CBM, 沿 b₁)", f"{me:.3f} m₀"))
            except Exception:  # noqa: BLE001 — 质量仅为附加信息
                pass
        if reference is not None:
            ref, tol = reference
            ok = gap.gap is not None and abs(gap.gap - ref) <= tol
            self.result_item.addChild(item(
                "文献参考", f"{ref:.3f} eV {'✓' if ok else '✗ 偏差>容差'}"))
        else:
            self.result_item.addChild(item("文献参考", self.preset.gap_note if snapshot is None else snapshot.gap_note))
        if snapshot is not None:
            self.result_item.addChild(item('任务编号', snapshot.task_id))
            self.result_item.addChild(item('任务材料', snapshot.material_name))
        for name, mass in results["effective_masses_m0"].items():
            self.result_item.addChild(item(name, f"{mass:.3f} m₀"))
        self.tree.expandAll()

        self.tabs.setCurrentIndex(1)
        self.progress.setRange(0, 1)
        self.progress.setValue(1)
        self._set_busy(False)
        self.status_label.setText(f"完成: {self.preset.name if snapshot is None else snapshot.material_name}")
        self.log("仿真完成。")

    def _on_error(self, msg: str) -> None:
        self.progress.setRange(0, 1)
        self.progress.setValue(0)
        self._set_busy(False)
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
            self._closing = True
            self.cancel_simulation()
            event.ignore()
            if not self._poll_timer.isActive():
                self._poll_timer.start(80)
            return
        super().closeEvent(event)
