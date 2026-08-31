from PyQt5.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout,
    QLabel, QPushButton, QDoubleSpinBox,
    QGroupBox, QCheckBox, QProgressBar
)


class ControlPanel(QWidget):
    """控制面板：文件操作、费米能级、能量范围、DOS 参数、导出、分析结果、进度"""

    def __init__(self, parent=None):
        super().__init__(parent)
        self._setup_ui()

    def _setup_ui(self):
        layout = QVBoxLayout(self)

        # === 文件操作 ===
        file_group = QGroupBox("File")
        file_layout = QVBoxLayout()
        self.btn_load = QPushButton("Load EIGENVAL")
        self.btn_load.clicked.connect(self._on_load)
        file_layout.addWidget(self.btn_load)
        file_group.setLayout(file_layout)
        layout.addWidget(file_group)

        # === 费米能级 ===
        fermi_group = QGroupBox("Fermi Level")
        fermi_layout = QVBoxLayout()
        self.spin_fermi = QDoubleSpinBox()
        self.spin_fermi.setRange(-100, 100)
        self.spin_fermi.setDecimals(4)
        self.spin_fermi.setSingleStep(0.1)
        self.spin_fermi.setValue(0.0)
        self.spin_fermi.valueChanged.connect(self._on_fermi_changed)
        fermi_layout.addWidget(self.spin_fermi)
        fermi_group.setLayout(fermi_layout)
        layout.addWidget(fermi_group)

        # === 能量范围 ===
        range_group = QGroupBox("Energy Range (eV)")
        range_layout = QVBoxLayout()

        range_layout.addWidget(QLabel("Min:"))
        self.spin_emin = QDoubleSpinBox()
        self.spin_emin.setRange(-50, 50)
        self.spin_emin.setValue(-5)
        self.spin_emin.valueChanged.connect(self._on_range_changed)
        range_layout.addWidget(self.spin_emin)

        range_layout.addWidget(QLabel("Max:"))
        self.spin_emax = QDoubleSpinBox()
        self.spin_emax.setRange(-50, 50)
        self.spin_emax.setValue(5)
        self.spin_emax.valueChanged.connect(self._on_range_changed)
        range_layout.addWidget(self.spin_emax)

        range_group.setLayout(range_layout)
        layout.addWidget(range_group)

        # === DOS 控制 ===
        dos_group = QGroupBox("DOS Settings")
        dos_layout = QVBoxLayout()

        dos_layout.addWidget(QLabel("Gaussian σ (eV):"))
        self.spin_dos_sigma = QDoubleSpinBox()
        self.spin_dos_sigma.setRange(0.001, 1.0)
        self.spin_dos_sigma.setDecimals(3)
        self.spin_dos_sigma.setSingleStep(0.01)
        self.spin_dos_sigma.setValue(0.05)
        self.spin_dos_sigma.valueChanged.connect(self._on_dos_sigma_changed)
        dos_layout.addWidget(self.spin_dos_sigma)

        self.chk_dos_total = QCheckBox("Show Total DOS")
        self.chk_dos_total.setChecked(True)
        self.chk_dos_total.stateChanged.connect(self._on_dos_visibility_changed)
        dos_layout.addWidget(self.chk_dos_total)

        self.chk_dos_vb = QCheckBox("Show VB DOS")
        self.chk_dos_vb.setChecked(True)
        self.chk_dos_vb.stateChanged.connect(self._on_dos_visibility_changed)
        dos_layout.addWidget(self.chk_dos_vb)

        self.chk_dos_cb = QCheckBox("Show CB DOS")
        self.chk_dos_cb.setChecked(True)
        self.chk_dos_cb.stateChanged.connect(self._on_dos_visibility_changed)
        dos_layout.addWidget(self.chk_dos_cb)

        dos_group.setLayout(dos_layout)
        layout.addWidget(dos_group)

        # === 导出 ===
        export_group = QGroupBox("Export")
        export_layout = QVBoxLayout()
        self.btn_export_png = QPushButton("Save PNG")
        self.btn_export_png.clicked.connect(self._on_export_png)
        self.btn_export_svg = QPushButton("Save SVG")
        self.btn_export_svg.clicked.connect(self._on_export_svg)
        export_layout.addWidget(self.btn_export_png)
        export_layout.addWidget(self.btn_export_svg)
        export_group.setLayout(export_layout)
        layout.addWidget(export_group)

        # === 分析结果 ===
        result_group = QGroupBox("Analysis")
        result_layout = QVBoxLayout()
        self.label_gap = QLabel("Band Gap: --")
        self.label_vbm = QLabel("VBM: --")
        self.label_cbm = QLabel("CBM: --")
        result_layout.addWidget(self.label_gap)
        result_layout.addWidget(self.label_vbm)
        result_layout.addWidget(self.label_cbm)
        result_group.setLayout(result_layout)
        layout.addWidget(result_group)

        # === 进度 ===
        progress_group = QGroupBox("Progress")
        progress_layout = QVBoxLayout()
        self.progress_bar = QProgressBar()
        self.progress_bar.setRange(0, 0)  # 无限循环样式
        self.progress_bar.setVisible(False)
        self.label_status = QLabel("Ready")
        progress_layout.addWidget(self.progress_bar)
        progress_layout.addWidget(self.label_status)
        progress_group.setLayout(progress_layout)
        layout.addWidget(progress_group)

        layout.addStretch()

        # 回调引用（由主窗口设置）
        self.main_window = None

    def set_main_window(self, mw):
        self.main_window = mw

    def get_energy_range(self) -> tuple:
        """返回当前能量范围 (emin, emax)"""
        return (self.spin_emin.value(), self.spin_emax.value())

    def set_progress(self, visible: bool, message: str = ""):
        """设置进度条状态"""
        self.progress_bar.setVisible(visible)
        self.label_status.setText(message)
        self.btn_load.setEnabled(not visible)

    # ------------------------------------------------------------------
    # 回调
    # ------------------------------------------------------------------
    def _on_load(self):
        if self.main_window:
            self.main_window.load_file()

    def _on_fermi_changed(self, value):
        if self.main_window:
            self.main_window.set_fermi_level(value)

    def _on_range_changed(self):
        if self.main_window:
            self.main_window.set_energy_range(
                self.spin_emin.value(),
                self.spin_emax.value()
            )

    def _on_dos_sigma_changed(self, value):
        if self.main_window:
            self.main_window.set_dos_sigma(value)

    def _on_dos_visibility_changed(self):
        if self.main_window:
            self.main_window.set_dos_visibility(
                total=self.chk_dos_total.isChecked(),
                vb=self.chk_dos_vb.isChecked(),
                cb=self.chk_dos_cb.isChecked()
            )

    def _on_export_png(self):
        if self.main_window:
            self.main_window.export_figure('png')

    def _on_export_svg(self):
        if self.main_window:
            self.main_window.export_figure('svg')

    def update_analysis(self, gap_info: dict):
        """更新分析结果显示"""
        if gap_info['gap'] is not None:
            self.label_gap.setText(f"Band Gap: {gap_info['gap']} eV")
            self.label_vbm.setText(f"VBM: {gap_info['vbm']} eV")
            self.label_cbm.setText(f"CBM: {gap_info['cbm']} eV")
        else:
            self.label_gap.setText("Band Gap: --")
            self.label_vbm.setText("VBM: --")
            self.label_cbm.setText("CBM: --")
