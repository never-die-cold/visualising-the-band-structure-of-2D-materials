from PyQt5.QtWidgets import QWidget, QVBoxLayout
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure

from core.dos_analyzer import DosData
from .figure_export import FigureExportMixin


class DosWidget(FigureExportMixin, QWidget):
    """态密度（DOS）绘图组件"""

    def __init__(self, parent=None):
        super().__init__(parent)
        self._setup_ui()
        self.dos_data: DosData = None
        self.energy_range = (-5, 5)
        self.show_vb = True
        self.show_cb = True
        self.show_total = True

    def _setup_ui(self):
        layout = QVBoxLayout(self)
        self.figure = Figure(figsize=(4, 6), dpi=100)
        self.canvas = FigureCanvas(self.figure)
        layout.addWidget(self.canvas)
        self.ax = self.figure.add_subplot(111)

    def set_data(self, dos_data: DosData):
        self.dos_data = dos_data
        self.draw()

    def set_energy_range(self, emin: float, emax: float):
        self.energy_range = (emin, emax)
        if self.dos_data is None:
            return
        # 全量数据已绘制，窗口只由 ylim 控制 → 不必重建 artist
        self.ax.set_ylim(emin, emax)
        self.canvas.draw_idle()

    def set_visibility(self, total: bool = True, vb: bool = True, cb: bool = True):
        self.show_total = total
        self.show_vb = vb
        self.show_cb = cb
        self.draw()

    def draw(self):
        if self.dos_data is None:
            return

        self.ax.clear()
        # 绘制全量数据；能量窗口由 set_ylim 控制，范围变化无需重绘
        e_plot = self.dos_data.energies
        t_plot = self.dos_data.total_dos
        vb_plot = self.dos_data.vb_dos
        cb_plot = self.dos_data.cb_dos
        emin, emax = self.energy_range

        # 绘制价带 DOS（蓝色填充）
        if self.show_vb and vb_plot is not None:
            self.ax.fill_betweenx(
                e_plot, 0, vb_plot,
                color='steelblue', alpha=0.4, label=self.dos_data.component_labels[0]
            )

        # 绘制导带 DOS（红色填充）
        if self.show_cb and cb_plot is not None:
            self.ax.fill_betweenx(
                e_plot, 0, cb_plot,
                color='firebrick', alpha=0.4, label=self.dos_data.component_labels[1]
            )

        # 绘制总 DOS（黑色实线）
        if self.show_total:
            self.ax.plot(t_plot, e_plot, color='black', linewidth=1.2, label='Total')

        # 费米能级参考线
        self.ax.axhline(y=0, color='red', linestyle='--', linewidth=1.0, label='Fermi Level')

        # 坐标轴设置（DOS 图：横轴为 DOS，纵轴为能量）
        self.ax.set_ylim(emin, emax)
        self.ax.set_ylabel('Energy (eV)', fontsize=11)
        physical = self.dos_data.scope == 'brillouin-zone'
        self.ax.set_xlabel('DOS (states/eV/cell)' if physical else 'Spectrum (states/eV)', fontsize=11)
        self.ax.set_title('Density of States' if physical else 'K-point spectrum (not BZ DOS)', fontsize=13, fontweight='bold')
        self.ax.text(.02, .98, f"σ={self.dos_data.sigma:g} eV; window N={self.dos_data.integral:.4g}",
                     transform=self.ax.transAxes, va='top', fontsize=8)
        self.setToolTip(self.dos_data.note)
        self.ax.grid(True, alpha=0.3)
        self.ax.legend(loc='upper right', fontsize=8)

        # 隐藏上方和右侧边框
        self.ax.spines['top'].set_visible(False)
        self.ax.spines['right'].set_visible(False)

        self.canvas.draw()
