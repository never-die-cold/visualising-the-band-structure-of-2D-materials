from PyQt5.QtWidgets import QWidget, QVBoxLayout
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
import numpy as np

from core.dos_analyzer import DosData


class DosWidget(QWidget):
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
        self.draw()

    def set_visibility(self, total: bool = True, vb: bool = True, cb: bool = True):
        self.show_total = total
        self.show_vb = vb
        self.show_cb = cb
        self.draw()

    def draw(self):
        if self.dos_data is None:
            return

        self.ax.clear()
        energies = self.dos_data.energies
        total = self.dos_data.total_dos
        vb = self.dos_data.vb_dos
        cb = self.dos_data.cb_dos

        # 在能量范围内切片
        emin, emax = self.energy_range
        mask = (energies >= emin) & (energies <= emax)
        e_plot = energies[mask]
        t_plot = total[mask]
        vb_plot = vb[mask] if vb is not None else None
        cb_plot = cb[mask] if cb is not None else None

        # 绘制价带 DOS（蓝色填充）
        if self.show_vb and vb_plot is not None:
            self.ax.fill_betweenx(
                e_plot, 0, vb_plot,
                where=(e_plot <= 0),
                color='steelblue', alpha=0.4, label='VB DOS'
            )

        # 绘制导带 DOS（红色填充）
        if self.show_cb and cb_plot is not None:
            self.ax.fill_betweenx(
                e_plot, 0, cb_plot,
                where=(e_plot > 0),
                color='firebrick', alpha=0.4, label='CB DOS'
            )

        # 绘制总 DOS（黑色实线）
        if self.show_total:
            self.ax.plot(t_plot, e_plot, color='black', linewidth=1.2, label='Total DOS')

        # 费米能级参考线
        self.ax.axhline(y=0, color='red', linestyle='--', linewidth=1.0, label='Fermi Level')

        # 坐标轴设置（DOS 图：横轴为 DOS，纵轴为能量）
        self.ax.set_ylim(emin, emax)
        self.ax.set_ylabel('Energy (eV)', fontsize=11)
        self.ax.set_xlabel('DOS (a.u.)', fontsize=11)
        self.ax.set_title('Density of States', fontsize=13, fontweight='bold')
        self.ax.grid(True, alpha=0.3)
        self.ax.legend(loc='upper right', fontsize=8)

        # 隐藏上方和右侧边框
        self.ax.spines['top'].set_visible(False)
        self.ax.spines['right'].set_visible(False)

        self.canvas.draw()

    def save_figure(self, filepath: str, dpi: int = 300):
        """导出高分辨率图片"""
        self.figure.savefig(filepath, dpi=dpi, bbox_inches='tight')
