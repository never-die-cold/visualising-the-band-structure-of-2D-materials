from PyQt5.QtWidgets import QWidget, QVBoxLayout
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
import numpy as np


class BandStructureWidget(QWidget):
    """能带结构绘图组件

    重绘分两条路径：数据或图形结构变化走 :meth:`rebuild` 完整重建；
    费米能级只更新已缓存 Line2D 的 y 数据，能量范围只改 ylim。后两者
    不重建 artist——否则每按一次 spinbox 都要重新创建整幅图（自旋夹具
    即 840 条线），在树莓派上是可感知的卡顿。
    """

    def __init__(self, parent=None):
        super().__init__(parent)
        self._setup_ui()
        self.band_data = None
        self.analyzer = None
        self.fermi_level = 0.0
        self.energy_range = (-5, 5)
        self.show_fermi = True
        self._curves = []           # (Line2D, 带索引, 段起点, 段终点)
        self._gap_text = None
        self._built = False

    def _setup_ui(self):
        layout = QVBoxLayout(self)
        self.figure = Figure(figsize=(8, 6), dpi=100)
        self.canvas = FigureCanvas(self.figure)
        layout.addWidget(self.canvas)
        self.ax = self.figure.add_subplot(111)

    def set_data(self, band_data, analyzer=None):
        self.band_data = band_data
        self.analyzer = analyzer
        self.rebuild()

    def set_fermi_level(self, efermi: float):
        self.fermi_level = efermi
        if self.analyzer:
            self.analyzer.set_fermi_level(efermi)
        self.refresh_fermi()

    def set_energy_range(self, emin: float, emax: float):
        self.energy_range = (emin, emax)
        self.refresh_range()

    # ------------------------------------------------------------------
    # 重绘路径
    # ------------------------------------------------------------------
    def rebuild(self):
        """完整重建图形（数据变化或图像结构变化时）。"""
        if self.band_data is None:
            return

        self._built = False
        self._curves = []
        self._gap_text = None
        self.ax.clear()
        kdist = self.band_data.kdistances
        energies = self.band_data.energies - self.fermi_level

        # Draw each segment separately so discontinuous paths are not joined.
        segments = self.band_data.segments or [(0, len(kdist) - 1)]
        for ib in range(self.band_data.num_bands):
            spin = ib // self.band_data.nbands
            for segment_index, (start, end) in enumerate(segments):
                label = None
                if self.band_data.ispin == 2 and ib % self.band_data.nbands == 0 and segment_index == 0:
                    label = 'Spin up' if spin == 0 else 'Spin down'
                line, = self.ax.plot(kdist[start:end + 1], energies[start:end + 1, ib],
                                     color='blue' if spin == 0 else 'darkorange',
                                     linewidth=1.2, alpha=0.8, label=label,
                                     marker='.' if start == end else None)
                self._curves.append((line, ib, start, end))

        # 费米能级线
        if self.show_fermi:
            self.ax.axhline(y=0, color='red', linestyle='--',
                          linewidth=1.0, label='Fermi Level')

        # Merge coincident ticks (e.g. X at a shared end, or X|M at a jump).
        ticks, labels = [], []
        for idx, label in self.band_data.kpoint_labels:
            position = kdist[idx]
            if ticks and np.isclose(position, ticks[-1], atol=1e-10, rtol=0):
                if label not in labels[-1].split('|'):
                    labels[-1] += '|' + label
            else:
                ticks.append(position)
                labels.append(label)
                self.ax.axvline(x=position, color='gray',
                               linestyle=':', linewidth=0.8, alpha=0.5)
        if ticks:
            self.ax.set_xticks(ticks)
            self.ax.set_xticklabels(labels, fontsize=11)

        if kdist[-1] > kdist[0]:
            self.ax.set_xlim(kdist[0], kdist[-1])
        else:
            self.ax.set_xlim(kdist[0] - 0.5, kdist[0] + 0.5)
        self.ax.set_ylim(self.energy_range[0], self.energy_range[1])
        axis_label = ('k-path (Å$^{-1}$)' if self.band_data.lattice_matrix is not None
                      else 'k-path (fractional; lattice unavailable)')
        self.ax.set_xlabel(axis_label, fontsize=12)
        self.ax.set_ylabel('Energy (eV)', fontsize=12)
        self.ax.set_title('Band Structure', fontsize=14, fontweight='bold')
        self.ax.grid(True, alpha=0.3)

        # 添加带隙标注
        caption = self._gap_caption()
        if caption is not None:
            self._gap_text = self.ax.text(0.02, 0.98, caption, transform=self.ax.transAxes,
                                         fontsize=10, verticalalignment='top',
                                         bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

        self.ax.legend()
        self._built = True
        self.canvas.draw()

    def refresh_fermi(self):
        """只更新曲线的 y 数据与带隙标注，不重建 artist。"""
        if not self._built:
            self.rebuild()
            return
        energies = self.band_data.energies
        for line, ib, start, end in self._curves:
            line.set_ydata(energies[start:end + 1, ib] - self.fermi_level)
        caption = self._gap_caption()
        if self._gap_text is not None and caption is not None:
            self._gap_text.set_text(caption)
        self.canvas.draw_idle()

    def refresh_range(self):
        """只改能量窗口。"""
        if not self._built:
            self.rebuild()
            return
        self.ax.set_ylim(self.energy_range[0], self.energy_range[1])
        self.canvas.draw_idle()

    def draw(self):
        """完整重建（兼容入口）。"""
        self.rebuild()

    # ------------------------------------------------------------------
    def _gap_caption(self):
        """带隙标注文字；无分析器时返回 None。"""
        if not self.analyzer:
            return None
        gap_info = self.analyzer.get_band_gap()
        status = gap_info.get('status', 'unknown')
        if status == 'insulator':
            kind = 'direct' if gap_info['direct'] else 'indirect'
            return f"Sampled Eg = {gap_info['gap']:.6g} eV ({kind})"
        state = {'metal': 'metal', 'zero-gap': 'zero gap', 'unknown': 'undetermined'}[status]
        return f"Sampled k points: {state}"

    def save_figure(self, filepath: str, dpi: int = 300):
        """导出高分辨率图片"""
        self.figure.savefig(filepath, dpi=dpi, bbox_inches='tight')
