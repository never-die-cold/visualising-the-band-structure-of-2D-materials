from PyQt5.QtWidgets import QWidget, QVBoxLayout
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
import numpy as np


class BandStructureWidget(QWidget):
    """能带结构绘图组件"""
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self._setup_ui()
        self.band_data = None
        self.analyzer = None
        self.fermi_level = 0.0
        self.energy_range = (-5, 5)
        self.show_fermi = True
        
    def _setup_ui(self):
        layout = QVBoxLayout(self)
        self.figure = Figure(figsize=(8, 6), dpi=100)
        self.canvas = FigureCanvas(self.figure)
        layout.addWidget(self.canvas)
        self.ax = self.figure.add_subplot(111)
        
    def set_data(self, band_data, analyzer=None):
        self.band_data = band_data
        self.analyzer = analyzer
        self.draw()
        
    def set_fermi_level(self, efermi: float):
        self.fermi_level = efermi
        if self.analyzer:
            self.analyzer.set_fermi_level(efermi)
        self.draw()
        
    def set_energy_range(self, emin: float, emax: float):
        self.energy_range = (emin, emax)
        self.draw()
        
    def draw(self):
        if self.band_data is None:
            return
            
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
                self.ax.plot(kdist[start:end + 1], energies[start:end + 1, ib],
                             color='blue' if spin == 0 else 'darkorange',
                             linewidth=1.2, alpha=0.8, label=label,
                             marker='.' if start == end else None)
        
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
        if self.analyzer:
            gap_info = self.analyzer.get_band_gap()
            status = gap_info.get('status', 'unknown')
            if status == 'insulator':
                kind = 'direct' if gap_info['direct'] else 'indirect'
                text = f"Sampled Eg = {gap_info['gap']:.6g} eV ({kind})"
            else:
                state = {'metal': 'metal', 'zero-gap': 'zero gap', 'unknown': 'undetermined'}[status]
                text = f"Sampled k points: {state}"
            self.ax.text(0.02, 0.98, text, transform=self.ax.transAxes,
                         fontsize=10, verticalalignment='top',
                         bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
        
        self.ax.legend()
        self.canvas.draw()
        
    def save_figure(self, filepath: str, dpi: int = 300):
        """导出高分辨率图片"""
        self.figure.savefig(filepath, dpi=dpi, bbox_inches='tight')
