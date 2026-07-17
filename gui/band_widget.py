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
        
        # 绘制能带
        for ib in range(self.band_data.num_bands):
            self.ax.plot(kdist, energies[:, ib], 
                        color='blue', linewidth=1.2, alpha=0.8)
        
        # 费米能级线
        if self.show_fermi:
            self.ax.axhline(y=0, color='red', linestyle='--', 
                          linewidth=1.0, label='Fermi Level')
        
        # 高对称点虚线
        for idx, label in self.band_data.kpoint_labels:
            self.ax.axvline(x=kdist[idx], color='gray', 
                          linestyle=':', linewidth=0.8, alpha=0.5)
        
        # 设置标签
        if self.band_data.kpoint_labels:
            ticks = [kdist[idx] for idx, _ in self.band_data.kpoint_labels]
            labels = [label for _, label in self.band_data.kpoint_labels]
            self.ax.set_xticks(ticks)
            self.ax.set_xticklabels(labels, fontsize=11)
        
        self.ax.set_xlim(kdist[0], kdist[-1])
        self.ax.set_ylim(self.energy_range[0], self.energy_range[1])
        self.ax.set_xlabel('k-path', fontsize=12)
        self.ax.set_ylabel('Energy (eV)', fontsize=12)
        self.ax.set_title('Band Structure', fontsize=14, fontweight='bold')
        self.ax.grid(True, alpha=0.3)
        
        # 添加带隙标注
        if self.analyzer:
            gap_info = self.analyzer.get_band_gap()
            if gap_info['gap'] is not None:
                text = f"Eg = {gap_info['gap']} eV"
                if gap_info['direct']:
                    text += " (direct)"
                else:
                    text += " (indirect)"
                self.ax.text(0.02, 0.98, text, transform=self.ax.transAxes,
                           fontsize=10, verticalalignment='top',
                           bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
        
        self.ax.legend()
        self.canvas.draw()
        
    def save_figure(self, filepath: str, dpi: int = 300):
        """导出高分辨率图片"""
        self.figure.savefig(filepath, dpi=dpi, bbox_inches='tight')