from PyQt5.QtWidgets import (QMainWindow, QWidget, QHBoxLayout, 
                            QFileDialog, QMessageBox)
from PyQt5.QtCore import Qt
from pathlib import Path

from .band_widget import BandStructureWidget
from .control_panel import ControlPanel
from core.parser import VASPEigenvalParser
from core.band_analyzer import BandAnalyzer


class MainWindow(QMainWindow):
    """主窗口"""
    
    def __init__(self):
        super().__init__()
        self.setWindowTitle("2D Material Band Structure Visualizer")
        self.setGeometry(100, 100, 1200, 800)
        
        # 中心部件
        central = QWidget()
        self.setCentralWidget(central)
        layout = QHBoxLayout(central)
        
        # 左侧文件树（简化版，可扩展）
        # 暂时留空或放简单列表
        
        # 中间图表
        self.band_widget = BandStructureWidget()
        layout.addWidget(self.band_widget, stretch=3)
        
        # 右侧控制面板
        self.control_panel = ControlPanel()
        self.control_panel.set_main_window(self)
        layout.addWidget(self.control_panel, stretch=1)
        
        self.current_data = None
        self.current_analyzer = None
        self.current_file = None
        
    def load_file(self):
        """加载 EIGENVAL 文件（同目录下 KPOINTS 会自动解析）"""
        filepath, _ = QFileDialog.getOpenFileName(
            self, "Open EIGENVAL", "", "EIGENVAL (EIGENVAL);;All Files (*)"
        )
        if not filepath:
            return

        try:
            file_path = Path(filepath)
            kpoints_path = file_path.parent / "KPOINTS"
            parser = VASPEigenvalParser(
                str(file_path),
                str(kpoints_path) if kpoints_path.exists() else None
            )

            self.current_data = parser.parse()
            self.current_analyzer = BandAnalyzer(self.current_data)
            self.current_file = file_path
            
            # 更新图表
            self.band_widget.set_data(self.current_data, self.current_analyzer)
            
            # 更新分析结果
            gap_info = self.current_analyzer.get_band_gap()
            self.control_panel.update_analysis(gap_info)
            
            self.setWindowTitle(f"Band Viz - {self.current_file.name}")
            
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Failed to parse file:\n{str(e)}")
            
    def set_fermi_level(self, efermi: float):
        """设置费米能级"""
        self.band_widget.set_fermi_level(efermi)
        if self.current_analyzer:
            gap_info = self.current_analyzer.get_band_gap()
            self.control_panel.update_analysis(gap_info)
            
    def set_energy_range(self, emin: float, emax: float):
        """设置能量范围"""
        self.band_widget.set_energy_range(emin, emax)
        
    def export_figure(self, fmt: str):
        """导出图片"""
        if not self.current_file:
            QMessageBox.warning(self, "Warning", "No data loaded!")
            return
            
        default_name = self.current_file.stem + f"_band.{fmt}"
        filepath, _ = QFileDialog.getSaveFileName(
            self, f"Save {fmt.upper()}", default_name,
            f"{fmt.upper()} (*.{fmt})"
        )
        if filepath:
            dpi = 300 if fmt == 'png' else 150
            self.band_widget.save_figure(filepath, dpi)
            QMessageBox.information(self, "Success", f"Saved to:\n{filepath}")