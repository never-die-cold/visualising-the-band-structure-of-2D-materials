"""
Background worker threads for parsing and computation.
Uses PyQt5 QThread to keep the GUI responsive.
"""

from PyQt5.QtCore import QThread, pyqtSignal
from pathlib import Path

from core.parser import VASPEigenvalParser
from core.band_analyzer import BandAnalyzer
from core.dos_analyzer import DosAnalyzer


class ParseWorker(QThread):
    """
    后台解析线程：解析 EIGENVAL + KPOINTS，计算 DOS。
    信号:
        progress(msg) -> 进度消息
        finished(data, analyzer, dos_data) -> 完成
        error(msg)    -> 错误
    """

    progress = pyqtSignal(str)
    finished = pyqtSignal(object, object, object)
    error = pyqtSignal(str)

    def __init__(self, eigenval_path: str, kpoints_path: str = None,
                 efermi: float = 0.0, emin: float = -5.0, emax: float = 5.0,
                 dos_sigma: float = 0.05, parent=None):
        super().__init__(parent)
        self.eigenval_path = eigenval_path
        self.kpoints_path = kpoints_path
        self.efermi = efermi
        self.emin = emin
        self.emax = emax
        self.dos_sigma = dos_sigma

    def run(self):
        try:
            self.progress.emit("Parsing EIGENVAL...")

            parser = VASPEigenvalParser(
                self.eigenval_path,
                self.kpoints_path
            )
            band_data = parser.parse()

            self.progress.emit("Analyzing band structure...")
            analyzer = BandAnalyzer(band_data)
            analyzer.set_fermi_level(self.efermi)

            self.progress.emit("Calculating DOS...")
            dos_analyzer = DosAnalyzer(band_data.energies, fermi_level=self.efermi)
            dos_data = dos_analyzer.calculate_dos(
                energy_range=(self.emin, self.emax),
                num_points=800,
                sigma=self.dos_sigma
            )

            self.progress.emit("Done")
            self.finished.emit(band_data, analyzer, dos_data)

        except Exception as e:
            self.error.emit(str(e))
