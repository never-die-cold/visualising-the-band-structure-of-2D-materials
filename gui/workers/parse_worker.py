"""
Background worker threads for parsing and computation.
Uses PyQt5 QThread to keep the GUI responsive.
"""

from PyQt5.QtCore import QThread, pyqtSignal
from pathlib import Path
from threading import Event
from uuid import uuid4

from core.parser import VASPEigenvalParser
from core.band_analyzer import BandAnalyzer
from core.dos_analyzer import load_spectrum


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
    cancelled = pyqtSignal()

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
        self.task_id = str(uuid4())
        self._cancel_requested = Event()

    def cancel(self):
        self._cancel_requested.set()
        self.requestInterruption()

    def check_interruption(self):
        if self._cancel_requested.is_set() or self.isInterruptionRequested():
            raise InterruptedError('Import cancelled')

    def run(self):
        try:
            self.progress.emit("Parsing EIGENVAL...")

            parser = VASPEigenvalParser(
                self.eigenval_path,
                self.kpoints_path, cancel_check=self.check_interruption
            )
            band_data = parser.parse()

            self.progress.emit("Analyzing band structure...")
            analyzer = BandAnalyzer(band_data)
            analyzer.set_fermi_level(self.efermi)
            analyzer.initial_mass_reports = analyzer.mass_fit_reports_at_gap()
            self.check_interruption()

            self.progress.emit("Calculating DOS...")
            dos_data = load_spectrum(
                band_data, self.eigenval_path, fermi_level=self.efermi,
                state_capacity=analyzer.state_capacity,
                cancel_check=self.check_interruption,
                energy_range=(self.emin, self.emax),
                num_points=800,
                sigma=self.dos_sigma
            )

            self.progress.emit("Done")
            self.check_interruption()
            self.finished.emit(band_data, analyzer, dos_data)

        except InterruptedError:
            self.cancelled.emit()

        except Exception as e:
            self.error.emit(str(e))
