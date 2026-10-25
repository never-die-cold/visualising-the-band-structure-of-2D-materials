"""Cancelable spectrum refresh; holds captured input and display parameters."""
import traceback
from PyQt5.QtCore import QThread, pyqtSignal
from core.dos_analyzer import load_spectrum
from .cancellation import CancelableWorker


class DosWorker(CancelableWorker, QThread):
    completed = pyqtSignal(object)
    error = pyqtSignal(str)
    cancel_message = 'Spectrum refresh cancelled'

    def __init__(self, data, path, request_id, *, fermi_level, energy_range,
                 sigma, state_capacity, parent=None):
        super().__init__(parent)
        self.data, self.path, self.request_id = data, path, request_id
        self.parameters = dict(fermi_level=fermi_level, energy_range=tuple(energy_range),
                               sigma=sigma, state_capacity=state_capacity)
        self.init_cancellation()

    def run(self):
        try:
            result = load_spectrum(self.data, self.path, **self.parameters, cancel_check=self.check_interruption)
            self.check_interruption()
            self.completed.emit(result)
        except InterruptedError:
            pass
        except Exception as exc:
            self.error.emit(f"{type(exc).__name__}: {exc}\n"
                            f"{traceback.format_exc()}")
