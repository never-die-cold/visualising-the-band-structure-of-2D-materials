"""Cancelable spectrum refresh; holds captured input and display parameters."""
from threading import Event
from PyQt5.QtCore import QThread, pyqtSignal
from core.dos_analyzer import load_spectrum


class DosWorker(QThread):
    completed = pyqtSignal(object)
    error = pyqtSignal(str)

    def __init__(self, data, path, request_id, *, fermi_level, energy_range,
                 sigma, state_capacity, parent=None):
        super().__init__(parent)
        self.data, self.path, self.request_id = data, path, request_id
        self.parameters = dict(fermi_level=fermi_level, energy_range=tuple(energy_range),
                               sigma=sigma, state_capacity=state_capacity, num_points=800)
        self._cancel_requested = Event()

    def cancel(self):
        self._cancel_requested.set()
        self.requestInterruption()

    def check_interruption(self):
        if self._cancel_requested.is_set() or self.isInterruptionRequested():
            raise InterruptedError('Spectrum refresh cancelled')

    def run(self):
        try:
            result = load_spectrum(self.data, self.path, **self.parameters, cancel_check=self.check_interruption)
            self.check_interruption()
            self.completed.emit(result)
        except InterruptedError:
            pass
        except Exception as exc:
            self.error.emit(str(exc))
