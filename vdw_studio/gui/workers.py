"""后台仿真线程。

实现说明：本模块刻意**不使用 Qt 信号**（pyqtSignal 的 emit 在部分
PyQt5 5.15.11 + Python 3.12 组合下存在 "native Qt signal is not
callable" 兼容性问题，包括树莓派常见的安装组合）。改用结果盒 +
主线程 QTimer 轮询模式：

- :meth:`SimulationWorker.run` 在子线程执行全部计算，把进度文本、
  最终结果与异常写入普通 Python 属性（GIL 保证赋值原子性）；
- 主窗口用 ``QTimer`` 周期读取 :attr:`progress_messages` / :attr:`done`
  并在**主线程**更新界面（Qt 界面操作永远留在主线程）。

对测试而言，直接同步调用 ``run()`` 即可获得 ``result``，无需事件循环。
"""

from __future__ import annotations

import traceback
from typing import Optional
from copy import deepcopy
from threading import Event

from PyQt5.QtCore import QThread

from ..simulation import simulate
from ..task_state import encode_model_state


class SimulationWorker(QThread):
    """按模型能力计算能带、DOS 与性质的后台线程。"""

    def __init__(self, model, n_per_segment: int = 40,
                 dos_mesh: tuple = (48, 48), dos_sigma: float = 0.05,
                 n_valence: Optional[int] = None, parent=None, *,
                 lattice=None, include_valley: bool = False, snapshot=None) -> None:
        super().__init__(parent)
        self.model = deepcopy(model)
        if snapshot is not None and encode_model_state(self.model) != snapshot.model_state_json:
            raise ValueError('Runtime model does not match the immutable task snapshot')
        self.n_per_segment = n_per_segment if snapshot is None else snapshot.n_per_segment
        self.dos_mesh = tuple(dos_mesh) if snapshot is None else snapshot.dos_mesh
        self.dos_sigma = dos_sigma if snapshot is None else snapshot.dos_sigma
        self.n_valence = n_valence if snapshot is None else snapshot.n_valence
        self.lattice = deepcopy(lattice) if snapshot is None else snapshot.make_structure().lattice
        self.include_valley = include_valley if snapshot is None else snapshot.include_valley
        self.snapshot = snapshot
        self._cancel_requested = Event()
        self.cancelled = False

        self.progress_messages: list = []   # 子线程写, 主线程读
        self.result: Optional[dict] = None
        self.error: Optional[str] = None
        self.done: bool = False

    # ------------------------------------------------------------------
    def log_progress(self, msg: str) -> None:
        self.progress_messages.append(msg)

    def cancel(self):
        self._cancel_requested.set()
        self.requestInterruption()

    def check_interruption(self):
        if self._cancel_requested.is_set() or self.isInterruptionRequested():
            raise InterruptedError('Simulation cancelled')

    def compute(self) -> dict:
        """完整计算流程（可在任意线程同步调用）。"""
        result = simulate(
            self.model, self.n_per_segment, self.dos_mesh, self.dos_sigma,
            self.n_valence, lattice=self.lattice,
            include_valley=self.include_valley,
            gap_mesh=(24, 24) if self.snapshot is None else self.snapshot.gap_mesh,
            cancel_check=self.check_interruption,
            on_progress=self.log_progress)
        if self.snapshot is not None:
            result['snapshot'] = self.snapshot
            result['task_id'] = self.snapshot.task_id
            result['structure'] = self.snapshot.make_structure()
        return result

    def run(self) -> None:  # noqa: D102
        try:
            self.result = self.compute()
            self.check_interruption()
        except InterruptedError:
            self.result = None
            self.cancelled = True
        except Exception as exc:  # noqa: BLE001 — 界面需要完整错误上报
            self.error = (f"{type(exc).__name__}: {exc}\n"
                          f"{traceback.format_exc()}")
        finally:
            self.done = True
