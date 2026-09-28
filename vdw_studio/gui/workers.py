"""后台仿真线程。

实现说明：本模块刻意**不使用 Qt 信号**（pyqtSignal 的 emit 在部分
PyQt5 5.15.11 + Python 3.12 组合下存在 "native Qt signal is not
callable" 兼容性问题，包括树莓派常见的安装组合）。改用结果盒 +
主线程 QTimer 轮询模式：

- :meth:`SimulationWorker.run` 在子线程执行全部计算，把进度文本、
  最终结果与异常写入普通 Python 属性（GIL 保证赋值原子性）；
- 主窗口用 ``QTimer`` 周期读取 :attr:`new_progress` / :attr:`done`
  并在**主线程**更新界面（Qt 界面操作永远留在主线程）。

对测试而言，直接同步调用 ``run()`` 即可获得 ``result``，无需事件循环。
"""

from __future__ import annotations

import traceback
from typing import Optional

from PyQt5.QtCore import QThread

from ..analysis.properties import GapResult, analyze_gap
from ..engine.kpath import KPath
from ..engine.solver import BandStructure, DOSResult, solve_bands, solve_dos


class SimulationWorker(QThread):
    """能带 + DOS + 带隙分析的后台线程（轮询模式，无 Qt 信号）。"""

    def __init__(self, model, n_per_segment: int = 40,
                 dos_mesh: tuple = (48, 48), dos_sigma: float = 0.05,
                 n_valence: Optional[int] = None, parent=None) -> None:
        super().__init__(parent)
        self.model = model
        self.n_per_segment = n_per_segment
        self.dos_mesh = dos_mesh
        self.dos_sigma = dos_sigma
        self.n_valence = n_valence

        self.progress_messages: list = []   # 子线程写, 主线程读
        self.result: Optional[dict] = None
        self.error: Optional[str] = None
        self.done: bool = False

    # ------------------------------------------------------------------
    def log_progress(self, msg: str) -> None:
        self.progress_messages.append(msg)

    def compute(self) -> dict:
        """完整计算流程（可在任意线程同步调用）。"""
        self.log_progress("生成 k 路径…")
        kpath = KPath.for_lattice(self.model.lattice)
        self.log_progress("求解能带…")
        band: BandStructure = solve_bands(self.model, kpath,
                                          self.n_per_segment)
        self.log_progress("计算态密度…")
        dos: DOSResult = solve_dos(self.model, mesh=self.dos_mesh,
                                   sigma=self.dos_sigma)
        self.log_progress("分析带隙与有效质量…")
        gap: GapResult = analyze_gap(self.model, kpath,
                                     n_valence=self.n_valence)
        return {"band": band, "dos": dos, "gap": gap, "kpath": kpath}

    def run(self) -> None:  # noqa: D102
        try:
            self.result = self.compute()
        except Exception as exc:  # noqa: BLE001 — 界面需要完整错误上报
            self.error = (f"{type(exc).__name__}: {exc}\n"
                          f"{traceback.format_exc()}")
        finally:
            self.done = True
