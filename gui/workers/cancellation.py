"""QThread 协作取消的公共部分。

``ParseWorker`` 与 ``DosWorker`` 此前各自维护一份完全相同的
``_cancel_requested`` / ``cancel()`` / ``check_interruption()``。
子类在 ``__init__`` 末尾调用 :meth:`init_cancellation`，并按需覆盖
``cancel_message``。
"""

from threading import Event


class CancelableWorker:
    """为 QThread 子类提供请求标志 + Qt 中断请求的统一检查。"""

    cancel_message = 'Cancelled'

    def init_cancellation(self):
        self._cancel_requested = Event()

    @property
    def cancel_requested(self) -> bool:
        return self._cancel_requested.is_set()

    def cancel(self):
        """请求停止：设置标志并发出 Qt 中断请求。"""
        self._cancel_requested.set()
        self.requestInterruption()

    def check_interruption(self):
        """在耗时循环的检查点调用；已请求取消时抛出 InterruptedError。"""
        if self._cancel_requested.is_set() or self.isInterruptionRequested():
            raise InterruptedError(self.cancel_message)
