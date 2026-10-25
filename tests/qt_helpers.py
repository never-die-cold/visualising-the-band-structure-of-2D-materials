"""共享的 Qt 测试助手（事件循环等待、隔离的根窗口）。

此前 ``wait_until`` 住在 ``test_task_state``、``root_window`` 住在
``test_lifecycle``，其余测试模块跨文件导入——与 vasp_fixtures 同样的
隐藏耦合问题。
"""


def wait_until(app, predicate, milliseconds=15000):
    """轮询事件循环直到 ``predicate`` 为真；超时则以断言失败收场。"""
    from PyQt5.QtCore import QEventLoop, QTimer
    loop, poll, timeout = QEventLoop(), QTimer(), QTimer()
    timeout.setSingleShot(True)
    timeout.timeout.connect(loop.quit)
    poll.timeout.connect(lambda: loop.quit() if predicate() else None)
    poll.start(20)
    timeout.start(milliseconds)
    loop.exec_()
    poll.stop()
    timeout.stop()
    assert predicate(), 'Timed out waiting for GUI state'


def root_window(tmp_path, monkeypatch):
    """构造配置与数据库都隔离到 ``tmp_path``、无示例数据的 MainWindow。"""
    import gui.main_window as module
    from storage.config_manager import ConfigManager
    from storage.database import Database
    monkeypatch.setattr(module, 'ConfigManager', lambda *args: ConfigManager(str(tmp_path / 'settings.json')))
    monkeypatch.setattr(module, 'Database', lambda *args: Database(str(tmp_path / 'test.db')))
    monkeypatch.setattr(module.MainWindow, '_init_example_data', lambda self: None)
    return module.MainWindow()
