# -*- coding: utf-8 -*-
"""离屏运行软件并截取主界面"""
import os, sys
from pathlib import Path
ROOT = Path(r"C:\Users\ASUS\visualising the band structure of 2D materials")
sys.path.insert(0, str(ROOT))

from PyQt5.QtWidgets import QApplication
from PyQt5.QtCore import QTimer
from PyQt5.QtGui import QFont
from gui.main_window import MainWindow

app = QApplication(sys.argv)
app.setStyle("Fusion")
app.setFont(QFont("Microsoft YaHei", 9))
win = MainWindow()
win.resize(1280, 800)
win.move(-3000, 100)  # 移到屏幕外，避免打扰用户
win.show()

state = {"loaded": False}

def load():
    win._load_eigenval(str(ROOT / "data" / "example" / "mos2" / "EIGENVAL"))

def grab():
    # 等后台线程解析完成后再截图
    if getattr(win, "current_data", None) is None and not state["loaded"]:
        QTimer.singleShot(500, grab)
        return
    pix = win.grab()
    out = str(ROOT / "paper_figures" / "fig_gui_main.png")
    pix.save(out)
    print("saved", out)
    app.quit()

QTimer.singleShot(800, load)
QTimer.singleShot(1500, grab)
QTimer.singleShot(20000, app.quit)  # 兜底退出
sys.exit(app.exec_())
