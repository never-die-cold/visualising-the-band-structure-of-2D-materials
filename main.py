import sys
from PyQt5.QtWidgets import QApplication
from gui.main_window import MainWindow
from utils.logger import setup_logger


def main():
    # 初始化日志
    logger = setup_logger("bandviz")
    logger.info("=" * 40)
    logger.info("2D Material Band Structure Visualizer")
    logger.info("=" * 40)

    app = QApplication(sys.argv)
    app.setStyle('Fusion')

    window = MainWindow()
    window.show()

    sys.exit(app.exec_())


if __name__ == '__main__':
    main()
