"""Run with -I to verify installed packages outside the source import path."""
import argparse
import importlib.metadata
from pathlib import Path
import os
import time


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--workdir', required=True)
    parser.add_argument('--eigenval', required=True)
    args = parser.parse_args()
    workdir = Path(args.workdir).resolve()
    workdir.mkdir(parents=True, exist_ok=True)
    eigenval = str(Path(args.eigenval).resolve())
    os.chdir(workdir)
    os.environ['BANDVIZ_DATA_DIR'] = str(workdir / 'app-data')
    os.environ['QT_QPA_PLATFORM'] = 'offscreen'
    os.environ['MPLBACKEND'] = 'Agg'

    import core
    import vdw_studio
    import gui
    for module in (core, vdw_studio, gui):
        assert 'site-packages' in Path(module.__file__).parts, module.__file__
    distribution = importlib.metadata.distribution('band-structure-2d')
    for name in ('bandviz', 'vdw-studio', 'vdw-studio-cli'):
        entry = next(ep for ep in distribution.entry_points if ep.name == name)
        assert callable(entry.load())

    from PyQt5.QtWidgets import QApplication
    from gui.main_window import MainWindow as BandViz
    from vdw_studio.gui.main_window import MainWindow as Studio
    app = QApplication([])
    app.setQuitOnLastWindowClosed(False)

    def wait(condition, timeout=90):
        deadline = time.monotonic() + timeout
        while not condition():
            app.processEvents()
            if time.monotonic() > deadline:
                raise AssertionError('Installed GUI task timed out')
            time.sleep(.01)
        app.processEvents()

    viewer = BandViz()
    viewer.show()
    viewer._load_eigenval(eigenval)
    wait(lambda: viewer.current_data is not None)
    assert viewer.current_data.nkpoints == 58
    assert viewer.current_dos_data is not None
    viewer.close()
    wait(lambda: not viewer.isVisible())

    studio = Studio()
    studio.show()
    studio.run_simulation()
    wait(lambda: bool(studio._results) and not studio._poll_timer.isActive())
    assert studio._results['snapshot'].material_key == 'mos2_kp'
    assert abs(studio._results['gap'].gap - 1.67) < 1e-8
    studio.close()
    wait(lambda: not studio.isVisible())
    print('Installed BandViz import and default vdW Studio task passed')


if __name__ == '__main__':
    main()
