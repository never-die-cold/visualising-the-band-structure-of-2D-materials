"""GUI 冒烟测试（QT_QPA_PLATFORM=offscreen 无头运行）。

验证：主窗口可构建、材料切换/结构绘制/工程树填充正常、
一次完整仿真（真实求解）能在后台线程完成后更新结果页。
"""

import os

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import pytest

pytest.importorskip("PyQt5.QtWidgets")

from vdw_studio.gui.main_window import MainWindow
from vdw_studio.gui.workers import SimulationWorker
from vdw_studio.presets import list_presets


@pytest.fixture(scope="module")
def app():
    from PyQt5.QtWidgets import QApplication
    app = QApplication.instance() or QApplication([])
    yield app


class TestMainWindow:
    def test_create_and_load_default(self, app):
        win = MainWindow()
        assert win.preset.key == "mos2_kp"
        assert win.structure.formula_str == "MoS2"
        assert win.model is not None
        assert win.tree.topLevelItemCount() == 3
        win.close()

    @pytest.mark.parametrize("key", list_presets())
    def test_switch_material(self, app, key):
        win = MainWindow()
        win.set_material(key)
        p = win.preset
        assert p.key == key
        assert win.structure is not None and win.model is not None
        win.close()

    def test_silicene_field_changes_model(self, app):
        win = MainWindow()
        win.set_material("silicene")
        win.field_spin.setValue(0.0)
        win._rebuild_model()
        gap0 = win.model.energies_at((1 / 3, 1 / 3))
        win.field_spin.setValue(1.0)
        win._rebuild_model()
        gap1 = win.model.energies_at((1 / 3, 1 / 3))
        assert (gap1[1] - gap1[0]) > (gap0[1] - gap0[0])   # 电场开隙
        win.close()

    def test_full_simulation_updates_results(self, app):
        """完整仿真流程（同步调用 compute，覆盖真实计算与结果页更新）。"""
        win = MainWindow()
        win.set_material("phosphorene")
        win._rebuild_model()
        worker = SimulationWorker(
            win.model, n_per_segment=12, dos_mesh=(12, 12),
            n_valence=win.preset.n_valence)
        results = worker.compute()          # 同步执行完整计算
        assert "gap" in results
        gap = results["gap"]
        assert gap.gap is not None
        assert gap.gap == pytest.approx(1.52, abs=0.02)
        win._on_results(results)
        assert win.result_item.childCount() >= 2
        win.close()

    def test_bilayer_field_opens_gap(self, app):
        """双层石墨烯预设：GUI 电场输入线性开隙（最小模型 E_g ≈ |U|）。"""
        win = MainWindow()
        win.set_material("bilayer_graphene")
        assert win.structure.formula_str == "C4"
        win.field_spin.setValue(0.5)
        win._rebuild_model()
        e = win.model.energies_at((1 / 3, 1 / 3))
        assert (e[2] - e[1]) == pytest.approx(0.5, abs=1e-8)
        win.close()

    def test_strain_changes_model(self, app):
        """GUI 应变输入 → 模型晶格实际缩放（仅 TB 引擎生效）。"""
        win = MainWindow()
        win.set_material("graphene")
        win.strain_x_spin.setValue(2.0)
        win.strain_y_spin.setValue(2.0)
        win._rebuild_model()
        assert win.model.lattice.parameters()[0] == pytest.approx(2.46 * 1.02)
        # hopping 已按 t∝d⁻² 重整
        assert win.model.hoppings[0].t == pytest.approx(-2.7 * 1.02 ** -2)
        # k·p 引擎不受应变输入影响
        win.set_material("mos2_kp")
        win.strain_x_spin.setValue(5.0)
        win.strain_y_spin.setValue(5.0)
        win._rebuild_model()
        gaps = [win.model.spin_block_energies((0, 0), +1, s)[1]
                - win.model.spin_block_energies((0, 0), +1, s)[0]
                for s in (+1, -1)]
        assert min(gaps) == pytest.approx(1.67, abs=1e-6)
        win.close()

    def test_valley_tab_visible_only_for_kp(self, app):
        """谷物理标签页仅 k·p 预设可见。"""
        win = MainWindow()
        win.set_material("mos2_kp")
        vidx = win.tabs.indexOf(win.valley_canvas)
        assert win.tabs.isTabVisible(vidx)
        win.set_material("graphene")
        assert not win.tabs.isTabVisible(vidx)
        win.close()

    def test_worker_thread_mode(self, app):
        """线程模式：run() 在子线程完成计算并填充结果盒（轮询模式）。"""
        win = MainWindow()
        win.set_material("hbn")
        win._rebuild_model()
        worker = SimulationWorker(win.model, n_per_segment=10,
                                  dos_mesh=(10, 10),
                                  n_valence=win.preset.n_valence)
        worker.start()
        worker.wait(60000)
        assert worker.done and worker.error is None
        assert worker.result["gap"].gap == pytest.approx(3.5, abs=1e-6)
        win.close()
