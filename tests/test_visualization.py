"""可视化模块测试（Agg 无头后端）。"""

import matplotlib
matplotlib.use("Agg")

import numpy as np
import pytest

from vdw_studio.analysis import analyze_gap
from vdw_studio.engine.kpath import KPath
from vdw_studio.engine.models import HoneycombModel, PhosphoreneRudenko
from vdw_studio.engine.solver import solve_bands, solve_dos
from vdw_studio.structure.builders import graphene, phosphorene, tmd
from vdw_studio.visualization import (
    brillouin_zone_polygon,
    plot_band_structure,
    plot_bz_path,
    plot_dos,
    plot_structure,
    polygon_area,
)


class TestBrillouinZone:
    def test_hexagonal_bz_shape_and_area(self):
        """六方 BZ：六边形，面积 = (2π)²/A_cell = 8π²/(√3 a²)。"""
        a = 2.46
        poly = brillouin_zone_polygon(HoneycombModel(a=a).lattice)
        assert poly.shape[0] == 6
        expected = 8 * np.pi ** 2 / (np.sqrt(3) * a ** 2)
        assert polygon_area(poly) == pytest.approx(expected, rel=1e-9)

    def test_rectangular_bz(self):
        a, b = 3.3136, 4.3763
        lat = phosphorene().lattice
        poly = brillouin_zone_polygon(lat)
        assert poly.shape[0] == 4
        expected = 4 * np.pi ** 2 / (a * b)
        assert polygon_area(poly) == pytest.approx(expected, rel=1e-9)

    def test_bz_plot(self):
        fig, ax = plot_bz_path(HoneycombModel(2.46).lattice,
                               KPath.for_lattice(HoneycombModel(2.46).lattice))
        assert len(ax.patches) == 1
        import matplotlib.pyplot as plt
        plt.close(fig)


class TestBandPlot:
    def test_lines_and_ticks(self):
        g = HoneycombModel(a=2.46, t=-2.7)
        bs = solve_bands(g, KPath.for_lattice(g.lattice), n_per_segment=12)
        fig, ax = plot_band_structure(bs)
        # 2 条能带 + 2 条 tick 竖线 + 1 条费米线
        assert len(ax.lines) == 2 + (len(bs.ticks) - 2) + 1
        assert [t.get_text() for t in ax.get_xticklabels()] == bs.tick_labels
        import matplotlib.pyplot as plt
        plt.close(fig)

    def test_energy_window(self):
        g = HoneycombModel(a=2.46, t=-2.7)
        bs = solve_bands(g, KPath.for_lattice(g.lattice), n_per_segment=10)
        fig, ax = plot_band_structure(bs, emin=-2, emax=2)
        ylim = ax.get_ylim()
        assert ylim[0] == pytest.approx(-2) and ylim[1] == pytest.approx(2)
        import matplotlib.pyplot as plt
        plt.close(fig)


class TestDosPlot:
    def test_dos_figure(self):
        p = PhosphoreneRudenko()
        dos = solve_dos(p, mesh=(16, 16), sigma=0.05, n_points=300)
        fig, ax = plot_dos(dos)
        assert len(ax.lines) >= 2   # DOS 曲线 + 费米线
        import matplotlib.pyplot as plt
        plt.close(fig)


class TestStructurePlot:
    def test_graphene_supercell_atoms(self):
        fig, ax = plot_structure(graphene(), supercell=(3, 3, 1))
        # 3×3 超胞 = 18 个原子，散点按元素分组绘制
        from mpl_toolkits.mplot3d.art3d import Path3DCollection
        scats = [c for c in ax.collections
                 if isinstance(c, Path3DCollection)]
        assert len(scats) == 1
        assert scats[0].get_offsets().shape[0] == 18
        import matplotlib.pyplot as plt
        plt.close(fig)

    def test_tmd_colors(self):
        fig, ax = plot_structure(tmd("MoS2"), supercell=(2, 2, 1))
        from mpl_toolkits.mplot3d.art3d import Path3DCollection
        scats = [c for c in ax.collections
                 if isinstance(c, Path3DCollection)]
        assert len(scats) == 2   # Mo 与 S 两种颜色
        import matplotlib.pyplot as plt
        plt.close(fig)

    def test_max_atoms_guard(self):
        with pytest.raises(ValueError):
            plot_structure(graphene(), supercell=(20, 20, 1))
