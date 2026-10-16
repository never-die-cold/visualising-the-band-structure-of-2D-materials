"""费米参考契约：能带图、DOS 与带隙分析必须共用同一个占据数。

非半填模型的教训（sp³d⁵ 单层 MoS₂ 占 54 态中的 18 条）：绘图与 DOS 曾
各自按半填推断零点（第 27 条带顶），而带隙分析用文档化的 18 —— 同一次
运行里图上的"费米面"与报告的带隙不在同一基准，且没有任何测试发现。
本文件固化统一后的契约：占据数由 :func:`vdw_studio.numerics.resolve_n_valence`
单点解析（显式参数 → 模型声明 → 实际带数半填）。
"""

import numpy as np
import pytest

from vdw_studio.analysis import analyze_path_gap
from vdw_studio.engine.kpath import KPath
from vdw_studio.engine.models import HoneycombModel
from vdw_studio.engine.solver import solve_bands, solve_dos
from vdw_studio.engine.zahid_mos2 import ZahidMoS2Model


class MisleadingSiteCount:
    """四个显式态，但 n_sites 与希尔伯特维度无关（见 test_dos_contract）。"""

    name = 'four states, misleading n_sites'
    n_sites = 999

    def bands(self, k):
        return np.tile([-2., -1., 1., 2.], (len(k), 1))


class TestValenceResolution:
    def test_half_filling_fallback_keeps_previous_reference(self):
        """无声明模型仍按实际带数半填：第 1 条带顶（石墨烯半填）。"""
        g = HoneycombModel(a=2.46, t=-2.7)
        bs = solve_bands(g, KPath.for_lattice(g.lattice), n_per_segment=20)
        assert bs.n_valence == 1
        assert bs.fermi_level() == pytest.approx(float(bs.energies[:, 0].max()))

    def test_site_count_is_not_used_as_filling(self):
        """n_sites 与带数不符时不得用它推算占据数（half of 999 会越界）。"""
        dos = solve_dos(MisleadingSiteCount(), mesh=(4, 4), sigma=.1, n_points=101)
        assert dos.efermi == pytest.approx(-1.0)   # 第 2 条带顶，而非 999//2 推导

    def test_explicit_argument_overrides_declaration(self):
        m = ZahidMoS2Model(soc=False)
        bs = solve_bands(m, KPath.for_lattice(m.lattice), 4, n_valence=20)
        assert bs.n_valence == 20

    def test_over_filling_is_rejected(self):
        m = ZahidMoS2Model(soc=False)
        with pytest.raises(ValueError):
            solve_bands(m, KPath.for_lattice(m.lattice), 4, n_valence=54)


class TestDeclaredFillingReference:
    """sp³d⁵ 模型：18/54 占据由模型自己声明，全流程共用。"""

    def test_model_declares_neutral_filling(self):
        assert ZahidMoS2Model.n_valence == 18

    def test_band_reference_is_declared_filling(self):
        m = ZahidMoS2Model(soc=False)
        bs = solve_bands(m, KPath.for_lattice(m.lattice), 4)
        assert bs.n_valence == 18
        assert bs.fermi_level() == pytest.approx(float(bs.energies[:, 17].max()))

    def test_plot_reference_matches_gap_analysis(self):
        """能带图的能量零点必须等于路径带隙所用的 VBM。"""
        m = ZahidMoS2Model(soc=False)
        kpath = KPath.for_lattice(m.lattice)
        bs = solve_bands(m, kpath, 4)
        gap = analyze_path_gap(m, kpath, n_per_segment=4, band=bs)
        assert gap.n_valence == 18
        assert gap.vbm == pytest.approx(bs.fermi_level())

    def test_dos_reference_is_declared_filling(self):
        """DOS 零点 = 第 18 带顶，而不是半填的第 27 条带顶。"""
        m = ZahidMoS2Model(soc=False)
        mesh = (4, 4)
        dos = solve_dos(m, mesh=mesh, sigma=.1, n_points=101)
        n1, n2 = mesh
        kf = np.array([[(i + .5) / n1, (j + .5) / n2]
                       for i in range(n1) for j in range(n2)])
        eig = m.bands(kf)
        assert dos.efermi == pytest.approx(float(eig[:, 17].max()))
        assert dos.efermi < float(eig[:, 26].max())
