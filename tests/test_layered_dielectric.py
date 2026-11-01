"""层状介质屏蔽相互作用与 Ovesen 2019 文献对账测试。

模型：Ovesen, Brem, Malic, Commun. Phys. 2, 23 (2019) 的介质板堆叠
（Supplementary Note 1；PDF 在 papers/ovesen2019/）。实现为 Poisson
方程的逐区指数展开直接解（``analysis.layered_dielectric``），配
导纳递推稳定化（大 k 不溢出）。

锚点分层：
1. 均匀介质 → 解析 2D 库仑（传输矩阵解的逐点对账）；
2. ε(k) 极限：k→0 → ε_top+ε_bottom；层内大 k → 2ε_l（bulk，
   float64 精确）；
3. 导纳递推 vs 矩阵解（k ≤ 1，矩阵解的可信域）；
4. 终锚（文献对账）：用 Ovesen 的介质参数 + Kormányos Table 3/4
   平均带质量，三条通道的 1s 束缚能复现原文 Table 1：
   层间 161 vs 150 meV、层内 216/195 vs 205/177 meV（+4~11%；
   容差 20% 覆盖堆叠顺序歧义与质量的不确定性）。
"""

import numpy as np
import pytest

from vdw_studio.analysis.layered_dielectric import (
    DielectricSlab,
    LayeredDielectricStack,
    E2_OVER_EPS0,
    dielectric_function_k,
    interaction_real_space,
    ovesen_stack,
    ovesen_slab_index,
    screened_interaction_k,
    _solve_potential_k,
)
from vdw_studio.analysis.exciton import solve_exciton_with_potential


def _uniform_stack(eps: float = 4.9) -> LayeredDielectricStack:
    return LayeredDielectricStack(
        top_env_epsilon=eps, bottom_env_epsilon=eps,
        slabs=(DielectricSlab(eps, 6.45), DielectricSlab(eps, 6.45)))


class TestBVPSolver:
    def test_uniform_medium_matches_analytic_coulomb(self):
        """所有 ε 相同 → 逐区解必须退回解析 2D 库仑 e^{−kΔz}E2/(2εk)。"""
        stack = _uniform_stack()
        ks = np.array([0.05, 0.2, 1.0])
        v = screened_interaction_k(stack, ks, 0, 1)
        v_ana = np.exp(-ks * 6.45) * E2_OVER_EPS0 / (2 * 4.9 * ks)
        np.testing.assert_allclose(v, v_ana, rtol=1e-12)

    def test_eps_k_zero_limit(self):
        """ε(k→0) → ε_top + ε_bottom（电荷场的两介质分摊）。"""
        stack = ovesen_stack(mose2_below=True)
        i_mo = ovesen_slab_index("MoSe2", True)
        eps_k = dielectric_function_k(stack, np.array([5e-4]), i_mo, i_mo)
        assert eps_k[0] == pytest.approx(1.0 + 3.9, rel=0.05)

    def test_intralayer_bulk_limit_exact(self):
        """层内大 k：ε(k) → 2ε_l（bulk），float64 逐位复现。"""
        stack = ovesen_stack(mose2_below=True)
        i_mo = ovesen_slab_index("MoSe2", True)
        ks = np.array([10.0, 25.0, 40.0])
        eps_k = dielectric_function_k(stack, ks, i_mo, i_mo)
        np.testing.assert_allclose(eps_k, 2 * 16.5, rtol=1e-10)

    def test_admittance_vs_matrix_cross_check(self):
        """导纳递推 vs 直接矩阵解（k ≤ 1 为矩阵解可信域）。"""
        stack = ovesen_stack(mose2_below=True)
        i_mo, i_ws = ovesen_slab_index("MoSe2", True), ovesen_slab_index("WSe2", True)
        regions, centers = stack.regions_for(i_mo)
        for k in (0.05, 0.2, 0.5, 1.0):
            v_mat = _solve_potential_k(regions, k, centers[i_mo], centers[i_ws])
            v_adm = screened_interaction_k(stack, [k], i_mo, i_ws)[0]
            assert v_adm == pytest.approx(v_mat, rel=1e-9)

    def test_interlayer_positive_and_decaying_in_k(self):
        """层间 V(k) 恒正且随 k 衰减。

        验证域分两段：k ≤ 3 由矩阵解逐点对账（导纳可信域）；
        k ≥ 10 为完全饱和区（g 钉在 ±1，纯指数）。k ∈ [4, 8] 的
        过渡带在 float64 下不可用——源平面 g 的 next-digit 结构
        （O(e^{−2kd})）低于机器精度，形状比被界面重标度的 |g|>1
        污染——层间 Hankel 因此截断在 k_max = 3（尾项 e^{−kΔz} ≤
        1e-11 相对贡献，见 interaction_real_space 的 docstring）。
        """
        stack = ovesen_stack(mose2_below=True)
        i_mo, i_ws = ovesen_slab_index("MoSe2", True), ovesen_slab_index("WSe2", True)
        ks = np.array([1.0, 2.0, 3.0, 10.0, 25.0])
        v = screened_interaction_k(stack, ks, i_mo, i_ws)
        assert np.all(v > 0)
        assert np.all(np.diff(v) < 0)

    def test_real_space_is_attractive(self):
        """实空间势（e-h 相互作用约定）全负且随 r 单调趋近 0。"""
        stack = ovesen_stack(mose2_below=True)
        i_mo, i_ws = ovesen_slab_index("MoSe2", True), ovesen_slab_index("WSe2", True)
        r = np.linspace(0.5, 300.0, 60)
        v = interaction_real_space(stack, r, i_mo, i_ws)
        assert np.all(v < 0)
        assert np.all(np.diff(v) > 0)          # 负值向 0 单调

    def test_k_max_truncation_insensitive_interlayer(self):
        """层间通道的 k_max 截断不敏感（e^{−kΔz} 使尾项可忽略）。"""
        stack = ovesen_stack(mose2_below=True)
        i_mo, i_ws = ovesen_slab_index("MoSe2", True), ovesen_slab_index("WSe2", True)
        r = np.linspace(0.5, 100.0, 40)
        v25 = interaction_real_space(stack, r, i_mo, i_ws, k_max=2.5)
        v3 = interaction_real_space(stack, r, i_mo, i_ws, k_max=3.0)
        np.testing.assert_allclose(v25, v3, rtol=1e-6, atol=1e-8)


class TestOvesenReconciliation:
    """终锚：复现 Ovesen 2019 Table 1（MoSe₂/WSe₂ 异质结构，SiO₂ 衬底）。"""

    @pytest.fixture(scope="class")
    def bindings(self):
        """三条通道的 1s 束缚能（meV），Kormányos 平均带质量。"""
        r_max = 500.0
        r_grid = np.linspace(0.0, r_max, 5001)[1:]
        stack = ovesen_stack(mose2_below=True)
        i_mo = ovesen_slab_index("MoSe2", True)
        i_ws = ovesen_slab_index("WSe2", True)
        m_e_mo, m_h_mo = (0.56 + 0.49) / 2, (0.59 + 0.70) / 2
        m_e_ws, m_h_ws = (0.28 + 0.39) / 2, (0.36 + 0.54) / 2
        channels = (
            ("MoSe2-intra", i_mo, i_mo, m_e_mo * m_h_mo / (m_e_mo + m_h_mo)),
            ("WSe2-intra", i_ws, i_ws, m_e_ws * m_h_ws / (m_e_ws + m_h_ws)),
            ("interlayer", i_mo, i_ws, m_e_mo * m_h_ws / (m_e_mo + m_h_ws)),
        )
        out = {}
        for label, a, b, mu in channels:
            vfun = interaction_real_space(stack, r_grid, a, b)
            res = solve_exciton_with_potential(
                lambda r, v=vfun, rg=r_grid: np.interp(r, rg, v),
                r_max, mu, n_levels=1, n_basis=200, n_quad=2500)
            out[label] = res.binding_1s * 1e3
        return out

    def test_interlayer_binding(self, bindings):
        """层间 1s：模型 161 vs 原文 150 meV（+7%）。"""
        assert bindings["interlayer"] == pytest.approx(150.0, rel=0.20)

    def test_intralayer_bindings(self, bindings):
        """层内 1s：216/195 vs 原文 205/177 meV（+5%/+10%）。"""
        assert bindings["MoSe2-intra"] == pytest.approx(205.0, rel=0.20)
        assert bindings["WSe2-intra"] == pytest.approx(177.0, rel=0.20)

    def test_interlayer_weaker_than_intralayer(self, bindings):
        """层间束缚能 < 两个层内（Ovesen 定性结论的模型复现）。"""
        assert bindings["interlayer"] < bindings["MoSe2-intra"]
        assert bindings["interlayer"] < bindings["WSe2-intra"]
