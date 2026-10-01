"""应变工程模块测试。

自检锚点（可解析推导，见 strain.py 模块文档）：
- 石墨烯双轴应变 ε：K 点保持无隙；v_F(ε) = v_F(0)·(1+ε)^{1−n}，
  n=2 时 = v_F(0)/(1+ε)；
- hopping 标度律：t' = t·(1+ε)^{−n} 精确成立；
- 单轴应变（NN 模型）：无隙、能带最小值存在；
- 黑磷烯：带隙对应变敏感（|Δgap| 随 |ε| 增大），不做符号断言
  （四带 TB 无形变势参数，定量符号需文献标定）。
"""

import numpy as np
import pytest

from vdw_studio.analysis import analyze_gap, fermi_velocity
from vdw_studio.engine.kpath import KPath, lattice_type
from vdw_studio.engine.models import HoneycombModel, PhosphoreneRudenko
from vdw_studio.engine.strain import apply_strain


class TestScalingLaw:
    def test_hopping_rescaling_exact(self):
        """t' = t·(1+ε)^{−n}（双轴应变下所有键等比拉长）。"""
        g = HoneycombModel(a=2.46, t=-2.7)
        eps = 0.02
        gs = apply_strain(g, ex=eps, ey=eps, exponent=2.0)
        for h0, h1 in zip(g.hoppings, gs.hoppings):
            assert h1.t == pytest.approx(h0.t * (1 + eps) ** -2, rel=1e-12)
            assert h1.dfrac == h0.dfrac          # 拓扑不变

    def test_lattice_scaled_vacuum_untouched(self):
        g = HoneycombModel(a=2.46)
        gs = apply_strain(g, ex=0.05, ey=0.0)
        a_new, _, c_new, _, _, gamma_new = gs.lattice.parameters()
        assert a_new == pytest.approx(2.46 * 1.05)
        assert c_new == pytest.approx(g.lattice.parameters()[2])  # 真空层不变
        assert gamma_new == pytest.approx(120.0, abs=1e-6)  # a1 沿 x 时 γ 不变

    def test_original_model_untouched(self):
        g = HoneycombModel(a=2.46, t=-2.7)
        _ = apply_strain(g, ex=0.05, ey=0.05)
        assert g.lattice.parameters()[0] == pytest.approx(2.46)
        assert g.hoppings[0].t == pytest.approx(-2.7)

    def test_zero_strain_returns_same(self):
        g = HoneycombModel(a=2.46)
        assert apply_strain(g, 0.0, 0.0) is g

    def test_rejects_non_site_model(self):
        from vdw_studio.engine.kp_tmd import TMDKpModel, TMDKpParams
        kp = TMDKpModel(TMDKpParams.from_library("MoS2"))
        with pytest.raises(TypeError):
            apply_strain(kp, ex=0.01)


class TestGrapheneStrain:
    def test_biaxial_stays_gapless(self):
        """双轴应变保持 C₃ 对称 → 石墨烯 NN 模型仍无隙。"""
        g = HoneycombModel(a=2.46, t=-2.7)
        gs = apply_strain(g, ex=0.03, ey=0.03)
        r = analyze_gap(gs, n_per_segment=30)
        assert r.gap is None

    def test_biaxial_fermi_velocity_scaling(self):
        """n=2: v_F(ε) = v_F(0)/(1+ε)（解析自检）。"""
        g = HoneycombModel(a=2.46, t=-2.7)
        v0 = fermi_velocity(g, (1 / 3, 1 / 3), band=1,
                            direction=(1 / 3, -1 / 3))
        for eps in (0.02, -0.02):
            gs = apply_strain(g, ex=eps, ey=eps, exponent=2.0)
            v = fermi_velocity(gs, (1 / 3, 1 / 3), band=1,
                               direction=(1 / 3, -1 / 3))
            assert v == pytest.approx(v0 / (1 + eps), rel=1e-4), f"ε={eps}"

    def test_uniaxial_stays_gapless_nn(self):
        """单轴应变（NN 模型）：Dirac 点移动但不打开能隙。

        严格验证：对 H(k) 的非对角元 f(k)=0 做 2D 求根，应存在精确零点
        （开隙需三阶近邻 hopping，超出 NN 范围——见模块文档的诚实标注）。
        """
        from scipy.optimize import fsolve
        g = HoneycombModel(a=2.46, t=-2.7)
        gs = apply_strain(g, ex=0.05, ey=0.0)

        def f(k):
            H = gs.hamiltonian(k[:2])
            return [H[0, 1].real, H[0, 1].imag]

        sol, info, ier, _ = fsolve(f, [0.7, 0.65], full_output=True)
        assert ier == 1, "Dirac 点求根失败"
        H = gs.hamiltonian(sol)
        assert abs(H[0, 1]) < 1e-8       # f(k_D) = 0 → 零隙
        assert abs(H[0, 1].imag) < 1e-8

    def test_strained_hexagonal_uses_hex_path(self):
        """轻微单轴应变的六方格子仍按 Γ-M-K-Γ 选取路径（文献惯例）。"""
        g = HoneycombModel(a=2.46, t=-2.7)
        gs = apply_strain(g, ex=0.03, ey=0.0)
        kp = KPath.for_lattice(gs.lattice)
        assert kp.path == ["G", "M", "K", "G"]
        assert lattice_type(gs.lattice, tol_rel=0.06, tol_deg=3.0) == "hexagonal"


class TestPhosphoreneStrain:
    p0 = PhosphoreneRudenko()
    gap0 = analyze_gap(p0, n_per_segment=30).gap

    @pytest.mark.parametrize("eps", [-0.03, -0.02, 0.02, 0.03])
    def test_gap_sensitivity(self, eps):
        """带隙对应变敏感：|gap(ε) − gap(0)| 随 |ε| 单调增大。

        （不做符号断言——四带 TB 无形变势参数，定量符号需文献标定。）
        """
        p = apply_strain(self.p0, ex=eps, ey=eps)
        gap = analyze_gap(p, n_per_segment=30).gap
        assert gap is not None
        dev = abs(gap - self.gap0)
        assert dev > 1e-3, f"ε={eps}: 带隙无响应"
        if abs(eps) == 0.03:
            dev_small = abs(analyze_gap(
                apply_strain(self.p0, ex=eps * 2 / 3, ey=eps * 2 / 3),
                n_per_segment=30).gap - self.gap0)
            assert dev > dev_small   # 灵敏度随 |ε| 增大

    def test_anisotropy_retained(self):
        """应变后主质量各向异性保留（黑磷烯的招牌性质不被应变抹掉）。"""
        from vdw_studio.analysis import principal_masses
        p = apply_strain(self.p0, ex=0.02, ey=-0.01)
        m1, m2, _ = principal_masses(p, (0, 0), band=2, dk=1e-4)
        assert max(m1, m2) / min(m1, m2) > 2.5
