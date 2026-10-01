"""双层石墨烯最小模型物理验证（McCann–Fal'ko AB 堆叠 + 垂直电场）。

解析锚点（模型精确结果，见 models.BilayerGrapheneModel 文档）：
- U=0：K 点能量 {0, 0, ±γ₁}；低能带抛物线触碰（k² 色散，m* 由 γ₁ 决定）；
- U≠0：K 点能量 {±U/2, ±√(γ₁²+U²/4)}，带隙 = |U|（最小模型线性开隙）；
- U=0 的带边 DOS 有限（2D 抛物线带的特征，区别于单层 Dirac 锥的零 DOS）。
"""

import numpy as np
import pytest

from vdw_studio.analysis import fermi_velocity
from vdw_studio.engine.models import BilayerGrapheneModel, HoneycombModel
from vdw_studio.engine.solver import solve_dos


class TestBilayerAnalytic:
    m0 = BilayerGrapheneModel(electric_field=0.0)

    def test_k_energies_zero_field(self):
        e = self.m0.energies_at((1 / 3, 1 / 3))
        assert sorted(e) == pytest.approx(sorted([0, 0, 0.4, -0.4]), abs=1e-10)

    def test_gap_linear_in_field(self):
        """最小模型：K 点带隙 = |U|（解析精确）。"""
        for u in (0.2, 0.5, 1.0):
            m = BilayerGrapheneModel(electric_field=u)
            e = m.energies_at((1 / 3, 1 / 3))
            assert (e[2] - e[1]) == pytest.approx(u, abs=1e-10)
            # dimer 带边位置 ±√(γ₁²+U²/4)
            assert e[3] == pytest.approx(np.sqrt(0.4 ** 2 + (u / 2) ** 2),
                                         abs=1e-10)
            assert e[0] == pytest.approx(-e[3], abs=1e-10)

    def test_gap_monotonic_even_in_u(self):
        gaps = []
        for u in (0.0, 0.3, 0.6, 0.9):
            m = BilayerGrapheneModel(electric_field=u)
            e = m.energies_at((1 / 3, 1 / 3))
            gaps.append(e[2] - e[1])
        assert np.all(np.diff(gaps) > 0)
        # 偶函数: gap(−U) = gap(U)
        m = BilayerGrapheneModel(electric_field=-0.5)
        e = m.energies_at((1 / 3, 1 / 3))
        assert (e[2] - e[1]) == pytest.approx(0.5, abs=1e-10)


class TestBilayerDispersion:
    m0 = BilayerGrapheneModel(electric_field=0.0)

    def test_quadratic_touching(self):
        """U=0 低能带色散 ~ k²（抛物线触碰，区别于单层线性 Dirac）。

        k 取得足够小使 vk ≪ γ₁（vk = 5.75×0.0104 ≈ 0.06 eV ≪ 0.4 eV）。
        """
        K = np.array([1 / 3, 1 / 3])
        d = np.array([0.3, -0.3])
        d /= np.linalg.norm(d)
        e1 = self.m0.energies_at(K + 0.0025 * d)[2]
        e2 = self.m0.energies_at(K + 0.005 * d)[2]
        assert e2 / e1 == pytest.approx(4.0, rel=0.05)   # k² 标度

    def test_two_band_dispersion_exact(self):
        """精确二带形式 E = √((γ₁/2)² + (ħv_F·q)²) − γ₁/2（解析）。"""
        K = np.array([1 / 3, 1 / 3])
        d = np.array([0.3, -0.3])
        d /= np.linalg.norm(d)
        recip = self.m0.lattice.reciprocal_matrix
        hv = np.sqrt(3) * 2.7 * 2.46 / 2          # ħv_F (eV·Å)
        for k in (0.005, 0.010):
            e = self.m0.energies_at(K + k * d)[2]
            k_cart = k * np.linalg.norm(d @ recip[:2, :2])
            expected = np.sqrt((0.4 / 2) ** 2 + (hv * k_cart) ** 2) - 0.4 / 2
            assert e == pytest.approx(expected, rel=2e-3)

    def test_effective_mass_literature(self):
        """m* = γ₁/(2v_F²) ≈ 0.046 m₀（双层石墨烯文献质量）。"""
        m = BilayerGrapheneModel(electric_field=0.0)
        from vdw_studio.analysis import effective_mass
        m_star = effective_mass(m, (1 / 3, 1 / 3), band=2, direction=(1, 1))
        assert m_star == pytest.approx(0.046, rel=0.05)

    def test_monolayer_is_linear_bilayer_is_quadratic(self):
        """同一小 k 采样下单层 E∝k（线性），双层 E∝k²。"""
        mono = HoneycombModel(a=2.46, t=-2.7)
        K = np.array([1 / 3, 1 / 3])
        d = np.array([0.3, -0.3])
        d /= np.linalg.norm(d)
        e_mono_1 = mono.energies_at(K + 0.005 * d)[1]
        e_mono_2 = mono.energies_at(K + 0.010 * d)[1]
        assert e_mono_2 / e_mono_1 == pytest.approx(2.0, rel=1e-3)
        e_bi_1 = self.m0.energies_at(K + 0.0025 * d)[2]
        e_bi_2 = self.m0.energies_at(K + 0.005 * d)[2]
        assert e_bi_2 / e_bi_1 == pytest.approx(4.0, rel=0.05)


class TestBilayerDOS:
    def test_finite_dos_at_touching(self):
        """U=0 抛物线触碰 → 带边 DOS 有限；单层 Dirac 点 DOS→0。

        判据：双层带边 DOS 应显著大于单层（比值 > 10）。
        """
        bi = BilayerGrapheneModel(electric_field=0.0)
        dos_bi = solve_dos(bi, mesh=(48, 48), sigma=0.03, n_points=1200,
                           e_min=-0.5, e_max=0.5)
        i0 = np.argmin(np.abs(dos_bi.energies))

        mono = HoneycombModel(a=2.46, t=-2.7)
        dos_mono = solve_dos(mono, mesh=(48, 48), sigma=0.03, n_points=1200,
                             e_min=-0.5, e_max=0.5)
        j0 = np.argmin(np.abs(dos_mono.energies))
        assert dos_bi.dos[i0] > 10 * dos_mono.dos[j0]
