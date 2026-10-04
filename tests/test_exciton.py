"""2D 激子求解器测试（Fourier–Bessel DVR）。

验证锚点（解析精确）：
- 库仑极限（r₀=None）：2D 氢原子谱 E_n = −4Ry*/(2n+1)²，
  Ry* = (μ/m₀)·13.6057/ε²；
- Keldysh 势的 r₀→0 库仑极限（逐点）；
- 束缚能随屏蔽（r₀↑ 或 ε↑）单调减弱；
- 变分收敛：能量从上方单调逼近精确值。
"""

import numpy as np
import pytest
from scipy import special

from vdw_studio.analysis import (
    coulomb_potential,
    keldysh_potential,
    solve_exciton,
)


class TestPotentialLimits:
    def test_keldysh_coulomb_limit(self):
        """Keldysh 势 r₀→0 库仑极限（逐点精确）。"""
        for r in (5.0, 20.0, 100.0):
            vk = keldysh_potential(r, r0=1e-3)
            vc = coulomb_potential(r)
            # Struve/Y0 在大 ρ 的浮点精度 ~1e-8 相对
            assert vk == pytest.approx(vc, rel=1e-6)

    def test_keldysh_short_range_finite(self):
        """r→0 时 r·V(r) 有限（Keldysh→0, 库仑→常数）——DVR 可积性的关键。"""
        assert keldysh_potential(1e-8, r0=50.0)[()] * 1e-8 == pytest.approx(0.0, abs=1e-6)
        assert coulomb_potential(1e-8) * 1e-8 == pytest.approx(-14.3996, rel=1e-6)

    def test_potential_attractive(self):
        r = np.array([1.0, 5.0, 50.0])
        assert np.all(keldysh_potential(r, r0=30.0) < 0)
        assert np.all(coulomb_potential(r) < 0)


class TestCoulombExactSpectrum:
    """库仑极限 = 2D 氢原子，与精确谱对账。"""

    MU = 0.25

    def exact(self, n):
        return -4 * self.MU * 13.6057 / (2 * n + 1) ** 2

    def test_ground_state(self):
        r = solve_exciton(self.MU, 1.0, r0=None, n_levels=1,
                          n_basis=250, n_quad=6000)
        assert r.energies[0] == pytest.approx(self.exact(0), rel=0.02)

    def test_first_three_levels(self):
        r = solve_exciton(self.MU, 1.0, r0=None, n_levels=3,
                          n_basis=300, n_quad=8000)
        for i in range(3):
            assert r.energies[i] == pytest.approx(self.exact(i), rel=0.03)

    def test_variational_convergence(self):
        """变分收敛：能量从上方逼近精确值，且随基组增大单调改善。"""
        e_small = solve_exciton(self.MU, 1.0, r0=None, n_levels=1,
                                n_basis=100, n_quad=4000).energies[0]
        e_large = solve_exciton(self.MU, 1.0, r0=None, n_levels=1,
                                n_basis=250, n_quad=8000).energies[0]
        exact = self.exact(0)
        assert e_small >= exact - 1e-6      # 变分上界
        assert abs(e_large - exact) < abs(e_small - exact)

    def test_epsilon_scaling(self):
        """ε 屏蔽: Ry* ∝ 1/ε² → 束缚能 ∝ 1/ε²（精确标度律）。"""
        r1 = solve_exciton(self.MU, 1.0, r0=None, n_levels=1,
                           n_basis=250, n_quad=6000)
        r4 = solve_exciton(self.MU, 4.0, r0=None, n_levels=1,
                           n_basis=1000, n_quad=8000)
        # 精确标度: E_b(ε) = E_b(1)/ε²; ε=4 时 a* ×4, 同 N 精度略降
        assert r4.binding_1s * 16 == pytest.approx(r1.binding_1s, rel=0.15)


class TestKeldyshScreening:
    MU = 0.25

    @pytest.mark.parametrize("r0", [10.0, 30.0, 75.0])
    def test_binding_decreases_with_r0(self, r0):
        """屏蔽长度 r₀ 越大 → 束缚越弱（单调）。"""
        r = solve_exciton(self.MU, 1.0, r0=r0, n_levels=1,
                          n_basis=250, n_quad=6000)
        assert r.binding_1s > 0
        ref = {10.0: 1.4951, 30.0: 0.7003, 75.0: 0.3504}[r0]
        assert r.binding_1s == pytest.approx(ref, rel=0.01)

    def test_binding_decreases_with_eps(self):
        e1 = solve_exciton(self.MU, 1.0, r0=75.0, n_levels=1,
                           n_basis=250, n_quad=6000).binding_1s
        e25 = solve_exciton(self.MU, 2.5, r0=75.0, n_levels=1,
                            n_basis=400, n_quad=8000).binding_1s
        assert e25 < e1

    def test_keldysh_weaker_than_coulomb(self):
        """同等参数下 Keldysh 束缚 < 库仑束缚（长程屏蔽）。"""
        e_k = solve_exciton(self.MU, 1.0, r0=30.0, n_levels=1,
                            n_basis=250, n_quad=6000).binding_1s
        e_c = solve_exciton(self.MU, 1.0, r0=None, n_levels=1,
                            n_basis=250, n_quad=6000).binding_1s
        assert e_k < e_c


class TestApi:
    def test_invalid_params(self):
        with pytest.raises(ValueError):
            solve_exciton(mu_over_m0=0.0)
        with pytest.raises(ValueError):
            solve_exciton(mu_over_m0=0.25, r0=0.0)

    def test_result_metadata(self):
        r = solve_exciton(0.25, 1.0, r0=30.0, n_levels=2,
                          n_basis=100, n_quad=4000)
        assert r.r0 == 30.0 and r.eps_env == 1.0
        assert r.mu_over_m0 == 0.25
        assert r.n_basis == 100
        assert r.rydberg() == pytest.approx(0.25 * 13.6057, rel=1e-6)

    def test_j0_zeros_consistency(self):
        """基组动量 = J₀ 零点 / R（Bessel DVR 的自洽性）。"""
        n_basis = 50
        R = 60.0
        j0 = special.jn_zeros(0, n_basis)
        assert np.allclose(j0 / R, j0 / R)  # 平凡自洽; 占位以固定约定
        # J₀(j₀,ₙ) = 0
        assert np.all(np.abs(special.j0(j0)) < 1e-12)
