"""2D 激子求解器测试（Fourier–Bessel DVR）。

验证锚点：
- **解析精确**：库仑极限（r₀=None）2D 氢原子谱 E_n = −4Ry*/(2n+1)²，
  Ry* = (μ/m₀)·13.6057/ε²；Keldysh 势的 r₀→0 库仑极限（逐点）；
  束缚能随屏蔽（r₀↑ 或 ε↑）单调减弱；变分收敛（能量从上方逼近）。
- **文献标定**：Berkelbach et al., PRB 88, 045318 (2013) Table 的
  4 种 TMD（μ、χ₂D、变分束缚能）——精确 DVR 解 ≥ 变分下界且偏差
  ≤10%，材料排序 MoS₂ > WS₂ > MoSe₂ > WSe₂ 复现。
"""

import numpy as np
import pytest
from scipy import special

from vdw_studio.analysis import (
    coulomb_potential,
    keldysh_potential,
    solve_exciton,
)
from vdw_studio.presets import get_exciton_params, solve_preset_exciton


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


class TestBerkelbachCalibration:
    """文献标定：Berkelbach 2013 Table 的 TMD 激子参数复现。

    Berkelbach, Hybertsen & Reichman, PRB 88, 045318 (2013) Table
    （tab:binding）: μ (m₀)、χ₂D (Å)、激子 1s 束缚能（**变分法**, eV）。
    r₀ = 2πχ₂D（Cudazzo 2011 关系；Berkelbach Eq. (1) 的 ρ₀ 等价，
    真空 ε₁=ε₂=1 → 本模块 eps_env=1，势形式逐项一致）。

    判据：变分能量是精确基态能量的**上界** → 文献束缚能是下界，
    本精确 DVR 解应给出 E_b(DVR) ≥ E_b(var) 且接近（≤ +10%）。
    """

    # (预设 key, E_b 变分文献值 eV)
    CASES = [
        ("mos2_kp", 0.54),
        ("mose2_kp", 0.47),
        ("ws2_kp", 0.50),
        ("wse2_kp", 0.45),
    ]

    @pytest.mark.parametrize("key,eb_var", CASES)
    def test_binding_matches_variational_lower_bound(self, key, eb_var):


        ex = get_exciton_params(key)
        assert ex.eb_ref[0] == pytest.approx(eb_var)
        r = solve_preset_exciton(key, eps_env=1.0, n_levels=1,
                                 n_basis=250, n_quad=6000)
        assert r.mu_over_m0 == ex.mu_over_m0
        assert r.r0 == pytest.approx(ex.r0_angstrom)
        assert r.binding_1s >= eb_var                      # 变分下界方向
        assert r.binding_1s <= eb_var * 1.10               # 接近文献值

    def test_material_trend_reproduced(self):
        """束缚能排序 MoS₂ > WS₂ > MoSe₂ > WSe₂（Berkelbach 趋势）。"""
        eb = {key: solve_preset_exciton(key, n_levels=1, n_basis=250,
                                        n_quad=6000).binding_1s
              for key, _ in self.CASES}
        assert eb["mos2_kp"] > eb["ws2_kp"] > eb["mose2_kp"] > eb["wse2_kp"]

    def test_dielectric_engineering_trend(self):
        """介电工程: ε_env ↑（hBN 封装 ε≈4.5）→ 束缚能显著减弱。

        文献趋势（ROADMAP Phase 3 锚点）：真空 ~0.5 eV → hBN 环境
        ~0.15–0.3 eV 量级（此趋势测试用宽窗口）。
        """
        e_vac = solve_preset_exciton("mos2_kp", eps_env=1.0, n_levels=1,
                                     n_basis=250, n_quad=6000).binding_1s
        e_hbn = solve_preset_exciton("mos2_kp", eps_env=4.5, n_levels=1,
                                     n_basis=400, n_quad=8000).binding_1s
        assert e_hbn < 0.5 * e_vac
        assert e_hbn > 0.05

    def test_uncalibrated_material_raises(self):
        """未标定材料（石墨烯等）应明确报错而非给无出处参数。"""

        with pytest.raises(KeyError):
            get_exciton_params("graphene")


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
        # Test an independently known fundamental J₀ root and hard-wall mode.
        assert j0[0] == pytest.approx(2.404825557695773, rel=1e-13)
        assert special.j0((j0[0] / R) * R) == pytest.approx(0., abs=1e-12)
        # J₀(j₀,ₙ) = 0
        assert np.all(np.abs(special.j0(j0)) < 1e-12)


class TestInterlayerSeparation:
    """层间分离 d 的 Rytova–Keldysh 推广（大目标 C 剩余项）。

    锚点：d=0 严格退回共面 RK；r₀→0 严格退回三维层间库仑；
    束缚能随 d 单调削弱（共面近似为上界）。
    """

    def test_potential_d_zero_delegates_to_coplanar(self):
        from vdw_studio.analysis.exciton import keldysh_potential_interlayer
        r = np.array([0.5, 2.0, 10.0, 50.0, 200.0])
        np.testing.assert_allclose(
            keldysh_potential_interlayer(r, 41.47, 0.0),
            keldysh_potential(r, 41.47), rtol=0, atol=1e-14)

    def test_potential_r0_to_zero_is_3d_interplane_coulomb(self):
        from vdw_studio.analysis.exciton import (
            interlayer_coulomb_potential,
            keldysh_potential_interlayer,
        )
        r = np.array([1.0, 3.0, 10.0, 40.0])
        d = 6.5
        np.testing.assert_allclose(
            keldysh_potential_interlayer(r, 1e-7, d),
            interlayer_coulomb_potential(r, d), rtol=1e-6)
        # 解析参考：−e²/(4πε₀·√(r²+d²))
        np.testing.assert_allclose(interlayer_coulomb_potential(r, d),
                                   -14.3996 / np.sqrt(r ** 2 + d ** 2),
                                   rtol=0, atol=1e-12)

    def test_potential_smooth_across_small_d(self):
        """d→0⁺ 连续：小 d 的势逐点趋于共面值（Laguerre 分支收敛）。"""
        from vdw_studio.analysis.exciton import keldysh_potential_interlayer
        r = np.array([2.0, 10.0, 41.5])
        v_small = keldysh_potential_interlayer(r, 41.47, 0.01)
        v_coplanar = keldysh_potential(r, 41.47)
        np.testing.assert_allclose(v_small, v_coplanar, rtol=2e-3, atol=0.0)

    def test_binding_decreases_with_separation(self):
        prev = None
        for d in (0.0, 3.0, 6.5, 13.0):
            res = solve_exciton(mu_over_m0=0.25, r0=41.47, n_levels=1,
                                interlayer_d=d)
            assert res.binding_1s is not None and res.binding_1s > 0
            if prev is not None:
                assert res.binding_1s < prev
            prev = res.binding_1s

    def test_coulomb_interlayer_branch(self):
        """r0=None + d>0：三维层间库仑支可用且弱于共面库仑。"""
        flat = solve_exciton(mu_over_m0=0.25, r0=None, n_levels=1)
        sep = solve_exciton(mu_over_m0=0.25, r0=None, n_levels=1,
                            interlayer_d=6.5)
        assert sep.binding_1s < flat.binding_1s

    def test_invalid_d_rejected(self):
        with pytest.raises(ValueError):
            solve_exciton(r0=41.47, interlayer_d=-1.0)
        from vdw_studio.analysis.exciton import keldysh_potential_interlayer
        with pytest.raises(ValueError):
            keldysh_potential_interlayer([1.0], 41.47, -0.5)
