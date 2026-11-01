"""转角双层 TMD moiré 模型的物理验证测试。

验证锚点全部可解析（见 engine/moire_tmd.py 模块文档）：
T=0 解耦精确谱、厄米性、C₃ 对称性、微扰劈裂 2|T|、平带化。
"""

import numpy as np
import pytest

from vdw_studio.engine.moire_tmd import (
    TwistedTMDKp,
    TwistedTMDBilayer,
    build_tmd_moire_geometry,
)


def _c3_rotate(k):
    """120° 旋转（moiré BZ 的 C₃ 轴）。"""
    c, s = np.cos(2 * np.pi / 3), np.sin(2 * np.pi / 3)
    return np.array([c * k[0] - s * k[1], s * k[0] + c * k[1]])


class TestGeometry:
    def test_moire_length_and_area(self):
        geom = build_tmd_moire_geometry(2.0, 3.184)
        expected = 3.184 / (2 * np.sin(np.radians(1.0)))
        assert geom.l_moire == pytest.approx(expected, rel=1e-12)
        # |q| = 2 K_D sin(θ/2)，K_D = 4π/(3a)
        kd = 4 * np.pi / (3 * 3.184)
        assert geom.k_theta == pytest.approx(2 * kd * np.sin(np.radians(1.0)))
        # moiré BZ 面积 = 8π²/(√3 L²)（2D 叉积手算，np.cross 不收 2D 向量）
        area = abs(geom.b1[0] * geom.b2[1] - geom.b1[1] * geom.b2[0])
        assert area == pytest.approx(8 * np.pi ** 2 / (np.sqrt(3) * expected ** 2), rel=1e-9)

    def test_invalid_inputs_rejected(self):
        with pytest.raises(ValueError):
            build_tmd_moire_geometry(0.0, 3.184)
        with pytest.raises(ValueError):
            build_tmd_moire_geometry(1.0, -1.0)
        with pytest.raises(ValueError):
            TwistedTMDBilayer(theta_deg=1.0, band="xx")
        with pytest.raises(ValueError):
            TwistedTMDBilayer(theta_deg=1.0, t_interlayer=-0.1)


class TestDecoupledLimit:
    def test_zero_coupling_gives_exact_parabolas(self):
        """T=0：两层解耦（严格分块对角），谱 = 对角元并集，任意截断精确。"""
        model = TwistedTMDBilayer(theta_deg=2.0, t_interlayer=0.0, n_shells=2)
        H = model.hamiltonian(np.array([0.0, 0.0]))
        assert np.abs(H - H.conj().T).max() < 1e-14
        e = np.linalg.eigvalsh(H)
        np.testing.assert_allclose(np.sort(e), np.sort(np.diag(H).real), atol=1e-12)

    def test_effective_mass_recovers_input_in_decoupled_limit(self):
        """T=0 时 Γ 点最低带 = 输入质量的纯抛物线：二阶差分质量必须精确还原。

        这是二阶差分质量因子 2 的精确锚点（m* = 2·(ħ²/2m₀)/d²E，
        与 analysis.properties._curvature_mass 同约定）：漏 2 会得 0.275。
        """
        m_in = 0.55
        model = TwistedTMDBilayer(theta_deg=2.0, t_interlayer=0.0, n_shells=2,
                                  effective_mass=m_in)
        assert model.moire_effective_mass(band_index=0) == pytest.approx(m_in, rel=1e-9)

    def test_t_zero_spectra_shift_with_k(self):
        """T=0：最低带 = 两层动量并集的最小值（动量守恒的自检）。"""
        model = TwistedTMDBilayer(theta_deg=3.0, t_interlayer=0.0, n_shells=3,
                                  effective_mass=0.55)
        from vdw_studio.constants import HBAR2_OVER_2M0
        dk = 0.005
        e = model.bands([[0.0, 0.0], [dk, 0.0]])[:, 0]
        # 最低态在 k=0 处应为 0（层 1 G=0）；移动 dk 后最小动量 = dk
        assert e[0] == pytest.approx(0.0, abs=1e-12)
        assert e[1] == pytest.approx(HBAR2_OVER_2M0 * dk ** 2 / 0.55, rel=1e-9)


class TestCoupledPhysics:
    @pytest.fixture(scope="class")
    def model(self):
        return TwistedTMDBilayer(theta_deg=1.5, t_interlayer=0.010, n_shells=3,
                                 effective_mass=0.55)

    def test_hermitian(self, model):
        for k in ([0.0, 0.0], [0.003, -0.002], [0.01, 0.004]):
            H = model.hamiltonian(k)
            assert np.abs(H - H.conj().T).max() < 1e-14

    def test_c3_symmetry_in_low_energy_window(self, model):
        """C₃：谱对 120° 旋转不变（限定低能窗口，避免截断邻带污染）。"""
        k = np.array([0.0035, 0.0011])
        e_k = model.bands([k])[0]
        e_r = model.bands([_c3_rotate(k)])[0]
        n_low = 8
        np.testing.assert_allclose(np.sort(e_k)[:n_low], np.sort(e_r)[:n_low],
                                   atol=1e-9)

    def test_weak_coupling_is_second_order(self):
        """弱耦合极限：k=0 两态动量差恒为 q₁（不简并）→ 带边修正是二阶的。

        解析锚点是比例：ΔE ∝ T²，两个 T 值的带边位移之比应为 (T₁/T₂)² = 25。
        （BM 石墨烯的"劈裂 2T"锚点依赖 Dirac 锥在 k=0 的简并；单带抛物线
        模型没有这个简并，不能照搬。）
        """
        def band_edge(t):
            m = TwistedTMDBilayer(theta_deg=1.0, t_interlayer=t, n_shells=3,
                                  effective_mass=0.55)
            return float(m.bands([[0.0, 0.0]])[0, 0])

        small, large = band_edge(1e-4), band_edge(5e-4)
        # 排斥使最低态下移：ΔE < 0 且 |ΔE| ∝ T²
        assert small < 0 and large < small
        assert abs(large / small) == pytest.approx(25.0, rel=0.05)

    def test_flat_band_limit(self):
        """平带化：θ 减小使动能标度低于 T，最低带带宽饱和于 O(T)。"""
        widths = {}
        for theta in (5.0, 1.0, 0.5):
            m = TwistedTMDBilayer(theta_deg=theta, t_interlayer=0.010, n_shells=3,
                                  effective_mass=0.55)
            k_list = np.column_stack((np.linspace(0, m.geom.k_theta, 12),
                                      np.zeros(12)))
            e = m.bands(k_list)
            widths[theta] = float(e[:, 0].max() - e[:, 0].min())
        # 大角度（弱耦合）带宽由色散决定且更大；小角度强耦合带宽有界
        assert widths[5.0] > widths[0.5]
        assert widths[0.5] < 6 * 0.010      # 平带化：O(T) 量级

    def test_effective_mass_at_gamma(self, model):
        m_star = model.moire_effective_mass(band_index=0)
        assert m_star > 0.55               # moiré 耦合使带质量变大
        assert m_star < 50.0


class TestTwistedTMDKp:
    """完整 k·p 层哈密顿量版 moiré 求解器（T17 剩余项）的物理验证。"""

    def test_decoupled_limit_is_kormanyos_union(self):
        """γ=0：谱 = 两层 Kormányos 谱在采样动量上的并集（含一般 k）。"""
        m = TwistedTMDKp(theta_deg=2.0, n_shells=2,
                         gamma_cb=0.0, gamma_vb=0.0)
        kp = m.kp
        for k in ([0.0, 0.0], [0.001, -0.002], [0.004, 0.001]):
            k = np.asarray(k, float)
            p1s = [k + g[0] * m.geom.b1 + g[1] * m.geom.b2 for g in m.pw]
            expected = np.sort(np.concatenate(
                [kp.energies(p, +1) for p in p1s]
                + [kp.energies(p + m.geom.q_vectors[0], +1) for p in p1s]))
            np.testing.assert_allclose(m.bands([k])[0], expected, atol=1e-10)

    def test_hermitian(self):
        m = TwistedTMDKp(theta_deg=1.5, n_shells=2,
                         gamma_cb=0.020, gamma_vb=0.020)
        for k in ([0.0, 0.0], [0.003, -0.002]):
            H = m.hamiltonian(k)
            assert np.abs(H - H.conj().T).max() < 1e-13

    def test_c3_converges_with_shells(self):
        """C₃ 低能窗口误差随壳层指数下降（kp_tmd 立方项修正后的回归）。"""
        c, s = np.cos(2 * np.pi / 3), np.sin(2 * np.pi / 3)
        rot = np.array([[c, -s], [s, c]])
        k = np.array([0.0035, 0.0011])
        errs = []
        for n_shells in (3, 4):
            m = TwistedTMDKp(theta_deg=1.5, n_shells=n_shells,
                             gamma_cb=0.020, gamma_vb=0.020)
            e0 = m.bands([k])[0]
            e1 = m.bands([rot @ k])[0]
            sl = slice(m.dim // 2 - 4, m.dim // 2 + 4)
            errs.append(np.abs(np.sort(e0[sl]) - np.sort(e1[sl])).max())
        assert errs[-1] < 1e-9
        assert errs[-1] < errs[0]          # 随壳层收敛
        assert errs[0] < 1e-6

    def test_weak_coupling_is_second_order(self):
        """弱耦合极限：cb 带边位移 |ΔE| ∝ γ²（比例 25）。"""
        def band_edge(g):
            m = TwistedTMDKp(theta_deg=1.0, n_shells=2,
                             gamma_cb=g, gamma_vb=g)
            return float(m.bands([[0.0, 0.0]])[0, m.i_cb_bottom])

        e0, small, large = band_edge(0.0), band_edge(1e-4), band_edge(5e-4)
        assert small < e0 and large < small        # 排斥使带边下移
        assert abs((large - e0) / (small - e0)) == pytest.approx(25.0, rel=0.05)

    def test_flat_band_limit(self):
        """平带化：θ 减小使 cb moiré 带宽饱和于 O(γ)。"""
        widths = {}
        for theta in (5.0, 1.0, 0.5):
            m = TwistedTMDKp(theta_deg=theta, n_shells=2,
                             gamma_cb=0.010, gamma_vb=0.010)
            ks = np.column_stack((np.linspace(0, m.geom.k_theta, 10),
                                  np.zeros(10)))
            e = m.bands(ks)[:, m.i_cb_bottom]
            widths[theta] = float(e.max() - e.min())
        assert widths[5.0] > widths[0.5]
        assert widths[0.5] < 6 * 0.010

    def test_decoupled_moire_mass_matches_kormanyos(self):
        """解耦 moiré 质量 = 单层 Kormányos 质量（衔接激子管线的锚点）。"""
        m = TwistedTMDKp(theta_deg=2.0, n_shells=2,
                         gamma_cb=0.0, gamma_vb=0.0)
        m_moire = m.moire_effective_mass(m.i_cb_bottom)
        m_kp = m.kp.effective_mass(band="cb", spin=-1, valley=+1,
                                   direction=(1.0, 0.0))
        assert m_moire == pytest.approx(m_kp, rel=1e-6)

    def test_coupled_mass_heavier_than_decoupled(self):
        """moiré 耦合使带质量增大（平带化方向）。"""
        dec = TwistedTMDKp(theta_deg=1.5, n_shells=2,
                           gamma_cb=0.0, gamma_vb=0.0)
        cpl = TwistedTMDKp(theta_deg=1.5, n_shells=2,
                           gamma_cb=0.020, gamma_vb=0.020)
        assert cpl.moire_effective_mass(cpl.i_cb_bottom) \
            > dec.moire_effective_mass(dec.i_cb_bottom)

    def test_invalid_inputs(self):
        with pytest.raises(ValueError):
            TwistedTMDKp(theta_deg=0.0)
        with pytest.raises(ValueError):
            TwistedTMDKp(theta_deg=2.0, gamma_cb=-0.1)
        with pytest.raises(ValueError):
            TwistedTMDKp(theta_deg=2.0, gamma_vb=-0.1)
        with pytest.raises(ValueError):
            TwistedTMDKp(theta_deg=2.0, n_shells=0)
        with pytest.raises(ValueError):
            TwistedTMDKp(theta_deg=2.0, valley=0)
        with pytest.raises(ValueError):
            TwistedTMDKp(theta_deg=2.0, material="GaAs")
