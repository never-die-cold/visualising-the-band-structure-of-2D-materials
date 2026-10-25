"""转角双层 TMD moiré 模型的物理验证测试。

验证锚点全部可解析（见 engine/moire_tmd.py 模块文档）：
T=0 解耦精确谱、厄米性、C₃ 对称性、微扰劈裂 2|T|、平带化。
"""

import numpy as np
import pytest

from vdw_studio.engine.moire_tmd import (
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
