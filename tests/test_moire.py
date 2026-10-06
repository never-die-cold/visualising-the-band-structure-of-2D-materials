"""转角双层石墨烯 BM 连续模型测试（Phase 5 莫尔超晶格）。

验证锚点（Bistritzer & MacDonald, PRB 84, 035440 (2011)）：
1. **几何**：k_θ = 2K_D·sin(θ/2)；moiré 晶格常数
   L_m = a₀/(2 sin(θ/2))（BM 正文 √3·a_cc/(2 sin θ/2)，a_cc = a₀/√3）；
   moiré 倒格矢 |b| = √3·k_θ。
2. **第一壳层解析速度**（BM Eq. (8)，8 带模型精确结果）::
       v*/v = (1−3α²)/(1+6α²)，α = w/(ħv·k_θ)
   同时验证 α 普适性（同 α 不同 (θ,w) → 同 v*/v，BM "moiré bands
   depend on a single parameter α"）。
3. **魔角**：全壳层数值 |v*| 在 θ = 1.05° 处取最小（BM Fig. 2/3：
   "the Dirac-point velocity vanishes already at θ ≈ 1.05°"，魔角序列
   1.05/0.5/0.35/0.24/0.2°）。
4. **k=0 零模**：k=0（moiré BZ 中心）恰有两个零能态（BM SI 解析）。
5. **魔角平带**：θ=1.05° 最低 moiré 带带宽 ≪ 一般转角
   （BM Fig. 2d "very flat moiré band"）。
"""

import numpy as np
import pytest

from vdw_studio.engine.moire import (
    A_LATTICE,
    HBAR_V_GRAPHENE,
    TwistedBilayerGraphene,
)


class TestMoireGeometry:
    def test_k_theta_and_moire_period(self):
        """k_θ = 2K_D sin(θ/2)；L_m = a₀/(2 sin(θ/2))（BM 正文公式）。"""
        m = TwistedBilayerGraphene(theta_deg=1.05)
        k_d = 4 * np.pi / (3 * A_LATTICE)
        assert m.geom.k_theta == pytest.approx(
            2 * k_d * np.sin(np.radians(1.05) / 2), rel=1e-12)
        assert m.geom.l_moire == pytest.approx(
            A_LATTICE / (2 * np.sin(np.radians(1.05) / 2)), rel=1e-12)
        # 1.05° → moiré 周期 ~134 Å（实验 ~13.4 nm 量级）
        assert 130 < m.geom.l_moire < 140

    def test_moire_reciprocal_vectors(self):
        """|b₁| = |b₂| = √3·k_θ（K 空间蜂巢格的布拉维格子）。"""
        m = TwistedBilayerGraphene(theta_deg=1.05)
        assert np.linalg.norm(m.geom.b1) == pytest.approx(
            np.sqrt(3) * m.geom.k_theta, rel=1e-12)
        assert np.linalg.norm(m.geom.b2) == pytest.approx(
            np.sqrt(3) * m.geom.k_theta, rel=1e-12)

    def test_default_hbar_v_from_reich(self):
        """默认 ħv = (√3/2)|t|a₀（Reich 2002: t=−2.7 eV, a₀=2.46 Å）。"""
        assert HBAR_V_GRAPHENE == pytest.approx(np.sqrt(3) / 2 * 2.7 * 2.46)
        assert HBAR_V_GRAPHENE == pytest.approx(5.755, abs=1e-2)


class TestFirstShellAnalytic:
    """8 带模型 vs BM Eq. (8) 解析速度（θ→0 精确）。"""

    @pytest.mark.parametrize("theta", [2.5, 1.5])
    def test_velocity_renormalization_analytic(self, theta):
        m = TwistedBilayerGraphene(theta_deg=theta, first_shell=True)
        alpha = m.alpha
        num = m.dirac_velocity() / m.hbar_v
        ana = (1 - 3 * alpha ** 2) / (1 + 6 * alpha ** 2)
        assert num == pytest.approx(ana, rel=1e-2), (theta, num, ana)

    def test_alpha_universality(self):
        """同 α 不同 (θ, w) → 同 v*/v（BM: 单参数 α 标度）。"""
        m1 = TwistedBilayerGraphene(theta_deg=1.05, w=0.110, n_shells=4)
        # θ 翻倍 → k_θ≈翻倍 → w 翻倍保持 α
        m2 = TwistedBilayerGraphene(
            theta_deg=2.1, w=0.110
            * np.sin(np.radians(1.05)) / np.sin(np.radians(0.525)),
            n_shells=4)
        assert m1.alpha == pytest.approx(m2.alpha, rel=1e-6)
        assert m1.dirac_velocity() / m1.hbar_v == pytest.approx(
            m2.dirac_velocity() / m2.hbar_v, rel=1e-4)


class TestFullModel:
    """全壳层模型的结构与魔角物理。"""

    def test_hermitian(self):
        m = TwistedBilayerGraphene(theta_deg=1.05, n_shells=3)
        for k in ([0.0, 0.0], [0.01, 0.003], [-0.008, 0.012]):
            H = m.hamiltonian(np.array(k))
            assert np.abs(H - H.conj().T).max() < 1e-12

    def test_zero_modes_at_k0(self):
        """k=0 恰有两个零能态（BM SI 解析结果，全壳层精确）。"""
        m = TwistedBilayerGraphene(theta_deg=1.05, n_shells=4)
        E = np.linalg.eigvalsh(m.hamiltonian(np.zeros(2)))
        d = len(E)
        assert abs(E[d // 2 - 1]) < 1e-8
        assert abs(E[d // 2]) < 1e-8

    def test_magic_angle_location(self):
        """|v*| 最小在 θ = 1.05°（BM Fig. 3 第一魔角）。"""
        thetas = [0.90, 0.95, 1.00, 1.05, 1.10, 1.15, 1.20]
        ratios = []
        for th in thetas:
            m = TwistedBilayerGraphene(theta_deg=th, n_shells=4)
            ratios.append(abs(m.dirac_velocity()) / m.hbar_v)
        i_min = int(np.argmin(ratios))
        assert thetas[i_min] == 1.05
        # 魔角处速度接近消失（<0.4% 裸值），且显著低于邻近点
        assert ratios[i_min] < 0.004
        assert ratios[i_min] < 0.25 * ratios[0]
        assert ratios[i_min] < 0.25 * ratios[-1]

    def test_flat_band_at_magic_angle(self):
        """魔角平带：W(1.05°) ≪ W(2°)（BM Fig. 2d）。"""
        m_magic = TwistedBilayerGraphene(theta_deg=1.05, n_shells=4)
        m_big = TwistedBilayerGraphene(theta_deg=2.0, n_shells=4)
        w_magic = m_magic.flat_band_width(n_grid=12)
        w_big = m_big.flat_band_width(n_grid=12)
        assert w_magic * 1e3 < 15.0          # meV 量级（文献 ~10 meV）
        assert w_big > 10 * w_magic

    def test_moire_path_endpoints(self):
        """高对称路径 Γ-M-K-Γ：起点终点为 BZ 中心，M/K 在 BZ 边界。"""
        m = TwistedBilayerGraphene(theta_deg=1.05)
        path, labels, ticks = m.moire_path(n_seg=8)
        assert labels == ["Γ", "M", "K", "Γ"]
        assert np.abs(path[0]).max() < 1e-14
        assert np.abs(path[-1]).max() < 1e-14
        # |K_m| = |b|/√3（BZ 角点）；|M_m| = |b|/2（边中点）
        assert np.linalg.norm(m.geom.bz_corner) == pytest.approx(
            np.linalg.norm(m.geom.b1) / np.sqrt(3), rel=1e-12)
        assert np.linalg.norm(m.geom.bz_edge_center) == pytest.approx(
            np.linalg.norm(m.geom.b1) / 2, rel=1e-12)

    def test_invalid_params(self):
        with pytest.raises(ValueError):
            TwistedBilayerGraphene(theta_deg=0.0)
        with pytest.raises(ValueError):
            TwistedBilayerGraphene(theta_deg=1.05, w=-0.1)
