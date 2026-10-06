"""轨道磁矩与谷 Zeeman 测试（Phase 4.2）。

验证策略（三路独立对账 + 文献锚点）：
1. **RMP 原式 vs 展开式**：Eq. (wave:m) 的 ∇u 有限差分直接实现
   （规范固定）与速度矩阵元展开式逐点一致（相互独立的两条实现路径）；
2. **两带解析**：大质量 Dirac 模型
   m_n(q) = τ·μ*_B·m²/(m²+g²q²)（μ*_B = m₀/m*）逐位吻合；
   带边极限 m(K) = τ·(m₀/m*)·μ_B，m* 由 effective_mass 独立计算；
3. **TMD k·p 模型**（MoS₂）：时间反演 m(K,q) = −m(−K,−q)（精确）；
   自旋块内两带关系（精确）；谷间符号相反；量级 ~μ_B·(m₀/m*)。

文献锚点：Xiao, Chang & Niu, Rev. Mod. Phys. 82, 1959 (2010)
（arXiv:0907.2021）：Eq. (wave:m) 定义、带边 m(τ_z) = τ_z μ*_B、
Zeeman 项 ε_M = ε − m·B。
"""

import numpy as np
import pytest

from vdw_studio.analysis.berry import berry_curvature
from vdw_studio.analysis.orbital_moment import (
    HBAR2_OVER_2M0,
    MU_B_MEV_PER_T,
    orbital_moment,
    orbital_moment_map,
    valley_zeeman_splitting,
)

from vdw_studio.engine.kp_tmd import TMDKpModel, TMDKpParams


def _massive_dirac(g: float = 5.75, m: float = 0.8, tau: int = 1):
    """大质量 Dirac 模型工厂（τ=±1 谷指标）。"""
    sx = np.array([[0, 1], [1, 0]], complex)
    sy = np.array([[0, -1j], [1j, 0]], complex)
    sz = np.array([[1, 0], [0, -1]], complex)

    def H(q):
        return g * (tau * q[0] * sx + q[1] * sy) + m * sz
    return H


def _moment_by_finite_difference(hamiltonian, q, band: int,
                                 dq: float = 1e-5) -> float:
    """RMP Eq. (wave:m) 的直接实现（独立校验路径）。

    m = −i(e/2ħ)⟨∇_q u| × [H−ε]|∇_q u⟩：∇u 用规范固定的有限差分，
    归一后 m[μ_B] = −Im(A)/(ħ²/2m₀)，A = ⟨∂_x u|(H−ε_n)|∂_y u⟩。
    （符号约定已与速度矩阵元展开式对账，见 tests 与模块文档。）
    """
    q = np.asarray(q, dtype=float)
    H0 = hamiltonian(q)
    E0, V0 = np.linalg.eigh(H0)

    def eig_gauge_fixed(H1):
        E1, V1 = np.linalg.eigh(H1)
        for i in range(V1.shape[1]):
            ov = V0.conj().T @ V1[:, i]
            j = int(np.argmax(np.abs(ov)))
            V1[:, i] *= np.exp(-1j * np.angle(ov[j]))
        return V1

    Vx = eig_gauge_fixed(hamiltonian(q + np.array([dq, 0.0])))
    Vy = eig_gauge_fixed(hamiltonian(q + np.array([0.0, dq])))
    dudx = (Vx - V0) / dq
    dudy = (Vy - V0) / dq
    O = hamiltonian(q) - E0[band] * np.eye(H0.shape[0])
    A = dudx[:, band].conj() @ O @ dudy[:, band]
    return float(-np.imag(A) / HBAR2_OVER_2M0)


class TestSyntheticMassiveDirac:
    """合成模型：三路独立实现 + 解析精确锚点。"""

    g, m = 5.75, 0.8
    H = staticmethod(_massive_dirac(g=5.75, m=0.8, tau=1))

    def test_finite_difference_matches_velocity_expansion(self):
        """RMP 原式（FD）与速度矩阵元展开式逐点一致。"""
        q = np.array([0.03, 0.02])
        for band in range(2):
            m_fd = _moment_by_finite_difference(self.H, q, band)
            m_ex = orbital_moment(self.H, q, band)
            assert m_fd == pytest.approx(m_ex, rel=5e-3), (band, m_fd, m_ex)

    def test_two_band_relation_with_berry_curvature(self):
        """两带关系 m_n = −(ε_n−ε_m)Ω_n/(2·ħ²/2m₀)（精确）。"""
        q = np.array([0.04, -0.03])
        H0 = self.H(q)
        E = np.linalg.eigvalsh(H0)
        for band in range(2):
            m_num = orbital_moment(self.H, q, band)
            om = berry_curvature(self.H, q, band=band)
            m_rel = -(E[band] - E[1 - band]) * om / 2 / HBAR2_OVER_2M0
            assert m_num == pytest.approx(m_rel, rel=1e-6)

    def test_q_dependence_analytic(self):
        """m_c(q) = τ·μ*_B·m²/(m²+g²q²)（解析精确）。"""
        for qv in (0.05, 0.15, 0.4):
            q = np.array([qv, 0.0])
            num = orbital_moment(self.H, q, band=1)
            eps2 = self.m ** 2 + (self.g * qv) ** 2
            ana = self.g ** 2 * self.m / (2 * eps2) / HBAR2_OVER_2M0
            assert num == pytest.approx(ana, rel=1e-5), qv

    def test_band_edge_bohr_magneton_enhancement(self):
        """带边极限 m(K) = (m₀/m*)·μ_B，m* 由色散二阶导独立求出。"""
        q = np.array([1e-4, 0.0])
        m_edge = orbital_moment(self.H, q, band=1)
        # m* 来自色散曲率（与磁矩完全独立的数值路径）:
        #   ħ²/2m* = (1/2)·d²E/dq² → m*[m₀] = (ħ²/2m₀)/((1/2)·d²E/dq²)
        h = 1e-4
        e_p = np.linalg.eigvalsh(self.H(np.array([h, 0.0])))[1]
        e_m = np.linalg.eigvalsh(self.H(np.array([-h, 0.0])))[1]
        e_0 = np.linalg.eigvalsh(self.H(np.array([0.0, 0.0])))[1]
        d2e = (e_p + e_m - 2 * e_0) / h ** 2
        m_star = HBAR2_OVER_2M0 / (d2e / 2)
        assert m_edge == pytest.approx(1.0 / m_star, rel=1e-3)

    def test_time_reversal_valley_odd(self):
        """TR: m(τ,q) = −m(−τ,−q)；谷心磁矩随 τ 变号。"""
        q = np.array([0.05, 0.02])
        h_k = _massive_dirac(g=self.g, m=self.m, tau=+1)
        h_km = _massive_dirac(g=self.g, m=self.m, tau=-1)
        m_k = orbital_moment(h_k, q, band=1)
        m_km = orbital_moment(h_km, -q, band=1)
        assert m_k == pytest.approx(-m_km, rel=1e-8)
        # 大质量 Dirac: m_c(τ, q) = τ·f(|q|) → 谷心值符号 = τ
        m_k_center = orbital_moment(h_k, np.array([1e-5, 0.0]), band=1)
        assert m_k_center > 0

    def test_moment_map_shape(self):
        """热图接口：形状与谷心值一致性。"""
        out = orbital_moment_map(self.H, band=1, qmax=0.2, n_grid=9)
        assert out["moment"].shape == (9, 9)
        center = orbital_moment(self.H, np.array([0.0, 0.0]), band=1)
        assert out["moment_center"] == pytest.approx(center, rel=1e-6)


class TestTMDValleyMagneticMoment:
    """MoS₂ k·p 模型（Kormányos 2015 参数）谷磁矩。"""

    kp = TMDKpModel(TMDKpParams.from_library("MoS2", "dft"))

    def test_time_reversal_exact(self):
        """TR: m_n^K(q) = −m_n^{−K}(−q)（全部 4 带，精确）。"""
        for q in ([0.05, 0.02], [0.1, -0.05]):
            q = np.array(q)
            for band in range(4):
                m1 = orbital_moment(
                    lambda qq: self.kp.hamiltonian(qq, +1), q, band=band)
                m2 = orbital_moment(
                    lambda qq: self.kp.hamiltonian(qq, -1), -q, band=band)
                assert m1 == pytest.approx(-m2, rel=1e-6), (band, q, m1, m2)

    def test_two_band_relation_in_spin_block(self):
        """自旋块内两带关系精确（跨自旋耦合严格为零 → 块内单一带内耦合）。

        带序号: vb↓(0), vb↑(1), cb↓(2), cb↑(3)。价带↑(1) 的块内伙伴是 cb↑(3)。
        """
        q = np.array([0.03, 0.02])
        E = np.linalg.eigvalsh(self.kp.hamiltonian(q, +1))
        for n, partner in ((1, 3), (0, 2)):
            m_num = orbital_moment(
                lambda qq: self.kp.hamiltonian(qq, +1), q, band=n)
            om = berry_curvature(
                lambda qq: self.kp.hamiltonian(qq, +1), q, band=n)
            m_rel = -(E[n] - E[partner]) * om / 2 / HBAR2_OVER_2M0
            assert m_num == pytest.approx(m_rel, rel=1e-6), n

    def test_valence_band_moment_magnitude(self):
        """|m_vb(K)| ~ (m₀/m*)·μ_B 量级：MoS₂ 价带 m* ≈ 0.55 m₀ → ~1.8 μ_B。

        k·p 含翘曲修正，取宽窗口 0.5–10 μ_B；两谷符号相反。
        """
        q = np.array([1e-4, 0.0])
        m_k = orbital_moment(lambda qq: self.kp.hamiltonian(qq, +1), q, band=1)
        m_km = orbital_moment(lambda qq: self.kp.hamiltonian(qq, -1), q, band=1)
        assert 0.5 < abs(m_k) < 10.0
        assert m_k * m_km < 0


class TestValleyZeeman:
    """Zeeman 劈裂换算与实验量级。"""

    def test_splitting_arithmetic(self):
        r = valley_zeeman_splitting(moment_mu_b=1.5, b_tesla=2.0)
        assert r.splitting_mev == pytest.approx(
            -2.0 * 1.5 * MU_B_MEV_PER_T * 2.0, rel=1e-12)
        assert r.g_valley == pytest.approx(-3.0)

    def test_zero_field_zero_splitting(self):
        assert valley_zeeman_splitting(2.0, 0.0).splitting_mev == 0.0

    def test_mos2_scale_measured_window(self):
        """MoS₂ 价带磁矩 → 劈裂量级与谷 Zeeman 实验窗口一致。

        实验（MacNeill 2015 / Aivazian 2015 等）：TMD 谷劈裂 ~0.1–0.3 meV/T
        （轨道+自旋贡献合计）。纯轨道部分取 |m| ≈ 1–3 μ_B → 0.1–0.35 meV/T。
        """
        q = np.array([1e-4, 0.0])
        kp = TMDKpModel(TMDKpParams.from_library("MoS2", "dft"))
        m_vb = orbital_moment(lambda qq: kp.hamiltonian(qq, +1), q, band=1)
        r = valley_zeeman_splitting(m_vb, b_tesla=1.0)
        assert 0.05 < abs(r.splitting_mev) < 0.4
