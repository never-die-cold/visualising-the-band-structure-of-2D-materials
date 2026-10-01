"""谷物理分析测试：Berry 曲率、谷半 Chern、谷选择光学定则。

验证策略：
1. **合成大质量 Dirac 模型**（解析精确锚点）：
   Ω(q) = −γ²m/(2(γ²q²+m²)^{3/2}) 逐位吻合；
   谷 Chern = ∓1/2（拓扑半量子化）；q→0 圆二色性比 → ∞。
2. **TMD k·p 模型**（MoS₂，Kormányos 2015 参数）：
   时间反演 Ω_{K,↑}(q) = −Ω_{−K,↓}(−q)（精确）；
   谷间 Berry 曲率符号相反（谷 Hall 效应基础）；
   圆偏振选择在 K 与 −K 相反（谷-光耦合定则）。
"""

import numpy as np
import pytest

from vdw_studio.analysis.berry import (
    berry_curvature,
    optical_circular_dichroism,
    valley_chern,
)
from vdw_studio.engine.kp_tmd import TMDKpModel, TMDKpParams


def _massive_dirac(g: float = 5.75, m: float = 0.8):
    sx = np.array([[0, 1], [1, 0]], complex)
    sy = np.array([[0, -1j], [1j, 0]], complex)
    sz = np.array([[1, 0], [0, -1]], complex)

    def H(q):
        return g * q[0] * sx + g * q[1] * sy + m * sz
    return H


class TestSyntheticMassiveDirac:
    """合成模型 vs 解析（精确锚点）。

    注意: H 必须用 staticmethod 包装（类属性函数会经描述符协议
    变成绑定方法, 调用时多出 self 参数）。
    """

    H = staticmethod(_massive_dirac(g=5.75, m=0.8))

    @pytest.mark.parametrize("q", [0.05, 0.15, 0.4])
    def test_berry_curvature_analytic(self, q):
        num = berry_curvature(self.H, np.array([q, 0.0]), band=1)
        ana = -5.75 ** 2 * 0.8 / (2 * (5.75 ** 2 * q ** 2 + 0.8 ** 2) ** 1.5)
        assert num == pytest.approx(ana, rel=1e-4)

    def test_valley_chern_half_quantized(self):
        """谷 Chern = ∓1/2（大质量 Dirac 半量子化，大 disk 极限）。"""
        c_cb, _ = valley_chern(self.H, band=1, qmax=8.0,
                               n_theta=36, n_rad=24)
        c_vb, _ = valley_chern(self.H, band=0, qmax=8.0,
                               n_theta=36, n_rad=24)
        assert c_cb == pytest.approx(-0.5, abs=0.02)
        assert c_vb == pytest.approx(+0.5, abs=0.02)

    def test_valley_chern_sign_flips_with_mass(self):
        H_neg = _massive_dirac(g=5.75, m=-0.8)   # 模块级函数, 无绑定问题
        c, _ = valley_chern(H_neg, band=1, qmax=8.0, n_theta=36, n_rad=24)
        assert c == pytest.approx(+0.5, abs=0.02)

    def test_circular_dichroism_diverges(self):
        """q→0 单一圆偏振主导（比值发散）。"""
        r_small = optical_circular_dichroism(
            self.H, np.array([0.001, 0.0]), band_v=0, band_c=1)
        assert r_small.ratio > 1e4
        r_mid = optical_circular_dichroism(
            self.H, np.array([0.05, 0.0]), band_v=0, band_c=1)
        assert r_mid.ratio > r_small.ratio * 0  # 结构性: 比值随 q 增大而减小
        assert r_mid.ratio < r_small.ratio


class TestTMDValleyPhysics:
    kp = TMDKpModel(TMDKpParams.from_library("MoS2", "dft"))

    def _omega_vb(self, valley, q):
        """价带 Berry 曲率（自旋↑块, 通过带序号定位）。"""
        return berry_curvature(
            lambda qq: self.kp.hamiltonian(qq, valley), q, band=1)

    def test_time_reversal_of_berry_curvature(self):
        """TR: Ω_n^{K}(q) = −Ω_n^{−K}(−q)（同带指标, 精确）。"""
        for q in ([0.05, 0.02], [0.1, -0.05]):
            q = np.array(q)
            for band in range(4):
                w1 = berry_curvature(
                    lambda qq: self.kp.hamiltonian(qq, +1), q, band=band)
                w2 = berry_curvature(
                    lambda qq: self.kp.hamiltonian(qq, -1), -q, band=band)
                assert w1 == pytest.approx(-w2, rel=1e-6), (band, w1, w2)

    def test_valley_contrasting_sign(self):
        """两谷价带顶 Berry 曲率符号相反（谷 Hall 的微观来源）。"""
        w_K = self._omega_vb(+1, np.array([0.05, 0.0]))    # vb↑ = band 1
        w_Kp = berry_curvature(                            # −K 谷 vb↓ = band 1
            lambda qq: self.kp.hamiltonian(qq, -1),
            np.array([0.05, 0.0]), band=1)
        assert w_K * w_Kp < 0
        assert abs(w_K) > 0.1    # 量级: 数 Å² (与大质量 Dirac 估计一致)

    def test_berry_curvature_magnitude(self):
        """|Ω_vb(K)| 与大质量 Dirac 估计同量级 (~γ²·2/Δ_eff²·?)."""
        w = self._omega_vb(+1, np.array([0.02, 0.0]))
        # 大质量 Dirac 量级: 2γ²Δ/Δ³ ~ 2γ²/Δ² ~ 2·2.76²/1.67² ≈ 5.5 Å²
        assert 1.0 < abs(w) < 30.0

    def test_optical_selection_rule_reversal(self):
        """q→0 谷选择圆偏振：K 与 −K 主导偏振相反（谷-光耦合定则）。

        带指标（MoS₂, K 点, 能量升序）: vb↓(0), vb↑(1), cb↓(2), cb↑(3)；
        A 激子通道 = 自旋守恒的 vb↑→cb↑ (1→3)。
        """
        d_K = optical_circular_dichroism(
            lambda qq: self.kp.hamiltonian(qq, +1),
            np.array([0.001, 0.0]), band_v=1, band_c=3)
        d_Kp = optical_circular_dichroism(
            lambda qq: self.kp.hamiltonian(qq, -1),
            np.array([0.001, 0.0]), band_v=1, band_c=3)
        # 各谷内单一偏振压倒性主导
        assert d_K.ratio > 100 and d_Kp.ratio > 100
        # 两谷主导偏振相反
        assert d_K.dominant != d_Kp.dominant

    def test_A_exciton_spin_conserving_transition(self):
        """自旋块内跃迁主导（A 激子: 自旋守恒通道），跨自旋跃迁为零。"""
        q = np.array([0.001, 0.0])
        H = self.kp.hamiltonian(q, +1)
        from vdw_studio.analysis.berry import _velocity_matrices
        Hx, Hy = _velocity_matrices(lambda qq: self.kp.hamiltonian(qq, +1), q)
        E, V = np.linalg.eigh(H)
        vx = V.conj().T @ Hx @ V
        # 跨自旋矩阵元 (↑块→↓块) 应严格为零 (块解耦)
        assert abs(vx[0, 3]) < 1e-12
        assert abs(vx[1, 2]) < 1e-12
