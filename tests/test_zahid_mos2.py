"""sp³d⁵ Slater-Koster 引擎测试（Zahid MoS₂）。

已验证内容：
- SK 角因子：轴向闭合式与求和规则（与 Slater-Koster 1954 一致）；
- 键几何：Mo–S 6 键（键长 2.414 Å）+ 同亚层 S–S/Mo–Mo 各 6 键；
- H(k)、S(k) 厄米性；
- on-site SOC 的 L·S 多重态结构（p: λ/2 与 −λ；d: λ 与 −1.5λ）；
- 参数转录抽查（对照 arXiv:1304.0074 Table 3）。

已知问题（见 docs/REFERENCES.md "待确认的约定"）：拟合重叠积分的
Nanoskif 相位约定未在原文给出，S(k) 正定性未达成，`bands()` 对非正定
S(k) 显式抛错；带隙验证（1.805/1.969 eV）待约定确认后进行。
"""

import numpy as np
import pytest
import scipy.linalg

from vdw_studio.engine.slater_koster import sk_element
from vdw_studio.engine.zahid_mos2 import (
    INTEGRALS_MOMO_H,
    INTEGRALS_MOMO_S,
    INTEGRALS_SMO_H,
    INTEGRALS_SMO_S,
    INTEGRALS_SS_H,
    INTEGRALS_SS_S,
    ONSITE,
    ZahidMoS2Model,
    angular_momentum_matrices,
    onsite_soc,
)

AX = np.array([1.0, 0, 0])
AY = np.array([0, 1.0, 0])
AZ = np.array([0, 0, 1.0])


class TestSKAngularFactors:
    """SK 角因子与闭合式/求和规则对账（Slater-Koster 1954 约定）。"""

    def test_pp_sigma_along_bond(self):
        V = {"ppσ": 1.0, "ppπ": 1.0}
        assert sk_element("px", "px", AX, V) == pytest.approx(1.0)   # σ
        assert sk_element("px", "px", AZ, V) == pytest.approx(1.0)   # π

    def test_pp_orthogonal(self):
        V = {"ppσ": 1.0, "ppπ": 1.0}
        assert sk_element("px", "py", AZ, V) == pytest.approx(0.0)
        assert sk_element("px", "py", AX, V) == pytest.approx(0.0)

    def test_sp_direction_cosine(self):
        assert sk_element("s", "px", AX, {"spσ": 1.0}) == pytest.approx(1.0)
        assert sk_element("s", "px", AY, {"spσ": 1.0}) == pytest.approx(0.0)
        assert sk_element("s", "pz", AZ, {"spσ": 1.0}) == pytest.approx(1.0)

    def test_sd_closed_form(self):
        # (s, dz2): [n² − (l²+m²)/2]
        assert sk_element("s", "dz2", AZ, {"sdσ": 1.0}) == pytest.approx(1.0)
        assert sk_element("s", "dz2", AX, {"sdσ": 1.0}) == pytest.approx(-0.5)
        assert sk_element("s", "dxy", AZ, {"sdσ": 1.0}) == pytest.approx(0.0)

    def test_pd_completeness(self):
        """p-d 通道完备性：Σ_β M[α,β]² = V_σ²C_{α,σ}² + V_π²|C_{α,π}|²。

        （一般方向下 α 的 σ/π 分解不纯，故为加权形式；纯 σ 方向
        退化为 V_σ²+V_π²。）
        """
        from vdw_studio.engine.zahid_mos2 import _channel_coeffs
        rng = np.random.default_rng(7)
        for _ in range(5):
            rhat = rng.normal(size=3)
            rhat /= np.linalg.norm(rhat)
            V = {"pdσ": 0.7, "pdπ": -0.4}
            ca = _channel_coeffs("px", rhat)
            expected = (0.7 ** 2 * ca["sigma"] ** 2
                        + 0.4 ** 2 * (ca["pi"][0] ** 2 + ca["pi"][1] ** 2))
            tot = sum(sk_element("px", d, rhat, V) ** 2
                      for d in ("dxy", "dyz", "dzx", "dx2-y2", "dz2"))
            assert tot == pytest.approx(expected, abs=1e-10)

    def test_dd_completeness(self):
        """d-d 通道完备性（含 δ 通道的加权求和）。"""
        from vdw_studio.engine.zahid_mos2 import _channel_coeffs
        rng = np.random.default_rng(11)
        for _ in range(5):
            rhat = rng.normal(size=3)
            rhat /= np.linalg.norm(rhat)
            V = {"ddσ": 0.9, "ddπ": 0.5, "ddδ": 0.2}
            ca = _channel_coeffs("dxy", rhat)
            expected = (0.9 ** 2 * ca["sigma"] ** 2
                        + 0.5 ** 2 * (ca["pi"][0] ** 2 + ca["pi"][1] ** 2)
                        + 0.2 ** 2 * (ca["delta"][0] ** 2 + ca["delta"][1] ** 2))
            tot = sum(sk_element("dxy", d, rhat, V) ** 2
                      for d in ("dxy", "dyz", "dzx", "dx2-y2", "dz2"))
            assert tot == pytest.approx(expected, abs=1e-10)

    def test_dd_axis_values(self):
        # (dz2, dz2) 沿 x: (1/2)²Vσ + (3/4)Vδ（π 通道为 0）
        V = {"ddσ": 1.0, "ddπ": 1.0, "ddδ": 1.0}
        assert sk_element("dz2", "dz2", AX, V) == pytest.approx(1.0)
        assert sk_element("dz2", "dz2", AZ, V) == pytest.approx(1.0)
        # (dxy, dxy) 沿 x = Vπ
        assert sk_element("dxy", "dxy", AX, V) == pytest.approx(1.0)


class TestSOC:
    def test_p_shell_multiplet(self):
        """p 壳 L·S：j=3/2 四重态 +λ/2，j=1/2 二重态 −λ。"""
        H = onsite_soc(0.3)  # λ_p = λ_d = 0.3
        w = scipy.linalg.eigvalsh(H)
        # p 壳本征值: {+0.15 ×4, −0.3 ×2}; s: 0 ×2
        for target, count in ((0.15, 4), (-0.3, 2), (0.0, 2)):
            hits = sum(1 for x in w if abs(x - target) < 1e-9)
            assert hits == count, f"p/d 多重态不符: {sorted(w)}"

    def test_d_shell_multiplet(self):
        """d 壳 L·S：j=5/2 六重态 +λ，j=3/2 四重态 −1.5λ。"""
        lam = 0.2
        H = onsite_soc(lam)
        w = scipy.linalg.eigvalsh(H)
        for target, count in ((lam, 6), (-1.5 * lam, 4)):
            hits = sum(1 for x in w if abs(x - target) < 1e-9)
            assert hits == count

    def test_hermitian(self):
        H = onsite_soc(0.5)
        assert np.allclose(H, H.conj().T)


class TestBondGeometry:
    m = ZahidMoS2Model(soc=False)

    def test_bond_counts(self):
        kinds = [k for _, _, _, k in self.m.bonds]
        assert kinds.count("SMo") == 6
        assert kinds.count("SS") == 6       # 两个 S 亚层各 3 个代表镜像
        assert kinds.count("MoMo") == 3     # 3 个代表镜像（±对折叠）

    def test_bond_length(self):
        """Mo–S 键长 = sqrt(a²/3 + dxx²/4) ≈ 2.414 Å。"""
        expected = np.sqrt(3.179 ** 2 / 3 + (3.135 / 2) ** 2)
        for (i, j, d, kind) in self.m.bonds:
            if kind != "SMo":
                continue
            dist = np.linalg.norm(self.m.lattice.frac_to_cart(d))
            assert dist == pytest.approx(expected, abs=1e-6)

    def test_same_species_distance(self):
        for (i, j, d, kind) in self.m.bonds:
            if kind in ("SS", "MoMo"):
                dist = np.linalg.norm(self.m.lattice.frac_to_cart(d))
                assert dist == pytest.approx(3.179, abs=1e-6)


class TestMatrixAssembly:
    m = ZahidMoS2Model(soc=False)

    @pytest.mark.parametrize("kf", [(0, 0), (1 / 3, 1 / 3), (0.2, 0.1)])
    def test_hermitian(self, kf):
        H, S = self.m.hamiltonian_overlap(kf)
        assert H.shape == (54, 54) and S.shape == (54, 54)
        assert np.allclose(H, H.conj().T, atol=1e-12)
        assert np.allclose(S, S.conj().T, atol=1e-12)

    def test_onsite_diagonal(self):
        """对角元 = on-site 能量 + Γ 点同种原子近邻 hopping 求和。

        Mo–s 对角元: E_s(Mo) + 6 × ssσ(Mo,Mo)；S–pz: E_p(S) + 6 × ppσ/π 组合。
        """
        H, S = self.m.hamiltonian_overlap((0, 0))
        from vdw_studio.engine.zahid_mos2 import INTEGRALS_MOMO_H
        expected_mo_s = ONSITE["Mo"]["s"] + 6 * INTEGRALS_MOMO_H["ssσ"]
        assert H[0, 0].real == pytest.approx(expected_mo_s)
        # 重叠矩阵对角 = 1 + 6 × ssσ(Mo,Mo) 重叠
        from vdw_studio.engine.zahid_mos2 import INTEGRALS_MOMO_S
        assert S[0, 0].real == pytest.approx(
            1.0 + 6 * INTEGRALS_MOMO_S["ssσ"])

    def test_parameter_transcription(self):
        """参数表抽查（防转录错误）。"""
        assert INTEGRALS_SMO_H["pdσ"] == -2.8732
        assert INTEGRALS_SMO_H["dpπ"] == -1.9408
        assert INTEGRALS_SMO_S["psσ"] == 0.1765
        assert INTEGRALS_SS_H["ddσ"] == 0.8347
        assert INTEGRALS_MOMO_H["spσ"] == 1.0910
        assert INTEGRALS_MOMO_S["ddδ"] == 0.0432
        assert ONSITE["Mo"]["d"] == 2.6429
        assert ONSITE["S"]["lambda"] == 0.2129


class TestKnownIssue:
    def test_overlap_positive_definite(self):
        """S(k) 在测试 k 点上正定（Gram 矩阵判据）。

        注：早期版本因未加入 on-site 单位重叠而出现非正定；加入后已解决。
        剩余问题：带隙数值与文献目标（1.805 eV）尚未吻合，
        待 Nanoskif 相位约定确认（见 docs/REFERENCES.md）。
        """
        m = ZahidMoS2Model(soc=False)
        for kf in [(0, 0), (1 / 3, 1 / 3), (0.2, 0.1), (0.4, 0.3)]:
            _, S = m.hamiltonian_overlap(kf)
            assert np.linalg.eigvalsh(S).min() > 1e-6

    def test_bands_runs(self):
        """广义本征值可求解（不抛错）。"""
        m = ZahidMoS2Model(soc=False)
        e = m.bands([(1 / 3, 1 / 3)])
        assert e.shape == (1, 54)
        assert np.all(np.diff(e[0]) >= -1e-10)   # 升序
