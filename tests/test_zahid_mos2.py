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


class TestGeometryAndFilling:
    """几何与填充约定（2026-10-14 独立核实后固化）。"""

    def test_trigonal_prismatic_eclipsed(self):
        """三棱柱配位：两个 S 位于同一面内空位（上下对齐，eclipsed）。

        1H 相的定义特征；此前的实现把两个 S 放在 (1/3,2/3)/(2/3,1/3)
        两个不同空位（60° 交错 = 八面体/1T 型），晶体场给反。
        参考：2H-MoS₂ 体相 P6₃/mmc（Mo 2c / S 4f）；两个 S 面内坐标
        相同、面外 ±dxx/2。
        """
        m = ZahidMoS2Model(soc=False)
        s_fracs = [site[1][:2] for site in m.sites if site[0] == "S"]
        assert len(s_fracs) == 2
        assert s_fracs[0] == pytest.approx(s_fracs[1], abs=1e-12)

    def test_s2_azimuth_matches_triangle(self):
        """两个 S 的方位角一致（相对 Mo），且面内距离 = a/√3。"""
        m = ZahidMoS2Model(soc=False)
        s1, s2 = [site[1] for site in m.sites if site[0] == "S"]
        c1 = m.lattice.frac_to_cart(s1)
        c2 = m.lattice.frac_to_cart(s2)
        assert np.hypot(*c1[:2]) == pytest.approx(m.a / np.sqrt(3), rel=1e-6)
        assert np.hypot(*c2[:2]) == pytest.approx(m.a / np.sqrt(3), rel=1e-6)
        az1 = np.arctan2(c1[1], c1[0])
        az2 = np.arctan2(c2[1], c2[0])
        assert az1 == pytest.approx(az2, abs=1e-12)

    def test_mo_sixfold_coordination(self):
        """Mo 六配位（三棱柱）、S 三配位（与结构构建器一致）。"""
        m = ZahidMoS2Model(soc=False)
        sm = [b for b in m.bonds if b[3] == "SMo"]
        assert len(sm) == 6
        per_s = {}
        for (i, j, _, _) in sm:
            per_s[i] = per_s.get(i, 0) + 1
        assert sorted(per_s.values()) == [3, 3]

    def test_neutral_filling_convention(self):
        """中性填充 = 18 电子/胞（Mo 4d⁵5s¹ + 2×S 3s²3p⁴）。

        含 SOC 时 54 态占 18（而非旧的"半填 27"）；无 SOC 时 9 个
        自旋简并能级对。预设 n_valence 已同步为 18。
        """
        from vdw_studio.presets import get_preset
        p = get_preset("mos2_sp3d5")
        assert p.n_valence == 18


class TestKnownIssue:
    def test_overlap_positive_definite(self):
        """S(k) 在测试 k 点上正定（Gram 矩阵判据，修正几何后复验）。"""
        m = ZahidMoS2Model(soc=False)
        for kf in [(0, 0), (1 / 3, 1 / 3), (0.2, 0.1), (0.4, 0.3)]:
            _, S = m.hamiltonian_overlap(kf)
            assert np.linalg.eigvalsh(S).min() > 1e-6

    def test_gap_not_yet_validated(self):
        """防幻觉声明：带隙/能级顺序尚未与文献对账（保持显式记录）。

        修正几何 + 填充 18 下：E[18]−E[17]（K 点，无 SOC）≈ 0.98 eV，
        文献目标 ~1.9 eV（含 SOC 1.805）；且 K 点晶体场顺序反转
        （带顶为 e′ 型 d_yz/d_zx）。装配自由度扫描（角色互换、
        通道族符号、106 组单/双通道符号）均未同时满足字符与带隙
        判据——待 Nanoskif 双中心坐标系约定。
        """
        m = ZahidMoS2Model(soc=False)
        E = m.energies_at([1 / 3, 1 / 3, 0.0])
        gap_now = E[18] - E[17]
        assert 0.5 < gap_now < 1.5          # 现实现的数值（非文献值）
        assert gap_now < 1.805 * 0.9        # 明确低于目标 → 不得作定量参考

    def test_bands_runs(self):
        """广义本征值可求解（不抛错）。"""
        m = ZahidMoS2Model(soc=False)
        e = m.bands([(1 / 3, 1 / 3)])
        assert e.shape == (1, 54)
        assert np.all(np.diff(e[0]) >= -1e-10)   # 升序

    def test_energies_at_shares_overlap_diagnostic(self, monkeypatch):
        """energies_at 与 bands 共用 S(k) 正定诊断。

        此前 energies_at 缺该检查，非正定 S(k) 会在 LAPACK 深处以
        无上下文的报错失败；而 analyze_gap / effective_mass 走的正是
        energies_at。
        """
        m = ZahidMoS2Model(soc=False)
        H, S = m.hamiltonian_overlap([1 / 3, 1 / 3, 0.0])
        monkeypatch.setattr(m, "hamiltonian_overlap", lambda k: (H, -S))
        with pytest.raises(RuntimeError, match="非正定"):
            m.energies_at([1 / 3, 1 / 3, 0.0])
        with pytest.raises(RuntimeError, match="非正定"):
            m.bands([[1 / 3, 1 / 3, 0.0]])
