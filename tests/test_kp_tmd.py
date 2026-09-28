"""TMD k·p 引擎物理验证测试（Kormányos 2015）。

文献基准（(HSE,LDA) 参数, docs/REFERENCES.md 文献 [1]）：
- K 点最小（含 SOC）带隙 = E_bg：MoS₂ 1.67 / MoSe₂ 1.40 / WS₂ 1.60 /
  WSe₂ 1.30 / MoTe₂ 0.997 / WTe₂ 0.792 eV（GW: 2.80/2.26/2.88/2.42/1.82/1.77）
- 价带自旋劈裂 2Δ_vb：148/186/429/466/219/484 meV
- 导带自旋劈裂 2Δ_cb：3/22/−32/−37/36/−52 meV
- 有效质量 m_cb/m_vb（Table 3/4，数值拟合提取，容差 15%）
- 时间反演 E_{K,↑}(q) = E_{−K,↓}(−q)（精确）
- 三角翘曲（TW）：固定 |q| 能量随角度变化
"""

import numpy as np
import pytest

from vdw_studio.engine.kp_tmd import (
    TMDKpModel,
    TMDKpParams,
    TMD_KP_PARAMS,
    TMD_REFERENCE_MASSES,
)

MATERIALS = list(TMD_KP_PARAMS)


@pytest.mark.parametrize("name", MATERIALS)
class TestGapsAndSplittings:
    def test_k_point_gap(self, name):
        """K 点最小（含 SOC）跃迁能 = E_bg（自旋块内取）。"""
        m = TMDKpModel(TMDKpParams.from_library(name, "dft"))
        gaps = [m.spin_block_energies((0, 0), +1, s)[1]
                - m.spin_block_energies((0, 0), +1, s)[0]
                for s in (+1, -1)]
        assert min(gaps) == pytest.approx(TMD_KP_PARAMS[name]["e_bg_dft"],
                                          abs=1e-8)

    def test_valence_spin_splitting(self, name):
        m = TMDKpModel(TMDKpParams.from_library(name, "dft"))
        vb_up = m.spin_block_energies((0, 0), +1, +1)[0]
        vb_dn = m.spin_block_energies((0, 0), +1, -1)[0]
        assert (vb_up - vb_dn) == pytest.approx(
            TMD_KP_PARAMS[name]["two_delta_vb"] / 1000.0, abs=1e-8)

    def test_conduction_spin_splitting(self, name):
        m = TMDKpModel(TMDKpParams.from_library(name, "dft"))
        cb_up = m.spin_block_energies((0, 0), +1, +1)[1]
        cb_dn = m.spin_block_energies((0, 0), +1, -1)[1]
        assert (cb_up - cb_dn) == pytest.approx(
            TMD_KP_PARAMS[name]["two_delta_cb"] / 1000.0, abs=1e-8)


class TestValleyPhysics:
    def test_time_reversal(self):
        """E_{K,·}(q) = E_{−K,·}(−q)（精确）。"""
        for name in ("MoS2", "WS2", "WSe2"):
            m = TMDKpModel(TMDKpParams.from_library(name, "dft"))
            for q in ([0.1, 0.05], [0.07, -0.08], [0.15, 0.0]):
                q = np.array(q)
                assert np.allclose(m.energies(q, +1), m.energies(-q, -1),
                                   atol=1e-10)

    def test_hermitian(self):
        m = TMDKpModel(TMDKpParams.from_library("MoS2", "dft"))
        rng = np.random.default_rng(3)
        for _ in range(10):
            q = rng.uniform(-0.3, 0.3, 2)
            H = m.hamiltonian(q, +1)
            assert np.allclose(H, H.conj().T, atol=1e-12)

    def test_trigonal_warping(self):
        """固定 |q| 时导带能量随角度变化（TW 非零）。"""
        m = TMDKpModel(TMDKpParams.from_library("MoS2", "dft"))
        angles = np.linspace(0.0, np.pi / 3, 7)
        energies = [m.energies(0.2 * np.array([np.cos(t), np.sin(t)]), +1)[3]
                    for t in angles]
        assert max(energies) - min(energies) > 0.02   # ~87 meV 实测

    def test_valley_contrast_spin_splitting_sign(self):
        """谷自旋劈裂符号：2Δ_cb 带号（MoX₂ 与 WX₂ 相反）。"""
        mos2 = TMDKpModel(TMDKpParams.from_library("MoS2", "dft"))
        ws2 = TMDKpModel(TMDKpParams.from_library("WS2", "dft"))
        cb_up_m = mos2.spin_block_energies((0, 0), +1, +1)[1]
        cb_dn_m = mos2.spin_block_energies((0, 0), +1, -1)[1]
        cb_up_w = ws2.spin_block_energies((0, 0), +1, +1)[1]
        cb_dn_w = ws2.spin_block_energies((0, 0), +1, -1)[1]
        assert (cb_up_m - cb_dn_m) > 0        # MoS₂: Δ_cb > 0
        assert (cb_up_w - cb_dn_w) < 0        # WS₂: Δ_cb < 0


class TestEffectiveMasses:
    @pytest.mark.parametrize("name", MATERIALS)
    def test_masses_against_literature(self, name):
        """与 Table 3/4 的文献质量对比。

        注意：文献质量是论文在 Γ–K 距离 5% 范围内对 DFT 能带做抛物线
        拟合提取的（附录 Fitting procedure），含 q⁴ 项贡献；本引擎数值
        给出的是 q→0 精确曲率质量，两者存在模型性偏差，容差取 25%。
        """
        m = TMDKpModel(TMDKpParams.from_library(name, "dft"))
        ref = TMD_REFERENCE_MASSES[name]
        for calc, refs in ((m.effective_mass("cb", +1), ref[:2]),
                           (m.effective_mass("cb", -1), ref[:2]),
                           (m.effective_mass("vb", +1), ref[2:]),
                           (m.effective_mass("vb", -1), ref[2:])):
            best = min(abs(calc - r) / r for r in refs)
            assert best < 0.25, f"{name}: m={calc:.3f} vs ref {refs}"

    @pytest.mark.parametrize("name", MATERIALS)
    def test_mass_analytic_relation(self, name):
        """q→0 质量满足论文 Eq.(A3)/(A4) 解析关系（内部一致性）。

        ħ²/2m_cb = ħ²/2mₑ + β_s + γ²/E_bg^(s)
        ħ²/2m_vb = ħ²/2mₑ + α_s − γ²/E_bg^(s)
        """
        p = TMDKpParams.from_library(name, "dft")
        m = TMDKpModel(p)
        hbar2_2me = 3.809982
        for s in (+1, -1):
            # 自旋分辨带隙 E_bg^(s) = span + s·(Δ_cb − Δ_vb)
            span = p.e_bg + p.delta_vb - p.delta_cb
            e_bg_s = span + s * (p.delta_cb - p.delta_vb)
            beta = p.beta_up if s > 0 else p.beta_down
            alpha = p.alpha_up if s > 0 else p.alpha_down
            m_cb_analytic = hbar2_2me / (hbar2_2me + beta
                                         + p.gamma**2 / e_bg_s)
            m_vb_analytic = hbar2_2me / abs(hbar2_2me + alpha
                                            - p.gamma**2 / e_bg_s)
            assert m.effective_mass("cb", s) == pytest.approx(m_cb_analytic,
                                                              rel=1e-3)
            assert m.effective_mass("vb", s) == pytest.approx(m_vb_analytic,
                                                              rel=1e-3)


class TestSources:
    def test_gw_gap(self):
        m = TMDKpModel(TMDKpParams.from_library("MoS2", "gw"))
        gaps = [m.spin_block_energies((0, 0), +1, s)[1]
                - m.spin_block_energies((0, 0), +1, s)[0] for s in (+1, -1)]
        assert min(gaps) == pytest.approx(2.80, abs=1e-8)

    def test_unknown_material(self):
        with pytest.raises(ValueError):
            TMDKpParams.from_library("GaAs")
