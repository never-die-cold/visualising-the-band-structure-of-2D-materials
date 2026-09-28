"""材料预设库端到端测试：结构 → 引擎 → 分析 → 文献参考对账。"""

import pytest

from vdw_studio.analysis import fermi_velocity, principal_masses
from vdw_studio.presets import PRESETS, get_preset, list_presets, run_preset


class TestRegistry:
    def test_all_presets_present(self):
        assert set(list_presets()) == {
            "graphene", "hbn", "silicene", "phosphorene",
            "mos2_kp", "mose2_kp", "ws2_kp", "wse2_kp", "mote2_kp", "wte2_kp",
            "mos2_sp3d5",
        }

    def test_lookup_case_insensitive(self):
        assert get_preset("MoS2_KP").key == "mos2_kp"
        with pytest.raises(ValueError):
            get_preset("NaCl")

    def test_source_attribution(self):
        """每个预设必须标注参数出处（防幻觉要求）。"""
        for key in list_presets():
            p = get_preset(key)
            assert len(p.source) > 10, f"{key} 缺少文献出处"


class TestEndToEnd:
    """每个预设：结构化学式一致 + 带隙与文献参考对账。"""

    # Hill 记法（字母序）：MoS2 天然有序，W 系材料为 S2W/Se2W/Te2W
    FORMULAS = {
        "graphene": "C2", "hbn": "BN", "silicene": "Si2",
        "phosphorene": "P4", "mos2_kp": "MoS2", "mose2_kp": "MoSe2",
        "ws2_kp": "S2W", "wse2_kp": "Se2W", "mote2_kp": "MoTe2",
        "wte2_kp": "Te2W", "mos2_sp3d5": "MoS2",
    }

    @pytest.mark.parametrize("key", list_presets())
    def test_structure_formula(self, key):
        """结构化学式正确（只建结构，不跑能带求解）。"""
        p = get_preset(key)
        assert p.make_structure(p.structure_key).formula_str == self.FORMULAS[key]

    @pytest.mark.parametrize("key", list_presets())
    def test_gap_against_reference(self, key):
        p = get_preset(key)
        r = run_preset(key, n_per_segment=30)
        gap = r["gap"]
        if p.gap_ref is not None:
            assert r["gap_matches_ref"] is True, \
                f"{key}: gap={gap.gap} vs ref {p.gap_ref}"
        elif key in ("graphene", "silicene"):
            assert gap.gap is None, f"{key} 应为零带隙"


class TestGraphenePreset:
    def test_fermi_velocity_reference(self):
        r = run_preset("graphene", n_per_segment=20)
        v = fermi_velocity(r["model"], (1 / 3, 1 / 3), band=1,
                           direction=(1 / 3, -1 / 3))
        assert v == pytest.approx(get_preset("graphene").v_fermi_ref, rel=0.02)


class TestPhosphorenePreset:
    def test_mass_reference(self):
        r = run_preset("phosphorene", n_per_segment=20)
        m1, m2, _ = principal_masses(r["model"], (0, 0), band=2, dk=1e-4)
        ref_light, ref_heavy = get_preset("phosphorene").mass_ref
        assert min(m1, m2) == pytest.approx(ref_light, rel=0.08)
        assert max(m1, m2) == pytest.approx(ref_heavy, rel=0.08)


class TestTMDKpPresets:
    @pytest.mark.parametrize("key", ["mos2_kp", "ws2_kp", "wse2_kp"])
    def test_direct_gap_at_K(self, key):
        r = run_preset(key, n_per_segment=20)
        assert r["gap"].direct is True
        assert r["gap"].vbm_label == "K"

    def test_valley_physics_available(self):
        """k·p 预设的模型应支持谷分析（我们特色功能的基础）。

        价带自旋序在两谷相反：K 谷 VB↑ 在上，−K 谷 VB↓ 在上
        （谷-自旋锁定，时间反演伴）。
        """
        r = run_preset("mos2_kp", n_per_segment=20)
        kp = r["model"]
        vb_up_K = kp.spin_block_energies((0, 0), +1, +1)[0]
        vb_dn_K = kp.spin_block_energies((0, 0), +1, -1)[0]
        vb_up_Kp = kp.spin_block_energies((0, 0), -1, +1)[0]
        vb_dn_Kp = kp.spin_block_energies((0, 0), -1, -1)[0]
        assert (vb_up_K - vb_dn_K) * (vb_up_Kp - vb_dn_Kp) < 0
