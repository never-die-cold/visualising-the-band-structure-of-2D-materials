"""多层堆叠引擎的物理验证测试（大目标 D 引擎侧）。

验证锚点全部可解析：
- Harrison 标度律 t(d) = t(d₀)·(d₀/d)^η（逐通道）；
- 蜂窝族：AB 双层 K 点谱 {0, 0, ±γ₁} 与 BilayerGrapheneModel 逐位一致；
  ABA/ABC 三层 K 点谱 {0×4, ±√2γ₁} / {0×2, ±γ₁×2}（f(K)=0 解耦 +
  γ₁ 链的精确结果，堆叠敏感性锚点）；二聚位耦合随间距按 d⁻² 标度；
- sp³d⁵ 族：N=1 与 ZahidMoS2Model（同几何）逐位一致；解耦极限下
  2H/3R 多层谱 = 单层谱平铺（含一般 k 点——rot180 层镜像规范回归）；
  conjugate_fill=False 的非厄米负结果保持；
- 诚实的阻塞声明：Zahid 表 + 现装配约定下单层在一般 k 点带交叠
  （金属化），gap-vs-N 定量对账被 T16-SK 阻塞——该状态被测试固化，
  约定修复后此测试应更新。
"""

import numpy as np
import pytest

from vdw_studio.structure.builders import graphene, hbn, tmd
from vdw_studio.structure.multilayer import (
    LayerStep,
    stack_2h,
    stack_3r,
    stack_bernald,
    stack_sequence,
)
from vdw_studio.engine.models import BilayerGrapheneModel, HoneycombModel
from vdw_studio.engine.multilayer import (
    HARRISON_EXPONENT,
    HoneycombMultilayerModel,
    MoS2MultilayerModel,
    harrison_scale,
    scaled_integrals,
)
from vdw_studio.engine.zahid_mos2 import ZahidMoS2Model

K_POINT = (1.0 / 3, 1.0 / 3)
GENERIC_K = [(0.1, 0.05), (0.2, -0.13), (0.5, 0.0), (0.0, 0.5), (1 / 3, 0.0)]


class TestHarrisonScaling:
    def test_power_law(self):
        assert harrison_scale(0.4, 3.35, 3.35) == pytest.approx(0.4)
        assert harrison_scale(0.4, 3.35, 6.70) == pytest.approx(0.1)
        assert harrison_scale(1.0, 2.0, 4.0, exponent=3.0) \
            == pytest.approx(0.125)

    def test_default_is_harrison_universal_two(self):
        assert HARRISON_EXPONENT == 2.0

    def test_scaled_integrals_per_channel(self):
        table = {"ppσ": 1.0, "ppπ": -0.5, "ddσ": 2.0}
        out = scaled_integrals(table, 2.0, 4.0,
                               {"ppσ": 2.0, "ppπ": 3.0})  # ddσ 用默认 2
        assert out["ppσ"] == pytest.approx(0.25)
        assert out["ppπ"] == pytest.approx(-0.5 * 0.125)
        assert out["ddσ"] == pytest.approx(0.5)

    def test_invalid_inputs(self):
        with pytest.raises(ValueError):
            harrison_scale(1.0, 0.0, 1.0)
        with pytest.raises(ValueError):
            harrison_scale(1.0, 2.0, -1.0)


class TestHoneycombMultilayer:
    def test_monolayer_pass_through(self):
        """N=1 堆叠 + 生成器 = 单层 HoneycombModel 谱（生成器不添乱）。"""
        st = stack_sequence(graphene(), [LayerStep()], 3.35)
        m_gen = HoneycombMultilayerModel(st)
        m_ref = HoneycombModel(a=2.46)
        ks = np.array([[0, 0], K_POINT, [0.5, 0.0], [0.2, 0.1]])
        np.testing.assert_allclose(m_gen.bands(ks), m_ref.bands(ks),
                                   atol=1e-10)

    def test_ab_bilayer_matches_mccann_model(self):
        """AB 双层与手写 BilayerGrapheneModel 逐位一致（取代目标的对账）。"""
        m_gen = HoneycombMultilayerModel(stack_bernald(graphene(), 2, 3.35))
        m_ref = BilayerGrapheneModel()
        ks = np.array([[0, 0], K_POINT, [0.5, 0.0], [0.2, 0.1], [1 / 3, 0.0]])
        np.testing.assert_allclose(m_gen.bands(ks), m_ref.bands(ks),
                                   atol=1e-12)

    def test_ab_bilayer_k_spectrum(self):
        m = HoneycombMultilayerModel(stack_bernald(graphene(), 2, 3.35))
        e = m.bands([K_POINT])[0]
        np.testing.assert_allclose(e, [-0.4, 0.0, 0.0, 0.4], atol=1e-10)

    def test_dimer_coupling_scales_harrison_d_minus_two(self):
        """二聚位耦合 γ₁(d) = 0.4·(3.35/d)²：K 点 max|E| 精确跟随。"""
        for d in (3.35, 4.0):
            m = HoneycombMultilayerModel(stack_bernald(graphene(), 2, d))
            e = m.bands([K_POINT])[0]
            assert np.abs(e).max() == pytest.approx(0.4 * (3.35 / d) ** 2,
                                                    rel=1e-10)

    def test_decoupled_limit_all_zero(self):
        """t = γ₁ = 0 且无在位能：全谱恒零（退化极限自检）。"""
        m = HoneycombMultilayerModel(stack_bernald(graphene(), 2, 3.35),
                                     t=0.0, gamma1=0.0)
        e = m.bands(np.array([[0, 0], K_POINT, [0.3, 0.2]]))
        np.testing.assert_allclose(e, 0.0, atol=1e-12)

    def test_aba_vs_abc_trilayer_k_spectra(self):
        """三层 K 点精确谱区分堆叠：ABA {0×4, ±√2γ₁}，ABC {0×2, ±γ₁×2}。"""
        aba = HoneycombMultilayerModel(stack_bernald(graphene(), 3, 3.35))
        abc = HoneycombMultilayerModel(stack_3r(graphene(), 3, 3.35))
        e_aba = aba.bands([K_POINT])[0]
        e_abc = abc.bands([K_POINT])[0]
        g1 = 0.4
        np.testing.assert_allclose(
            e_aba, [-np.sqrt(2) * g1, 0, 0, 0, 0, np.sqrt(2) * g1], atol=1e-10)
        np.testing.assert_allclose(
            e_abc, [-g1, -g1, 0, 0, g1, g1], atol=1e-10)

    def test_hbn_gap_from_onsite(self):
        """hBN 型在位能差打开 K 点带隙 = |Δ|（与 HoneycombModel 同锚点）。"""
        delta = 3.0
        st = stack_sequence(hbn(), [LayerStep()], 3.35)
        m = HoneycombMultilayerModel(st, element_onsite={"B": +delta / 2,
                                                         "N": -delta / 2})
        gap = m.bands([K_POINT])[0]
        assert (gap[1] - gap[0]) == pytest.approx(delta, rel=1e-6)

    def test_rejects_non_honeycomb_stack(self):
        with pytest.raises(ValueError):
            HoneycombMultilayerModel(stack_2h(tmd("MoS2"), 2, 6.15))


class TestMoS2Multilayer:
    @pytest.fixture(scope="class")
    def matched_zahid(self):
        """与 tmd() 构建器同几何的 Zahid 单层（a=3.1604, dxx=3.170）。"""
        return ZahidMoS2Model(a=3.1604, dxx=3.170, soc=True)

    def test_monolayer_matches_zahid(self, matched_zahid):
        st = stack_sequence(tmd("MoS2"), [LayerStep()], 6.15)
        m = MoS2MultilayerModel(st, cutoff=0.1)
        ks = np.array([[0, 0], K_POINT, [0.5, 0.5], [0.1, 0.05], [0.2, -0.13]])
        np.testing.assert_allclose(m.bands(ks), matched_zahid.bands(ks),
                                   atol=1e-9)

    def test_decoupled_bilayer_is_tiled_monolayer(self, matched_zahid):
        """解耦极限：2H 双层谱 = 单层谱 ×2（含一般 k 点）。

        rot180 层的同亚层镜像规范必须随层旋转（C₂），否则一般 k 点
        两层谱不同（实测差 0.37 eV，Γ/K/M 上看不出来）——本测试是该
        规范的回归锚点。
        """
        st = stack_2h(tmd("MoS2"), 2, 6.15)
        m = MoS2MultilayerModel(st, cutoff=0.1)
        ks = np.array([[0, 0], K_POINT, [0.5, 0.5], [0.1, 0.05],
                       [0.2, -0.13], [0.5, 0.0]])
        tiled = np.sort(np.concatenate(
            [matched_zahid.bands(ks)] * 2, axis=1), axis=1)
        np.testing.assert_allclose(m.bands(ks), tiled, atol=1e-9)

    def test_decoupled_3r_trilayer_is_tiled_monolayer(self, matched_zahid):
        st = stack_3r(tmd("MoS2"), 3, 6.15)
        m = MoS2MultilayerModel(st, cutoff=0.1)
        ks = np.array([[0, 0], [0.1, 0.05], [0.5, 0.0]])
        tiled = np.sort(np.concatenate(
            [matched_zahid.bands(ks)] * 3, axis=1), axis=1)
        np.testing.assert_allclose(m.bands(ks), tiled, atol=1e-9)

    def test_coupled_bilayer_hermitian_and_s_positive(self):
        st = stack_2h(tmd("MoS2"), 2, 6.15)
        m = MoS2MultilayerModel(st)
        assert m.n_interlayer_bonds > 0
        assert m.n_valence == 36
        for kf in ([0, 0], K_POINT, [0.13, 0.07]):
            H, S = m.hamiltonian_overlap(kf)
            assert np.abs(H - H.conj().T).max() < 1e-12
            assert np.abs(S - S.conj().T).max() < 1e-12
            assert np.linalg.eigvalsh(S).min() > -1e-8

    def test_interlayer_coupling_changes_gamma_spectrum(self):
        """层间耦合开启后 Γ 点谱 ≠ 解耦谱（p_z 杂化通道确实工作）。"""
        ks = np.array([[0.0, 0.0], [0.05, 0.0]])
        dec = MoS2MultilayerModel(stack_2h(tmd("MoS2"), 2, 6.15), cutoff=0.1)
        cpl = MoS2MultilayerModel(stack_2h(tmd("MoS2"), 2, 6.15))
        diff = np.abs(cpl.bands(ks) - dec.bands(ks)).max()
        assert diff > 0.05

    def test_conjugate_fill_false_breaks_hermiticity(self):
        """负结果保持：同亚层不做共轭填充 → H 非厄米（与 Zahid 同结论）。"""
        st = stack_2h(tmd("MoS2"), 2, 6.15)
        m = MoS2MultilayerModel(st, soc=False, conjugate_fill=False)
        H, _ = m.hamiltonian_overlap(K_POINT)
        assert np.abs(H - H.conj().T).max() > 1e-6

    def test_rejects_non_tmd_monolayer(self):
        with pytest.raises(ValueError):
            MoS2MultilayerModel(stack_bernald(graphene(), 2, 3.35))

    def test_unknown_integral_table_rejected(self):
        with pytest.raises(ValueError):
            MoS2MultilayerModel(stack_2h(tmd("MoS2"), 2, 6.15),
                                integrals={"XX_H": {}})


class TestDocumentedBlockage:
    """T16-SK 阻塞状态的固化（诚实声明，防"悄悄变好/变坏"）。

    Zahid 表 + 现装配约定下单层在一般 k 点**跨 k 带交叠**（网格间接
    间隙 < 0：实测 −0.07 eV @ SOC、−0.24 eV @ 无 SOC，builder 几何；
    各 k 点的同 k 谱隙可为正，交叠发生在不同 k 之间）——多层
    gap-vs-N 定量对账因此不可达（inherits T16-SK）。
    若未来修复装配约定使此测试失败：同步更新
    examples/run_multilayer_mos2.py 的对账表与 docs/REFERENCES.md。
    """

    def test_monolayer_is_band_overlapping_across_k(self):
        m = MoS2MultilayerModel(stack_2h(tmd("MoS2"), 1, 6.15))
        ks = np.array([[0.15, 0.15], [0.25, 0.25], [0.1, 0.05],
                       [1 / 3, 0.0], [0.2, 0.0]])
        e = m.bands(ks)
        # 网格间接间隙 = min(CB) − max(VB)：负值即跨 k 带交叠（金属化）
        gap_grid = e[:, m.n_valence].min() - e[:, m.n_valence - 1].max()
        assert gap_grid < 0.05
