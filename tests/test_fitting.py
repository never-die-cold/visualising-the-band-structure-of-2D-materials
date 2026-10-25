"""DFT→TB 拟合管线的回归测试。

核心是**合成数据往返**：扰动若干参数构造"真值"模型，生成 (k, E, W) 参考
数据，再从未扰动初值拟合，检验能否恢复参数。这验证的是拟合器与残差实现
的正确性（可辨识性的全局问题属于科研范畴，不属于回归范围）。

投影权重的一个数学事实（测试固化）：不同 (原子, 壳层) 子空间**不相互
S-正交**（spσ 等通道耦合它们），因此分组权重之和不守恒——"字符归属"
天然依赖分组方案；恒等的只有单组=全基时 w = ψ†Sψ = 1。
"""

import numpy as np
import pytest

from vdw_studio.engine.zahid_mos2 import ZahidMoS2Model
from vdw_studio.fitting import (
    BandReference,
    apply_overrides,
    fit,
    get_parameter,
    get_vector,
    group_weights,
    parameter_names,
    set_vector,
)
from vdw_studio.fitting.driver import band_residuals

# 拟合测试共用的小网格（3×3，含 Γ/M/K）与带窗口
KPOINTS = np.array([[i / 3, j / 3] for i in range(3) for j in range(3)])
N_BANDS = (14, 23)

# 往返测试扰动的"真值"参数：都是带边附近的强耦合量
TRUTH_OVERRIDES = {
    "ONSITE:Mo.d": 2.6429 + 0.30,
    "SMo_H:ddσ": -3.1425 - 0.40,
    "SMo_H:pdσ": -2.8732 + 0.30,
    "SS_H:ddσ": 0.8347 + 0.20,
    "MoMo_H:ddσ": 0.3602 - 0.15,
}


@pytest.fixture(scope="module")
def truth_and_start():
    truth = ZahidMoS2Model(soc=False)
    apply_overrides(truth, TRUTH_OVERRIDES)
    reference = BandReference.from_model(truth, KPOINTS, n_bands=N_BANDS)
    start = ZahidMoS2Model(soc=False)          # 未扰动初值
    return truth, reference, start


class TestParameterVector:
    def test_roundtrip_preserves_tables(self):
        model = ZahidMoS2Model(soc=False)
        x = get_vector(model)
        set_vector(model, x + np.arange(len(x)) * 1e-9)
        np.testing.assert_allclose(get_vector(model), x + np.arange(len(x)) * 1e-9)

    def test_names_cover_every_table_entry(self):
        model = ZahidMoS2Model(soc=False)
        names = parameter_names(model)
        assert len(names) == 2 * 4 + 2 * 14 + 2 * 10 + 2 * 10   # ONSITE + 6 张积分表
        assert names[:4] == ["ONSITE:S.s", "ONSITE:S.p", "ONSITE:S.d", "ONSITE:S.lambda"]
        assert "SMo_H:spσ" in names and "MoMo_H:ddσ" in names

    def test_apply_overrides_rejects_unknown(self):
        model = ZahidMoS2Model(soc=False)
        with pytest.raises(ValueError, match="未知参数"):
            apply_overrides(model, {"NOPE:xx": 1.0})
        apply_overrides(model, {"SMo_H:ddσ": -3.0})
        assert get_parameter(model, "SMo_H:ddσ") == pytest.approx(-3.0)
        # 模块常量不被污染
        from vdw_studio.engine.zahid_mos2 import INTEGRALS_SMO_H
        assert INTEGRALS_SMO_H["ddσ"] == pytest.approx(-3.1425)

    def test_tables_are_instance_private(self):
        """两个模型的积分表互不影响，模块常量也不被就地修改波及。

        ONSITE 是两层嵌套 dict，浅拷贝会让内层跨实例共享——拟合改
        ONSITE:Mo.d 曾把模块 ONSITE 一并改掉（全量跑序下
        test_parameter_transcription 因此失败）。
        """
        import vdw_studio.engine.zahid_mos2 as zahid
        first = ZahidMoS2Model(soc=False)
        second = ZahidMoS2Model(soc=False)
        apply_overrides(first, {"ONSITE:Mo.d": 9.9, "SMo_H:ddσ": -9.9})
        assert get_parameter(second, "ONSITE:Mo.d") == pytest.approx(2.6429)
        assert get_parameter(second, "SMo_H:ddσ") == pytest.approx(-3.1425)
        assert zahid.ONSITE["Mo"]["d"] == pytest.approx(2.6429)
        assert zahid.INTEGRALS_SMO_H["ddσ"] == pytest.approx(-3.1425)


class TestProjection:
    def test_single_full_basis_group_is_exactly_one(self):
        """单组=全基时 S_A=S，权重恒等于 ψ†Sψ=1（公式的解析锚点）。"""
        model = ZahidMoS2Model(soc=False)
        d_all = ("dxy", "dyz", "dzx", "dx2-y2", "dz2")
        groups = {"all": ((0, 1, 2), ("s", "px", "py", "pz") + d_all)}
        _, weights = group_weights(model, KPOINTS[:2], groups)
        np.testing.assert_allclose(weights["all"][0], 1.0, atol=1e-10)
        np.testing.assert_allclose(weights["all"][1], 1.0, atol=1e-10)

    def test_group_weights_are_bounded(self):
        """投影范数不超过总范数：0 ≤ w ≤ 1 + 数值容差。"""
        model = ZahidMoS2Model(soc=False)
        _, weights = group_weights(model, KPOINTS)
        for name, w in weights.items():
            assert w.min() > -1e-10, name
            assert w.max() < 1.0 + 1e-8, name

    def test_d_weight_dominates_valence_top(self):
        """价带顶（带 17，无 SOC）的 Mo-d 权重显著超过 Mo-s/Mo-p。"""
        model = ZahidMoS2Model(soc=False)
        _, weights = group_weights(model, np.array([[1 / 3, 1 / 3]]))
        assert weights["Mo_d"][0, 17] > 0.1
        assert weights["Mo_d"][0, 17] > weights["Mo_p"][0, 17]
        assert weights["Mo_d"][0, 17] > weights["Mo_s"][0, 17]


class TestSyntheticRoundtrip:
    def test_reference_carries_window_and_source(self, truth_and_start):
        _, reference, _ = truth_and_start
        assert reference.energies.shape == (9, 9)
        assert reference.band_window == N_BANDS
        assert reference.source == "synthetic"
        assert set(reference.weights) == {"Mo_s", "Mo_p", "Mo_d", "S_s", "S_p", "S_d"}

    def test_roundtrip_recovers_parameters(self, truth_and_start):
        truth, reference, start = truth_and_start
        free = list(TRUTH_OVERRIDES)
        result = fit(start, reference, free_names=free, weight_weight=0.0,
                     max_nfev=200)
        for name, truth_value in TRUTH_OVERRIDES.items():
            assert get_parameter(start, name) == pytest.approx(truth_value, abs=5e-3), name
        assert result.success or result.cost < 1e-12
        # 带隙（无 SOC 用 E[18]-E[17]）同步恢复
        e_k = start.energies_at([1 / 3, 1 / 3])
        e_t = truth.energies_at([1 / 3, 1 / 3])
        assert (e_k[18] - e_k[17]) == pytest.approx(e_t[18] - e_t[17], abs=1e-4)

    def test_projection_residuals_discriminate_truth(self, truth_and_start):
        """带投影残差时，真值参数的代价必须远低于未扰动初值（字符信息有效）。"""
        truth, reference, start = truth_and_start
        probe = ZahidMoS2Model(soc=False)
        cost_truth = float((band_residuals(truth, reference, weight_weight=1.0) ** 2).sum())
        cost_start = float((band_residuals(probe, reference, weight_weight=1.0) ** 2).sum())
        assert cost_truth < 1e-16
        assert cost_start > 1e-4
