"""层间激子管线的物理验证测试。

锚点：
1. t=0（无 moiré 耦合）→ μ 回到单层激子的 Berkelbach 标定值 0.253 m₀，
   束缚能与既有 4-TMD 验证链的 MoS₂ 数值衔接；
2. 平带化单调性：θ 减小 → moiré 带平坦 → 质量增大 → 束缚能增大；
3. 壳层收敛检查在弱耦合（大 θ）下应通过。
"""

import pytest

from vdw_studio.analysis.interlayer_exciton import (
    MOS2_M_E,
    MOS2_M_H,
    MOS2_R0,
    interlayer_exciton_report,
    interlayer_exciton_report_kp,
)
from vdw_studio.analysis.exciton import solve_exciton

# MoS₂ 单层激子（无 moiré 耦合）的参考解：μ = 0.55·0.47/1.02 ≈ 0.2535
MU_SINGLE = MOS2_M_E * MOS2_M_H / (MOS2_M_E + MOS2_M_H)


def test_decoupled_limit_matches_single_layer_exciton():
    """t=0：moiré 质量退回单层质量，束缚能 == 直接求解的同参数激子。"""
    report = interlayer_exciton_report(5.0, t_interlayer=0.0, n_levels=1)
    assert report.m_e_m0 == pytest.approx(MOS2_M_E, rel=1e-6)
    assert report.m_h_m0 == pytest.approx(MOS2_M_H, rel=1e-6)
    assert report.mu_over_m0 == pytest.approx(MU_SINGLE, rel=1e-6)
    direct = solve_exciton(mu_over_m0=MU_SINGLE, eps_env=1.0, r0=MOS2_R0, n_levels=1)
    assert report.binding_ev == pytest.approx(float(direct.binding_1s), rel=1e-9)
    # 与 Berkelbach 标定衔接：精确解 ≥ 变分下界 0.54 eV 且接近（≤10%）
    assert 0.54 <= report.binding_ev <= 0.54 * 1.10


def test_binding_grows_toward_small_angle():
    """平带化：θ 减小 → moiré 质量增大 → 束缚能增大（层间激子的转角响应）。"""
    wide = interlayer_exciton_report(5.0, t_interlayer=0.020)
    flat = interlayer_exciton_report(1.0, t_interlayer=0.020)
    assert flat.m_e_m0 > wide.m_e_m0
    assert flat.m_h_m0 > wide.m_h_m0
    assert flat.binding_ev > wide.binding_ev


def test_shell_convergence_flag_in_weak_coupling(self=None):
    """弱耦合（大 θ、小 T）：moiré 带宽对壳层不敏感 → converged=True。"""
    report = interlayer_exciton_report(5.0, t_interlayer=0.002, shells=(2, 3))
    assert report.converged is True


def test_invalid_inputs_rejected():
    with pytest.raises(ValueError, match="θ 必须为正"):
        interlayer_exciton_report(0.0)
    with pytest.raises(ValueError, match="shells"):
        interlayer_exciton_report(2.0, shells=(3,))


class TestKpPipeline:
    """完整 k·p 层哈密顿量版的层间激子管线（T17 剩余项）。"""

    def test_decoupled_masses_match_kormanyos(self):
        """γ=0：moiré 质量精确退回单层 Kormányos 质量（自旋分辨带边）。"""
        from vdw_studio.engine.kp_tmd import TMDKpModel, TMDKpParams

        report = interlayer_exciton_report_kp(5.0, gamma_cb=0.0, gamma_vb=0.0,
                                              n_levels=1)
        kp = TMDKpModel(TMDKpParams.from_library("MoS2", "dft"))
        m_e_ref = kp.effective_mass(band="cb", spin=-1, valley=+1,
                                    direction=(1.0, 0.0))
        m_h_ref = kp.effective_mass(band="vb", spin=+1, valley=+1,
                                    direction=(1.0, 0.0))
        assert report.m_e_m0 == pytest.approx(m_e_ref, rel=1e-6)
        assert report.m_h_m0 == pytest.approx(m_h_ref, rel=1e-6)
        # 与 Berkelbach 标定链衔接（μ ≈ 0.25）
        assert report.mu_over_m0 == pytest.approx(0.25, abs=0.01)
        assert 0.54 <= report.binding_ev <= 0.54 * 1.10

    def test_binding_grows_toward_small_angle(self):
        """平带化单调性：θ 减小 → moiré 质量增大 → 束缚能增大。"""
        wide = interlayer_exciton_report_kp(5.0, n_levels=1)
        flat = interlayer_exciton_report_kp(1.0, n_levels=1)
        assert flat.m_e_m0 > wide.m_e_m0
        assert flat.m_h_m0 > wide.m_h_m0
        assert flat.binding_ev > wide.binding_ev

    def test_invalid_inputs_rejected(self):
        with pytest.raises(ValueError, match="θ 必须为正"):
            interlayer_exciton_report_kp(0.0)
        with pytest.raises(ValueError, match="shells"):
            interlayer_exciton_report_kp(2.0, shells=(3,))

    def test_interlayer_separation_weakens_binding(self):
        """层间分离 d>0 的束缚能低于共面近似（上界），且随 d 单调。"""
        coplanar = interlayer_exciton_report_kp(2.0, n_levels=1)
        separated = interlayer_exciton_report_kp(2.0, interlayer_d=6.5,
                                                 n_levels=1)
        farther = interlayer_exciton_report_kp(2.0, interlayer_d=13.0,
                                               n_levels=1)
        assert separated.binding_ev < coplanar.binding_ev
        assert farther.binding_ev < separated.binding_ev
        # MoS₂ 默认参数、θ=2° 的量级（诚实记录，不凑数）：共面 ~0.73，
        # d=6.5 Å 时 ~0.50 eV 量级
        assert 0.40 < separated.binding_ev < coplanar.binding_ev
