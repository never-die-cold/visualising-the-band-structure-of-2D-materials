"""层间激子：转角双层 TMD 的束缚能与有效质量随转角 θ 的演化。

物理链条（每个环节都有独立验证）：

1. **θ → moiré 有效质量**：``engine.moire_tmd`` 的平面波求解给出导带/
   价带 moiré 带，k=0 二阶差分取有效质量（moiré 耦合使质量随 θ 减小
   而增大——平带化）；
2. **质量 → 约化质量**：μ = m_e·m_h/(m_e + m_h)；
3. **μ → 激子束缚能**：现有 Rytova–Keldysh 精确 DVR 求解器
   （``analysis.exciton.solve_exciton``，库仑极限与文献变分对账）。

近似说明（诚实标注）：层间激子的电子/空穴分居两层，真实相互作用含
层间距离 d 的修正；本管线用**共面 2D 近似**（d → 0 的束缚能上界），
d 的定量削弱与 moiré 势的空间结构（电子-空穴在不同层位阱）留待
Keldysh 势的层间推广。

文献参数：MoS₂ 单层 m_e = 0.55、m_h = 0.47 m₀（Kormányos 2015
Table 3/4）；r₀ = 2πχ₂D = 6.60 Å（Berkelbach 2013/Cudazzo 2011，
与 presets 的激子标定同源）。一个有用的自检：t=0（无 moiré 耦合）
时 μ = 0.55·0.47/1.02 ≈ 0.253 m₀——正是 Berkelbach 标定表用的
μ = 0.25 m₀，层间激子管线因此无缝衔接既有的 4-TMD 激子验证链。
"""

from dataclasses import dataclass
from typing import List, Sequence

import numpy as np

from ..engine.moire_tmd import TwistedTMDBilayer
from .exciton import solve_exciton

# MoS₂ 单层参数（Kormányos 2015 Table 3/4；Berkelbach 2013 的 r₀）
MOS2_A_LATTICE = 3.184          # Å
MOS2_M_E = 0.55                 # m₀
MOS2_M_H = 0.47                 # m₀
MOS2_CHI2D = 6.60               # Å（Berkelbach 2013 Table 的 χ₂D）
MOS2_R0 = 2 * np.pi * MOS2_CHI2D   # r₀ = 2πχ₂D ≈ 41.5 Å（Cudazzo 2011 关系；
                                # 与 presets 激子标定同一条换算链——χ₂D 本身
                                # 不是 r₀，误用会把束缚能高估 ~3.5 倍）


@dataclass
class InterlayerExcitonResult:
    """单个转角下的层间激子报告。"""

    theta_deg: float
    m_e_m0: float
    m_h_m0: float
    mu_over_m0: float
    binding_ev: float
    flat_band_width_ev: float
    widths_by_shells: dict
    converged: bool


def _flat_band_width(model, n_points: int = 12) -> float:
    """最低 moiré 带在 Γ–M 方向的带宽（eV）。"""
    k_list = np.column_stack((np.linspace(0.0, model.geom.k_theta, n_points),
                              np.zeros(n_points)))
    e = model.bands(k_list)
    return float(e[:, 0].max() - e[:, 0].min())


def interlayer_exciton_report(theta_deg: float, *,
                              a_lattice: float = MOS2_A_LATTICE,
                              m_e_single: float = MOS2_M_E,
                              m_h_single: float = MOS2_M_H,
                              t_interlayer: float = 0.020,
                              r0: float = MOS2_R0,
                              eps_env: float = 1.0,
                              shells: Sequence[int] = (2, 3),
                              n_levels: int = 1) -> InterlayerExcitonResult:
    """单个转角下的层间激子报告（含 moiré 带宽的壳层收敛检查）。"""
    if not np.isfinite(theta_deg) or theta_deg <= 0:
        raise ValueError("θ 必须为正")
    shells = sorted({int(s) for s in shells})
    if len(shells) < 2 or shells[0] < 1:
        raise ValueError("shells 需要至少两个递增壳层做收敛检查")

    m_e = TwistedTMDBilayer(theta_deg, a_lattice, m_e_single, t_interlayer,
                            n_shells=shells[-1], band="cb").moire_effective_mass(0)
    # vb 谱升序且 ≤ 0，价带顶在最后一条（见 moire_effective_mass 的说明）
    m_h = TwistedTMDBilayer(theta_deg, a_lattice, m_h_single, t_interlayer,
                            n_shells=shells[-1], band="vb").moire_effective_mass(-1)
    mu = m_e * m_h / (m_e + m_h)
    exciton = solve_exciton(mu_over_m0=mu, eps_env=eps_env, r0=r0,
                            n_levels=n_levels)

    widths = {}
    for n_shells in shells:
        model = TwistedTMDBilayer(theta_deg, a_lattice, m_e_single, t_interlayer,
                                  n_shells=n_shells, band="cb")
        widths[n_shells] = _flat_band_width(model)
    w = widths[shells[-1]]
    converged = abs(widths[shells[-1]] - widths[shells[0]]) < 0.05 * max(w, 1e-4)

    return InterlayerExcitonResult(
        theta_deg=float(theta_deg), m_e_m0=float(m_e), m_h_m0=float(m_h),
        mu_over_m0=float(mu), binding_ev=float(exciton.binding_1s),
        flat_band_width_ev=float(w),
        widths_by_shells={int(k): float(v) for k, v in widths.items()},
        converged=bool(converged),
    )


def interlayer_exciton_vs_theta(theta_list: Sequence[float], **kwargs) -> List[InterlayerExcitonResult]:
    """转角序列的层间激子报告（逐角计算，供 Eb(θ) 曲线使用）。"""
    return [interlayer_exciton_report(theta, **kwargs) for theta in theta_list]
