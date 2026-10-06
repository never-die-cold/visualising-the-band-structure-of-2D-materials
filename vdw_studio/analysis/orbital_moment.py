"""轨道磁矩与谷 Zeeman 效应（谷物理深化，Phase 4.2）。

物理模型（Xiao, Chang & Niu, Rev. Mod. Phys. **82**, 1959 (2010) 的波包
表述；arXiv 版 arXiv:0907.2021）：

- **轨道磁矩**（波包自旋转角动量，RMP Eq. (11)）::

      m_n(q) = −i(e/2ħ) ⟨∇_q u_n| × [H(q) − ε_n(q)] |∇_q u_n⟩

  经微扰论 |∂_i u_n⟩ = Σ_{m≠n} |u_m⟩⟨u_m|v_i|u_n⟩/(ε_n−ε_m)（v_i = ∂H/∂q_i）
  展开为速度矩阵元形式::

      m_n(q) = (e/ħ) · Im Σ_{m≠n} ⟨n|v_x|m⟩⟨m|v_y|n⟩ / (ε_n − ε_m)

  实现注记：本展开式与 RMP 原式的有限差分直接实现（∇u 规范固定后
  逐点差分）在测试中相互对账一致，并与两带模型解析关系一致::

      m_n[μ_B] = −(ε_n − ε_m)·Ω_n / (2·ħ²/2m₀)     （两带：单一带内耦合）

- **带边磁矩 = 有效玻尔磁子**（RMP: "m(τ_z) = τ_z μ*_B, μ*_B = eħ/2m*"）::

      大质量 Dirac 带边  m_n(K) = τ_z·(m₀/m*)·μ_B

  即磁矩以质量比 m₀/m* 增强玻尔磁子——谷磁矩可比自旋磁矩大数倍
  （MoS₂ 价带 m* ≈ 0.55 m₀ → m ≈ 1.8 μ_B）。

- **Zeeman 耦合**（RMP: "ε_M(k) = ε(k) − m(k)·B"）::

      谷劈裂  ΔE = E(K) − E(−K) = −2·m_n(K)·B

  （时间反演要求 m_n(−K) = −m_n(K)，故劈裂为每谷磁矩的两倍。）
  数值上 μ_B = 57.8838 μeV/T = 0.0578838 meV/T。

单位约定：m 以玻尔磁子 μ_B 计。速度矩阵元 S₁ = Σ⟨n|v_x|m⟩⟨m|v_y|n⟩/(ε_n−ε_m)
以 eV·Å² 计（v 以 eV·Å 计、q 以 Å⁻¹ 计），换算::

      m[μ_B] = Im(S₁) / (ħ²/2m₀)，  ħ²/2m₀ = 3.809982 eV·Å²

适用对象：任何提供厄米矩阵 ``hamiltonian(q)`` 的模型（与 berry 模块相同
的接口约定）。
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .berry import _velocity_matrices

# ħ²/(2m₀) = 3.809982 eV·Å²（与 exciton.py 共享同一常数）
HBAR2_OVER_2M0 = 3.809982
# 玻尔磁子 μ_B = eħ/2m₀ = 57.8838 μeV/T（Zeeman 劈裂换算用）
MU_B_MEV_PER_T = 0.0578838


def orbital_moment(hamiltonian, q, band: int, dq: float = 1e-6) -> float:
    """单 k 点、单带的轨道磁矩 m (μ_B)。

    参数:
        hamiltonian: callable(q 2-vector) → 厄米矩阵 (eV)。
        band: 按能量升序的带序号。
        q: 相对谷中心的笛卡尔波矢 (Å⁻¹)。
    """
    q = np.asarray(q, dtype=float)
    H0 = hamiltonian(q)
    Hx, Hy = _velocity_matrices(hamiltonian, q, dq)
    E, V = np.linalg.eigh(H0)
    vx = V.conj().T @ Hx @ V
    vy = V.conj().T @ Hy @ V
    n = band
    s1 = 0.0 + 0.0j
    for m in range(len(E)):
        if m == n:
            continue
        s1 += vx[n, m] * vy[m, n] / (E[n] - E[m])
    return float(np.imag(s1) / HBAR2_OVER_2M0)


def orbital_moment_map(hamiltonian, band: int, qmax: float,
                       n_grid: int = 41) -> dict:
    """轨道磁矩热图（与 berry.valley_report 的接口约定一致）。

    返回:
        dict(q=qs, moment=(n,n) 网格, moment_center=谷心磁矩)
        行 = qy（绘图时转置，与 valley_report 相同）。
    """
    qs = np.linspace(-qmax, qmax, n_grid)
    moment = np.zeros((n_grid, n_grid))
    for i, qx in enumerate(qs):
        for j, qy in enumerate(qs):
            moment[i, j] = orbital_moment(hamiltonian,
                                          np.array([qx, qy]), band)
    return {
        "q": qs,
        "moment": moment,
        "moment_center": float(moment[n_grid // 2, n_grid // 2]),
    }


@dataclass
class ValleyZeemanResult:
    """谷 Zeeman 劈裂结果。"""

    moment_mu_b: float      # 谷心轨道磁矩 m_n(K) (μ_B)
    b_tesla: float          # 磁场 (T, 垂直面外)
    splitting_mev: float    # E(K) − E(−K) (meV) = −2·m·μ_B·B
    g_valley: float         # 等效谷 g 因子（ΔE = g·μ_B·B 约定，数值 = −2m[μ_B]）


def valley_zeeman_splitting(moment_mu_b: float, b_tesla: float) -> ValleyZeemanResult:
    """由谷心磁矩计算 Zeeman 谷劈裂 ΔE = −2·m·μ_B·B (meV)。

    时间反演: m(−K) = −m(K) → 两谷能移相反，劈裂为单谷磁移的 2 倍
    （RMP: ε_M = ε − m·B）。
    """
    split = -2.0 * moment_mu_b * MU_B_MEV_PER_T * b_tesla
    return ValleyZeemanResult(moment_mu_b=moment_mu_b, b_tesla=b_tesla,
                              splitting_mev=float(split),
                              g_valley=-2.0 * moment_mu_b)
