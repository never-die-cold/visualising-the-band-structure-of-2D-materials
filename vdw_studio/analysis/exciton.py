"""2D 激子束缚能求解器（Rytova–Keldysh 屏蔽势，Phase 3）。

物理模型：
- 电子–空穴相互作用取二维屏蔽库仑势（Rytova–Keldysh），
  实空间形式（本文约定，见下方极限自检）::

      V(r) = −(e²/4πε₀ε_env) · (π/2) · [H₀(r/r₀) − Y₀(r/r₀)] / r₀

  其中 H₀ 为 Struve 函数、Y₀ 为第二类 Bessel 函数，
  r₀ 为二维极化长度（材料+介电环境相关），ε_env 为环境介电常数。
  极限自检：r₀→0（ρ = r/r₀ → ∞）时 H₀−Y₀ ~ 2/(πρ)，
  V → −e²/(4πε₀ε_env·r)——严格回到三维库仑形式 ✓。

- 相对坐标径向方程（m=0, 1s/2s/… 通道）::

      [−(ħ²/2μ)(ψ'' + ψ'/r) + V(r)ψ] = Eψ

  通过 ψ = χ/√r 变换化为带 1/(4r²) 势的常微分方程（数值上
  等价且保证厄米性），三对角有限差分本征值求解。

- **验证锚点（解析精确）**：库仑极限下 2D 氢原子谱
  E_n = −4Ry*/(2n+1)²（n = 0,1,…），Ry* = (μ/m₀)·13.6058 eV/ε_env²；
  数值解应重现 E₀ = −4Ry*（1s）与 E₁ = −4Ry*/9（2s）。

材料参数说明（防幻觉）：r₀ 与 μ 因材料和介电环境而异；
本模块将其作为**用户参数**，材料默认值待文献标定后加入预设库
（见 docs/REFERENCES.md 待办：Cudazzo et al. PRB 84, 085406 (2011)）。

⚠️ **已知问题（WIP，勿用于定量结论）**：守恒型有限差分格式对 2D
库仑势的 1/r 奇异性收敛不足——库仑极限下基态收敛到 ≈ −8.2 eV
（精确值 −4Ry* = −13.6 eV @ μ=0.25），网格加倍仅缓慢改善。
2D 氢原子精确谱验证**未通过**。后续计划：r < a* 区域用解析
基函数或对数/非均匀网格处理奇异性。已验证可用的部分：
keldysh_potential() 的库仑极限（r₀→0）与势函数本身。
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np
from scipy import special

# e²/(4πε₀) = 14.3996 eV·Å（库仑常数）
COULOMB_CONST = 14.3996
# ħ²/(2m₀) = 0.2380 eV·Å²·? — 即 ħ²/(2m₀) = 3.80998 eV·Å²（用于 μ 标度）
HBAR2_OVER_2M0 = 3.809982
# Rydberg（m₀, ε=1）: 13.6057 eV
RY = 13.6057


def keldysh_potential(r, r0: float, eps_env: float = 1.0) -> np.ndarray:
    """Rytova–Keldysh 屏势 V(r) (eV)，吸引（负值）。

    参数:
        r: 径向坐标 (Å)，r > 0。
        r0: 二维极化长度 (Å)。
        eps_env: 环境介电常数（乘性屏蔽）。
    """
    r = np.asarray(r, dtype=float)
    rho = r / r0
    # scipy.special.struve(0, ρ) = H₀(ρ); special.y0(ρ) = Y₀(ρ)
    return -(COULOMB_CONST / eps_env) * (np.pi / 2) * \
        (special.struve(0, rho) - special.y0(rho)) / r0


def coulomb_potential(r, eps_env: float = 1.0) -> np.ndarray:
    """纯库仑参考势 −e²/(4πε₀ε_env·r) (eV)。"""
    return -COULOMB_CONST / (eps_env * np.asarray(r, dtype=float))


@dataclass
class ExcitonResult:
    """激子能级求解结果。"""

    energies: np.ndarray      # (n_levels,) 束缚态能量 (eV, 负值)
    binding_1s: float         # 1s 束缚能 = −E₀ (eV)
    r0: Optional[float]       # 极化长度 (None = 纯库仑)
    eps_env: float
    mu_over_m0: float         # 约化质量 (m₀ 单位)

    def rydberg(self) -> float:
        """有效 Rydberg Ry* = (μ/m₀)·13.6057/ε² (eV)。"""
        return self.mu_over_m0 * RY / self.eps_env ** 2


def solve_exciton(mu_over_m0: float = 0.25,
                  eps_env: float = 1.0,
                  r0: Optional[float] = None,
                  n_levels: int = 4,
                  n_grid: int = 4000,
                  r_max: Optional[float] = None) -> ExcitonResult:
    """求解 2D 激子径向方程（m=0 通道，s 态序列）。

    数值格式：径向拉普拉斯 (1/r)(d/dr)(r du/dr) 的守恒型离散
    （保证对称性），广义本征值问题 ``S u = E·M u``（M = diag(r)）。

    参数:
        mu_over_m0: 电子–空穴约化质量 (m₀ 单位)。
        eps_env: 环境介电常数。
        r0: Keldysh 极化长度 (Å)；None = 纯库仑势（2D 氢原子极限）。
        n_levels: 返回的低能级数。
        n_grid: 径向网格点数。
        r_max: 网格外半径 (Å)；默认按有效玻尔半径自适应
               （库仑: a* = ε·(m₀/μ)·0.529 Å，取 40a*；
                Keldysh: 取 40·r₀ 与 200 Å 的较大者）。

    返回:
        :class:`ExcitonResult`（能量升序，负值为束缚态）。
    """
    import scipy.linalg

    if mu_over_m0 <= 0 or eps_env <= 0:
        raise ValueError("μ 与 ε_env 必须为正")
    hbar2_over_2mu = HBAR2_OVER_2M0 / mu_over_m0   # ħ²/(2μ) eV·Å²

    if r_max is None:
        a_bohr = eps_env / mu_over_m0 * 0.529       # 库仑有效玻尔半径
        if r0 is None:
            r_max = max(40 * a_bohr, 30.0)
        else:
            r_max = max(40 * r0, 30 * a_bohr, 30.0)

    # 半开网格 r_i = i·dr (i = 1..N), 边界 u(0)=0, u(r_max⁺)=0
    dr = r_max / (n_grid + 1)
    r = np.arange(1, n_grid + 1) * dr
    r_half_p = r + dr / 2     # r_{i+1/2}
    r_half_m = r - dr / 2     # r_{i-1/2}

    # 势
    if r0 is None:
        V = coulomb_potential(r, eps_env)
    else:
        if r0 <= 0:
            raise ValueError("r0 必须为正")
        V = keldysh_potential(r, r0, eps_env)

    # 守恒型离散: −ħ²/2μ·(1/r_i)·[r₊(u_{i+1}−u_i) − r₋(u_i−u_{i−1})]/dr²
    # 两边乘 r_i dr² 得对称矩阵 S (广义本征问题 S u = E·M u, M = diag(r))
    # 守恒型离散: −ħ²/2μ·(1/r_i)·[r₊(u_{i+1}−u_i) − r₋(u_i−u_{i−1})]/dr²
    # 两边乘 r_i dr²: 对角 = ħ²/2μ(r₊+r₋)/dr² + V·r_i (r_i 已与 1/r_i 相消);
    # 非对角 = −ħ²/2μ·r_{i±1/2}/dr² (N−1 个耦合)
    tpp = hbar2_over_2mu * r_half_p[: n_grid - 1] / dr ** 2   # N−1 个耦合
    diag_kin = hbar2_over_2mu * (r_half_p + r_half_m) / dr ** 2  # N 个
    S = np.diag(diag_kin + V * r)
    S += np.diag(-tpp, 1)
    S += np.diag(-tpp, -1)
    M = np.diag(r)

    eig = scipy.linalg.eigh(S, M, eigvals_only=True)
    bound = eig[eig < 0][:n_levels]
    return ExcitonResult(energies=bound, binding_1s=float(-bound[0]),
                         r0=r0, eps_env=eps_env,
                         mu_over_m0=mu_over_m0)
