"""2D 激子束缚能求解器（Rytova–Keldysh 屏蔽势，Phase 3）。

物理模型：
- 电子–空穴相互作用取二维屏蔽库仑势（Rytova–Keldysh），
  实空间形式::

      V(r) = −(e²/4πε₀ε_env) · (π/2) · [H₀(r/r₀) − Y₀(r/r₀)] / r₀

  其中 H₀ 为 Struve 函数、Y₀ 为第二类 Bessel 函数，
  r₀ 为二维极化长度（材料+介电环境相关），ε_env 为环境介电常数。
  极限自检：r₀→0（ρ = r/r₀ → ∞）时 H₀−Y₀ ~ 2/(πρ)，
  V → −e²/(4πε₀ε_env·r)——严格回到库仑形式 ✓（tests 对账）。

- 相对坐标径向方程（m=0, 1s/2s/… 通道）::

      [−(ħ²/2μ)(1/r)(d/dr)(r d/dr) + V(r)] u = E u

- **数值方法：Fourier–Bessel 基组（径向 DVR）+ 广义本征值**。
  展开 u(r) = Σ cₙ J₀(kₙr)，kₙ = j₀,ₙ/R（j₀,ₙ 为 J₀ 的第 n 个零点，
  满足 u(R)=0 边界）。基函数以 ``r dr`` 为度量正交：
  Nₙ = ∫J₀² r dr = (R²/2)J₁(j₀,ₙ)²；
  动能矩阵 (ħ²kₙ²/2μ)·Nₙ δₙₘ（严格对角）；
  势能矩阵 Vₙₘ = ∫ J₀(kₙr)·V(r)·J₀(kₘr)·r dr（梯形求积，
  积分核 r·V(r) 在 r→0 处有限——库仑: → −e²/4πε₀ε_env；
  Keldysh: → 0——完全规避 1/r 奇异性）。
  求解广义本征值问题 Hc = E·Nc（scipy generalized eigh）。

  ⚠ 实现注记：动能矩阵必须含 Nₙ 权重（广义度量），且基组内
  不得混入其它约定（如 1/√N 归一化或列/行向量转置）——
  这些约定混用曾导致"各向异性/不收敛"的假象（见 git 历史）。

材料参数说明（防幻觉）：r₀ 与 μ 因材料和介电环境而异。
本模块将其作为**用户参数**；四种 TMD（MoS₂/MoSe₂/WS₂/WSe₂）的文献
标定默认值已入库 `presets/materials.py`（Berkelbach et al., PRB 88,
045318 (2013) Table：μ、χ₂D；r₀ = 2πχ₂D 取 Cudazzo et al., PRB 84,
085406 (2011) 关系），并经 `tests/test_exciton.py` 与该文献的变分
束缚能对账（精确解 ≥ 变分下界，偏差 2–5%）。势的形式与
Berkelbach Eq. (1)：V = πe²/((ε₁+ε₂)ρ₀)·[H₀−Y₀] 逐项一致
（本模块 ε_env = (ε₁+ε₂)/2，真空下 ε_env = 1）。
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np
import scipy.linalg
from scipy import special

# e²/(4πε₀) = 14.3996 eV·Å（库仑常数）
COULOMB_CONST = 14.3996
# ħ²/(2m₀) = 3.809982 eV·Å²（用于 μ 标度）
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


def _r_times_potential(r: np.ndarray, r0: Optional[float],
                       eps_env: float) -> np.ndarray:
    """积分核 w(r) = r·V(r)：r→0 极限解析处理（避免 0·∞）。

    - 纯库仑：r·V → −e²/(4πε₀ε_env)（有限常数）；
    - Keldysh：ρ(H₀−Y₀) → 0，故 r·V → 0。
    """
    w = np.empty_like(r)
    if r0 is None:
        w[1:] = coulomb_potential(r[1:], eps_env) * r[1:]
        w[0] = -COULOMB_CONST / eps_env
    else:
        w[0] = 0.0
        w[1:] = keldysh_potential(r[1:], r0, eps_env) * r[1:]
    return w


@dataclass
class ExcitonResult:
    """激子能级求解结果。"""

    energies: np.ndarray      # (n_levels,) 束缚态能量 (eV, 负值)
    binding_1s: float         # 1s 束缚能 = −E₀ (eV)
    r0: Optional[float]       # 极化长度 (None = 纯库仑)
    eps_env: float
    mu_over_m0: float         # 约化质量 (m₀ 单位)
    r_max: float = 0.0        # 径向盒子尺寸 (Å)
    n_basis: int = 0          # Bessel 基函数数目

    def rydberg(self) -> float:
        """有效 Rydberg Ry* = (μ/m₀)·13.6057/ε² (eV)。"""
        return self.mu_over_m0 * RY / self.eps_env ** 2


def solve_exciton(mu_over_m0: float = 0.25,
                  eps_env: float = 1.0,
                  r0: Optional[float] = None,
                  n_levels: int = 4,
                  n_basis: int = 250,
                  n_quad: int = 6000,
                  r_max: Optional[float] = None) -> ExcitonResult:
    """求解 2D 激子径向方程（m=0 通道，s 态序列；Fourier–Bessel DVR）。

    参数:
        mu_over_m0: 电子–空穴约化质量 (m₀ 单位)。
        eps_env: 环境介电常数。
        r0: Keldysh 极化长度 (Å)；None = 纯库仑势（2D 氢原子极限）。
        n_levels: 返回的低能级数。
        n_basis: Bessel 基函数数目（越大越精确，收敛 ~ O(N⁻²)）。
        n_quad: 势能积分的求积点数。
        r_max: 径向盒子半径 (Å)；默认按有效玻尔半径自适应
               （库仑: a* = ε·(m₀/μ)·0.529 Å，取 40a*；
                Keldysh: 取 40·r₀ 与 30a* 的较大者）。

    返回:
        :class:`ExcitonResult`（能量升序，负值为束缚态）。
    """
    if mu_over_m0 <= 0 or eps_env <= 0:
        raise ValueError("μ 与 ε_env 必须为正")
    hbar2_over_2mu = HBAR2_OVER_2M0 / mu_over_m0   # ħ²/(2μ) eV·Å²

    if r_max is None:
        a_bohr = eps_env / mu_over_m0 * 0.529       # 库仑有效玻尔半径
        if r0 is None:
            r_max = max(40 * a_bohr, 30.0)
        else:
            r_max = max(40 * r0, 30 * a_bohr, 30.0)

    # Fourier–Bessel 基组: u(r) = Σ cₙ J₀(kₙr), kₙ = j₀,ₙ/R
    zeros = special.jn_zeros(0, n_basis)
    kn = zeros / r_max
    norms = 0.5 * r_max ** 2 * special.j1(zeros) ** 2   # ∫J₀² r dr
    T = np.diag(hbar2_over_2mu * kn ** 2 * norms)       # 动能 (含 Nₙ 权重!)

    # 势能矩阵 (梯形求积; 核 r·V(r) 在 r=0 有限)
    r = np.linspace(0.0, r_max, n_quad)
    dr = r[1] - r[0]
    w = _r_times_potential(r, r0, eps_env)              # r·V(r)
    qw = np.full(n_quad, dr)
    qw[0] *= 0.5
    qw[-1] *= 0.5
    J = special.j0(np.outer(kn, r))                     # (N, n_quad)
    V = ((J * w) * qw) @ J.T

    # 广义本征值问题 H c = E·N c (度量 = diag(Nₙ))
    H = T + V
    eig = scipy.linalg.eigh(H, np.diag(norms), eigvals_only=True)
    bound = eig[eig < 0][:n_levels]
    return ExcitonResult(energies=bound, binding_1s=float(-bound[0]),
                         r0=r0, eps_env=eps_env,
                         mu_over_m0=mu_over_m0,
                         r_max=float(r_max), n_basis=int(n_basis))
