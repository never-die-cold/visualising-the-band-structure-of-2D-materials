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
from ..constants import HBAR2_OVER_2M0
from ..numerics import finite_scalar, integer

# e²/(4πε₀) = 14.3996 eV·Å（库仑常数）
COULOMB_CONST = 14.3996
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
    r0 = finite_scalar(r0, 'r0', positive=True)
    eps_env = finite_scalar(eps_env, 'eps_env', positive=True)
    if not np.isfinite(r).all() or np.any(r <= 0):
        raise ValueError('r must contain finite positive radii')
    rho = r / r0
    # scipy.special.struve(0, ρ) = H₀(ρ); special.y0(ρ) = Y₀(ρ)
    return -(COULOMB_CONST / eps_env) * (np.pi / 2) * \
        (special.struve(0, rho) - special.y0(rho)) / r0


def coulomb_potential(r, eps_env: float = 1.0) -> np.ndarray:
    """纯库仑参考势 −e²/(4πε₀ε_env·r) (eV)。"""
    eps_env = finite_scalar(eps_env, 'eps_env', positive=True)
    r = np.asarray(r, dtype=float)
    if not np.isfinite(r).all() or np.any(r <= 0):
        raise ValueError('r must contain finite positive radii')
    return -COULOMB_CONST / (eps_env * r)


# Gauss–Laguerre 节点（模块级缓存；层间势的光滑指数型积分一次算好）
_LAGUERRE_CACHE: dict = {}


def _laguerre_nodes(n: int):
    if n not in _LAGUERRE_CACHE:
        _LAGUERRE_CACHE[n] = special.roots_laguerre(n)
    return _LAGUERRE_CACHE[n]


def keldysh_potential_interlayer(r, r0: float, d: float,
                                 eps_env: float = 1.0) -> np.ndarray:
    """层间分离 d 的 Rytova–Keldysh 屏势 (eV)，吸引（负值）。

    电子与空穴分居相距 d (Å) 的两层：动量空间在共面势上乘真空传播
    因子 e^{−qd}（2D 屏蔽单层近似：屏蔽来自同一 r₀ 的极化层，层间
    传播为真空）::

        V_d(q) = V_RK(q)·e^{−qd}，  V_RK(q) = −(e²/4πε₀ε_env)/(q(1+r₀q))

    经 Laplace–Bessel 恒等式（∫J₀(qr)e^{−βq}dq = 1/√(r²+β²)）化为
    Gauss–Laguerre 可积的光滑形式::

        V_d(r) = −(e²/4πε₀ε_env) ∫₀^∞ e^{−u} / √(r² + (d+r₀u)²) du

    两个解析锚点（tests/test_exciton.py）：

    - d = 0：严格退回 :func:`keldysh_potential`（∫e^{−u}/√(x²+u²)du
      = (π/2)[H₀−Y₀] 正是 RK 势的积分表示）——实现中直接委托；
    - r₀ → 0：严格退回三维层间库仑 −e²/(4πε₀ε_env·√(r²+d²))
      （屏蔽层不再起作用，真空传播保留）。

    近似层级（诚实标注）：单层屏蔽 + 精确真空传播；未含双层各自
    极化的更细结构。
    """
    r0 = finite_scalar(r0, 'r0', positive=True)
    d = finite_scalar(d, 'd')
    if d < 0:
        raise ValueError('层间分离 d 必须 ≥ 0')
    eps_env = finite_scalar(eps_env, 'eps_env', positive=True)
    r = np.asarray(r, dtype=float)
    if not np.isfinite(r).all() or np.any(r <= 0):
        raise ValueError('r must contain finite positive radii')
    if d == 0.0:
        return keldysh_potential(r, r0, eps_env)
    nodes, weights = _laguerre_nodes(200)
    u = d + r0 * nodes
    g = 1.0 / np.sqrt(r[..., None] ** 2 + u[None, :] ** 2)
    return -(COULOMB_CONST / eps_env) * (g @ weights)


def interlayer_coulomb_potential(r, d: float, eps_env: float = 1.0) -> np.ndarray:
    """三维层间库仑势 −e²/(4πε₀ε_env·√(r²+d²)) (eV)。

    :func:`keldysh_potential_interlayer` 的 r₀→0 极限，独立给出以便
    极限锚点直接对账。
    """
    d = finite_scalar(d, 'd')
    if d < 0:
        raise ValueError('层间分离 d 必须 ≥ 0')
    eps_env = finite_scalar(eps_env, 'eps_env', positive=True)
    r = np.asarray(r, dtype=float)
    if not np.isfinite(r).all() or np.any(r <= 0):
        raise ValueError('r must contain finite positive radii')
    return -COULOMB_CONST / (eps_env * np.sqrt(r ** 2 + d ** 2))


def _r_times_potential(r: np.ndarray, r0: Optional[float], eps_env: float,
                       interlayer_d: Optional[float] = None) -> np.ndarray:
    """积分核 w(r) = r·V(r)：r→0 极限解析处理（避免 0·∞）。

    - 纯库仑：r·V → −e²/(4πε₀ε_env)（有限常数）；
    - Keldysh（共面或层间）：V 在 r→0 有限（共面为 0，层间为
      −C∫e^{−u}/(d+r₀u)du），故 r·V → 0。
    """
    w = np.empty_like(r)
    if r0 is None and interlayer_d is None:
        w[1:] = coulomb_potential(r[1:], eps_env) * r[1:]
        w[0] = -COULOMB_CONST / eps_env
    elif r0 is None:
        w[0] = 0.0
        w[1:] = interlayer_coulomb_potential(r[1:], interlayer_d,
                                             eps_env) * r[1:]
    else:
        w[0] = 0.0
        w[1:] = keldysh_potential_interlayer(r[1:], r0, interlayer_d or 0.0,
                                             eps_env) * r[1:]
    return w


@dataclass
class ExcitonResult:
    """激子能级求解结果。"""

    energies: np.ndarray      # (n_levels,) 束缚态能量 (eV, 负值)
    binding_1s: Optional[float]  # None when the finite basis finds no bound state
    r0: Optional[float]       # 极化长度 (None = 纯库仑)
    eps_env: float
    mu_over_m0: float         # 约化质量 (m₀ 单位)
    r_max: float = 0.0        # 径向盒子尺寸 (Å)
    n_basis: int = 0          # Bessel 基函数数目
    interlayer_d: float = 0.0  # 层间分离 (Å)；0 = 共面
    status: str = 'bound-states-found'
    requested_levels: int = 0
    bound_states_in_basis: int = 0

    def rydberg(self) -> float:
        """有效 Rydberg Ry* = (μ/m₀)·13.6057/ε² (eV)。"""
        return self.mu_over_m0 * RY / self.eps_env ** 2


def solve_exciton(mu_over_m0: float = 0.25,
                  eps_env: float = 1.0,
                  r0: Optional[float] = None,
                  n_levels: int = 4,
                  n_basis: int = 250,
                  n_quad: int = 6000,
                  r_max: Optional[float] = None,
                  interlayer_d: Optional[float] = None) -> ExcitonResult:
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
        interlayer_d: 电子–空穴层间分离 (Å)；None/0 = 共面（原行为，
            束缚能上界）。>0 时势取 Rytova–Keldysh 的层间推广
            （见 :func:`keldysh_potential_interlayer`；r0=None 时为
            三维层间库仑）。层间激子的真实束缚能低于共面近似。

    返回:
        :class:`ExcitonResult`（能量升序，负值为束缚态）。
    """
    mu_over_m0 = finite_scalar(mu_over_m0, 'mu_over_m0', positive=True)
    eps_env = finite_scalar(eps_env, 'eps_env', positive=True)
    if r0 is not None:
        r0 = finite_scalar(r0, 'r0', positive=True)
    n_basis = integer(n_basis, 'n_basis')
    n_quad = integer(n_quad, 'n_quad', minimum=3)
    n_levels = integer(n_levels, 'n_levels')
    if n_levels > n_basis:
        raise ValueError('n_levels cannot exceed n_basis')
    if n_quad < n_basis + 1:
        raise ValueError('n_quad must be at least n_basis + 1 to resolve the basis')
    if r_max is not None:
        r_max = finite_scalar(r_max, 'r_max', positive=True)
    if interlayer_d is not None:
        interlayer_d = finite_scalar(interlayer_d, 'interlayer_d')
        if interlayer_d < 0:
            raise ValueError('interlayer_d 必须 ≥ 0')
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
    w = _r_times_potential(r, r0, eps_env, interlayer_d)   # r·V(r)
    qw = np.full(n_quad, dr)
    qw[0] *= 0.5
    qw[-1] *= 0.5
    J = special.j0(np.outer(kn, r))                     # (N, n_quad)
    V = ((J * w) * qw) @ J.T

    # 广义本征值问题 H c = E·N c (度量 = diag(Nₙ))
    H = T + V
    eig = scipy.linalg.eigh(H, np.diag(norms), eigvals_only=True)
    all_bound = eig[eig < 0]
    bound = all_bound[:n_levels]
    status = 'no-bound-state' if not len(bound) else 'partial-bound-spectrum' if len(bound) < n_levels else 'bound-states-found'
    return _make_result(bound, len(all_bound), status, r0, eps_env,
                        mu_over_m0, float(r_max), int(n_basis), interlayer_d,
                        n_levels)


def solve_exciton_with_potential(potential, r_max: float,
                                 mu_over_m0: float,
                                 n_levels: int = 4,
                                 n_basis: int = 250,
                                 n_quad: int = 6000) -> ExcitonResult:
    """以**任意径向势**求解激子（DVR 核与 :func:`solve_exciton` 相同）。

    参数:
        potential: 可调用对象 V(r) → eV（r > 0 的数组；r→0 处须有限或
            弱于 1/r——核 r·V(r) 在 r=0 有限即可）。
        r_max: 径向盒子半径 (Å)，须显式给出（自适应逻辑依赖 r₀ 分支，
            自定义势无此信息）。
        mu_over_m0 / n_levels / n_basis / n_quad: 同 :func:`solve_exciton`。

    典型用途：层状介质模型（``analysis.layered_dielectric``）的实空间
    势——文献对账（Ovesen 2019）即由此进入。
    """
    r_max = finite_scalar(r_max, 'r_max', positive=True)
    mu_over_m0 = finite_scalar(mu_over_m0, 'mu_over_m0', positive=True)
    n_basis = integer(n_basis, 'n_basis')
    n_quad = integer(n_quad, 'n_quad', minimum=3)
    n_levels = integer(n_levels, 'n_levels')
    if n_levels > n_basis:
        raise ValueError('n_levels cannot exceed n_basis')
    if n_quad < n_basis + 1:
        raise ValueError('n_quad must be at least n_basis + 1 to resolve the basis')
    hbar2_over_2mu = HBAR2_OVER_2M0 / mu_over_m0

    zeros = special.jn_zeros(0, n_basis)
    kn = zeros / r_max
    norms = 0.5 * r_max ** 2 * special.j1(zeros) ** 2
    T = np.diag(hbar2_over_2mu * kn ** 2 * norms)

    r = np.linspace(0.0, r_max, n_quad)
    dr = r[1] - r[0]
    v_vals = np.asarray(potential(r[1:]), dtype=float)
    if v_vals.shape != (n_quad - 1,) or not np.isfinite(v_vals).all():
        raise ValueError('potential 必须返回 (len(r)-1,) 的有限数组')
    w = np.zeros(n_quad)
    w[1:] = v_vals * r[1:]
    qw = np.full(n_quad, dr)
    qw[0] *= 0.5
    qw[-1] *= 0.5
    J = special.j0(np.outer(kn, r))
    V = ((J * w) * qw) @ J.T

    H = T + V
    eig = scipy.linalg.eigh(H, np.diag(norms), eigvals_only=True)
    all_bound = eig[eig < 0]
    bound = all_bound[:n_levels]
    status = 'no-bound-state' if not len(bound) else 'partial-bound-spectrum' if len(bound) < n_levels else 'bound-states-found'
    return _make_result(bound, len(all_bound), status, None, 1.0,
                        mu_over_m0, float(r_max), int(n_basis), None,
                        n_levels)


def _make_result(bound, n_all_bound: int, status: str, r0, eps_env: float,
                 mu_over_m0: float, r_max: float, n_basis: int,
                 interlayer_d, n_levels: int) -> ExcitonResult:
    return ExcitonResult(energies=bound, binding_1s=float(-bound[0]) if len(bound) else None,
                         r0=r0, eps_env=eps_env,
                         mu_over_m0=mu_over_m0,
                         r_max=float(r_max), n_basis=int(n_basis),
                         interlayer_d=float(interlayer_d or 0.0), status=status,
                         requested_levels=n_levels,
                         bound_states_in_basis=int(n_all_bound))
