"""层状介质堆叠的屏蔽相互作用（Poisson 传输矩阵解，Ovesen 2019 模型）。

物理模型（Ovesen, Brem, Malic, Commun. Phys. 2, 23 (2019)，
Supplementary Note 1；PDF 在 ``papers/ovesen2019/``）：

把异质结构建模为 z 方向的一维介质堆叠（半无限环境 + 有限厚度
均匀介质板），对每个面内波矢 k 解一维 Poisson 方程::

    d/dz (ε(z) dφ/dz) − ε(z)·k² φ = −q·δ(z−z_src)/ε₀

（2D 傅里叶变换后；界面条件：φ 连续、ε∂φ/∂z 连续；外边界：衰减）。
电荷平面取介质板**中心**（原文"neglecting the in-layer z dependence"）。
屏蔽相互作用与介电函数按原文式 (2) 定义::

    V_ll'(k) = e²/(ε₀·k·ε_ll'(k))，  ε_ll'(k) = e²/(ε₀·k·V_ll'(k))

实现为逐区指数展开的直接线性系统（每区解用区中心作指数参考点
避免溢出）；**不转录原文的 ε(k) 闭式**（Eqs. (3)-(4)，PDF 提取的
分式结构易错），而是直接解同一 Poisson 问题——用原文声明的四个
极限做回归锚点：

1. 均匀介质（所有 ε 相等）→ 解析 2D 库仑 e²e^{−kΔz}/(2ε₀εk)；
2. k → 0：ε(k) → ε_top + ε_bottom（电荷场的两介质分摊）；
3. R → 0（板贴合）+ 环境介质 ≪ 板介质：ε(k) 线性化
   ε(k) ≈ (ε_top+ε_bottom) + k·Σ(ε_l−1)·d_l（单层 Keldysh 化，
   χ₂D,eff/ε₀ = Σ(ε_l−1)d_l）；
4. 终锚：复现原文 Table 1 的 MoSe₂/WSe₂ 激子束缚能
   （层间 1s 150 meV、层内 205/177 meV）——介质参数全固定，
   唯一自由度是约化质量 μ（原文未给出数值），由束缚能反推并
   与 Kormányos 带质量交叉检验（tests/test_layered_dielectric.py）。

实空间势由数值 Hankel 变换 V(r) = (1/2π)∫V(k)J₀(kr)k dk 得到，
供 :func:`vdw_studio.analysis.exciton.solve_exciton_with_potential`
求解激子束缚能。

**文献对账结果（2026-10-30，大目标 C 收官）**：用 Ovesen 的介质参数
（d₁=d₂=0.645 nm、R=0、ε=16.5/15.1/1/3.9）+ Kormányos Tables 3/4
平均带质量（MoSe₂ μ=0.289、WSe₂ μ=0.192、层间 μ=0.242），三条通道
的 1s 束缚能（``tests/test_layered_dielectric.py::TestOvesenReconciliation``）：

- 层间：**161 meV vs 原文 150 meV（+7%）**——对堆叠顺序、r_max、
  k_max 全部不敏感（160.85 meV 逐位稳定）；
- 层内：216/195 meV vs 原文 205/177 meV（+5%/+10%）——原文未指明
  哪层贴衬底，两种顺序给出 +5%/+11% 与 +4%/+10%（歧义范围一致）。

两条独立参数链（他们的 DFT 拟合介质+质量 vs 本实现 + Kormányos 表）
在 ~10% 水平定量闭合。层内大 k 的 bulk 极限 E2/(2ε_l·k) 被 float64
逐位复现；层间 V(k) 的 e^{−kΔz} 压制使 Hankel 在 k ≳ 3 饱和（截断
尾差 <1e-8 相对）。k ∈ [4, 8] 的过渡带在 float64 下不可用（源平面
g 的 O(e^{−2kd}) 修正低于机器精度），已在代码内显式规避并测试固化。
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Sequence, Tuple

import numpy as np
from scipy import special

from ..numerics import finite_scalar

# e²/ε₀ = 4π × 14.3996 eV·Å（库仑常数 × 4π）
E2_OVER_EPS0 = 4 * np.pi * 14.3996


@dataclass(frozen=True)
class DielectricSlab:
    """有限厚度均匀介质板。"""

    epsilon: float
    thickness: float          # Å

    def __post_init__(self) -> None:
        finite_scalar(self.epsilon, 'epsilon', positive=True)
        finite_scalar(self.thickness, 'thickness', positive=True)


@dataclass(frozen=True)
class LayeredDielectricStack:
    """1D 介质堆叠：半无限底环境 / 若干板（自下而上）/ 半无限顶环境。

    电荷平面取各板中心；``outer_padding`` 是把半无限区截断为有限区
    的厚度（须 ≫ 1/k_max，衰减边界条件在截断处施加）。
    """

    top_env_epsilon: float
    bottom_env_epsilon: float
    slabs: Tuple[DielectricSlab, ...]   # 自下而上
    outer_padding: float = 120.0

    def __post_init__(self) -> None:
        finite_scalar(self.top_env_epsilon, 'top_env_epsilon', positive=True)
        finite_scalar(self.bottom_env_epsilon, 'bottom_env_epsilon', positive=True)
        if not self.slabs:
            raise ValueError("至少要有一层介质板")
        finite_scalar(self.outer_padding, 'outer_padding', positive=True)

    # ------------------------------------------------------------------
    def regions_for(self, source_slab: Optional[int] = None
                    ) -> Tuple[List[Tuple[float, float, float, bool]], List[float]]:
        """组装有限区列表 [(ε, z_lo, z_hi, has_source)] 与板中心 z。

        ``source_slab`` 标记源电荷所在的板（-1 无；板索引按 slabs 顺序）。
        """
        regions: List[Tuple[float, float, float, bool]] = []
        centers: List[float] = []
        pad = self.outer_padding
        # 底环境（有限截断）
        z0 = 0.0
        regions.append((self.bottom_env_epsilon, z0 - pad, z0, False))
        z = z0
        for i, slab in enumerate(self.slabs):
            regions.append((slab.epsilon, z, z + slab.thickness,
                            i == source_slab))
            centers.append(z + slab.thickness / 2)
            z += slab.thickness
        regions.append((self.top_env_epsilon, z, z + pad, False))
        return regions, centers

    def slab_center(self, index: int) -> float:
        _, centers = self.regions_for()
        return centers[index]


def _phi_particular(eps: float, k: float, z_src: float, has_src: bool,
                    z: float) -> float:
    if not has_src:
        return 0.0
    return E2_OVER_EPS0 / (2 * eps * k) * np.exp(-k * abs(z - z_src))


def _dphi_particular(eps: float, k: float, z_src: float, has_src: bool,
                     z: float) -> float:
    if not has_src:
        return 0.0
    sgn = 1.0 if z > z_src else -1.0
    return E2_OVER_EPS0 / (2 * eps * k) * (-k * sgn) * np.exp(-k * abs(z - z_src))


def _solve_potential_k(regions, k: float, z_src: float, z_eval: float) -> float:
    """单个 k 的势（eV，单位电荷源）。

    每区解写为 A_j e^{k(z−z_lo,j)} + B_j e^{−k(z−z_lo,j)}（指数以本区
    下边界为参考点）——大 k·厚度下 e^{±kz} 直接用绝对 z 会溢出
    （k_max=40 × 堆叠 120 Å → e^{4800}），锚定后指数 ≤ e^{k·厚度}。
    """
    n = len(regions)
    M = np.zeros((2 * n, 2 * n))
    rhs = np.zeros(2 * n)

    def idx(j: int, which: int) -> int:
        return 2 * j + which

    def exp_pair(j: int, z: float):
        z_ref = regions[j][1]
        return np.exp(k * (z - z_ref)), np.exp(-k * (z - z_ref))

    # 外边界衰减：底部区 B=0（e^{−k(z−z_ref)} 在 z→−∞ 发散），顶部区 A=0
    M[0, idx(0, 1)] = 1.0
    M[1, idx(n - 1, 0)] = 1.0
    r = 2
    for j in range(n - 1):
        z_i = regions[j][2]
        eps_j, eps_p = regions[j][0], regions[j + 1][0]
        ej_lo, ep_lo = exp_pair(j, z_i)       # 区 j 的上边界 = 其 z_ref → 1
        ej_hi, ep_hi = exp_pair(j + 1, z_i)   # 区 j+1 的下边界 = 其 z_ref → 1
        pp_j = _phi_particular(regions[j][0], k, z_src, regions[j][3], z_i)
        pp_p = _phi_particular(regions[j + 1][0], k, z_src,
                               regions[j + 1][3], z_i)
        M[r, idx(j, 0)] = ej_lo
        M[r, idx(j, 1)] = ep_lo
        M[r, idx(j + 1, 0)] = -ej_hi
        M[r, idx(j + 1, 1)] = -ep_hi
        rhs[r] = -(pp_j - pp_p)
        r += 1
        dp_j = _dphi_particular(regions[j][0], k, z_src, regions[j][3], z_i)
        dp_p = _dphi_particular(regions[j + 1][0], k, z_src,
                                regions[j + 1][3], z_i)
        M[r, idx(j, 0)] = eps_j * k * ej_lo
        M[r, idx(j, 1)] = -eps_j * k * ep_lo
        M[r, idx(j + 1, 0)] = -eps_p * k * ej_hi
        M[r, idx(j + 1, 1)] = eps_p * k * ep_hi
        rhs[r] = -(eps_j * dp_j - eps_p * dp_p)
        r += 1
    sol = np.linalg.solve(M, rhs)
    for j, (eps, z_lo, z_hi, has_src) in enumerate(regions):
        if z_lo <= z_eval <= z_hi:
            ej, ep = exp_pair(j, z_eval)
            return (sol[idx(j, 0)] * ej + sol[idx(j, 1)] * ep
                    + _phi_particular(eps, k, z_src, has_src, z_eval))
    raise ValueError("z_eval 不在任何区域内")


def _g_propagate(g: float, kd: float, downward: bool) -> float:
    """区内 g 传播（厚度 kd = k·d，向下/向上两支）。

    代数式 (g∓T)/(1∓gT)（T = tanh(kd)）在 T 饱和到 1.0 且 g = ∓1 时
    出现 0/0——此时物理极限是 g 钉在饱和端（上行/下行解解耦），
    直接 snap。kd ≳ 18 时 tanh 在 float64 中饱和，guard 接管。
    """
    T = np.tanh(kd)
    if downward:
        den = 1.0 - g * T
        if abs(den) < 1e-200:
            return -1.0 if g < 1.0 else 1.0
        return (g - T) / den
    den = 1.0 + g * T
    if abs(den) < 1e-200:
        return 1.0 if g > -1.0 else -1.0
    return (g + T) / den


def screened_interaction_k(stack: LayeredDielectricStack,
                           k: Sequence[float], source_slab: int,
                           test_slab: int) -> np.ndarray:
    """屏蔽相互作用 V(k)（eV；式 (2) 形式，含 e²）。

    电荷取源板中心单位正电荷，势在测试板中心评估。实现用导纳递推
    （tanh 稳定形式，大 k·厚度不溢出）；``tests`` 将其与直接矩阵解
    在中等 k 对账（矩阵解在 k·厚度 ≳ 300 失去精度，仅作参考实现）。
    """
    k = np.atleast_1d(np.asarray(k, dtype=float))
    if not np.isfinite(k).all() or np.any(k <= 0):
        raise ValueError("k 必须为正")
    if not (0 <= source_slab < len(stack.slabs)
            and 0 <= test_slab < len(stack.slabs)):
        raise ValueError("板索引越界")
    out = np.empty(len(k))
    for i, kk in enumerate(k):
        out[i] = _solve_potential_k_admittance(stack, float(kk),
                                               source_slab, test_slab)
    return out


def _solve_potential_k_admittance(stack: LayeredDielectricStack,
                                  k: float, source_slab: int,
                                  test_slab: int) -> float:
    """导纳（Riccati）递推版的单 k 势——大 k 稳定。

    g ≡ (1/k)φ'/φ（对数导数归一）。上/下两支衰减解在源平面的
    g⁺/g⁻ 由顶/底环境向内递推（界面：εg 连续；区内传播：
    g_lo = (g_hi − T)/(1 − g_hi·T) 等，T = tanh(k·厚度)——
    tanh 饱和于 ±1，无 e^{k·厚度} 溢出）。源平面势::

        V(z_src) = E2/(k·ε_s·(g⁻ − g⁺))

    （跨源平面的 εφ' 跳变 = −q/ε₀）。到测试平面的形状比按
    cosh(kδ) ± g·sinh(kδ) 的**因式分解**形式传播（无对消；
    k·厚度 ≤ k_max·板厚 有界，不溢出）。
    """
    regions, centers = stack.regions_for(source_slab)
    n = len(regions)
    z_src = centers[source_slab]
    z_eval = centers[test_slab]
    src_region = next(j for j, rg in enumerate(regions)
                      if rg[1] <= z_src <= rg[2])

    def region_ratio_up(g: float, delta: float) -> float:
        """φ(z+δ)/φ(z)（向上解，g 在 z 处）。

        cosh(kδ) + g·sinh(kδ) 的因式分解 = e^{kδ}(1+g)/2 +
        e^{−kδ}(1−g)/2——两项同号，无灾难性对消（g ≈ −1 时
        cosh−sinh 直接相减会丢光有效数字）。
        """
        em, ep = np.exp(-k * delta), np.exp(k * delta)
        return (1.0 + g) / 2 * ep + (1.0 - g) / 2 * em

    def region_ratio_down(g: float, delta: float) -> float:
        """φ(z−δ)/φ(z)（向下解，g 在 z 处）= cosh(kδ) − g·sinh(kδ)。"""
        em, ep = np.exp(-k * delta), np.exp(k * delta)
        return (1.0 - g) / 2 * ep + (1.0 + g) / 2 * em

    # ---- g⁺：自顶环境向下递推到源平面 ----
    g = -1.0                       # 顶环境衰减解
    for j in range(n - 1, src_region, -1):
        # 从区 j 的下边界穿界面进入区 j−1：ε_{j−1} g_{j−1} = ε_j g_j
        g = regions[j][0] / regions[j - 1][0] * g
        if j - 1 > src_region:
            g = _g_propagate(g, k * (regions[j - 1][2] - regions[j - 1][1]),
                             downward=True)
    g_plus = _g_propagate(g, k * (regions[src_region][2] - z_src),
                          downward=True)

    # ---- g⁻：自底环境向上递推到源平面 ----
    g = 1.0                        # 底环境衰减解
    for j in range(0, src_region):
        # 从区 j 的上边界穿界面进入区 j+1：ε_{j+1} g_{j+1} = ε_j g_j
        g = regions[j][0] / regions[j + 1][0] * g
        if j + 1 < src_region:
            g = _g_propagate(g, k * (regions[j + 1][2] - regions[j + 1][1]),
                             downward=False)
    g_minus = _g_propagate(g, k * (z_src - regions[src_region][1]),
                           downward=False)

    if g_minus <= g_plus:
        raise RuntimeError(
            f"导纳递推异常：g⁻={g_minus} ≤ g⁺={g_plus}（k={k}）——堆叠参数有误")

    v_src = E2_OVER_EPS0 / (k * regions[src_region][0] * (g_minus - g_plus))

    # ---- 形状比：源平面 → 测试平面（φ 跨界面连续，比值 1）----
    ratio = 1.0
    if z_eval > z_src:
        g, z, j = g_plus, z_src, src_region
        while z < z_eval - 1e-12:
            z_next = min(z_eval, regions[j][2])
            delta = z_next - z
            if delta > 1e-12:
                if k * delta > 30.0:
                    # 饱和段：上行解纯指数衰减（ growing 分量被
                    # e^{−2kδ} 压到 float64 之下，g 的下一位数字
                    # 不可用——直接取纯衰减）
                    ratio *= np.exp(-k * delta)
                    g = -1.0
                else:
                    ratio *= region_ratio_up(g, delta)
                    g = _g_propagate(g, k * delta, downward=False)
            z = z_next
            if z < z_eval - 1e-12:
                # 穿界面（区 j 顶 → 区 j+1 底）：εg 连续
                g = regions[j][0] / regions[j + 1][0] * g
                j += 1
    elif z_eval < z_src:
        g, z, j = g_minus, z_src, src_region
        while z > z_eval + 1e-12:
            z_next = max(z_eval, regions[j][1])
            delta = z - z_next
            if delta > 1e-12:
                if k * delta > 30.0:
                    ratio *= np.exp(-k * delta)
                    g = 1.0
                else:
                    ratio *= region_ratio_down(g, delta)
                    g = _g_propagate(g, k * delta, downward=True)
            z = z_next
            if z > z_eval + 1e-12:
                g = regions[j][0] / regions[j - 1][0] * g
                j -= 1
    return v_src * ratio


def dielectric_function_k(stack: LayeredDielectricStack,
                          k: Sequence[float], source_slab: int,
                          test_slab: int) -> np.ndarray:
    """介电函数 ε_ll'(k) = e²/(ε₀·k·V(k))（原文式 (2) 的反读）。"""
    k = np.atleast_1d(np.asarray(k, dtype=float))
    v = screened_interaction_k(stack, k, source_slab, test_slab)
    return E2_OVER_EPS0 / (k * v)


def interaction_real_space(stack: LayeredDielectricStack,
                           r: Sequence[float], source_slab: int,
                           test_slab: int, *,
                           k_max: Optional[float] = None, n_k: int = 4000
                           ) -> np.ndarray:
    """实空间势 V(r)（eV）：数值 Hankel 变换
    V(r) = (1/2π)∫ V(k) J₀(kr) k dk。

    通道相关的大 k 处理（数值稳定性，均已验证）：

    - **层内**（source == test）：V(k) → E2/(2ε_s·k)（bulk 极限，
      float64 精确复现）。减除该渐近项后积分核 ~1/k³ 绝对收敛，
      解析库仑尾 −E2/(4πε_s·r) = −14.4/(ε_s·r) 解析加回；
      ``k_max`` 默认 25。
    - **层间**：V(k) ~ e^{−k·Δz}·bulk 指数衰减（float64 下 k=10 处
      已达 1e-29）——积分在 k ≲ 3 就饱和，``k_max`` 默认 3
      （导纳递推在该域由矩阵解逐点验证，见 tests）。

    ``n_k``：Gauss–Legendre 节点数。
    """
    r = np.atleast_1d(np.asarray(r, dtype=float))
    if not np.isfinite(r).all() or np.any(r <= 0):
        raise ValueError("r 必须为正")
    intralayer = (source_slab == test_slab)
    if k_max is None:
        k_max = 25.0 if intralayer else 3.0
    finite_scalar(k_max, 'k_max', positive=True)
    n_k = int(n_k)
    if n_k < 8:
        raise ValueError("n_k ≥ 8")

    eps_s = stack.slabs[source_slab].epsilon
    x, w = special.roots_legendre(n_k)
    kk = k_max * (x + 1.0) / 2.0
    wk = k_max * w / 2.0
    vk = screened_interaction_k(stack, kk, source_slab, test_slab)
    if intralayer:
        # 大 k 渐近（bulk 屏蔽，float64 精确复现）减除 → 核 ~1/k³ 收敛
        asym = E2_OVER_EPS0 / (2 * eps_s * kk)
        integrand = (vk - asym) * kk * wk
    else:
        integrand = vk * kk * wk
    out = np.empty(len(r))
    chunk = max(1, int(2e7 // max(len(kk), 1)))
    for start in range(0, len(r), chunk):
        rr = r[start:start + chunk]
        J = special.j0(np.outer(rr, kk))
        out[start:start + chunk] = J @ integrand
    out /= (2 * np.pi)
    if intralayer:
        # 渐近项的 Hankel 解析加回：e·φ₊ 的 bulk 库仑 = +E2/(4πε_s r)
        out += E2_OVER_EPS0 / (4 * np.pi * eps_s * r)
    # 符号约定：本函数返回**电子-空穴相互作用能**（负=吸引），与
    # ``exciton.keldysh_potential`` 同约定 = −e·φ₊（screened_interaction_k
    # 返回的是正电荷的势 e·φ₊，恒正）。
    return -out


# ----------------------------------------------------------------------
# Ovesen 2019 的 MoSe₂/WSe₂ 堆叠（Supplementary Note 1 参数）
# ----------------------------------------------------------------------
OVesen_D_SLAB = 6.45        # Å（d₁ = d₂ = 0.645 nm）
OVesen_EPS_MOSE2 = 16.5
OVesen_EPS_WSE2 = 15.1
OVesen_EPS_SUBSTRATE = 3.9  # SiO₂
OVesen_EPS_AIR = 1.0


def ovesen_stack(mose2_below: bool = True) -> LayeredDielectricStack:
    """Ovesen 2019 的 MoSe₂/WSe₂ 堆叠（SiO₂ 衬底 + 空气环境）。

    ``mose2_below``：MoSe₂ 是否贴衬底（原文未指明堆叠顺序；两种顺序
    都算并对比——见 tests/test_layered_dielectric.py 的诚实记录）。
    """
    lower = DielectricSlab(OVesen_EPS_MOSE2 if mose2_below else OVesen_EPS_WSE2,
                           OVesen_D_SLAB)
    upper = DielectricSlab(OVesen_EPS_WSE2 if mose2_below else OVesen_EPS_MOSE2,
                           OVesen_D_SLAB)
    return LayeredDielectricStack(top_env_epsilon=OVesen_EPS_AIR,
                                  bottom_env_epsilon=OVesen_EPS_SUBSTRATE,
                                  slabs=(lower, upper))


# 层索引约定：0 = 下板（贴衬底），1 = 上板
def ovesen_slab_index(material: str, mose2_below: bool = True) -> int:
    key = material.strip().lower()
    is_mose2 = key in ("mose2", "mose₂")
    if key not in ("mose2", "mose₂", "wse2", "wse₂"):
        raise ValueError(f"未知材料 {material!r}（MoSe2/WSe2）")
    return (0 if is_mose2 else 1) if mose2_below else (1 if is_mose2 else 0)
