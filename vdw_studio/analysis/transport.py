"""玻尔兹曼输运：2D 电导/迁移率张量（恒定弛豫时间近似，Phase 4.4）。

物理模型（标准半经典玻尔兹曼输运，Ashcroft–Mermin Ch.13 表述）：

- **电导张量**（恒定弛豫时间近似 CRTA）::

      σ_ij = e² τ ∫_BZ (d²k/(2π)²) v_i v_j (−∂f₀/∂ε)

  - v_i = (1/ħ)·∂ε/∂k_i：本实现用均匀网格邻点的周期性中心差分
    （抛物线带对中心差分**严格精确**——二次型的中心差分即导数），
    免去额外对角化；
  - −∂f₀/∂ε 为 Fermi–Dirac 展宽因子，T→0 退化为费米面 δ 函数；
  - 2D 载流子密度 n = g_s·∫ (d²k/(2π)²) f₀（费米能级由给定 n 二分
    反解；符号约定 n > 0 电子型、n < 0 空穴型，中性参考 = 带隙中点）；
  - 迁移率张量 μ = σ/(e·|n|)。

- **单位约定**：内部 eV / Å / fs，输出
  电导以 e²/h（2D 片电导量子单位，h/e² = 25.813 kΩ）与
  迁移率 [cm²/Vs]。

- **解析极限（测试锚点）**：
  1. **Drude 精确**：单带抛物线 + 热窗不触及邻带时，CRTA 迁移率与
     温度/简并度无关，μ = eτ/m*（任意温度/密度下严格成立）——
     黑磷烯空穴带是干净的检验对象（Γ 谷单带、最近邻带 1.5 eV 之外），
     T=100 K 下数值与解析值吻合 1–3%；
  2. **各向异性**：黑磷烯 μ 比 = 质量比（CRTA 各向异性完全由色散决定），
     高迁移率方向为 y 轴（4.38 Å）——与 DFT 结论方向一致
     （Qiao et al., PRL 113, 256803 (2014)：单层黑磷空穴迁移率高方向
      为 y，10000–26000 vs x 640–700 cm²/Vs；注意其各向异性比主要来自
      变形势+弹性模量，CRTA 无此机制，仅质量比可对账）；
  3. 时间反演对称模型（无磁场）σ_xy = 0。

**触碰带注记**（双层石墨烯的情形）：μ = eτ/m* 恒等式要求费米热窗内
只有单一抛物线带贡献。双层石墨烯 U=0 的触碰带（m* ≈ 0.046 m₀）在
低密度下另一支（镜像带）也落在热窗内，两带共同贡献电导而载流子数
只计净过剩——CRTA 迁移率可显著高于 eτ/m*（n=5×10¹¹ cm⁻² 时 ~2.5×，
细网格 N=800 复核确认，为模型物理而非数值误差）。此类体系应作
各向同性/τ 线性等结构检验，绝对 Drude 对账需热窗满足 μ ≫ 4kT。

实现注记：网格取 BZ 分数坐标 [0,1)² 均匀网格（胞中心）。低密度/light-mass
体系费米面小，需 Δk 网格分辨 ~4kT 能量窗口内的色散（能量间距 ≲ kT），
否则数值收敛不足——经验准则 N ≳ |b|·dε/dk(k_F)/(kT)。
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ..constants import ELEMENTARY_CHARGE, EV_ANGSTROM_TO_V, H_OVER_E2
from ..numerics import finite_scalar, integer, resolve_n_valence


@dataclass
class TransportResult:
    """CRTA 输运结果（2D 片电导与迁移率张量）。"""

    n_cm2: float             # 载流子密度 (cm⁻², +电子 / −空穴)
    fermi_ev: float          # 费米能级 (eV, 模型绝对能量零点)
    sigma_xx: float          # 片电导 σ_xx (e²/h)
    sigma_yy: float          # σ_yy (e²/h)
    sigma_xy: float          # σ_xy (e²/h)；无磁场时间反演下 ≈ 0
    mu_xx: float             # 迁移率 μ_xx (cm²/Vs)
    mu_yy: float             # μ_yy (cm²/Vs)
    mu_xy: float             # μ_xy (cm²/Vs)
    temperature: float       # K
    tau_fs: float            # 弛豫时间 (fs)
    n_mesh: int              # 网格边长（n_mesh × n_mesh）


def solve_transport(model, n_cm2: float, tau_fs: float = 10.0,
                    temperature: float = 300.0, n_mesh: int = 160,
                    n_valence: int | None = None,
                    g_spin: float = 2.0) -> TransportResult:
    """玻尔兹曼 CRTA 输运：给定载流子密度求电导/迁移率张量。

    参数:
        model: 提供 ``hamiltonian(kfrac)``（BZ 分数坐标）与
            ``lattice``（含 ``reciprocal_matrix``）的紧束缚模型。
        n_cm2: 载流子密度 (cm⁻²)；正 = 电子，负 = 空穴。
        tau_fs: 弛豫时间 (fs)——CRTA 唯一唯象参数，迁移率 ∝ τ。
        temperature: 温度 (K)。
        n_mesh: BZ 网格边长（总点数 n_mesh²）。
        n_valence: 占据带数（自旋无关计带）；默认 n_sites//2。
        g_spin: 自旋简并度（模型不含自旋时取 2）。

    返回:
        :class:`TransportResult`
    """
    # NaN 与任何值的比较都为假，会穿过朴素的范围校验并一路算出 NaN 结果。
    n_cm2 = finite_scalar(n_cm2, 'n_cm2')
    tau_fs = finite_scalar(tau_fs, 'tau_fs', positive=True)
    temperature = finite_scalar(temperature, 'temperature', positive=True)
    g_spin = finite_scalar(g_spin, 'g_spin', positive=True)
    n_mesh = integer(n_mesh, 'n_mesh', minimum=8)
    if n_cm2 == 0:
        raise ValueError("载流子密度不能为零")

    B = model.lattice.reciprocal_matrix[:2, :2]   # 行 = b1, b2 (Å⁻¹)
    a_bz_m2 = abs(np.linalg.det(B)) * 1e20        # BZ 面积 (m⁻²)

    # ---- 网格能量（胞中心） ------------------------------------------
    N = n_mesh
    idx = (np.arange(N) + 0.5) / N
    kf1, kf2 = np.meshgrid(idx, idx, indexing="ij")
    E = model.bands(np.stack([kf1.ravel(), kf2.ravel()], axis=1))  # (N², nb)
    nb = E.shape[1]
    n_val = resolve_n_valence(model, nb, n_valence)
    if not np.isfinite(E).all():
        raise ValueError("模型返回的网格能量含非有限值")

    # ---- 速度：网格邻点中心差分（周期），抛物线带严格精确 -------------
    # ∂ε/∂kfrac_j = ±邻差×N/2；∂ε/∂q = B⁻¹·∂ε/∂kfrac（q = kfrac·B 行约定）
    dEdk = np.empty((N * N, nb, 2))
    for j, axis in enumerate((0, 1)):             # axis 0 = k1, 1 = k2
        for b in range(nb):
            grid = E[:, b].reshape(N, N)
            dEdk[:, b, j] = ((np.roll(grid, -1, axis=axis)
                              - np.roll(grid, +1, axis=axis)) * (N / 2.0)
                             ).ravel()
    dEdq = dEdk @ np.linalg.inv(B).T              # (N², nb, 2) eV·Å
    vel = dEdq * EV_ANGSTROM_TO_V                # m/s

    # ---- 中性参考与费米能级二分 --------------------------------------
    e_vbm = E[:, n_val - 1].max()
    e_cbm = E[:, n_val].min()
    e_ref = 0.5 * (e_vbm + e_cbm)

    kbT = 8.617333262e-5 * temperature            # eV
    w_cell = a_bz_m2 / (N * N) / (2 * np.pi) ** 2  # m⁻² per mesh point
    occupied_ref = int((E < e_ref).sum())          # T=0 中性占据（自旋无关）

    def excess(e_f: float) -> float:
        with np.errstate(over="ignore"):   # 深尾 exp 上溢 → 结果仍正确 (0)
            occ = (1.0 / (np.exp((E - e_f) / kbT) + 1.0)).sum()
        return g_spin * w_cell * (occ - occupied_ref)

    n_target = n_cm2 * 1e4                        # m⁻²
    lo, hi = float(E.min()) - 1.0 - 20 * kbT, float(E.max()) + 1.0 + 20 * kbT
    # 括号检查：密度超出网格可达范围时，二分法会静默收敛到区间端点并报出
    # 一个看似合理的费米能级与迁移率。先确认目标确实被 [lo, hi] 夹住。
    n_lo, n_hi = excess(lo), excess(hi)
    if not n_lo < n_target < n_hi:
        raise ValueError(
            f"载流子密度 {n_cm2:g} cm⁻² 超出该网格可达范围 "
            f"({n_lo / 1e4:.4g} … {n_hi / 1e4:.4g} cm⁻²)；"
            "调大 n_mesh 或改用范围内的密度")
    for _ in range(200):
        mid = 0.5 * (lo + hi)
        if excess(mid) < n_target:
            lo = mid
        else:
            hi = mid
        if hi - lo < 1e-12:
            break
    e_fermi = 0.5 * (lo + hi)

    # ---- 电导张量 ----------------------------------------------------
    # σ_ij = e²·τ·g·w_cell·Σ v_i v_j (−∂f/∂ε_J)；eV⁻¹→J⁻¹: /(1.602e-19)
    with np.errstate(over="ignore"):   # |E−ε_F|/2kT 大时 cosh 上溢 → 权重 0
        dfde = (1.0 / np.cosh((E - e_fermi) / (2 * kbT)) ** 2) / (4 * kbT)
    pref = ELEMENTARY_CHARGE ** 2 * (tau_fs * 1e-15) * g_spin * w_cell \
        / ELEMENTARY_CHARGE                               # = e·τ·g·w (C·s·m⁻²)
    v0, v1 = vel[:, :, 0], vel[:, :, 1]           # (N², nb) m/s
    sigma_xx = pref * float(np.sum(v0 * v0 * dfde))
    sigma_yy = pref * float(np.sum(v1 * v1 * dfde))
    sigma_xy = pref * float(np.sum(v0 * v1 * dfde))

    mu_xx = sigma_xx / (ELEMENTARY_CHARGE * abs(n_target)) * 1e4   # cm²/Vs
    mu_yy = sigma_yy / (ELEMENTARY_CHARGE * abs(n_target)) * 1e4
    mu_xy = sigma_xy / (ELEMENTARY_CHARGE * abs(n_target)) * 1e4

    return TransportResult(
        n_cm2=n_cm2, fermi_ev=float(e_fermi),
        sigma_xx=sigma_xx * H_OVER_E2,           # → e²/h
        sigma_yy=sigma_yy * H_OVER_E2,
        sigma_xy=sigma_xy * H_OVER_E2,
        mu_xx=mu_xx, mu_yy=mu_yy, mu_xy=mu_xy,
        temperature=temperature, tau_fs=tau_fs, n_mesh=n_mesh,
    )
