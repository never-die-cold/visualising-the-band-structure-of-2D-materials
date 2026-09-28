"""材料性质分析：带隙（直接/间接）、有效质量、费米速度。

分析对象是"位点型"紧束缚模型（具有 ``lattice`` / ``n_sites`` /
``energies_at(kfrac)`` 接口，见 ``engine.models`` 与
``engine.zahid_mos2``）；TMD k·p 模型自带自旋分辨的质量分析
（``engine.kp_tmd.TMDKpModel.effective_mass``），由预设库分别接线。

占据数约定：每位点轨道 2 自旋 → 占据带数 = n_sites（非自旋极化）；
调用方可通过 ``n_valence`` 覆盖（如特殊填充）。

换算常数：ħ²/(2m₀) = 3.809982 eV·Å²；1 eV·Å 的能带斜率对应
1.51927×10⁵ m/s（除以 ħ 的单位换算）。
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Sequence, Tuple

import numpy as np

from ..engine.kpath import DISPLAY_SYMBOLS, KPath
from ..structure.lattice import Lattice

HBAR2_OVER_2M0 = 3.809982          # eV·Å²
SLOPE_TO_VELOCITY = 1.51927e5      # (m/s) per (eV·Å)


@dataclass
class GapResult:
    """带隙分析结果。"""

    gap: Optional[float]          # eV；None = 无带隙（金属/半金属）
    direct: Optional[bool]        # 直接/间接带隙
    vbm: float                    # 价带顶 (eV)
    cbm: float                    # 导带底 (eV)
    vbm_k: np.ndarray             # VBM 的倒格分数坐标 (2,)
    cbm_k: np.ndarray             # CBM 的倒格分数坐标 (2,)
    vbm_label: str                # VBM 最近高对称点符号（Γ/M/K/…）
    cbm_label: str
    n_valence: int                # 使用的占据带数


def _nearest_hsym_label(kfrac, kpath: Optional[KPath],
                        lattice: Lattice) -> str:
    """k 点最近的高对称点显示符号（无路径时返回 '·'）。"""
    if kpath is None:
        return "·"
    recip = lattice.reciprocal_matrix
    best, best_d = "·", np.inf
    for name, pt in kpath.points.items():
        kc = np.array([*pt, 0.0]) @ recip
        kk = np.array([*kfrac, 0.0]) @ recip
        d = np.linalg.norm(kc - kk)
        if d < best_d:
            best, best_d = DISPLAY_SYMBOLS.get(name, name), d
    return best


def analyze_gap(model, kpath: Optional[KPath] = None,
                n_per_segment: int = 40,
                n_valence: Optional[int] = None) -> GapResult:
    """沿 k 路径（或默认路径）求解能带并分析带隙。

    参数:
        model: 位点型紧束缚模型（``energies_at(kfrac)`` 接口）。
        kpath: k 路径；默认按模型晶格自动选取（Γ-M-K-Γ 等）。
        n_per_segment: 每段采样数。
        n_valence: 占据带数；默认 = 模型自由度的一半（每位点/基函数
            按非自旋极化双占据）。

    返回:
        :class:`GapResult`。带隙 < 1 meV 视为零带隙（gap=None, direct=None）。
    """
    if kpath is None:
        kpath = KPath.for_lattice(model.lattice)
    if n_valence is None:
        # 位点型 TB 模型为无自旋基（n_sites 条带、n_sites 个电子 →
        # 占据 n_sites/2 条带）；sp³d⁵ 等含自旋基组同理取自由度的一半。
        n_bands = getattr(model, "n_basis", model.n_sites)
        n_valence = getattr(model, "n_sites", n_bands) // 2

    # 沿路径采样（分数坐标）
    kfracs: List[np.ndarray] = []
    for (p0, p1) in kpath.segments():
        for t in np.linspace(0.0, 1.0, n_per_segment, endpoint=False):
            kfracs.append([p0[0] + t * (p1[0] - p0[0]),
                           p0[1] + t * (p1[1] - p0[1])])
    kfracs.append(list(kpath.points[kpath.path[-1]]))
    kfrac = np.asarray(kfracs)

    energies = model.bands(kfrac)                    # (N, n_bands)
    vbm_idx = n_valence - 1
    cbm_idx = n_valence
    if cbm_idx >= energies.shape[1]:
        raise ValueError(f"占据带数 {n_valence} 超出可求能带数 "
                         f"{energies.shape[1]}")

    vbm_ki = int(np.argmax(energies[:, vbm_idx]))
    cbm_ki = int(np.argmin(energies[:, cbm_idx]))
    vbm = float(energies[vbm_ki, vbm_idx])
    cbm = float(energies[cbm_ki, cbm_idx])
    gap = cbm - vbm

    zero_tol = 1e-3   # eV
    if gap < zero_tol:
        return GapResult(gap=None, direct=None, vbm=vbm, cbm=cbm,
                         vbm_k=kfrac[vbm_ki], cbm_k=kfrac[cbm_ki],
                         vbm_label=_nearest_hsym_label(kfrac[vbm_ki], kpath,
                                                      model.lattice),
                         cbm_label=_nearest_hsym_label(kfrac[cbm_ki], kpath,
                                                      model.lattice),
                         n_valence=n_valence)

    # 直接/间接：VBM 与 CBM 的 k 点是否重合（用笛卡尔距离判断）
    recip = model.lattice.reciprocal_matrix
    kv = kfrac[vbm_ki] @ recip[:2, :2]
    kc = kfrac[cbm_ki] @ recip[:2, :2]
    direct = bool(np.linalg.norm(kv - kc) < 1e-6)

    return GapResult(gap=float(gap), direct=direct, vbm=vbm, cbm=cbm,
                     vbm_k=kfrac[vbm_ki], cbm_k=kfrac[cbm_ki],
                     vbm_label=_nearest_hsym_label(kfrac[vbm_ki], kpath,
                                                  model.lattice),
                     cbm_label=_nearest_hsym_label(kfrac[cbm_ki], kpath,
                                                  model.lattice),
                     n_valence=n_valence)


def effective_mass(model, k0: Sequence[float], band: int,
                   direction: Sequence[float] = (1.0, 0.0),
                   dk: float = 1e-4) -> float:
    """k0 处第 ``band`` 条带沿给定方向的（|m*|）有效质量，单位 m₀。

    中心差分计算曲率（分数坐标差分后换算到笛卡尔），m* = ħ²/λ。
    """
    d = np.asarray(direction, dtype=float)
    d = d / np.linalg.norm(d)
    k0 = np.asarray(k0, dtype=float)
    recip = model.lattice.reciprocal_matrix
    scale = float(np.linalg.norm(d @ recip[:2, :2])) ** 2   # 分数→笛卡尔

    e0 = model.energies_at(k0)[band]
    ep = model.energies_at(k0 + dk * d)[band]
    em = model.energies_at(k0 - dk * d)[band]
    curv_frac = (ep + em - 2 * e0) / dk ** 2
    return abs(2 * HBAR2_OVER_2M0 / (curv_frac / scale))


def principal_masses(model, k0: Sequence[float], band: int,
                     dk: float = 1e-4) -> Tuple[float, float, float]:
    """k0 处曲率张量的主质量 (m₁, m₂, θ)。

    对曲率张量 H_ij = ∂²E/∂k_i∂k_j（笛卡尔）做本征分解，
    返回 (|m₁|, |m₂|, 主轴角度 rad)——各向异性材料的特征量。
    """
    k0 = np.asarray(k0, dtype=float)
    recip = model.lattice.reciprocal_matrix
    e00 = model.energies_at(k0)[band]

    # 直接在笛卡尔 k 坐标差分：k_cart = kfrac @ B[:2,:2]
    B = recip[:2, :2]
    Binv = np.linalg.inv(B)

    def energy_at_cart(dc: np.ndarray) -> float:
        return model.energies_at(k0 + Binv @ dc)[band]

    h = dk
    e_xp = energy_at_cart(np.array([+h, 0.0]))
    e_xm = energy_at_cart(np.array([-h, 0.0]))
    e_yp = energy_at_cart(np.array([0.0, +h]))
    e_ym = energy_at_cart(np.array([0.0, -h]))
    h2 = h / np.sqrt(2)
    e_xpyp = energy_at_cart(np.array([+h2, +h2]))
    e_xpym = energy_at_cart(np.array([+h2, -h2]))
    e_xmyp = energy_at_cart(np.array([-h2, +h2]))
    e_xmym = energy_at_cart(np.array([-h2, -h2]))

    hxx = (e_xp + e_xm - 2 * e00) / h ** 2
    hyy = (e_yp + e_ym - 2 * e00) / h ** 2
    hxy = (e_xpyp - e_xpym - e_xmyp + e_xmym) / (4 * h2 ** 2)

    Hmat = np.array([[hxx, hxy], [hxy, hyy]])
    vals, vecs = np.linalg.eigh(Hmat)
    # 主轴角度（相对 x 轴）
    v = vecs[:, 0]
    theta = float(np.arctan2(v[1], v[0]))
    m1 = abs(2 * HBAR2_OVER_2M0 / vals[0])
    m2 = abs(2 * HBAR2_OVER_2M0 / vals[1])
    return m1, m2, theta


def fermi_velocity(model, k0: Sequence[float],
                   band: int, direction: Sequence[float] = (1.0, 0.0),
                   dk: float = 1e-5) -> float:
    """带交叉点处的费米速度 |v| = |∇E|/ħ (m/s)，单侧差分。

    适用于 Dirac 锥顶点（如石墨烯 K 点）：在 k0 处能量过零、
    两侧对称，取 ``E(k0 + dk·d̂)/dk`` 的斜率。
    """
    d = np.asarray(direction, dtype=float)
    d = d / np.linalg.norm(d)
    k0 = np.asarray(k0, dtype=float)
    e = model.energies_at(k0 + dk * d)[band]
    slope_frac = e / dk
    recip = model.lattice.reciprocal_matrix
    slope_cart = slope_frac / np.linalg.norm(d @ recip[:2, :2])
    return abs(slope_cart) * SLOPE_TO_VELOCITY
