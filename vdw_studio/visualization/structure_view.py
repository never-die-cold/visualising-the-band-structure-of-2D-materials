"""3D 球棍模型结构可视化（matplotlib）。

绘制约定（VESTA 风格）：
- 原子 = 球（半径 ∝ 共价半径 × ``atom_scale``，颜色 = Jmol/CPK）；
- 化学键 = 线段（共价半径判据，容差 ``bond_tol``，见
  ``structure.Crystal.bonds``）；
- 晶胞 = 蓝灰色棱线；默认显示 (3×3×1) 超胞以直观展示二维延展性。

所有绘图函数返回 ``(fig, ax)``，便于脚本保存或嵌入 PyQt 界面。
"""

from __future__ import annotations

from typing import Optional, Tuple

import numpy as np
from matplotlib import pyplot as plt
from matplotlib.figure import Figure

from ..elements import get_element
from ..structure.crystal import Crystal

_DEFAULT_CELL_COLOR = "#7a8ba3"


def plot_structure(crystal: Crystal,
                   ax=None,
                   supercell: Tuple[int, int, int] = (3, 3, 1),
                   atom_scale: float = 0.35,
                   bond_tol: float = 1.15,
                   max_atoms: int = 400,
                   elev: float = 24.0,
                   azim: float = -58.0,
                   title: Optional[str] = None) -> Tuple[Figure, plt.Axes]:
    """绘制二维材料的 3D 球棍模型。

    参数:
        crystal: 结构（一般为原胞；超胞在函数内部展开）。
        ax: 可选的已有 3D 坐标轴。
        supercell: 展示用超胞倍数。
        atom_scale: 原子球半径缩放（× 共价半径，单位 Å）。
        bond_tol: 成键判据容差（× 共价半径之和）。
        max_atoms: 超胞原子数上限（防止超大展示卡顿）。
    """
    from mpl_toolkits.mplot3d.art3d import Line3D

    sc = crystal.supercell(supercell)
    n = sc.n_atoms
    if n > max_atoms:
        raise ValueError(f"展示超胞含 {n} 个原子 > 上限 {max_atoms}，"
                         "请减小 supercell")

    created_fig = False
    if ax is None:
        fig = plt.figure(figsize=(7, 6))
        ax = fig.add_subplot(111, projection="3d")
        created_fig = True
    fig = ax.figure

    cart = sc.cart_coords
    symbols = sc.symbols

    # --- 键（先画，压在球下面）---
    for (i, j, image, dist) in sc.bonds(tol_factor=bond_tol):
        p = cart[i]
        q = cart[j] + image @ sc.lattice.matrix
        line = Line3D(*zip(p, q), color="#3a3f44", lw=1.6, alpha=0.85,
                      solid_capstyle="round")
        ax.add_line(line)

    # --- 原子球 ---
    for sym in sorted(set(symbols)):
        el = get_element(sym)
        mask = np.array([s == sym for s in symbols])
        pts = cart[mask]
        sizes = np.full(len(pts), el.covalent_radius * atom_scale * 2)
        ax.scatter(pts[:, 0], pts[:, 1], pts[:, 2], s=260 * sizes ** 2,
                   c=[el.color], edgecolors="#22262a", linewidths=0.6,
                   depthshade=True, label=f"{sym}", zorder=3)

    # --- 晶胞棱线（二维材料：z 范围裁剪到原子分布附近，避免真空层
    #     把盒子拉得过高、原子被压扁）---
    m = sc.lattice.matrix
    zlo = float(cart[:, 2].min() - 0.9)
    zhi = float(cart[:, 2].max() + 0.9)
    if (zhi - zlo) >= 0.9 * abs(m[2, 2]):
        zlo, zhi = 0.0, float(abs(m[2, 2]))   # 三维体材料则画全胞
    ij_list = [(0, 0), (1, 0), (0, 1), (1, 1)]
    c_bottom = {ij: np.array([ij[0] * m[0, 0] + ij[1] * m[1, 0],
                              ij[0] * m[0, 1] + ij[1] * m[1, 1], zlo])
                for ij in ij_list}
    c_top = {ij: c_bottom[ij] + np.array([0, 0, zhi - zlo]) for ij in ij_list}
    for a, b in (((0, 0), (1, 0)), ((0, 0), (0, 1)),
                 ((1, 0), (1, 1)), ((0, 1), (1, 1))):
        ax.plot(*zip(c_bottom[a], c_bottom[b]),
                color=_DEFAULT_CELL_COLOR, lw=1.2, alpha=0.9)
        ax.plot(*zip(c_top[a], c_top[b]),
                color=_DEFAULT_CELL_COLOR, lw=1.2, alpha=0.9)
    for ij in ij_list:
        ax.plot(*zip(c_bottom[ij], c_top[ij]),
                color=_DEFAULT_CELL_COLOR, lw=1.2, alpha=0.9)

    ax.set_box_aspect((np.ptp(cart[:, 0]) or 1.0,
                       np.ptp(cart[:, 1]) or 1.0,
                       max(np.ptp(cart[:, 2]), 0.25 * max(np.ptp(cart[:, 0]), 1.0))))
    ax.set_axis_off()
    ax.view_init(elev=elev, azim=azim)
    if title:
        from .fonts import plot_text
        ax.set_title(plot_text(title, f'Structure — {crystal.formula_str}'))
    if created_fig:
        ax.legend(loc="upper right", frameon=False, fontsize=9)
    return fig, ax
