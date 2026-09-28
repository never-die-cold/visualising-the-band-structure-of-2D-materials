"""二维布里渊区与 k 路径绘图。

布里渊区 = 倒格子的 Wigner-Seitz 原胞，用半平面裁剪法构造：
对每个倒格矢 G，保留满足 ``k·G ≤ |G|²/2`` 的半平面，将一个大正方形
逐次裁剪成凸多边形（Sutherland–Hodgman 思路，稳健且无需 Voronoi 库）。
"""

from __future__ import annotations

from typing import List, Optional, Tuple

import numpy as np
from matplotlib import pyplot as plt
from matplotlib.patches import Polygon as MplPolygon

from ..engine.kpath import DISPLAY_SYMBOLS, KPath
from ..structure.lattice import Lattice


def brillouin_zone_polygon(lattice: Lattice) -> np.ndarray:
    """计算二维布里渊区多边形顶点（逆时针，笛卡尔坐标 Å⁻¹）。"""
    B = lattice.reciprocal_matrix[:2, :2]
    # 足够大的初始正方形
    gmax = max(np.linalg.norm(B[0]), np.linalg.norm(B[1]))
    half = 2.5 * gmax
    poly = [np.array([-half, -half]), np.array([half, -half]),
            np.array([half, half]), np.array([-half, half])]

    def clip(poly: List[np.ndarray], G: np.ndarray) -> List[np.ndarray]:
        out: List[np.ndarray] = []
        c = np.dot(G, G) / 2
        vals = [np.dot(p, G) - c for p in poly]
        n = len(poly)
        for i in range(n):
            j = (i + 1) % n
            pi, pj = poly[i], poly[j]
            vi, vj = vals[i], vals[j]
            if vi <= 0:
                out.append(pi)
            if (vi < 0 < vj) or (vi > 0 > vj):
                t = vi / (vi - vj)
                out.append(pi + t * (pj - pi))
        return out

    for n1 in range(-2, 3):
        for n2 in range(-2, 3):
            if n1 == 0 and n2 == 0:
                continue
            G = n1 * B[0] + n2 * B[1]
            poly = clip(poly, G)
            if not poly:
                raise RuntimeError("布里渊区裁剪失败")

    # 清理：去除重复顶点与共线顶点（裁剪可能在角点留下冗余点）
    cleaned: List[np.ndarray] = []
    for p in poly:
        if not cleaned or np.linalg.norm(p - cleaned[-1]) > 1e-9:
            cleaned.append(p)
    if len(cleaned) > 1 and np.linalg.norm(cleaned[0] - cleaned[-1]) < 1e-9:
        cleaned.pop()
    result: List[np.ndarray] = []
    n = len(cleaned)
    for i in range(n):
        p_prev, p, p_next = cleaned[i - 1], cleaned[i], cleaned[(i + 1) % n]
        v1, v2 = p - p_prev, p_next - p
        cross = v1[0] * v2[1] - v1[1] * v2[0]
        if abs(cross) > 1e-9:            # 非共线才保留
            result.append(p)
    return np.array(result)


def polygon_area(poly: np.ndarray) -> float:
    """鞋带公式计算多边形面积。"""
    x, y = poly[:, 0], poly[:, 1]
    return 0.5 * abs(np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1)))


def plot_bz_path(lattice: Lattice,
                 kpath: Optional[KPath] = None,
                 ax=None,
                 show_path: bool = True,
                 title: Optional[str] = None,
                 ) -> Tuple[plt.Figure, plt.Axes]:
    """绘制布里渊区、高对称点与 k 路径。"""
    created = False
    if ax is None:
        fig, ax = plt.subplots(figsize=(4.6, 4.4))
        created = True
    fig = ax.figure

    poly = brillouin_zone_polygon(lattice)
    ax.add_patch(MplPolygon(poly, closed=True, facecolor="#eef3fa",
                            edgecolor="#1f4e79", lw=1.6, zorder=1))

    B = lattice.reciprocal_matrix[:2, :2]
    if kpath is not None:
        # 高对称点
        for name, pt in kpath.points.items():
            kc = np.array(pt) @ B
            ax.scatter(*kc, s=42, color="#c0392b", zorder=4)
            ax.annotate(DISPLAY_SYMBOLS.get(name, name), kc + [0.06, 0.06],
                        fontsize=12, fontweight="bold", color="#7b241c",
                        ha="center", zorder=5)
        # 路径折线（分段画以便区分）
        if show_path:
            for (p0, p1) in kpath.segments():
                a = np.array(p0) @ B
                b = np.array(p1) @ B
                ax.plot([a[0], b[0]], [a[1], b[1]], color="#c0392b",
                        lw=2.2, alpha=0.85, zorder=3,
                        solid_capstyle="round")
    ax.set_aspect("equal")
    ax.set_axis_off()
    ax.set_title(title or "Brillouin zone & k-path", fontsize=11)
    pad = 0.35 * max(np.linalg.norm(B[0]), np.linalg.norm(B[1]))
    lim = float(np.abs(poly).max() + pad)
    ax.set_xlim(-lim, lim)
    ax.set_ylim(-lim, lim)
    if created:
        fig.tight_layout()
    return fig, ax
