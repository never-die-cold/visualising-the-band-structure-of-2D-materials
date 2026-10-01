"""布里渊区高对称点与 k 路径生成。

约定：高对称点用**倒格子分数坐标** (k1, k2) 表示，笛卡尔波矢
k_cart = (k1, k2, 0) @ B，其中 B = 2π(A⁻¹)ᵀ 为倒格矩阵（见
``structure.lattice.Lattice.reciprocal_matrix``）。

内置高对称点集：
- 六方 (hexagonal)：Γ=(0,0), M=(1/2,0), K=(1/3,1/3)，标准路径 Γ-M-K-Γ；
- 正交 (rectangular)：Γ=(0,0), X=(1/2,0), Y=(0,1/2), S=(1/2,1/2)，
  标准路径 Γ-X-S-Y-Γ；
- 正方 (square)：Γ, X, M，路径 Γ-X-M-Γ。
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np

from ..structure.lattice import Lattice

# 倒格子分数坐标 (k1, k2)
HIGH_SYMMETRY_POINTS: Dict[str, Dict[str, Tuple[float, float]]] = {
    "hexagonal": {"G": (0.0, 0.0), "M": (0.5, 0.0), "K": (1 / 3, 1 / 3)},
    "rectangular": {"G": (0.0, 0.0), "X": (0.5, 0.0), "Y": (0.0, 0.5),
                    "S": (0.5, 0.5)},
    "square": {"G": (0.0, 0.0), "X": (0.5, 0.0), "M": (0.5, 0.5)},
}

DEFAULT_PATHS: Dict[str, List[str]] = {
    "hexagonal": ["G", "M", "K", "G"],
    "rectangular": ["G", "X", "S", "Y", "G"],
    "square": ["G", "X", "M", "G"],
}

# 绘图用显示符号（Γ 需要特殊字符）
DISPLAY_SYMBOLS = {"G": "Γ", "M": "M", "K": "K", "X": "X", "Y": "Y", "S": "S"}


def lattice_type(lattice: Lattice, tol_rel: float = 1e-4,
                 tol_deg: float = 1e-3) -> str:
    """按面内对称性判定格子类型（hexagonal / square / rectangular）。

    参数:
        tol_rel: 晶格长度相对容差（|a−b|/a）。
        tol_deg: 角度容差 (°)。
    """
    a, b, _, _, _, gamma = lattice.parameters()
    if (abs(a - b) <= tol_rel * a
            and abs(abs(gamma) - 120.0) <= tol_deg):
        return "hexagonal"
    if (abs(a - b) <= tol_rel * a
            and abs(gamma - 90.0) <= tol_deg):
        return "square"
    return "rectangular"


def high_symmetry_points(lattice: Lattice) -> Dict[str, Tuple[float, float]]:
    """返回给定晶格的高对称点（倒格子分数坐标）。"""
    return HIGH_SYMMETRY_POINTS[lattice_type(lattice)]


@dataclass
class KPath:
    """一条 k 路径：高对称点序列 + 采样。

    属性:
        points: 名称 -> 倒格子分数坐标 (k1, k2)
        path: 高对称点名称序列，如 ["G", "M", "K", "G"]
    """

    points: Dict[str, Tuple[float, float]]
    path: List[str]

    # ------------------------------------------------------------------
    @classmethod
    def for_lattice(cls, lattice: Lattice,
                    path: Optional[List[str]] = None) -> "KPath":
        """按晶格类型自动选择高对称点集与默认路径。

        路径选择的对称性判定使用放宽容差（长度 6%、角度 3°），
        使轻微应变的六方格子仍按 Γ-M-K-Γ 处理（文献惯例）。
        """
        ltype = lattice_type(lattice, tol_rel=0.06, tol_deg=3.0)
        pts = HIGH_SYMMETRY_POINTS[ltype]
        path = list(path) if path is not None else list(DEFAULT_PATHS[ltype])
        for name in path:
            if name not in pts:
                raise ValueError(f"路径点 {name!r} 不在 {ltype} 格子"
                                 f"的高对称点集 {sorted(pts)} 中")
        return cls(points=pts, path=path)

    # ------------------------------------------------------------------
    def segments(self) -> List[Tuple[Tuple[float, float], Tuple[float, float]]]:
        """路径切分为相邻高对称点对。"""
        return [(self.points[self.path[i]], self.points[self.path[i + 1]])
                for i in range(len(self.path) - 1)]

    def generate(self, lattice: Lattice, n_per_segment: int = 60
                 ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """沿路径采样 k 点。

        参数:
            lattice: 实空间晶格（用于把分数坐标转成笛卡尔 Å⁻¹）。
            n_per_segment: 每段高对称点间的采样数（不含终点）。

        返回:
            (kpoints, x_axis, ticks)
            kpoints: (N, 3) 笛卡尔 k 点 (Å⁻¹)，z 分量为 0；
            x_axis: (N,) 累积路径长度（绘图横轴，单位 Å⁻¹）；
            ticks: 高对称点在 x_axis 上的位置。
        """
        if n_per_segment < 2:
            raise ValueError("每段至少需要 2 个采样点")
        recip = lattice.reciprocal_matrix
        kfrac_list: List[np.ndarray] = []
        for (p0, p1) in self.segments():
            for t in np.linspace(0.0, 1.0, n_per_segment, endpoint=False):
                kfrac_list.append(np.array([p0[0] + t * (p1[0] - p0[0]),
                                            p0[1] + t * (p1[1] - p0[1])]))
        # 补上最后一个点
        last = self.points[self.path[-1]]
        kfrac_list.append(np.array(last))
        kfrac = np.array(kfrac_list)

        kcart = np.c_[kfrac @ recip[:2, :2], np.zeros(len(kfrac))]
        # 累积长度（段间断开处长度照常累计，绘图用 ticks 定位即可）
        seg = np.linalg.norm(np.diff(kcart[:, :2], axis=0), axis=1)
        x_axis = np.concatenate([[0.0], np.cumsum(seg)])
        ticks = [0.0]
        for i in range(len(self.path) - 1):
            start = i * n_per_segment
            end = start + n_per_segment
            ticks.append(x_axis[end])
        return kcart, x_axis, np.array(ticks)
