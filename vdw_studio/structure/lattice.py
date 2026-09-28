"""晶格（Lattice）与倒空间表示。

二维材料的仿真在"真空层 slab"模型下进行：面内两支晶格矢量张成周期方向，
第三支沿 z（面外）提供真空层，使周期性求解器（紧束缚、k·p 或 DFT）可以
按三维周期体系处理二维晶体。

约定：
- 晶格矩阵 ``A`` 为 3×3，**行向量**为晶格基矢：``A[0] = a1``，``A[1] = a2``，
  ``A[2] = a3``；
- 倒格基矢按晶体学约定 ``B = 2π (A^{-1})^T``（同样行向量）；
- 分数坐标 ``f`` 与笛卡尔坐标 ``r`` 的关系为 ``r = f @ A``。
"""

from __future__ import annotations

import math
from typing import Optional

import numpy as np

# pi 的短别名，公式更接近教材写法
PI = math.pi


class Lattice:
    """三维晶格（二维材料使用 a3 方向的真空层）。"""

    def __init__(self, matrix) -> None:
        """参数:
            matrix: 3×3 晶格矩阵，行向量为 a1, a2, a3（单位 Å）。
        """
        m = np.asarray(matrix, dtype=float)
        if m.shape != (3, 3):
            raise ValueError(f"晶格矩阵必须为 3×3, 收到 shape={m.shape}")
        det = np.linalg.det(m)
        if abs(det) < 1e-8:
            raise ValueError("晶格矩阵奇异（体积为零），请检查晶格基矢")
        self.matrix = m

    # ------------------------------------------------------------------
    # 常用构造
    # ------------------------------------------------------------------
    @classmethod
    def from_parameters(
        cls,
        a: float,
        b: float,
        c: float,
        alpha: float = 90.0,
        beta: float = 90.0,
        gamma: float = 90.0,
    ) -> "Lattice":
        """由晶格常数 (Å) 与夹角 (°) 构造（晶体学标准取向：a 沿 x）。"""
        if min(a, b, c) <= 0:
            raise ValueError("晶格常数必须为正")
        al, be, ga = (math.radians(x) for x in (alpha, beta, gamma))
        if al + be + ga >= 2 * PI or min(al, be, ga) <= 0:
            raise ValueError("晶格夹角组合无物理意义")

        va = np.array([a, 0.0, 0.0])
        vb = np.array([b * math.cos(ga), b * math.sin(ga), 0.0])
        cx = c * math.cos(be)
        cy = c * (math.cos(al) - math.cos(be) * math.cos(ga)) / math.sin(ga)
        cz2 = c**2 - cx**2 - cy**2
        if cz2 <= 0:
            raise ValueError("晶格夹角组合无法构成右手系三维晶格")
        vc = np.array([cx, cy, math.sqrt(cz2)])
        return cls(np.vstack([va, vb, vc]))

    @classmethod
    def hexagonal(cls, a: float, vacuum: float = 15.0) -> "Lattice":
        """六方晶格（2H/1H 二维材料的面内格子），a3 沿 z 提供真空层。"""
        return cls.from_parameters(a, a, vacuum, 90, 90, 120)

    @classmethod
    def square(cls, a: float, vacuum: float = 15.0) -> "Lattice":
        """正方晶格。"""
        return cls.from_parameters(a, a, vacuum, 90, 90, 90)

    @classmethod
    def rectangular(cls, a: float, b: float, vacuum: float = 15.0) -> "Lattice":
        """正交（长方）晶格。"""
        return cls.from_parameters(a, b, vacuum, 90, 90, 90)

    # ------------------------------------------------------------------
    # 基本量
    # ------------------------------------------------------------------
    @property
    def a1(self) -> np.ndarray:
        return self.matrix[0]

    @property
    def a2(self) -> np.ndarray:
        return self.matrix[1]

    @property
    def a3(self) -> np.ndarray:
        return self.matrix[2]

    @property
    def volume(self) -> float:
        """胞体积 (Å³)。二维材料下为 面积 × 真空层厚度。"""
        return float(abs(np.linalg.det(self.matrix)))

    @property
    def area_2d(self) -> float:
        """面内原胞面积 |a1 × a2| (Å²)。"""
        return float(np.linalg.norm(np.cross(self.a1, self.a2)))

    def parameters(self) -> tuple:
        """返回 (a, b, c, alpha, beta, gamma)，长度 Å、角度 °。"""
        la, lb, lc = (np.linalg.norm(v) for v in self.matrix)
        alpha = math.degrees(math.acos(
            np.dot(self.a2, self.a3) / (lb * lc)))
        beta = math.degrees(math.acos(
            np.dot(self.a1, self.a3) / (la * lc)))
        gamma = math.degrees(math.acos(
            np.dot(self.a1, self.a2) / (la * lb)))
        return la, lb, lc, alpha, beta, gamma

    # ------------------------------------------------------------------
    # 倒空间
    # ------------------------------------------------------------------
    @property
    def reciprocal_matrix(self) -> np.ndarray:
        """倒格矩阵 B = 2π (A⁻¹)ᵀ，行向量为 b1, b2, b3 (Å⁻¹)。"""
        return 2 * PI * np.linalg.inv(self.matrix).T

    def reciprocal(self) -> "Lattice":
        """返回以倒格矢为基矢的 :class:`Lattice`（便于复用几何方法）。"""
        return Lattice(self.reciprocal_matrix)

    # ------------------------------------------------------------------
    # 坐标变换
    # ------------------------------------------------------------------
    def frac_to_cart(self, frac) -> np.ndarray:
        """分数坐标 → 笛卡尔坐标。输入 (…, 3)。"""
        f = np.asarray(frac, dtype=float)
        return f @ self.matrix

    def cart_to_frac(self, cart) -> np.ndarray:
        """笛卡尔坐标 → 分数坐标。输入 (…, 3)。"""
        r = np.asarray(cart, dtype=float)
        return r @ np.linalg.inv(self.matrix)

    def d_hkl_plane_spacing(self) -> float:
        """a3 方向的重复周期（二维材料即 slab + 真空层总厚度）。"""
        return float(np.linalg.norm(self.a3))

    # ------------------------------------------------------------------
    # 应变（二维材料仿真常用调控手段）
    # ------------------------------------------------------------------
    def scaled_xy(self, sx: float = 1.0, sy: float = 1.0) -> "Lattice":
        """面内双轴/单轴应变后的新晶格（不缩放真空层方向）。"""
        m = self.matrix.copy()
        m[0, :2] *= sx
        m[1, :2] *= sy
        return Lattice(m)

    def is_hexagonal(self, tol: float = 1e-4) -> bool:
        """是否为 120° 六方面内格子（|a1| = |a2|, γ = 120°）。"""
        la, lb, _, _, _, gamma = self.parameters()
        return abs(la - lb) < tol and abs(gamma - 120.0) < tol

    # ------------------------------------------------------------------
    def __repr__(self) -> str:
        a, b, c, al, be, ga = self.parameters()
        return (f"Lattice(a={a:.4f}, b={b:.4f}, c={c:.4f} Å, "
                f"α={al:.2f}°, β={be:.2f}°, γ={ga:.2f}°)")

    def __eq__(self, other) -> bool:
        if not isinstance(other, Lattice):
            return NotImplemented
        return bool(np.allclose(self.matrix, other.matrix, atol=1e-8))
