"""常见二维材料结构构建器。

所有默认结构参数均取自文献实验值/标准 DFT 值（出处见
``docs/REFERENCES.md`` 与各函数 docstring），保证"建模 → 仿真"全流程
的参数可追溯。

包含的二维材料家族：

- 六方蜂窝家族：石墨烯、h-BN、硅烯（低翘曲）
- 过渡金属硫族化合物 1H 相：MoS₂ / MoSe₂ / WS₂ / WSe₂ / MoTe₂ / WTe₂
- 黑磷烯（褶皱正交结构，4 原子/胞）
"""

from __future__ import annotations

from typing import Optional

import numpy as np

from .crystal import Crystal
from .lattice import Lattice

# ----------------------------------------------------------------------
# 六方蜂窝家族
# ----------------------------------------------------------------------
# 石墨烯晶格常数（实验值, a = 2.46 Å, C-C 键长 1.42 Å）
GRAPHENE_A = 2.46
# h-BN 晶格常数（实验值, a = 2.504 Å, B-N 键长 1.446 Å）
HBN_A = 2.504
# 硅烯（低翘曲相）：a = 3.86 Å, buckling Δ = 0.44 Å
# (标准 DFT 值, 键长 sqrt((a/√3)² + Δ²) ≈ 2.27 Å)
SILICENE_A = 3.86
SILICENE_BUCKLING = 0.44


def _honeycomb(symbol_a: str, symbol_b: str, a: float,
               dz_a: float = 0.0, dz_b: float = 0.0,
               vacuum: float = 15.0) -> Crystal:
    """蜂窝双原子基元（可用于石墨烯/hBN/硅烯等）。

    A 位放在原点，B 位放在 (2/3, 1/3)（等价地 (1/3, 2/3)，取决于约定），
    键长 = a/√3（平面情形）。
    """
    lat = Lattice.hexagonal(a, vacuum=vacuum)
    c = Crystal(lat)
    c.add_atom(symbol_a, (0.0, 0.0, 0.5 + dz_a))
    c.add_atom(symbol_b, (2 / 3, 1 / 3, 0.5 + dz_b))
    return c


def graphene(a: float = GRAPHENE_A, vacuum: float = 15.0) -> Crystal:
    """单层石墨烯（2 原子/胞）。

    键长 = a/√3 = 1.420 Å (a = 2.46 Å)。
    """
    return _honeycomb("C", "C", a, vacuum=vacuum)


def hbn(a: float = HBN_A, vacuum: float = 15.0) -> Crystal:
    """单层六方氮化硼 h-BN（2 原子/胞，B 在 A 位、N 在 B 位）。"""
    return _honeycomb("B", "N", a, vacuum=vacuum)


def silicene(a: float = SILICENE_A, buckling: float = SILICENE_BUCKLING,
             vacuum: float = 15.0) -> Crystal:
    """低翘曲（low-buckled）硅烯（2 原子/胞）。

    两个子格在面外方向错开 ``buckling``（默认 0.44 Å），
    为电场开启带隙（Kane-Mele 型交错势）提供结构基础。
    """
    return _honeycomb("Si", "Si", a,
                      dz_a=+buckling / 2 / vacuum,
                      dz_b=-buckling / 2 / vacuum,
                      vacuum=vacuum)


# ----------------------------------------------------------------------
# 过渡金属硫族化合物 (1H 相, P-6m2)
# ----------------------------------------------------------------------
# 晶格常数 a0 (Å) 与面外 X-X 距离 dXX (Å)：实验值优先，
# 取自 Kormányos et al., 2D Mater. 2, 022001 (2015) Table 1；
# WTe2 实验值缺失，使用其 HSE 计算值。
TMD_STRUCTURAL_PARAMS = {
    "MoS2":  {"metal": "Mo", "chalcogen": "S", "a": 3.1604, "dxx": 3.170},
    "MoSe2": {"metal": "Mo", "chalcogen": "Se", "a": 3.288, "dxx": 3.335},
    "WS2":   {"metal": "W", "chalcogen": "S", "a": 3.154, "dxx": 3.140},
    "WSe2":  {"metal": "W", "chalcogen": "Se", "a": 3.286, "dxx": 3.340},
    "MoTe2": {"metal": "Mo", "chalcogen": "Te", "a": 3.519, "dxx": 3.604},
    "WTe2":  {"metal": "W", "chalcogen": "Te", "a": 3.521, "dxx": 3.5999},
}


def tmd(name: str = "MoS2", a: Optional[float] = None,
        dxx: Optional[float] = None, vacuum: float = 15.0) -> Crystal:
    """单层过渡金属硫族化合物 MX₂（1H 相，P-6m2，3 原子/胞）。

    结构：金属 M 位于 (0,0)，两个硫族 X 位于 (1/3, 2/3) 与 (2/3, 1/3)、
    面外高度 ±dxx/2（相对金属原子平面）。M-X 键长 = sqrt(a²/3 + dxx²/4)。

    参数:
        name: 材料名（"MoS2" / "MoSe2" / "WS2" / "WSe2" / "MoTe2" / "WTe2"）。
        a: 面内晶格常数 (Å)，默认取文献实验值。
        dxx: 面外 X-X 距离 (Å)，默认取文献值。
        vacuum: 真空层厚度 (Å)。
    """
    key = name.replace("₂", "2").strip()
    if key not in TMD_STRUCTURAL_PARAMS:
        raise ValueError(f"未知 TMD 材料 {name!r}，可选: {list(TMD_STRUCTURAL_PARAMS)}")
    p = TMD_STRUCTURAL_PARAMS[key]
    a = p["a"] if a is None else a
    dxx = p["dxx"] if dxx is None else dxx
    lat = Lattice.hexagonal(a, vacuum=vacuum)
    c = Crystal(lat)
    half = dxx / 2
    c.add_atom(p["metal"], (0.0, 0.0, 0.5))
    c.add_atom(p["chalcogen"], (1 / 3, 2 / 3, 0.5 + half / vacuum))
    c.add_atom(p["chalcogen"], (2 / 3, 1 / 3, 0.5 - half / vacuum))
    return c


# ----------------------------------------------------------------------
# 黑磷烯 (monolayer black phosphorus)
# ----------------------------------------------------------------------
# 结构：Brown & Rundqvist 实验体相结构投影到单层
# (a = 3.3136 Å 链方向, c = 4.3763 Å 褶皱方向, 4 原子/胞)。
# 与 Rudenko & Katsnelson PRB 89, 201408(R) (2014) Table 1 的
# 5 个 hopping 距离/配位数完全自洽（见 docs/REFERENCES.md）。
PHOSPHORENE_A = 3.3136     # 链 (zigzag) 方向晶格常数 (Å)
PHOSPHORENE_C = 4.3763     # 褶皱方向晶格常数 (Å)
PHOSPHORENE_H = 1.0654     # 两个 P 亚层到中面的高度 (Å)


def phosphorene(a: float = PHOSPHORENE_A, c: float = PHOSPHORENE_C,
                height: float = PHOSPHORENE_H,
                vacuum: float = 15.0) -> Crystal:
    """单层黑磷（黑磷烯，4 原子/胞，正交褶皱结构）。

    坐标约定：x = 链方向（zigzag，a = 3.3136 Å），
    y = 褶皱方向（c = 4.3763 Å），z = 面外（真空层）。
    四个原子分属上下两个 P 亚层（±height），上亚层原子在
    (x, y) = (0, +0.3525) 与 (a/2, +1.8356)，下亚层关于中面反演对称。

    该几何自动给出文献键长 d₁ = 2.224 Å（链内）、d₂ = 2.244 Å（跨亚层）
    与键角 θ₁ = 96.3°、θ₂ = 102.1°（实验体相值）。
    """
    lat = Lattice.rectangular(a, c, vacuum=vacuum)
    c_ = Crystal(lat)
    # 分数坐标 (x, y) 与面外高度 (Å)
    sites = [
        (0.0, +0.3525 / c, +height),
        (0.5, +1.8356 / c, +height),
        (0.0, -0.3525 / c, -height),
        (0.5, -1.8356 / c, -height),
    ]
    for fx, fy, h in sites:
        c_.add_atom("P", (fx, fy, 0.5 + h / vacuum))
    return c_


# ----------------------------------------------------------------------
# 注册表：供 GUI/CLI 按名称构建材料
# ----------------------------------------------------------------------
BUILDERS = {
    "graphene": graphene,
    "hbn": hbn,
    "silicene": silicene,
    "phosphorene": phosphorene,
    **{k: (lambda n=k: tmd(n)) for k in TMD_STRUCTURAL_PARAMS},
}


def build(name: str, **kwargs) -> Crystal:
    """按名称构建材料结构。``build("MoS2")``、``build("graphene")`` 等。

    材料名大小写不敏感（``"mos2"`` 等价于 ``"MoS2"``）。
    """
    key = name.strip()
    lookup = {k.lower(): k for k in BUILDERS}
    if key.lower() not in lookup:
        raise ValueError(f"未知材料 {name!r}，可选: {sorted(BUILDERS)}")
    return BUILDERS[lookup[key.lower()]](**kwargs)
