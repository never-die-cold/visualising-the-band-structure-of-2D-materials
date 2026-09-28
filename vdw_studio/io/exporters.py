"""结构文件导出：VASP POSCAR 与 XYZ。

导出的 POSCAR 可直接用于 VASP / Quantum ESPRESSO（经转换）等
第一性原理软件，使 vdW Studio 的建模结果能无缝接入 DFT 工作流；
XYZ 为通用交换格式。
"""

from __future__ import annotations

from collections import Counter
from typing import Optional

import numpy as np

from ..structure.crystal import Crystal


def poscar_string(crystal: Crystal, title: Optional[str] = None,
                  direct: bool = True) -> str:
    """生成 VASP POSCAR（VASP 5 格式：含元素符号行）字符串。

    参数:
        crystal: 结构。
        title: 首行注释，默认使用化学式。
        direct: True 输出分数坐标 (Direct)，False 输出笛卡尔坐标。
    """
    lines = [title or crystal.formula_str or "structure"]
    lines.append("1.0")
    for v in crystal.lattice.matrix:
        lines.append(f"  {v[0]:>18.12f}  {v[1]:>18.12f}  {v[2]:>18.12f}")
    counts = Counter(crystal.symbols)
    # 保持出现顺序，避免打乱用户习惯
    species = list(dict.fromkeys(crystal.symbols))
    lines.append("  " + "  ".join(f"{s:>4}" for s in species))
    lines.append("  " + "  ".join(f"{counts[s]:>4}" for s in species))
    lines.append("Direct" if direct else "Cartesian")
    for sym, f in zip(crystal.symbols, crystal.frac_coords):
        if direct:
            lines.append(f"  {f[0]:>18.12f}  {f[1]:>18.12f}  {f[2]:>18.12f}")
        else:
            cart = crystal.lattice.frac_to_cart(f)
            lines.append(f"  {cart[0]:>18.12f}  {cart[1]:>18.12f}  {cart[2]:>18.12f}")
    return "\n".join(lines) + "\n"


def write_poscar(crystal: Crystal, path: str, **kwargs) -> None:
    """把结构写成 POSCAR 文件。"""
    with open(path, "w", encoding="utf-8", newline="\n") as fh:
        fh.write(poscar_string(crystal, **kwargs))


def xyz_string(crystal: Crystal, title: Optional[str] = None) -> str:
    """生成 XYZ 字符串（笛卡尔坐标, Å）。真空层方向原样导出。"""
    cart = crystal.cart_coords
    lines = [str(crystal.n_atoms),
             title or f"{crystal.formula_str} (vdW Studio export)"]
    for sym, r in zip(crystal.symbols, cart):
        lines.append(f"{sym:<3} {r[0]:>16.8f} {r[1]:>16.8f} {r[2]:>16.8f}")
    return "\n".join(lines) + "\n"


def write_xyz(crystal: Crystal, path: str, **kwargs) -> None:
    """把结构写成 XYZ 文件。"""
    with open(path, "w", encoding="utf-8", newline="\n") as fh:
        fh.write(xyz_string(crystal, **kwargs))
