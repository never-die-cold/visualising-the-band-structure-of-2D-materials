"""晶体结构（Crystal）：原子位点 + 周期性近邻搜索 + 超胞。

本模块是结构建模的核心数据结构，设计参考 ASE/pymatgen 的
``Structure`` 概念但做了面向二维材料的精简：

- 原子坐标统一使用**分数坐标**存储，便于施加周期性操作；
- 近邻搜索通过周期镜像枚举实现（二维原胞原子数少，暴力 27 镜像
  搜索已足够快；超胞后晶格更大，仍适用）；
- 成键判定采用共价半径之和 × 容差系数的经验规则（VESTA 同款思路）。
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field
from itertools import product
from typing import Iterable, List, Optional, Sequence, Tuple

import numpy as np

from ..elements import get_element
from .lattice import Lattice


@dataclass
class Atom:
    """一个原子位点（分数坐标存储）。"""

    symbol: str                      # 元素符号
    frac: np.ndarray                 # 分数坐标 (3,)
    label: Optional[str] = None      # 可选位点标签（如 "Mo1"）

    def __post_init__(self) -> None:
        get_element(self.symbol)  # 校验元素符号
        self.frac = np.asarray(self.frac, dtype=float).reshape(3)

    def __repr__(self) -> str:
        f = np.array2string(self.frac, precision=4, suppress_small=True)
        return f"Atom({self.symbol}, frac={f})"


@dataclass
class Neighbor:
    """周期性近邻记录。"""

    index: int            # 原胞内原子序号
    symbol: str           # 元素符号
    distance: float       # 键长 (Å)
    image: np.ndarray     # 镜像偏移（整数晶格矢系数）


class Crystal:
    """周期性晶体结构（二维材料 = slab + 真空层）。"""

    def __init__(self, lattice: Lattice, atoms: Iterable[Atom] = ()) -> None:
        self.lattice = lattice
        self.atoms: List[Atom] = list(atoms)

    # ------------------------------------------------------------------
    # 构建原子位点
    # ------------------------------------------------------------------
    def add_atom(self, symbol: str, frac: Sequence[float], label: str = None) -> Atom:
        """按分数坐标添加原子，返回新建的 :class:`Atom`。"""
        atom = Atom(symbol, np.asarray(frac, dtype=float), label)
        self.atoms.append(atom)
        return atom

    def add_atoms(self, symbols, fracs) -> None:
        """批量添加原子。"""
        for s, f in zip(symbols, fracs):
            self.add_atom(s, f)

    # ------------------------------------------------------------------
    # 基本性质
    # ------------------------------------------------------------------
    @property
    def n_atoms(self) -> int:
        return len(self.atoms)

    @property
    def symbols(self) -> List[str]:
        return [a.symbol for a in self.atoms]

    @property
    def frac_coords(self) -> np.ndarray:
        """(n, 3) 分数坐标数组。"""
        if not self.atoms:
            return np.zeros((0, 3))
        return np.array([a.frac for a in self.atoms])

    @property
    def cart_coords(self) -> np.ndarray:
        """(n, 3) 笛卡尔坐标数组 (Å)。"""
        return self.lattice.frac_to_cart(self.frac_coords)

    @property
    def formula(self) -> Counter:
        """化学式计数字典，如 ``Counter({'Mo': 1, 'S': 2})``。"""
        return Counter(self.symbols)

    @property
    def formula_str(self) -> str:
        """Hill 记法化学式字符串（C 在前、H 其次、其余按字母序）。"""
        counts = self.formula
        if not counts:
            return ""
        keys = sorted(counts)
        ordered = [k for k in ("C", "H") if k in counts] + \
                  [k for k in keys if k not in ("C", "H")]
        return "".join(f"{k}{counts[k] if counts[k] > 1 else ''}" for k in ordered)

    @property
    def species_set(self) -> set:
        """结构中出现的元素集合。"""
        return set(self.symbols)

    # ------------------------------------------------------------------
    # 周期性近邻
    # ------------------------------------------------------------------
    def neighbors(self, center_frac, cutoff: float) -> List[Neighbor]:
        """以 ``center_frac`` 为球心，搜集半径 ``cutoff`` (Å) 内的全部周期镜像原子。

        返回按键长升序的 :class:`Neighbor` 列表；包含中心原子自身（image=0）。
        对小原胞结构这是 O(n × 27) 的暴力枚举，对二维原胞完全够用。
        """
        center = np.asarray(center_frac, dtype=float)
        # 沿各晶格方向的搜索半径（分数坐标单位），保守取 cutoff / 最短法向周期
        lat = self.lattice
        lens = np.array([np.linalg.norm(lat.matrix[i]) for i in range(3)])
        n_max = np.ceil(cutoff / lens).astype(int) + 1
        fracs = self.frac_coords
        out: List[Neighbor] = []
        for i in range(self.n_atoms):
            for nx, ny, nz in product(range(-n_max[0], n_max[0] + 1),
                                      range(-n_max[1], n_max[1] + 1),
                                      range(-n_max[2], n_max[2] + 1)):
                d_frac = fracs[i] + np.array([nx, ny, nz]) - center
                d_cart = d_frac @ lat.matrix
                dist = float(np.linalg.norm(d_cart))
                if dist <= cutoff:
                    out.append(Neighbor(i, self.atoms[i].symbol, dist,
                                        np.array([nx, ny, nz])))
        out.sort(key=lambda nb: nb.distance)
        return out

    def bonds(self, tol_factor: float = 1.15) -> List[Tuple[int, int, np.ndarray, float]]:
        """按共价半径规则判定成键对。

        键判据：``d ≤ tol_factor × (r_cov_i + r_cov_j)``，且原子对不相同位点。
        返回 ``[(i, j, image, dist), …]``，``j`` 取自镜像 ``image``。
        同一对原子只保留一条最近镜像键。
        """
        from ..elements import covalent_radius

        bonds = []
        seen = set()
        for i in range(self.n_atoms):
            for nb in self.neighbors(self.frac_coords[i],
                                     2.6 * max(covalent_radius(s) for s in self.species_set)):
                j = nb.index
                if j == i and not np.any(nb.image):
                    continue
                rcut = tol_factor * (covalent_radius(self.atoms[i].symbol)
                                     + covalent_radius(self.atoms[j].symbol))
                if nb.distance > rcut:
                    continue
                key = (i, j, tuple(nb.image))
                rkey = (j, i, tuple(-nb.image))
                if key in seen or rkey in seen:
                    continue
                seen.add(key)
                bonds.append((i, j, nb.image, nb.distance))
        return bonds

    # ------------------------------------------------------------------
    # 超胞
    # ------------------------------------------------------------------
    def supercell(self, rep: Sequence[int]) -> "Crystal":
        """生成超胞。``rep`` 为 (nx, ny, nz) 整数倍数，必须 ≥ 1。"""
        nx, ny, nz = (int(v) for v in rep)
        if min(nx, ny, nz) < 1:
            raise ValueError(f"超胞倍数必须 ≥ 1, 收到 {rep}")
        m = self.lattice.matrix.copy()
        m[0] *= nx
        m[1] *= ny
        m[2] *= nz
        super_lattice = Lattice(m)
        out = Crystal(super_lattice)
        for ix, iy, iz in product(range(nx), range(ny), range(nz)):
            for a in self.atoms:
                f = (a.frac + np.array([ix, iy, iz])) / np.array([nx, ny, nz])
                out.add_atom(a.symbol, f, a.label)
        return out

    # ------------------------------------------------------------------
    def copy(self) -> "Crystal":
        c = Crystal(self.lattice)
        for a in self.atoms:
            c.add_atom(a.symbol, a.frac.copy(), a.label)
        return c

    def wrap(self) -> None:
        """将所有分数坐标折回 [0, 1)（就地操作）。"""
        for a in self.atoms:
            a.frac = np.mod(a.frac, 1.0)

    def __repr__(self) -> str:
        return (f"Crystal({self.formula_str}, {self.n_atoms} atoms, "
                f"{self.lattice!r})")

    def __len__(self) -> int:
        return self.n_atoms
