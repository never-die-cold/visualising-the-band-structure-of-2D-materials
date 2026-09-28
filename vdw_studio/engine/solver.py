"""能带结构与态密度 (DOS) 求解器。

- :func:`solve_bands`：沿 k 路径求解能带，返回可绘制的
  :class:`BandStructure` 结果对象；
- :func:`solve_dos`：均匀 BZ 网格采样 + 高斯展宽得到总 DOS。

所有模型哈密顿量均为小矩阵（≤ 20×20），直接使用
``numpy.linalg.eigvalsh`` 逐点对角化；二维采样网格 (48×48) 在
普通 PC 上毫秒级完成，无需并行。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional, Tuple

import numpy as np

from .kpath import DISPLAY_SYMBOLS, KPath
from .models import TightBindingModel


@dataclass
class BandStructure:
    """沿 k 路径的能带计算结果。"""

    model_name: str
    kpoints: np.ndarray          # (N, 3) 笛卡尔 k 点 (Å⁻¹)
    x_axis: np.ndarray           # (N,) 绘图横轴
    ticks: np.ndarray            # 高对称点横轴位置
    tick_labels: List[str]       # 高对称点显示符号 (Γ/M/K/...)
    path: List[str]              # 高对称点名称序列
    energies: np.ndarray         # (N, n_bands) 能量 (eV)，每行升序

    @property
    def n_bands(self) -> int:
        return self.energies.shape[1]

    def fermi_level(self) -> float:
        """以占据带顶估计费米能级（每位点 2 个自旋态）。

        简单取法：价带顶 = 所有 k 点中第 (n_occ−1) 条带的最高值，
        其中 n_occ = n_bands // 2（非自旋极化、每态双占据）。
        """
        n_occ = self.energies.shape[1] // 2
        if n_occ == 0:
            return float(self.energies.min())
        return float(self.energies[:, n_occ - 1].max())


@dataclass
class DOSResult:
    """态密度计算结果。"""

    model_name: str
    energies: np.ndarray     # (M,) 能量轴 (eV)
    dos: np.ndarray          # (M,) 态密度 (states/eV/cell)
    sigma: float             # 高斯展宽 (eV)
    mesh: Tuple[int, int]    # k 网格
    efermi: float = 0.0      # 费米能级估计 (eV，占据带顶)


def solve_bands(model: TightBindingModel, kpath: KPath,
                n_per_segment: int = 60) -> BandStructure:
    """沿 k 路径求解能带。

    参数:
        model: 紧束缚模型。
        kpath: k 路径（应与模型晶格匹配）。
        n_per_segment: 每段采样数。
    """
    kcart, x_axis, ticks = kpath.generate(model.lattice, n_per_segment)
    # 直接按倒格分数坐标采样（哈密顿量相位约定使用分数坐标）
    kfrac_list = []
    for (p0, p1) in kpath.segments():
        for t in np.linspace(0.0, 1.0, n_per_segment, endpoint=False):
            kfrac_list.append([p0[0] + t * (p1[0] - p0[0]),
                               p0[1] + t * (p1[1] - p0[1])])
    last = kpath.points[kpath.path[-1]]
    kfrac_list.append(list(last))
    kfrac = np.array(kfrac_list)

    energies = model.bands(kfrac)
    labels = [DISPLAY_SYMBOLS.get(nm, nm) for nm in kpath.path]
    return BandStructure(
        model_name=model.name, kpoints=kcart, x_axis=x_axis, ticks=ticks,
        tick_labels=labels, path=list(kpath.path), energies=energies,
    )


def solve_dos(model: TightBindingModel, mesh: Tuple[int, int] = (48, 48),
              sigma: float = 0.05, n_points: int = 800,
              e_min: Optional[float] = None,
              e_max: Optional[float] = None) -> DOSResult:
    """均匀网格采样 BZ + 高斯展宽计算总 DOS。

    参数:
        model: 紧束缚模型。
        mesh: 面内 k 网格 (n1, n2)。
        sigma: 高斯展宽宽度 (eV)。
        n_points: 能量轴点数。
        e_min / e_max: 能量窗口 (eV)，默认取本征值范围 ±10σ。
    """
    if min(mesh) < 1:
        raise ValueError("k 网格维度必须 ≥ 1")
    n1, n2 = mesh
    kf = np.array([[ (i + 0.5) / n1, (j + 0.5) / n2 ]
                   for i in range(n1) for j in range(n2)])
    eig = model.bands(kf).ravel()

    if e_min is None:
        e_min = float(eig.min() - 10 * sigma)
    if e_max is None:
        e_max = float(eig.max() + 10 * sigma)
    energies = np.linspace(e_min, e_max, n_points)

    # 高斯展宽（向量化分块计算，避免 N_k × N_E 大矩阵）
    # 每个 k 点代表一个 BZ 体积 → DOS = (1/N_k) Σ_s g(E−E_s)，
    # 积分 = 每原胞态数 (n_sites)
    n_kpoints = len(eig) // model.n_sites
    pref = 1.0 / (sigma * np.sqrt(2 * np.pi) * n_kpoints)
    dos = np.zeros_like(energies)
    chunk = max(1, int(2e5 / n_points))
    for start in range(0, len(eig), chunk):
        block = eig[start:start + chunk]
        dos += pref * np.exp(-0.5 * ((energies[None, :] - block[:, None]) / sigma) ** 2
                            ).sum(axis=0)

    result = DOSResult(model_name=model.name, energies=energies, dos=dos,
                       sigma=sigma, mesh=mesh)
    # 费米能级：非自旋极化 → 占据带数 = 总带数的一半；
    # 全局排序后占据/非占据的分界在总本征值数的一半处
    sorted_eig = np.sort(eig)
    n_total = len(sorted_eig)
    result.efermi = float(sorted_eig[n_total // 2 - 1])
    return result
