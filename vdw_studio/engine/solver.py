"""能带结构与态密度 (DOS) 求解器。

- :func:`solve_bands`：沿 k 路径求解能带，返回可绘制的
  :class:`BandStructure` 结果对象；
- :func:`solve_dos`：均匀 BZ 网格采样 + 高斯展宽得到总 DOS。

按模型接口逐点对角化；SK 等较大基组的时间需单独测量。
Gaussian 展宽复用固定预算的数组，避免完整 N_k × N_E 矩阵。
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
    scope: str = "brillouin-zone"
    spin_degeneracy: float = 1.0
    expected_states: Optional[float] = None
    n_bands: Optional[int] = None


def sample_bands(model, points, cancel_check=None):
    if cancel_check is None:
        return model.bands(points)
    blocks = []
    chunk = 1 if getattr(model, 'n_basis', 0) > 20 else 32
    for start in range(0, len(points), chunk):
        cancel_check()
        blocks.append(model.bands(points[start:start + chunk]))
    cancel_check()
    return np.vstack(blocks)


def solve_bands(model: TightBindingModel, kpath: KPath,
                n_per_segment: int = 60, *, cancel_check=None) -> BandStructure:
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

    energies = sample_bands(model, kfrac, cancel_check)
    labels = [DISPLAY_SYMBOLS.get(nm, nm) for nm in kpath.path]
    return BandStructure(
        model_name=model.name, kpoints=kcart, x_axis=x_axis, ticks=ticks,
        tick_labels=labels, path=list(kpath.path), energies=energies,
    )


def solve_dos(model: TightBindingModel, mesh: Tuple[int, int] = (48, 48),
              sigma: float = 0.05, n_points: int = 800,
              e_min: Optional[float] = None,
              e_max: Optional[float] = None, *, spin_degeneracy: float = 1.0, cancel_check=None,
              max_workspace_mb: float = 32.) -> DOSResult:
    """均匀网格采样 BZ + 高斯展宽计算总 DOS。

    参数:
        model: 紧束缚模型。
        mesh: 面内 k 网格 (n1, n2)。
        sigma: 高斯展宽宽度 (eV)。
        n_points: 能量轴点数。
        e_min / e_max: 能量窗口 (eV)，默认取本征值范围 ±10σ。
        spin_degeneracy: 默认 1；仅未显式包含自旋的模型可选择乘以 2。
    """
    if len(mesh) != 2 or any(isinstance(n, bool) or not isinstance(n, (int, np.integer)) or n < 1 for n in mesh):
        raise ValueError("mesh must contain two positive integers")
    if not np.isfinite(sigma) or sigma <= 0:
        raise ValueError("sigma must be finite and positive")
    if isinstance(n_points, bool) or not isinstance(n_points, (int, np.integer)) or n_points < 2:
        raise ValueError("n_points must be an integer >= 2")
    if spin_degeneracy not in (1., 2.):
        raise ValueError("spin_degeneracy must be 1 or 2")
    if not np.isfinite(max_workspace_mb) or max_workspace_mb <= 0:
        raise ValueError('max_workspace_mb must be finite and positive')
    n1, n2 = mesh
    kf = np.array([[ (i + 0.5) / n1, (j + 0.5) / n2 ]
                   for i in range(n1) for j in range(n2)])
    samples = np.asarray(sample_bands(model, kf, cancel_check), dtype=float)
    if samples.ndim != 2 or samples.shape[0] != len(kf) or samples.shape[1] < 1 or not np.isfinite(samples).all():
        raise ValueError("Model must return finite (nk, bands) energies")
    n_bands = samples.shape[1]
    eig = samples.ravel()

    if e_min is None:
        e_min = float(eig.min() - 10 * sigma)
    if e_max is None:
        e_max = float(eig.max() + 10 * sigma)
    if not np.isfinite([e_min, e_max]).all() or e_min >= e_max:
        raise ValueError("Energy window must have finite increasing bounds")
    energies = np.linspace(e_min, e_max, n_points)

    # 高斯展宽（向量化分块计算，避免 N_k × N_E 大矩阵）
    # 每个 k 点代表一个 BZ 体积 → DOS = (1/N_k) Σ_s g(E−E_s)，
    # 积分 = 实际返回的带数 × 显式自旋简并；默认每个模型态计一次。
    n_kpoints = len(kf)
    pref = spin_degeneracy / (sigma * np.sqrt(2 * np.pi) * n_kpoints)
    dos = np.zeros_like(energies)
    chunk = max(1, int(max_workspace_mb * 1024 ** 2 / (8 * n_points)))
    workspace = np.empty((min(chunk, len(eig)), n_points))
    for start in range(0, len(eig), chunk):
        if cancel_check is not None:
            cancel_check()
        block = eig[start:start + chunk]
        gaussian = workspace[:len(block)]
        np.subtract(energies[None, :], block[:, None], out=gaussian)
        gaussian /= sigma
        np.square(gaussian, out=gaussian)
        gaussian *= -.5
        np.exp(gaussian, out=gaussian)
        dos += pref * gaussian.sum(axis=0)

    result = DOSResult(model_name=model.name, energies=energies, dos=dos,
                       sigma=sigma, mesh=tuple(mesh), spin_degeneracy=float(spin_degeneracy),
                       expected_states=float(n_bands * spin_degeneracy), n_bands=n_bands)
    # 费米能级：非自旋极化 → 占据带数 = 总带数的一半；
    # 全局排序后占据/非占据的分界在总本征值数的一半处
    sorted_eig = np.sort(eig)
    n_total = len(sorted_eig)
    result.efermi = float(sorted_eig[max(0, n_total // 2 - 1)])
    return result
