"""Wannier90 实空间哈密顿量的周期插值模型。

把 :func:`vdw_studio.io.wannier.read_hr` 读出的 H(R) 封装成与
``engine.models`` 同一套模型协议（``hamiltonian`` / ``bands`` /
``lattice`` / ``n_sites``），从而接入现有的求解器、带隙分析、DOS、
有效质量与激子全链——DFT 精度的能带由此进入平台。

插值公式（Wannier90 user guide 的约定）::

    H(k)_{mn} = Σ_R  w_R · e^{i 2π k·R} · H(R)_{mn}

``k`` 为倒格分数坐标、``R`` 为整数晶格矢（``_hr.dat`` 的 R 列表是完整
的，不需要截断/镜像搜索）；``w_R`` 是简并因子（全 1 即可）。Wannier
函数正交，因此这是普通本征值问题（S = I）。
"""

import numpy as np

from ..io.wannier import read_hr
from ..structure.lattice import Lattice


class WannierModel:
    """由 ``_hr.dat`` 驱动的插值 TB 模型。

    参数:
        hr_path: ``_hr.dat`` 路径。
        lattice: 晶格（Wannier90 文件不含晶格信息，必须由调用方提供；
                 单位 Å，行向量约定与平台一致）。
        n_valence: 占据带数（NELECT/2，自旋无关计带）——带隙分析、
                 费米参考与输运共用。
        name: 模型显示名。
    """

    def __init__(self, hr_path, lattice: Lattice, n_valence: int,
                 name: str = "Wannier90 model") -> None:
        self.hr_path = str(hr_path)
        self.lattice = lattice
        self.n_valence = int(n_valence)
        self.name = name
        self.r_vectors, self.h_r, self.degeneracy, _ = read_hr(hr_path)
        self.n_sites = self.h_r.shape[1]

    # ------------------------------------------------------------------
    def hamiltonian(self, kfrac) -> np.ndarray:
        """周期插值 H(k)（厄米复矩阵，eV）。

        ``kfrac`` 为倒格分数坐标 (2,) 或 (3,)——第三分量沿真空方向，
        平面内插值不使用它。
        """
        k = np.asarray(kfrac, dtype=float).reshape(-1)[:2]
        phase = np.exp(2j * np.pi * (self.r_vectors[:, :2] @ k))
        weighted = self.h_r * (phase / self.degeneracy)[:, None, None]
        return weighted.sum(axis=0)

    def bands(self, kfracs) -> np.ndarray:
        """逐 k 点本征值（(nk, nw)，升序）。"""
        kfracs = np.asarray(kfracs, dtype=float)
        out = np.empty((len(kfracs), self.n_sites))
        for ik, k in enumerate(kfracs):
            out[ik] = np.linalg.eigvalsh(self.hamiltonian(k))
        return out

    def energies_at(self, kfrac) -> np.ndarray:
        return np.linalg.eigvalsh(self.hamiltonian(kfrac))

    def __repr__(self) -> str:
        return (f"WannierModel(nw={self.n_sites}, nrpt={len(self.r_vectors)}, "
                f"n_valence={self.n_valence}, hr={self.hr_path!r})")
