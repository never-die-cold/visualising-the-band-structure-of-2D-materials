"""拟合参考数据：能带 + 轨道投影的 (k, E, W) 三元组。

本交付包含合成生成（``from_model``：从"真值"模型造数据，用于拟合器的
自验证——标准做法是扰动参数造数据、从未扰动初值拟合、检验能否恢复）。
真实 DFT 数据的接入点预留为 ``from_procar``（需要 EIGENVAL/PROCAR/KPOINTS
三件套与相位无关的投影处理，后续交付）。
"""

from dataclasses import dataclass, field
from typing import Dict, Optional, Tuple

import numpy as np

from .projection import DEFAULT_GROUPS, group_weights


@dataclass
class BandReference:
    """窗口化的参考数据。``band_window`` 是它在 TB 带序列中的 [start, end)。"""

    kpoints: np.ndarray                                  # (nk, 2) 分数坐标
    energies: np.ndarray                                 # (nk, n_ref) eV
    weights: Optional[Dict[str, np.ndarray]]             # 组名 -> (nk, n_ref)
    band_window: Tuple[int, int]
    n_valence: int
    source: str = "synthetic"
    meta: dict = field(default_factory=dict)

    @classmethod
    def from_model(cls, model, kpoints, *, n_bands: Tuple[int, int] = (14, 22),
                   groups: Optional[dict] = None, source: str = "synthetic") -> "BandReference":
        """从"真值"模型生成合成参考数据（拟合器自验证的标准输入）。"""
        start, end = (int(n_bands[0]), int(n_bands[1]))
        if not 0 <= start < end <= model.n_basis:
            raise ValueError(f"n_bands 窗口 [{start}, {end}) 越界（n_basis={model.n_basis}）")
        kpoints = np.asarray(kpoints, dtype=float)
        energies_full, weights_full = group_weights(
            model, kpoints, DEFAULT_GROUPS if groups is None else groups)
        weights = None
        if groups is None or groups is DEFAULT_GROUPS or groups:
            weights = {name: w[:, start:end] for name, w in weights_full.items()}
        return cls(
            kpoints=kpoints[:, :2],
            energies=energies_full[:, start:end],
            weights=weights,
            band_window=(start, end),
            n_valence=int(getattr(model, "n_valence", model.n_basis // 2)),
            source=source,
            meta={"groups": sorted(weights) if weights else []},
        )
