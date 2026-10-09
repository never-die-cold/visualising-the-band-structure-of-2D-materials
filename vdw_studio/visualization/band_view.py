"""能带结构绘图。"""

from __future__ import annotations

from typing import Optional, Tuple

import numpy as np
from matplotlib import pyplot as plt
from matplotlib.figure import Figure

from ..engine.solver import BandStructure
from .fonts import plot_text


def plot_band_structure(bs: BandStructure,
                        ax=None,
                        emin: Optional[float] = None,
                        emax: Optional[float] = None,
                        shift_fermi: bool = True,
                        color: str = "#1f4e79",
                        title: Optional[str] = None,
                        ) -> Tuple[Figure, plt.Axes]:
    """绘制沿 k 路径的能带图。

    参数:
        bs: :class:`BandStructure` 求解结果。
        emin/emax: 能量窗口 (eV, 相对费米能级)。
        shift_fermi: 以费米能级为能量零点。
    """
    created = False
    if ax is None:
        fig, ax = plt.subplots(figsize=(6, 4.5))
        created = True
    fig = ax.figure

    efermi = bs.fermi_level() if shift_fermi else 0.0
    energies = bs.energies - efermi

    for b in range(bs.n_bands):
        ax.plot(bs.x_axis, energies[:, b], color=color, lw=1.5)

    for x in bs.ticks[1:-1]:
        ax.axvline(x, color="#b8bec6", lw=0.8, ls=(0, (4, 3)))

    ax.axhline(0.0, color="#c0392b", lw=0.9, ls="--", alpha=0.8)
    ax.set_xticks(bs.ticks)
    ax.set_xticklabels(bs.tick_labels)
    ax.set_xlim(bs.x_axis[0], bs.x_axis[-1])
    if emin is not None or emax is not None:
        lo = -np.inf if emin is None else emin
        hi = np.inf if emax is None else emax
        ax.set_ylim(lo, hi)
    else:
        # 默认窗口：费米面附近 ±(带宽的一半再留 10%)
        span = energies.max() - energies.min()
        center = (energies.max() + energies.min()) / 2
        ax.set_ylim(center - 0.6 * span, center + 0.6 * span)
    ax.set_ylabel("E − E$_F$ (eV)")
    fallback = f"Band structure — {bs.model_name}"
    ax.set_title(plot_text(title or fallback, fallback))
    if created:
        fig.tight_layout()
    return fig, ax
