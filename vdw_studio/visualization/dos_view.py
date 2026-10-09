"""态密度 (DOS) 绘图。"""

from __future__ import annotations

from typing import Optional, Tuple

import numpy as np
from matplotlib import pyplot as plt

from ..engine.solver import DOSResult
from .fonts import plot_text


def plot_dos(dos: DOSResult,
             ax=None,
             emin: Optional[float] = None,
             emax: Optional[float] = None,
             color: str = "#2e7d32",
             title: Optional[str] = None,
             ) -> Tuple[plt.Figure, plt.Axes]:
    """绘制总态密度（含费米能级虚线与积分标注）。"""
    created = False
    if ax is None:
        fig, ax = plt.subplots(figsize=(4.2, 4.5))
        created = True
    fig = ax.figure

    e = dos.energies - dos.efermi
    ax.fill_betweenx(e, 0, dos.dos, color=color, alpha=0.25)
    ax.plot(dos.dos, e, color=color, lw=1.6)
    ax.axhline(0.0, color="#c0392b", lw=0.9, ls="--", alpha=0.8)

    if emin is not None or emax is not None:
        ax.set_ylim(emin if emin is not None else e.min(),
                    emax if emax is not None else e.max())
    ax.set_xlabel("DOS (states/eV/cell)")
    ax.set_ylabel("E − E$_F$ (eV)")
    integral = float(np.trapezoid(dos.dos, dos.energies))
    fallback = (f"DOS — {dos.model_name}\n"
                          f"(mesh {dos.mesh[0]}×{dos.mesh[1]}, "
                          f"σ={dos.sigma} eV, N={integral:.1f})")
    ax.set_title(plot_text(title or fallback, fallback), fontsize=10)
    if created:
        fig.tight_layout()
    return fig, ax
