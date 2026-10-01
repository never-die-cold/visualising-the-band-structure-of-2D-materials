"""示例：石墨烯与黑磷烯的应变工程 — 带隙与费米速度随应变的响应。

运行::

    python examples/run_strain.py

输出保存到 ``examples/output/``。

物理内容：
- 石墨烯双轴应变：K 点保持无隙（C₃ 对称），费米速度 v_F(ε) = v_F(0)/(1+ε)
  （键长标度律 t ∝ d⁻²，n=2）；输出 v_F(ε) 曲线；
- 黑磷烯双轴应变：带隙随应变的响应曲线（四带 TB + 键长标度的近似结果，
  定量符号需文献形变势标定，图注已如实标注）。
"""

import matplotlib
matplotlib.use("Agg")

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
from matplotlib import pyplot as plt

from vdw_studio.analysis import analyze_gap, fermi_velocity
from vdw_studio.engine.models import HoneycombModel, PhosphoreneRudenko
from vdw_studio.engine.strain import apply_strain
from vdw_studio.visualization import setup_cjk_fonts


def graphene_vf_curve() -> None:
    out = Path(__file__).parent / "output"
    out.mkdir(exist_ok=True)
    has_cjk = setup_cjk_fonts()

    g = HoneycombModel(a=2.46, t=-2.7)
    v0 = fermi_velocity(g, (1 / 3, 1 / 3), band=1, direction=(1 / 3, -1 / 3))
    strains = np.linspace(-0.05, 0.05, 21)
    vfs, gaps = [], []
    for eps in strains:
        gs = apply_strain(g, ex=float(eps), ey=float(eps), exponent=2.0)
        vfs.append(fermi_velocity(gs, (1 / 3, 1 / 3), band=1,
                                  direction=(1 / 3, -1 / 3)))
        r = analyze_gap(gs, n_per_segment=24)
        gaps.append(0.0 if r.gap is None else r.gap)   # 零隙画 0

    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    axes[0].plot(strains * 100, np.array(vfs) / 1e5, "o-", ms=4,
                 color="#1f4e79", label="TB (t∝d$^{-2}$)")
    axes[0].plot(strains * 100, v0 / 1e5 / (1 + strains), "--",
                 color="#c0392b", label="解析: v$_0$/(1+ε)")
    axes[0].set_xlabel("双轴应变 ε (%)" if has_cjk else "biaxial strain (%)")
    axes[0].set_ylabel("v$_F$ (10⁵ m/s)")
    axes[0].set_title("石墨烯费米速度 vs 双轴应变" if has_cjk else
                      "graphene Fermi velocity vs biaxial strain")
    axes[0].legend(loc="upper right", fontsize=9)

    axes[1].plot(strains * 100, gaps, "o-", ms=4, color="#2e7d32")
    axes[1].set_xlabel("双轴应变 ε (%)")
    axes[1].set_ylabel("带隙 (eV)")
    axes[1].set_title("石墨烯带隙 vs 双轴应变\n(C₃ 对称下保持零隙)")
    fig.tight_layout()
    fig.savefig(out / "graphene_strain.png", dpi=150)
    print(f"v_F(0) = {v0:.3e} m/s;  v_F(5%) = {vfs[-1]:.3e} m/s "
          f"(解析 {v0/1.05:.3e})")
    print(f"已保存: {out / 'graphene_strain.png'}")


def phosphorene_gap_curve() -> None:
    out = Path(__file__).parent / "output"
    has_cjk = setup_cjk_fonts()
    p0 = PhosphoreneRudenko()
    gap0 = analyze_gap(p0, n_per_segment=24).gap
    strains = np.linspace(-0.04, 0.04, 17)
    gaps = []
    for eps in strains:
        p = apply_strain(p0, ex=float(eps), ey=float(eps))
        gaps.append(analyze_gap(p, n_per_segment=24).gap)

    fig, ax = plt.subplots(figsize=(6, 4.2))
    ax.plot(strains * 100, gaps, "o-", ms=4, color="#2e7d32")
    ax.axhline(gap0, color="#888", lw=0.8, ls=":")
    ax.set_xlabel("双轴应变 ε (%)")
    ax.set_ylabel("带隙 (eV)")
    ax.set_title("黑磷烯带隙 vs 双轴应变\n(四带 TB + 键长标度近似；"
                 "定量符号需文献形变势标定)")
    fig.tight_layout()
    fig.savefig(out / "phosphorene_strain.png", dpi=150)
    print(f"黑磷烯: gap(0) = {gap0:.3f} eV, "
          f"gap(+4%) = {gaps[-1]:.3f} eV, gap(−4%) = {gaps[0]:.3f} eV")
    print(f"已保存: {out / 'phosphorene_strain.png'}")


if __name__ == "__main__":
    graphene_vf_curve()
    phosphorene_gap_curve()
