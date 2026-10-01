"""示例：单层 MoS₂ 的 Berry 曲率与谷选择光学定则（谷物理，Phase 4）。

运行::

    python examples/run_valley_physics.py

输出保存到 ``examples/output/``。

物理内容：
- 价带 Berry 曲率热图（K 谷附近）：谷热点与符号；
- 谷 Chern 数（半量子化）；
- 圆偏振光学选择定则：K 与 −K 谷的 σ⁺/σ⁻ 跃迁强度比。

公式：Xiao, Chang & Niu, Rev. Mod. Phys. 82, 1959 (2010) 的速度算符
表述；模型参数：Kormányos et al., 2D Mater. 2, 022001 (2015)。
"""

import matplotlib
matplotlib.use("Agg")

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
from matplotlib import pyplot as plt

from vdw_studio.analysis.berry import (
    berry_curvature,
    optical_circular_dichroism,
    valley_chern,
)
from vdw_studio.engine.kp_tmd import TMDKpModel, TMDKpParams
from vdw_studio.presets import run_preset
from vdw_studio.visualization import setup_cjk_fonts


def main() -> None:
    out = Path(__file__).parent / "output"
    out.mkdir(exist_ok=True)
    has_cjk = setup_cjk_fonts()

    r = run_preset("mos2_kp")
    kp = r["model"]
    print(f"材料: 单层 MoS2 (k·p, Kormányos 2015); K 点带隙 {r['gap'].gap:.3f} eV")

    # --- 价带 Berry 曲率热图 (K 谷附近, |q| ≤ 0.3 Å⁻¹) ---
    n = 61
    qs = np.linspace(-0.3, 0.3, n)
    omega = np.zeros((n, n))
    for i, qx in enumerate(qs):
        for j, qy in enumerate(qs):
            omega[i, j] = berry_curvature(
                lambda qq: kp.hamiltonian(qq, +1),
                np.array([qx, qy]), band=1)
    omega_T = omega.T   # 行 = qy

    fig, ax = plt.subplots(figsize=(6.2, 5))
    vmax = np.abs(omega_T).max()
    im = ax.imshow(omega_T, origin="lower", extent=[-0.3, 0.3, -0.3, 0.3],
                   cmap="RdBu_r", vmin=-vmax, vmax=vmax)
    ax.scatter([0], [0], marker="x", color="black", s=60, label="K valley")
    ax.set_xlabel("q$_x$ (Å$^{-1}$)")
    ax.set_ylabel("q$_y$ (Å$^{-1}$)")
    ax.set_title("MoS$_2$ valence-band Berry curvature Ω (Å²)\n"
                 "near K (spin-up block)")
    ax.legend(loc="upper right")
    fig.colorbar(im, ax=ax, label="Ω (Å²)")
    fig.tight_layout()
    fig.savefig(out / "mos2_berry_curvature.png", dpi=150)
    print(f"Berry 曲率峰值: {omega_T.max():.2f} / {omega_T.min():.2f} Å² "
          f"(K 点符号: {'+' if omega_T[n//2, n//2] > 0 else '-'})")
    print(f"已保存: {out / 'mos2_berry_curvature.png'}")

    # --- 谷 Chern (半量子化) ---
    for valley in (+1, -1):
        c, flux = valley_chern(
            lambda qq: kp.hamiltonian(qq, valley), band=1,
            qmax=0.6, n_theta=24, n_rad=16)
        print(f"谷 {valley:+d}: Berry 通量 {flux:+.3f} rad "
              f"(C/2π·disk → 半量子化; 完整极限 ±π)")

    # --- 圆偏振选择定则 ---
    print("\n谷选择光学定则 (A 激子通道 vb↑→cb↑):")
    for valley, tag in ((+1, "K"), (-1, "-K")):
        d = optical_circular_dichroism(
            lambda qq: kp.hamiltonian(qq, valley),
            np.array([0.005, 0.0]), band_v=1, band_c=3)
        print(f"  {tag:2s} 谷: 主导偏振 {d.dominant}, "
              f"强度比 {d.ratio:.0f}:1")

    # --- σ+/σ− 强度柱状对比 ---
    fig, ax = plt.subplots(figsize=(4.6, 4))
    labels, fplus, fminus = [], [], []
    for valley, tag in ((+1, "K"), (-1, "-K")):
        d = optical_circular_dichroism(
            lambda qq: kp.hamiltonian(qq, valley),
            np.array([0.005, 0.0]), band_v=1, band_c=3)
        labels.append(tag)
        fplus.append(d.f_plus)
        fminus.append(d.f_minus)
    x = np.arange(2)
    ax.bar(x - 0.18, fplus, 0.36, label="σ$^+$", color="#c0392b")
    ax.bar(x + 0.18, fminus, 0.36, label="σ$^-$", color="#1f4e79")
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylabel("|⟨c|v$_x$±iv$_y$|v⟩|² (eV²Å²)")
    ax.set_title("MoS$_2$ valley-selective optical excitation\n"
                 "(q = 0.005 Å⁻¹, A-exciton channel)")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out / "mos2_dichroism.png", dpi=150)
    print(f"已保存: {out / 'mos2_dichroism.png'}")


if __name__ == "__main__":
    main()
