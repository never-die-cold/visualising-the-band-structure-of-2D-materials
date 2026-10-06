"""示例：转角双层石墨烯 moiré 平带（Phase 5，Bistritzer–MacDonald 模型）。

运行::

    python examples/run_moire.py

输出保存到 ``examples/output/``。

物理内容：
- 魔角 θ=1.05° 与一般转角 2° 的 moiré 能带（Γ_m-M-K-Γ_m 路径）：
  魔角处最低 moiré 带极端平化；
- Dirac 点速度 v*/v 随 α²=w²/(ħv·k_θ)² 的变化：数值扫描 + BM Eq. (8)
  第一壳层解析曲线 (1−3α²)/(1+6α²)，第一魔角 α₁≈1/√3；
- 平带带宽与 moiré 周期报告。

模型：Bistritzer & MacDonald, Phys. Rev. B 84, 035440 (2011)；
参数 w = 110 meV（AB 堆叠标定）、ħv = (√3/2)|t|a₀ = 5.755 eV·Å
（Reich 2002，与 HoneycombModel 一致）。
"""

import matplotlib
matplotlib.use("Agg")

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
from matplotlib import pyplot as plt

from vdw_studio.engine.moire import TwistedBilayerGraphene
from vdw_studio.visualization import setup_cjk_fonts


def main() -> None:
    out = Path(__file__).parent / "output"
    out.mkdir(exist_ok=True)
    setup_cjk_fonts()

    fig, (ax_band, ax_vel) = plt.subplots(1, 2, figsize=(11, 4.4))

    # ---- (a) moiré 能带：魔角 vs 一般转角 -----------------------------
    for theta, color, label in ((2.0, "#888888", "2.0°（一般转角）"),
                                (1.05, "#c0392b", "1.05°（第一魔角）")):
        m = TwistedBilayerGraphene(theta_deg=theta, n_shells=4)
        path, labels, ticks = m.moire_path(n_seg=30)
        E = m.bands(list(path))
        dim = E.shape[1]
        n0 = dim // 2
        x = np.arange(len(path))
        for b in range(n0 - 4, min(n0 + 5, dim)):
            ax_band.plot(x, E[:, b] * 1e3, lw=1.0, color=color, alpha=0.9,
                         label=label if b == n0 else None)
    ax_band.axhline(0.0, color="k", lw=0.6, ls="--", alpha=0.6)
    for t, lab in zip(ticks, labels):
        ax_band.axvline(t, color="k", lw=0.5, alpha=0.3)
    ax_band.set_xticks(ticks)
    ax_band.set_xticklabels(labels)
    ax_band.set_ylabel("E (meV)")
    ax_band.set_title("(a) moiré 能带（w = 110 meV）")
    ax_band.legend(loc="upper right", fontsize=9)
    ax_band.set_ylim(-80, 80)

    # ---- (b) v*/v vs α²：数值扫描 + BM Eq. (8) ------------------------
    thetas = np.linspace(0.7, 2.6, 27)
    ratios = []
    for th in thetas:
        m = TwistedBilayerGraphene(theta_deg=float(th), n_shells=4)
        ratios.append(m.dirac_velocity() / m.hbar_v)
    ratios = np.array(ratios)
    alphas = np.array([TwistedBilayerGraphene(theta_deg=float(th)).alpha
                       for th in thetas])

    a_fit = np.linspace(0.0, 1.0, 200)
    ax_vel.plot(a_fit ** 2, (1 - 3 * a_fit ** 2) / (1 + 6 * a_fit ** 2),
                "k--", lw=1.2, label="BM Eq. (8)：$(1-3\\alpha^2)/(1+6\\alpha^2)$")
    ax_vel.plot(alphas ** 2, ratios, "o", ms=4.5, color="#2980b9",
                label="全壳层数值（$N_G$ 截断）")
    ax_vel.axhline(0.0, color="k", lw=0.6, alpha=0.5)
    ax_vel.axvline(1 / 3, color="#c0392b", lw=0.8, ls=":",
                   label=r"第一魔角 $\alpha_1=1/\sqrt{3}$")
    ax_vel.set_xlabel(r"$\alpha^2 = [w/(\hbar v k_\theta)]^2$")
    ax_vel.set_ylabel(r"$v^{\ast}/v$")
    ax_vel.set_title("(b) Dirac 点速度重整与魔角")
    ax_vel.set_ylim(-0.6, 1.0)
    ax_vel.legend(fontsize=8, loc="upper right")

    fig.tight_layout()
    fig.savefig(out / "moire_bands.png", dpi=160)
    print(f"已保存 {out / 'moire_bands.png'}")

    # ---- 报告 ---------------------------------------------------------
    m_magic = TwistedBilayerGraphene(theta_deg=1.05, n_shells=5)
    w_flat = m_magic.flat_band_width(n_grid=12)
    print("\n=== 转角双层石墨烯（BM 连续模型）报告 ===")
    print(f"  w = 110 meV，ħv = 5.755 eV·Å（Reich 2002）")
    print(f"  魔角 θ = 1.05°（BM Fig. 3）：|v*|/v = "
          f"{abs(m_magic.dirac_velocity())/m_magic.hbar_v:.4f}")
    print(f"  魔角平带带宽（12×12 BZ 采样）：{w_flat*1e3:.2f} meV")
    print(f"  moiré 周期 L_m(1.05°) = {m_magic.geom.l_moire:.1f} Å"
          f"（实验 ~13.4 nm）")
    print(f"  α(1.05°) = {m_magic.alpha:.3f}（第一魔角解析值 1/√3 ≈ 0.577，"
          "高壳层修正使数值魔角略偏）")


if __name__ == "__main__":
    main()
