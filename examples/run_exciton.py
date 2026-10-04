"""示例：2D 激子束缚能 — 介电环境工程（Phase 3 演示）。

运行::

    python examples/run_exciton.py

输出保存到 ``examples/output/``。

物理内容（Rytova–Keldysh 屏蔽势 + 有效质量近似）：
- 束缚能随 Keldysh 极化长度 r₀ 的变化（材料极化越强束缚越弱）；
- 束缚能随环境介电常数 ε_env 的变化（衬底工程：高介电衬底
  压制激子束缚——"介电工程"调谐激子，二维材料特色物理）。

⚠ 参数说明：μ 与 r₀ 为示例值（μ=0.25 m₀ 为 TMD 文献常见区间；
r₀ 的材料精确值待 Cudazzo 2011 标定，见 docs/REFERENCES.md）。
"""

import matplotlib
matplotlib.use("Agg")

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
from matplotlib import pyplot as plt

from vdw_studio.analysis import solve_exciton
from vdw_studio.visualization import setup_cjk_fonts


def main() -> None:
    out = Path(__file__).parent / "output"
    out.mkdir(exist_ok=True)
    setup_cjk_fonts()

    mu = 0.25   # 约化质量 (m₀), TMD 文献常见区间 0.2–0.3

    # --- 束缚能 vs r₀（不同环境）---
    r0s = np.linspace(5, 120, 24)
    fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.2))
    for eps, style in ((1.0, "o-"), (2.5, "s--"), (4.0, "^:")):
        eb = [solve_exciton(mu, eps, r0=float(r0), n_levels=1,
                            n_basis=220, n_quad=5000).binding_1s
              for r0 in r0s]
        axes[0].plot(r0s, eb, style, ms=4, label=f"ε$_{{env}}$ = {eps}")
    axes[0].set_xlabel("Keldysh 极化长度 r$_0$ (Å)")
    axes[0].set_ylabel("1s 束缚能 (eV)")
    axes[0].set_title("激子束缚能 vs 屏蔽长度 (μ = 0.25 m$_0$)")
    axes[0].legend()

    # --- 束缚能 vs ε_env（衬底工程）---
    epss = np.linspace(1, 8, 22)
    for r0, style in ((30, "o-"), (75, "s--")):
        eb = [solve_exciton(mu, float(e), r0=r0, n_levels=1,
                            n_basis=220, n_quad=5000).binding_1s
              for e in epss]
        axes[1].plot(epss, eb, style, ms=4, label=f"r$_0$ = {r0:.0f} Å")
    axes[1].set_xlabel("环境介电常数 ε$_{env}$")
    axes[1].set_ylabel("1s 束缚能 (eV)")
    axes[1].set_title("衬底介电工程：高 ε 环境压制激子束缚")
    axes[1].legend()
    fig.tight_layout()
    fig.savefig(out / "exciton_binding.png", dpi=150)
    print(f"已保存: {out / 'exciton_binding.png'}")

    # --- 数值表 ---
    print("\n束缚能示例 (μ = 0.25 m$_0$):")
    for eps in (1.0, 2.5):
        for r0 in (30.0, 75.0):
            r = solve_exciton(mu, eps, r0=r0, n_levels=1,
                              n_basis=250, n_quad=6000)
            print(f"  ε_env={eps:<4} r₀={r0:<5.0f}: "
                  f"E_b = {r.binding_1s:.3f} eV")


if __name__ == "__main__":
    main()
