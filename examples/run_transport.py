"""示例：黑磷烯各向异性输运（Phase 4.4，CRTA 玻尔兹曼输运）。

运行::

    python examples/run_transport.py

输出保存到 ``examples/output/``。

物理内容：
- 空穴迁移率随载流子密度的变化（x 与 y 两个晶格方向）；
- 各向异性比 μ$_y$/μ$_x$ 与质量比的对账；
- 与文献方向结论对照（Qiao et al., PRL 113, 256803 (2014)：
  单层黑磷空穴迁移率高方向为 y 轴，10⁴–2.6×10⁴ vs x 640–700 cm²/Vs；
  注意其比值主要来自变形势/弹性模量，CRTA 只含质量因素）。

参数：Rudenko 四带模型（PRB 89, 201408(R) (2014)），
τ = 10 fs 唯象弛豫时间（CRTA）。
"""

import matplotlib
matplotlib.use("Agg")

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
from matplotlib import pyplot as plt

from vdw_studio.analysis import effective_mass, solve_transport
from vdw_studio.engine.models import PhosphoreneRudenko
from vdw_studio.visualization import setup_cjk_fonts


def main() -> None:
    out = Path(__file__).parent / "output"
    out.mkdir(exist_ok=True)
    setup_cjk_fonts()

    model = PhosphoreneRudenko()
    m_x = effective_mass(model, (0, 0), band=1, direction=(1, 0))
    m_y = effective_mass(model, (0, 0), band=1, direction=(0, 1))
    mu_x_dr = 1.602176634e-19 * 1e-14 / (m_x * 9.1093837015e-31) * 1e4
    mu_y_dr = 1.602176634e-19 * 1e-14 / (m_y * 9.1093837015e-31) * 1e4

    densities = np.logspace(11.0, 12.3, 12)      # cm⁻²
    res = [solve_transport(model, n_cm2=-n, tau_fs=10.0,
                           temperature=100.0, n_mesh=160)
           for n in densities]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10.5, 4.2))
    ax1.loglog(densities, [r.mu_xx for r in res], "o-",
               label=f"μ$_x$（3.31 Å 轴，m$^*$ = {m_x:.2f} m$_0$）")
    ax1.loglog(densities, [r.mu_yy for r in res], "s-",
               label=f"μ$_y$（4.38 Å 轴，m$^*$ = {m_y:.2f} m$_0$）")
    ax1.axhline(mu_x_dr, color="C0", ls="--", lw=0.8, alpha=0.7,
                label=r"Drude $e\tau/m_x$")
    ax1.axhline(mu_y_dr, color="C1", ls="--", lw=0.8, alpha=0.7,
                label=r"Drude $e\tau/m_y$")
    ax1.set_xlabel("空穴密度 n (cm$^{-2}$)")
    ax1.set_ylabel("迁移率 μ (cm$^2$/Vs)")
    ax1.set_title("(a) 黑磷烯空穴迁移率（CRTA, τ = 10 fs, T = 100 K）")
    ax1.legend(fontsize=8)

    ratios = [r.mu_yy / r.mu_xx for r in res]
    ax2.plot(densities, ratios, "d-", color="C2")
    ax2.axhline(m_x / m_y, color="k", ls="--", lw=0.9,
                label=f"质量比 $m_x/m_y$ = {m_x / m_y:.2f}")
    ax2.set_xscale("log")
    ax2.set_xlabel("空穴密度 n (cm$^{-2}$)")
    ax2.set_ylabel("各向异性比 μ$_y$/μ$_x$")
    ax2.set_title("(b) 各向异性 = 质量比（CRTA）")
    ax2.legend(fontsize=9)

    fig.tight_layout()
    fig.savefig(out / "phosphorene_transport.png", dpi=160)
    print(f"已保存 {out / 'phosphorene_transport.png'}")

    print("\n=== 黑磷烯输运报告（CRTA, τ = 10 fs）===")
    print(f"  带边质量: m_x = {m_x:.3f} m0（3.31 Å 轴）, "
          f"m_y = {m_y:.3f} m0（4.38 Å 轴）")
    r = res[len(res) // 2]
    print(f"  n = {r.n_cm2:.2e} cm⁻², T = {r.temperature:.0f} K:")
    print(f"    μ_x = {r.mu_xx:.1f}, μ_y = {r.mu_yy:.1f} cm2/Vs"
          f"  （比 {r.mu_yy / r.mu_xx:.2f}）")
    print(f"    σ_xx = {r.sigma_xx:.3f} e2/h, σ_xy = {r.sigma_xy:.2e} e2/h")
    print("  文献方向（Qiao 2014）：高迁移率方向为 y 轴 ✓"
          "（其比值由变形势主导，CRTA 比值为质量比）")


if __name__ == "__main__":
    main()
