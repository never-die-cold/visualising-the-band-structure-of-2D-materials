"""示例：转角双层 TMD 的 moiré 平带与层间激子 — Eb(θ) 响应。

运行::

    python examples/run_twisted_tmd.py

产出（保存到 ``examples/output/``）：

- (a) θ = 2° 的 moiré 能带（最低 6 条，Γ–M 方向）；
- (b) moiré 有效质量与层间激子束缚能随 θ 的变化——小转角平带化
  使质量暴涨、束缚能增强（Wu–Lovorn–MacDonald 式连续近似的单带
  最小模型；层间激子用共面 2D 近似，见 analysis.interlayer_exciton
  的诚实标注）。

模型与验证锚点见 ``engine.moire_tmd``（T=0 解耦、C₃、二阶弱耦合、
平带化）与 ``tests/test_interlayer_exciton.py``（t=0 衔接 Berkelbach
标定链）。
"""

import matplotlib
matplotlib.use("Agg")

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np

from vdw_studio.analysis.interlayer_exciton import interlayer_exciton_vs_theta
from vdw_studio.engine.moire_tmd import TwistedTMDBilayer
from vdw_studio.visualization import setup_cjk_fonts


def moire_bands_panel(ax, theta_deg: float = 2.0) -> None:
    """(a) θ=2° 的 moiré 能带（Γ–M 方向，最低 6 条）。"""
    model = TwistedTMDBilayer(theta_deg=theta_deg, t_interlayer=0.020,
                              n_shells=3, band="cb")
    k = np.linspace(0.0, model.geom.k_theta, 60)
    energies = model.bands(np.column_stack((k, np.zeros_like(k))))
    x = k / model.geom.k_theta
    for band in range(6):
        ax.plot(x, energies[:, band] * 1e3, lw=1.3, color="#1f4e79",
                alpha=0.9 if band == 0 else 0.55)
    ax.set_xlabel("Γ → M (k/k$_M$)")
    ax.set_ylabel("E (meV)")
    ax.set_title(f"moiré 导带（θ = {theta_deg}°，T = 20 meV）" if setup_cjk_fonts()
                 else f"moire conduction bands (θ = {theta_deg}°, T = 20 meV)")
    ax.grid(True, alpha=0.3)


def exciton_panel(ax, thetas: np.ndarray) -> None:
    """(b) moiré 质量与层间激子束缚能随 θ。"""
    reports = interlayer_exciton_vs_theta(thetas, t_interlayer=0.020, shells=(2, 3))
    mass_e = [r.m_e_m0 for r in reports]
    binding = [r.binding_ev * 1e3 for r in reports]
    ax2 = ax.twinx()
    ax.plot(thetas, mass_e, "o-", ms=4, color="#1f4e79", label="m*(θ)")
    ax2.plot(thetas, binding, "s--", ms=4, color="#c0392b", label="E$_b$(θ)")
    ax.set_xscale("log")
    ax.set_xlabel("转角 θ (°)" if setup_cjk_fonts() else "twist angle θ (°)")
    ax.set_ylabel("moiré 有效质量 (m$_0$)", color="#1f4e79")
    ax2.set_ylabel("层间激子束缚能 (meV)", color="#c0392b")
    ax.set_title("平带化增强层间激子束缚" if setup_cjk_fonts()
                 else "flat-band enhancement of interlayer exciton binding")
    ax.grid(True, alpha=0.3)
    lines = ax.get_lines() + ax2.get_lines()
    ax.legend(lines, [line.get_label() for line in lines], fontsize=8)


def main() -> None:
    out = Path(__file__).parent / "output"
    out.mkdir(exist_ok=True)
    has_cjk = setup_cjk_fonts()

    fig, axes = __import__("matplotlib.pyplot", fromlist=["plt"]).subplots(1, 2, figsize=(10, 4))
    moire_bands_panel(axes[0])
    exciton_panel(axes[1], np.array([0.5, 0.75, 1.0, 1.5, 2.0, 3.0, 5.0]))
    fig.tight_layout()
    target = out / "twisted_tmd_exciton.png"
    fig.savefig(target, dpi=200, bbox_inches="tight")
    print(f"saved: {target}")

    # 终端摘要
    reports = interlayer_exciton_vs_theta([1.0, 2.0, 5.0], t_interlayer=0.020, shells=(2, 3))
    print("θ(°)   m_e(m0)  m_h(m0)  E_b(eV)  带宽(meV)  converged")
    for r in reports:
        print(f"{r.theta_deg:5.2f}  {r.m_e_m0:7.3f}  {r.m_h_m0:7.3f}  "
              f"{r.binding_ev:7.4f}  {r.flat_band_width_ev * 1e3:8.1f}  {r.converged}")
    if not has_cjk:
        print("(CJK fonts unavailable; figure labels fell back to English)")


if __name__ == "__main__":
    main()
