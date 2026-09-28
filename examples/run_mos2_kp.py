"""示例：单层 MoS₂ 的 k·p 谷物理仿真 — 能带 / DOS / 布里渊区一键出图。

运行::

    python examples/run_mos2_kp.py

输出保存到 ``examples/output/``。材料参数出处见
``docs/REFERENCES.md``（Kormányos et al., 2D Mater. 2, 022001 (2015)）。
"""

import matplotlib
matplotlib.use("Agg")

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


from pathlib import Path

import numpy as np

from vdw_studio.analysis import analyze_gap, effective_mass, principal_masses
from vdw_studio.engine.kp_tmd import TMDKpModel, TMDKpParams
from vdw_studio.engine.kpath import KPath
from vdw_studio.engine.solver import solve_bands, solve_dos
from vdw_studio.presets import run_preset
from vdw_studio.visualization import plot_band_structure, plot_bz_path, plot_dos


def main() -> None:
    out = Path(__file__).parent / "output"
    out.mkdir(exist_ok=True)

    r = run_preset("mos2_kp")
    kp: TMDKpModel = r["model"]
    gap = r["gap"]
    print(f"材料: 单层 MoS2 (k·p, Kormányos 2015)")
    print(f"K 点最小带隙: {gap.gap:.3f} eV (文献 1.670 eV, "
          f"匹配: {r['gap_matches_ref']})")

    # --- 能带（k·p 有效模型仅在 K 谷附近 |q| ≲ 0.25 Å⁻¹ 有效）---
    # 取以 K 为原点的两支径向切割：朝 Γ 方向与朝 M 方向
    from vdw_studio.structure.builders import tmd
    B = tmd("MoS2").lattice.reciprocal_matrix[:2, :2]
    K = np.array([1 / 3, 1 / 3]) @ B            # K 点 (笛卡尔)
    M = np.array([1 / 2, 0.0]) @ B              # M 点 (笛卡尔)
    qmax = 0.25   # Å⁻¹, k·p 有效范围
    u_gamma = -K / np.linalg.norm(K)            # K → Γ 单位矢
    u_m = (M - K)
    u_m /= np.linalg.norm(u_m)                  # K → M 单位矢

    def radial(u, n=50):
        return [qmax * t / (n - 1) * u for t in range(n)]

    qpath = radial(u_gamma) + radial(-u_gamma)[1:] + radial(u_m, 50)[1:]
    E = np.array([kp.energies(q, +1) for q in qpath])
    x = np.concatenate([[0], np.cumsum(np.linalg.norm(
        np.diff(np.array(qpath), axis=0), axis=1))])
    n_half = 50
    ticks_x = [x[0], x[n_half - 1], x[2 * n_half - 2]]

    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(6, 4.5))
    for b in range(4):
        ax.plot(x, E[:, b], lw=1.6, color="#1f4e79")
    for xx in ticks_x[1:]:
        ax.axvline(xx, color="#b8bec6", lw=0.8, ls=(0, (4, 3)))
    ax.axhline(0, color="#c0392b", lw=0.9, ls="--")
    ax.set_xticks(ticks_x)
    ax.set_xticklabels(["K", "0.25 Å⁻¹\n→Γ", "0.25 Å⁻¹\n→M"])
    ax.set_ylabel("E (eV)")
    ax.set_title("Monolayer MoS$_2$ — k·p bands near K (spin-resolved,\n"
                 "validity window |q| ≤ 0.25 Å⁻¹)")
    fig.tight_layout()
    fig.savefig(out / "mos2_kp_bands.png", dpi=150)
    print(f"已保存: {out / 'mos2_kp_bands.png'}")

    # --- 有效质量（与文献 Table 3/4 对账）---
    me_up = kp.effective_mass("cb", +1)
    me_dn = kp.effective_mass("cb", -1)
    mh_up = kp.effective_mass("vb", +1)
    mh_dn = kp.effective_mass("vb", -1)
    print(f"有效质量 m_e: {me_up:.3f} / {me_dn:.3f} m0 (文献 0.46/0.43)")
    print(f"           m_h: {mh_up:.3f} / {mh_dn:.3f} m0 (文献 0.54/0.61)")

    # --- 布里渊区与谷位置 ---
    lat = tmd("MoS2").lattice
    kp_path = KPath.for_lattice(lat)
    fig, ax = plot_bz_path(lat, kp_path,
                           title="MoS$_2$ BZ — K/-K valleys")
    fig.savefig(out / "mos2_bz.png", dpi=150)
    print(f"已保存: {out / 'mos2_bz.png'}")


if __name__ == "__main__":
    main()
