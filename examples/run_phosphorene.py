"""示例：黑磷烯四带 TB 仿真 — 各向异性能带 / DOS / 结构图。

运行::

    python examples/run_phosphorene.py

输出保存到 ``examples/output/``。模型出处：
Rudenko & Katsnelson, PRB 89, 201408(R) (2014)。
"""

import matplotlib
matplotlib.use("Agg")

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


from pathlib import Path

from vdw_studio.analysis import analyze_gap, principal_masses
from vdw_studio.engine.kpath import KPath
from vdw_studio.engine.models import PhosphoreneRudenko
from vdw_studio.engine.solver import solve_bands, solve_dos
from vdw_studio.presets import run_preset
from vdw_studio.structure.builders import phosphorene
from vdw_studio.visualization import (
    plot_band_structure,
    plot_bz_path,
    plot_dos,
    plot_structure,
)


def main() -> None:
    out = Path(__file__).parent / "output"
    out.mkdir(exist_ok=True)

    r = run_preset("phosphorene")
    model: PhosphoreneRudenko = r["model"]
    gap = r["gap"]
    print(f"材料: 黑磷烯 (四带 TB, Rudenko 2014)")
    print(f"Γ 点直接带隙: {gap.gap:.3f} eV (文献模型值 1.52 eV, "
          f"匹配: {r['gap_matches_ref']})")

    # --- 能带 ---
    bs = solve_bands(model, KPath.for_lattice(model.lattice),
                     n_per_segment=60)
    fig, ax = plot_band_structure(bs, emin=-4, emax=4,
                                  title="Monolayer black phosphorus — "
                                        "4-band TB (Rudenko 2014)")
    fig.savefig(out / "phosphorene_bands.png", dpi=150)
    print(f"已保存: {out / 'phosphorene_bands.png'}")

    # --- DOS ---
    dos = solve_dos(model, mesh=(48, 48), sigma=0.05, n_points=900)
    fig, ax = plot_dos(dos, emin=-4, emax=4)
    fig.savefig(out / "phosphorene_dos.png", dpi=150)
    print(f"已保存: {out / 'phosphorene_dos.png'}")

    # --- 布里渊区 ---
    fig, ax = plot_bz_path(model.lattice, KPath.for_lattice(model.lattice))
    fig.savefig(out / "phosphorene_bz.png", dpi=150)
    print(f"已保存: {out / 'phosphorene_bz.png'}")

    # --- 各向异性主质量（黑磷烯的招牌性质）---
    m1, m2, theta = principal_masses(model, (0, 0), band=2, dk=1e-4)
    light, heavy = (m1, m2) if m1 < m2 else (m2, m1)
    print(f"导带主质量: {light:.3f} / {heavy:.3f} m0 "
          f"(各向异性比 {heavy/light:.1f}×；文献 ~0.17 / ~0.85)")

    # --- 结构图 ---
    fig, ax = plot_structure(phosphorene(), supercell=(3, 2, 1),
                             title="Monolayer black phosphorus (P)")
    fig.savefig(out / "phosphorene_structure.png", dpi=150)
    print(f"已保存: {out / 'phosphorene_structure.png'}")


if __name__ == "__main__":
    main()
