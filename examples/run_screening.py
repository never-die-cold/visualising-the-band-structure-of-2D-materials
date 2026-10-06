"""示例：二维材料高通量筛选 — 能带 × 应变 × 介电环境 → 结果数据库。

运行::

    python examples/run_screening.py [--db screening.db]

演示 ROADMAP Phase 6 的高通量工作流：

1. **能带筛选**：四种 TB 材料 × 双轴应变 {−2%, 0, +2%} → 带隙表
   （应变工程：带隙可调性一览）；
2. **激子筛选**：Keldysh 屏蔽势 × 环境介电常数 {1, 2.5, 4} →
   束缚能表（介电工程：衬底选择压制/增强激子束缚）；
3. 全部结果写入 SQLite 数据库（`ResultsDB`），可随时查询/导出。

⚠ 参数说明：激子部分的 μ/r₀ 为示例值（材料精确值待文献标定，
见 docs/REFERENCES.md）；应变效应基于键长标度律近似。
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import argparse

from vdw_studio.analysis import analyze_gap, solve_exciton
from vdw_studio.engine.models import (
    BilayerGrapheneModel,
    BuckledHoneycombModel,
    HoneycombModel,
    PhosphoreneRudenko,
)
from vdw_studio.engine.strain import apply_strain
from vdw_studio.storage import ResultsDB

# 能带筛选: 材料 → (模型工厂, 占据带数)
TB_MATERIALS = {
    "graphene": (lambda: HoneycombModel(a=2.46, t=-2.7), 1),
    "hbn": (lambda: HoneycombModel(a=2.504, t=-2.7, delta_onsite=3.5), 1),
    "silicene": (lambda: BuckledHoneycombModel(a=3.86, buckling=0.44,
                                               t=-1.6), 1),
    "phosphorene": (lambda: PhosphoreneRudenko(), 2),
}
STRAINS = (-0.02, 0.0, 0.02)
EPSS = (1.0, 2.5, 4.0)


def screen_bands(db: ResultsDB) -> None:
    print("=" * 66)
    print("能带筛选: TB 材料 × 双轴应变 (键长标度律, n=2)")
    print("=" * 66)
    print(f"{'材料':<14}{'ε=−2%':>10}{'ε=0':>10}{'ε=+2%':>10}   可调性")
    for name, (factory, n_occ) in TB_MATERIALS.items():
        gaps = []
        for eps in STRAINS:
            model = apply_strain(factory(), ex=eps, ey=eps, exponent=2.0)
            g = analyze_gap(model, n_per_segment=24, n_valence=n_occ).gap
            gaps.append(g)
        cells = [(f"{g:.3f}" if g is not None else "0") for g in gaps]
        tunable = ("金属→隙" if gaps[0] is None and gaps[-1] is not None
                   else "隙→金属" if gaps[0] is not None and gaps[-1] is None
                   else "—" if gaps[0] is None
                   else f"Δ={abs(gaps[-1]-gaps[0]):.2f} eV")
        print(f"{name:<14}{cells[0]:>10}{cells[1]:>10}{cells[2]:>10}   {tunable}")
        for eps, g in zip(STRAINS, gaps):
            db.record(material=name, engine="tb",
                      formula=name, gap_eV=g,
                      gap_direct=(g is not None),
                      payload={"strain": eps, "screen": "bands"})
    print()


def screen_excitons(db: ResultsDB) -> None:
    print("=" * 66)
    print("激子筛选: Keldysh 势 × 环境介电常数 (μ=0.25, r₀=30 Å 示例参数)")
    print("=" * 66)
    print(f"{'ε_env':<10}{'E_b 1s (eV)':>14}{'相对 (vs ε=1)':>16}")
    base = None
    for eps in EPSS:
        r = solve_exciton(mu_over_m0=0.25, eps_env=eps, r0=30.0,
                          n_levels=1, n_basis=250, n_quad=5000)
        if base is None:
            base = r.binding_1s
        rel = r.binding_1s / base * 100
        print(f"{eps:<10.1f}{r.binding_1s:>14.4f}{rel:>15.1f}%")
        db.record(material="exciton_demo", engine="keldysh",
                  formula="—", gap_eV=None, gap_direct=None,
                  payload={"screen": "excitons", "eps_env": eps,
                           "r0_A": 30.0, "mu_over_m0": 0.25,
                           "binding_1s_eV": r.binding_1s})
    print("  → 介电工程: 高 ε 衬底 (hBN 封装/高介电栅介) 压制激子束缚\n")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--db", default="screening.db",
                        help="结果数据库路径")
    args = parser.parse_args()
    db = ResultsDB(args.db)
    screen_bands(db)
    screen_excitons(db)
    print(f"全部结果已写入数据库: {args.db}")
    print(f"查询: python -m vdw_studio.cli history --db {args.db}")


if __name__ == "__main__":
    main()
