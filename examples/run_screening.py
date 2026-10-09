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
from vdw_studio.presets import get_preset
from vdw_studio.task_state import SimulationSnapshot, _json_state
from vdw_studio.reproducibility import provenance, task_key, canonical_hash

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


def screen_bands(db: ResultsDB, *, resume=False, mesh=(24, 24)) -> int:
    print("能带筛选: TB 材料 × 双轴应变 (指数 2；全区带边搜索)")
    failed = 0
    code = provenance()
    for name, (factory, n_occ) in TB_MATERIALS.items():
        for eps in STRAINS:
            identity = canonical_hash({'request': [name, eps, list(mesh), n_occ], 'code': code})
            payload = {'schema_version': 1, 'provenance': code, 'screen': 'bands',
                       'strain': eps, 'strain_exponent': 2., 'mesh': list(mesh)}
            try:
                model = apply_strain(factory(), ex=eps, ey=eps, exponent=2.)
                preset = get_preset(name)
                snapshot = SimulationSnapshot.capture(preset, model,
                    preset.make_structure(preset.structure_key), strain=(eps, eps),
                    n_per_segment=24, gap_mesh=mesh)
                payload['task'] = snapshot.to_dict()
                identity = task_key(payload['task'], code)
                previous = db.find_task(identity) if resume else None
                if previous and previous['status'] == 'completed':
                    print(f"{name} {eps:+.1%}: 恢复已有任务")
                    continue
                gap = analyze_gap(model, n_per_segment=24, n_valence=n_occ, mesh=mesh)
                payload['gap'] = _json_state(gap)
                db.record(name, 'tb', formula=snapshot.make_structure().formula_str,
                    gap_eV=gap.gap, gap_direct=gap.direct, payload=payload, task_key=identity)
                print(f"{name} {eps:+.1%}: gap={gap.gap}, direct={gap.direct}, status={gap.status}")
            except Exception as exc:
                failed += 1
                db.record(name, 'tb', payload=payload, task_key=identity, status='failed',
                          error=f'{type(exc).__name__}: {exc}')
                print(f"{name} {eps:+.1%}: 失败 {exc}")
    return failed


def screen_excitons(db: ResultsDB, *, resume=False) -> int:
    print("激子筛选: 示例 μ=0.25, r₀=30 Å；不代替材料标定")
    failed = 0
    code = provenance()
    for eps in EPSS:
        params = dict(mu_over_m0=.25, eps_env=eps, r0=30.,
                      n_levels=1, n_basis=250, n_quad=5000,
                      r_max=max(1200., 30 * eps / .25 * .529))
        identity = canonical_hash({'exciton': params, 'code': code})
        if resume:
            previous = db.find_task(identity)
            if previous and previous['status'] == 'completed':
                print(f"eps_env={eps}: 恢复已有任务")
                continue
        payload = {'schema_version': 1, 'provenance': code, 'screen': 'excitons',
                   'parameters': params, 'source': 'demonstration parameters, not material calibrated'}
        try:
            result = solve_exciton(**params)
            payload['result'] = _json_state(result)
            db.record('exciton_demo', 'keldysh', payload=payload, task_key=identity)
            print(f"eps_env={eps}: binding_1s={result.binding_1s}, status={result.status}")
        except Exception as exc:
            failed += 1
            db.record('exciton_demo', 'keldysh', payload=payload, task_key=identity,
                      status='failed', error=f'{type(exc).__name__}: {exc}')
            print(f"eps_env={eps}: 失败 {exc}")
    return failed


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--db", default="results/screening.db",
                        help="结果数据库路径")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--csv", default=None)
    args = parser.parse_args()
    db = ResultsDB(args.db)
    failed = screen_bands(db, resume=args.resume) + screen_excitons(db, resume=args.resume)
    if args.csv:
        db.export_csv(args.csv)
    print(f"全部结果已写入数据库: {args.db}")
    print(f"查询: python -m vdw_studio.cli history --db {args.db}")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
