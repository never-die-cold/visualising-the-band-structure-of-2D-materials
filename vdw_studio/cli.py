"""vdW Studio 命令行接口。

用法::

    # 列出全部材料预设
    python -m vdw_studio.cli list

    # 单材料仿真: 能带 + DOS + 带隙分析, 图片与 JSON 保存到 out/
    python -m vdw_studio.cli run mos2_kp --out results/mos2

    # 指定能带采样与 DOS 网格
    python -m vdw_studio.cli run phosphorene --out r_bp --npoints 60 --mesh 64

    # 全部预设批量运行
    python -m vdw_studio.cli run-all --out results/batch

    # 结构导出 (POSCAR/XYZ)
    python -m vdw_studio.cli export MoS2 --out structures
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

from .analysis.properties import GapResult
from .io import write_poscar, write_xyz
from .presets import get_preset, list_presets
from .simulation import simulate
from .structure.crystal import Crystal
from .visualization import (
    plot_band_structure,
    plot_bz_path,
    plot_dos,
    plot_structure,
    setup_cjk_fonts,
)


def _gap_dict(gap: GapResult) -> dict:
    return {
        "gap_eV": gap.gap,
        "direct": gap.direct,
        "vbm_eV": gap.vbm,
        "cbm_eV": gap.cbm,
        "vbm_klabel": gap.vbm_label,
        "cbm_klabel": gap.cbm_label,
        "vbm_kfrac": gap.vbm_k.tolist(),
        "cbm_kfrac": gap.cbm_k.tolist(),
        "scope": gap.scope,
        "status": gap.status,
        "raw_gap_eV": gap.raw_gap,
        "search": gap.search_metadata,
        "n_valence": gap.n_valence,
    }


def cmd_list(args) -> int:
    print(f"{'预设键':<14s}{'名称':<18s}{'引擎':<8s}{'带隙参考 (eV)'}")
    print("-" * 62)
    for key in list_presets():
        p = get_preset(key)
        if p.gap_ref is not None:
            ref = f"{p.gap_ref[0]:.3f}"
        elif "无" in p.gap_note and "隙" in p.gap_note:
            ref = "0 (零隙/电场可调)"
        else:
            ref = "待验证"
        print(f"{key:<14s}{p.name:<18s}{p.engine:<8s}{ref}")
    return 0


def _record_summary(args, summary):
    if args.db:
        from .storage import ResultsDB
        ResultsDB(args.db).record(material=summary['material'], engine=summary['engine'],
            formula=summary['formula'], gap_eV=summary['gap'].get('gap_eV'),
            gap_direct=summary['gap'].get('direct'), payload=summary,
            task_key=summary['task_key'])


def cmd_run(args) -> int:
    args._task_key = None
    try:
        return _cmd_run(args)
    except Exception as exc:
        if getattr(args, 'db', None):
            from .storage import ResultsDB
            from .reproducibility import canonical_hash, provenance
            request = {'material': args.material, 'npoints': args.npoints,
                       'mesh': args.mesh, 'gap_mesh': getattr(args, 'gap_mesh', 24),
                       'sigma': args.sigma}
            code = provenance()
            # Model construction may fail before a model-specific task key exists.
            key = getattr(args, '_task_key', None) or canonical_hash({'request': request, 'code': code})
            ResultsDB(args.db).record(args.material, get_preset(args.material).engine,
                payload={'request': request, 'provenance': code}, task_key=key,
                status='failed', error=f'{type(exc).__name__}: {exc}')
        raise


def _cmd_run(args) -> int:
    p = get_preset(args.material)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    print(f"[vdW Studio] 材料: {p.name} (引擎: {p.engine})")
    structure: Crystal = p.make_structure(p.structure_key)
    print(f"  结构: {structure.formula_str}, {structure.n_atoms} 原子/胞")

    from .task_state import SimulationSnapshot, align_structure
    model = p.make_model()
    structure = align_structure(structure, model)
    snapshot = SimulationSnapshot.capture(p, model, structure, n_per_segment=args.npoints,
        dos_mesh=(args.mesh,) * 2, dos_sigma=args.sigma, gap_mesh=(getattr(args, 'gap_mesh', 24),) * 2)
    from .reproducibility import provenance, task_key
    import hashlib
    code = provenance()
    identity = task_key(snapshot.to_dict(), code)
    args._task_key = identity
    args._resumed = False
    summary_file = out / 'summary.json'
    if getattr(args, 'resume', False) and summary_file.exists():
        try:
            previous = json.loads(summary_file.read_text(encoding='utf-8'))
            intact = previous.get('task_key') == identity and bool(previous.get('artifacts'))
            required = {'structure.png', 'bands.png', 'bz.png', 'bands.npz', f'{p.structure_key}.POSCAR'}
            if p.engine != 'kp':
                required.update(['dos.png', 'dos.npz'])
            intact = intact and set(previous.get('artifacts', {})) == required
            intact = intact and task_key(previous['task'], previous['provenance']) == identity
            intact = intact and all(Path(name).name == name and
                hashlib.sha256((out / name).read_bytes()).hexdigest() == digest
                for name, digest in previous.get('artifacts', {}).items())
            if intact:
                _record_summary(args, previous)
                args._resumed = True
                print(f'  恢复已有完整任务: {identity[:12]}')
                return 0
        except (OSError, ValueError, TypeError, KeyError):
            pass

    r = simulate(
        model, n_per_segment=args.npoints,
        dos_mesh=(args.mesh, args.mesh), dos_sigma=args.sigma,
        n_valence=p.n_valence, lattice=structure.lattice,
        gap_mesh=(getattr(args, "gap_mesh", 24),) * 2,
        on_progress=lambda message: print(f"  {message}"))

    summary: dict = {
        "material": p.key,
        "name": p.name,
        "engine": p.engine,
        "formula": structure.formula_str,
        "n_atoms": structure.n_atoms,
        "lattice": {
            "a_A": structure.lattice.parameters()[0],
            "b_A": structure.lattice.parameters()[1],
            "gamma_deg": structure.lattice.parameters()[5],
        },
        "source": p.source,
        "calculation_domain": r["domain"],
        "experimental": r["experimental"],
        "task": snapshot.to_dict(),
        "schema_version": 1,
        "provenance": code,
        "task_key": identity,
        "status": "completed",
        "dos": {"available": r["dos"] is not None},
    }

    # --- 出图公共部分 ---
    import matplotlib
    matplotlib.use("Agg")
    from matplotlib import pyplot as plt
    setup_cjk_fonts()

    def save_figure(fig, filename):
        try:
            fig.savefig(out / filename, dpi=150)
        finally:
            plt.close(fig)

    fig, ax = plot_structure(structure, title=f"{p.name}")
    save_figure(fig, "structure.png")
    fig, ax = plot_bz_path(r["lattice"], r["kpath"], title=f"{p.name} — BZ")
    save_figure(fig, "bz.png")

    band = r["band"]
    title = f"{p.name} — bands"
    if p.engine == "kp":
        title = f"{p.name} — k·p bands (|q| ≤ {r['qmax']} Å$^{{-1}}$)"
        summary["qmax_A_inv"] = r["qmax"]
        summary["effective_masses_m0"] = {
            name: round(mass, 3)
            for name, mass in r["effective_masses_m0"].items()
        }
    fig, ax = plot_band_structure(band, title=title)
    save_figure(fig, "bands.png")
    np.savez(out / "bands.npz", x=band.x_axis, energies=band.energies,
             ticks=band.ticks, tick_labels=band.tick_labels, kpoints=band.kpoints)
    dos = r["dos"]
    if dos is not None:
        summary["dos"].update(scope=dos.scope, spin_degeneracy=dos.spin_degeneracy,
                              expected_states=dos.expected_states, n_bands=dos.n_bands,
                              mesh=list(dos.mesh), sigma_eV=dos.sigma,
                              energy_points=len(dos.energies),
                              energy_window_eV=[float(dos.energies[0]), float(dos.energies[-1])],
                              integral_in_window=float(np.trapezoid(dos.dos, dos.energies)))
        fig, ax = plot_dos(dos, title=f"{p.name} — DOS")
        save_figure(fig, "dos.png")
        np.savez(out / "dos.npz", energies=dos.energies, dos=dos.dos)
    else:
        summary["dos"]["note"] = r["dos_reason"]

    gap = r["gap"]
    summary["gap"] = _gap_dict(gap)
    summary["path_gap"] = _gap_dict(r["path_gap"]) if r["path_gap"] is not None else None
    if p.engine == "kp":
        summary["gap"].update(kpoint="K", note="K 点最小（含 SOC）带隙")
    if gap.gap is None:
        state = "零隙" if gap.status == "zero-gap" else "金属/能带重叠"
        print(f"  带隙: {state} ({gap.scope})")
    else:
        kind = "直接" if gap.direct else "间接"
        print(f"  带隙: {gap.gap:.3f} eV ({kind}, {gap.cbm_label})")
    if gap.search_metadata and not gap.search_metadata.get("converged", True):
        print("  带边搜索尚未收敛，请增加 --gap-mesh 并检查结果")
    if p.gap_ref is not None:
        ref, tol = p.gap_ref
        ok = gap.gap is not None and abs(gap.gap - ref) <= tol
        summary["gap"]["reference_eV"] = ref
        summary["gap"]["matches_reference"] = bool(ok)
        print(f"  文献参考: {ref:.3f} eV {'✓' if ok else '✗'}")

    # --- 数据文件 ---
    write_poscar(structure, str(out / f"{p.structure_key}.POSCAR"),
                 title=p.name)
    artifacts = ['structure.png', 'bands.png', 'bz.png', 'bands.npz', f'{p.structure_key}.POSCAR']
    if dos is not None:
        artifacts.extend(['dos.png', 'dos.npz'])
    summary['artifacts'] = {name: hashlib.sha256((out / name).read_bytes()).hexdigest() for name in artifacts}
    from utils.atomic_io import write_json_atomic
    write_json_atomic(out / 'summary.json', summary)
    plots = "structure/bands/bz" + ("/dos" if dos is not None else "")
    print(f"  输出目录: {out} ({plots} PNG + npz + POSCAR + summary.json)")

    # --- 结果数据库 ---
    _record_summary(args, summary)
    return 0


def cmd_run_all(args) -> int:
    base = Path(args.out)
    base.mkdir(parents=True, exist_ok=True)
    runs = []
    for key in list_presets():
        p = get_preset(key)
        if p.engine == "sp3d5":
            reason = "sp3d5 引擎带隙待 Nanoskif 约定确认"
            print(f"[跳过] {key}: {reason}")
            runs.append({"material": key, "status": "skipped", "reason": reason})
            _save_batch(base, runs)
            continue
        print()
        ns = argparse.Namespace(material=key, out=str(base / key),
                                npoints=args.npoints, mesh=args.mesh,
                                sigma=args.sigma, db=args.db,
                                resume=getattr(args, 'resume', False),
                                gap_mesh=getattr(args, "gap_mesh", 24))
        try:
            cmd_run(ns)
        except Exception as exc:  # 单材料失败不阻止其余材料计算
            error = f"{type(exc).__name__}: {exc}"
            print(f"[失败] {key}: {error}")
            runs.append({"material": key, "status": "failed", "error": error})
        else:
            runs.append({"material": key, "status": "completed",
                         "output": str(base / key), "resumed": ns._resumed})
        _save_batch(base, runs)
    batch = _save_batch(base, runs)
    print(f"批量完成: 成功 {batch['completed']}，失败 {batch['failed']}，"
          f"跳过 {batch['skipped']}；汇总: {base / 'batch_summary.json'}")
    return 1 if batch["failed"] else 0


def _save_batch(base, runs):
    batch = {status: sum(r["status"] == status for r in runs)
             for status in ("completed", "failed", "skipped")}
    batch["runs"] = runs
    from utils.atomic_io import write_json_atomic
    write_json_atomic(base / 'batch_summary.json', batch)
    return batch


def cmd_history(args) -> int:
    from .storage import ResultsDB
    db = ResultsDB(args.db)
    rows = db.history(material=args.material, limit=args.limit)
    if not rows:
        print("(无记录)")
        return 0
    print(f"{'id':<5}{'时间 (UTC)':<21}{'材料':<12}{'引擎':<8}{'带隙 (eV)':<12}直接")
    print("-" * 68)
    for row in rows:
        gap = "—" if row["gap_eV"] is None else f"{row['gap_eV']:.3f}"
        direct = "—" if row["gap_direct"] is None else ("是" if row["gap_direct"] else "否")
        print(f"{row['id']:<5}{row['timestamp']:<21}{row['material']:<12}"
              f"{row['engine']:<8}{gap:<12}{direct}")
    return 0


def cmd_export_records(args):
    from .storage import ResultsDB
    count = ResultsDB(args.db).export_csv(args.out, material=args.material)
    print(f'已导出 {count} 条记录: {args.out}')
    return 0


def cmd_replay(args):
    from .reproducibility import replay_task, provenance
    summary = json.loads(Path(args.record).read_text(encoding='utf-8'))
    if summary.get('schema_version') != 1:
        raise ValueError('Unsupported summary schema for replay')
    current = provenance()
    same_code = summary['provenance']['source_sha256'] == current['source_sha256']
    result = replay_task(summary['task'])
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    np.savez(out / 'bands.npz', energies=result['band'].energies, x=result['band'].x_axis)
    comparison = {'schema_version': 1, 'original_task_key': summary['task_key'],
        'same_source': same_code, 'provenance': current, 'gap': _gap_dict(result['gap']),
        'original_gap': summary['gap']}
    (out / 'replay.json').write_text(json.dumps(comparison, ensure_ascii=False, indent=2), encoding='utf-8')
    print(f'已重建保存的模型与参数；源代码一致: {same_code}；结果: {out}')
    return 0


def cmd_export(args) -> int:
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    from .structure.builders import build
    c = build(args.material)
    write_poscar(c, str(out / f"{c.formula_str}.POSCAR"), title=args.material)
    write_xyz(c, str(out / f"{c.formula_str}.xyz"), title=args.material)
    print(f"[vdW Studio] {args.material}: {c.formula_str}, "
          f"{c.n_atoms} 原子 → {out}/")
    return 0


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(
        prog="vdw-studio",
        description="vdW Studio — 二维半导体仿真平台命令行接口")
    sub = parser.add_subparsers(dest="command", required=True)

    sub.add_parser("list", help="列出全部材料预设").set_defaults(func=cmd_list)

    p_run = sub.add_parser("run", help="运行单材料仿真 (能带+DOS+带隙)")
    p_run.add_argument("material", help="预设键 (见 list)")
    p_run.add_argument("--out", default="results/run", help="输出目录")
    p_run.add_argument("--npoints", type=int, default=40,
                       help="k 路径每段点数")
    p_run.add_argument("--mesh", type=int, default=48, help="DOS 网格边长")
    p_run.add_argument("--gap-mesh", type=int, default=24,
                       help="带边搜索初始网格边长（另用两倍网格检查收敛）")
    p_run.add_argument("--sigma", type=float, default=0.05, help="DOS 展宽 eV")
    p_run.add_argument("--db", default=None,
                       help="结果数据库路径 (指定后自动记录本次仿真)")
    p_run.add_argument('--resume', action='store_true', help='只复用参数/代码一致且完整的输出')
    p_run.set_defaults(func=cmd_run)

    p_all = sub.add_parser("run-all", help="批量运行全部预设")
    p_all.add_argument("--out", default="results/batch")
    p_all.add_argument("--npoints", type=int, default=40)
    p_all.add_argument("--mesh", type=int, default=48)
    p_all.add_argument("--gap-mesh", type=int, default=24)
    p_all.add_argument("--sigma", type=float, default=0.05)
    p_all.add_argument("--db", default=None, help="结果数据库路径")
    p_all.add_argument('--resume', action='store_true')
    p_all.set_defaults(func=cmd_run_all)

    p_exp = sub.add_parser("export", help="导出材料结构 (POSCAR/XYZ)")
    p_exp.add_argument("material", help="材料名 (graphene/MoS2/…)")
    p_exp.add_argument("--out", default="structures")
    p_exp.set_defaults(func=cmd_export)

    p_his = sub.add_parser("history", help="查询仿真历史 (SQLite)")
    p_his.add_argument("--db", default="vdw_results.db", help="数据库路径")
    p_his.add_argument("--material", default=None, help="按材料过滤")
    p_his.add_argument("--limit", type=int, default=20)
    p_his.set_defaults(func=cmd_history)

    p_csv = sub.add_parser('export-records', help='导出完整记录 CSV')
    p_csv.add_argument('--db', required=True)
    p_csv.add_argument('--out', required=True)
    p_csv.add_argument('--material', default=None)
    p_csv.set_defaults(func=cmd_export_records)
    p_replay = sub.add_parser('replay', help='从 summary.json 重建已验证 TB/k·p 任务')
    p_replay.add_argument('record')
    p_replay.add_argument('--out', default='results/replay')
    p_replay.set_defaults(func=cmd_replay)

    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
