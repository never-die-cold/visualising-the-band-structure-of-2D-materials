"""vdW Studio 命令行接口。

用法::

    # 列出全部材料预设
    python -m vdw_studio.cli list

    # 单材料仿真: 能带 + DOS + 带隙分析, 图片与 JSON 保存到 out/
    python -m vdw_studio.cli run mos2_kp --out results/mos2

    # 指定能带窗口与网格
    python -m vdw_studio.cli run phosphorene --out r_bp --emin -4 --emax 4 --mesh 64

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

from .analysis.properties import GapResult, analyze_gap
from .engine.kpath import KPath
from .engine.solver import solve_bands, solve_dos
from .io import write_poscar, write_xyz
from .presets import PRESETS, get_preset, list_presets, run_preset
from .structure.crystal import Crystal
from .visualization import (
    plot_band_structure,
    plot_bz_path,
    plot_dos,
    plot_structure,
)


def _gap_dict(gap: GapResult) -> dict:
    return {
        "gap_eV": gap.gap,
        "direct": gap.direct,
        "vbm_eV": gap.vbm,
        "cbm_eV": gap.cbm,
        "vbm_klabel": gap.vbm_label,
        "cbm_klabel": gap.cbm_label,
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


def cmd_run(args) -> int:
    p = get_preset(args.material)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    print(f"[vdW Studio] 材料: {p.name} (引擎: {p.engine})")
    r = run_preset(args.material)
    structure: Crystal = r["structure"]
    print(f"  结构: {structure.formula_str}, {structure.n_atoms} 原子/胞")

    model = r["model"]

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
    }

    # --- 出图公共部分 ---
    import matplotlib
    matplotlib.use("Agg")
    fig, ax = plot_structure(structure, title=f"{p.name}")
    fig.savefig(out / "structure.png", dpi=150)
    fig, ax = plot_bz_path(structure.lattice, KPath.for_lattice(
        structure.lattice), title=f"{p.name} — BZ")
    fig.savefig(out / "bz.png", dpi=150)

    if p.engine == "kp":
        # k·p 有效模型: K 谷径向切割能带 (|q| ≤ 0.25 Å⁻¹), 无全 BZ DOS
        kp = model
        B = structure.lattice.reciprocal_matrix[:2, :2]
        K = np.array([1 / 3, 1 / 3]) @ B
        M = np.array([1 / 2, 0.0]) @ B
        qmax = 0.25
        u_g = -K / np.linalg.norm(K)
        u_m = (M - K) / np.linalg.norm(M - K)

        def radial(u, n=40):
            return [qmax * t / (n - 1) * u for t in range(n)]

        qpath = radial(u_g) + radial(-u_g)[1:] + radial(u_m, 40)[1:]
        E = np.array([kp.energies(q, +1) for q in qpath])
        x = np.concatenate([[0], np.cumsum(np.linalg.norm(
            np.diff(np.array(qpath), axis=0), axis=1))])
        nh = 40
        import matplotlib
        matplotlib.use("Agg")
        from matplotlib import pyplot as plt
        fig, ax = plt.subplots(figsize=(6, 4.5))
        for b in range(4):
            ax.plot(x, E[:, b], lw=1.6, color="#1f4e79")
        ax.axhline(0, color="#c0392b", lw=0.9, ls="--")
        ax.set_xticks([x[0], x[nh - 1], x[2 * nh - 2]])
        ax.set_xticklabels(["K", "→Γ 0.25 Å⁻¹", "→M 0.25 Å⁻¹"])
        ax.set_ylabel("E (eV)")
        ax.set_title(f"{p.name} — k·p bands (|q| ≤ 0.25 Å⁻¹)")
        fig.tight_layout()
        fig.savefig(out / "bands.png", dpi=150)
        np.savez(out / "bands.npz", x=x, energies=E)

        gaps = [kp.spin_block_energies((0, 0), +1, s)[1]
                - kp.spin_block_energies((0, 0), +1, s)[0] for s in (+1, -1)]
        gap_min = float(min(gaps))
        summary["gap"] = {
            "gap_eV": gap_min, "direct": True, "kpoint": "K",
            "note": "K 点最小（含 SOC）带隙",
        }
        summary["effective_masses_m0"] = {
            "m_e_up": round(kp.effective_mass("cb", +1), 3),
            "m_e_down": round(kp.effective_mass("cb", -1), 3),
            "m_h_up": round(kp.effective_mass("vb", +1), 3),
            "m_h_down": round(kp.effective_mass("vb", -1), 3),
        }
        print(f"  K 点带隙: {gap_min:.3f} eV")
        if p.gap_ref is not None:
            ref, tol = p.gap_ref
            ok = abs(gap_min - ref) <= tol
            summary["gap"]["reference_eV"] = ref
            summary["gap"]["matches_reference"] = bool(ok)
            print(f"  文献参考: {ref:.3f} eV {'✓' if ok else '✗'}")
    else:
        kpath = KPath.for_lattice(model.lattice)
        print("  求解能带…")
        band = solve_bands(model, kpath, n_per_segment=args.npoints)
        print("  计算态密度…")
        dos = solve_dos(model, mesh=(args.mesh, args.mesh), sigma=args.sigma)
        print("  分析带隙…")
        fig, ax = plot_band_structure(band, title=f"{p.name} — bands")
        fig.savefig(out / "bands.png", dpi=150)
        fig, ax = plot_dos(dos, title=f"{p.name} — DOS")
        fig.savefig(out / "dos.png", dpi=150)
        np.savez(out / "bands.npz", x=band.x_axis, energies=band.energies,
                 ticks=band.ticks)
        np.savez(out / "dos.npz", energies=dos.energies, dos=dos.dos)
        gap = analyze_gap(model, kpath, n_valence=p.n_valence)
        summary["gap"] = _gap_dict(gap)
        if gap.gap is None:
            print(f"  带隙: 无 ({p.gap_note})")
        else:
            kind = "直接" if gap.direct else "间接"
            print(f"  带隙: {gap.gap:.3f} eV ({kind}, {gap.cbm_label})")
        if p.gap_ref is not None:
            ref, tol = p.gap_ref
            ok = gap.gap is not None and abs(gap.gap - ref) <= tol
            summary["gap"]["reference_eV"] = ref
            summary["gap"]["matches_reference"] = bool(ok)
            print(f"  文献参考: {ref:.3f} eV {'✓' if ok else '✗'}")

    # --- 数据文件 ---
    write_poscar(structure, str(out / f"{p.structure_key}.POSCAR"),
                 title=p.name)
    (out / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"  输出目录: {out} (structure/bands/dos/bz PNG + npz + "
          "POSCAR + summary.json)")
    return 0


def cmd_run_all(args) -> int:
    base = Path(args.out)
    for key in list_presets():
        p = get_preset(key)
        if p.engine == "sp3d5":
            print(f"[跳过] {key}: sp3d5 引擎带隙待 Nanoskif 约定确认")
            continue
        print()
        ns = argparse.Namespace(material=key, out=str(base / key),
                                npoints=args.npoints, mesh=args.mesh,
                                sigma=args.sigma)
        cmd_run(ns)
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
    p_run.add_argument("--sigma", type=float, default=0.05, help="DOS 展宽 eV")
    p_run.set_defaults(func=cmd_run)

    p_all = sub.add_parser("run-all", help="批量运行全部预设")
    p_all.add_argument("--out", default="results/batch")
    p_all.add_argument("--npoints", type=int, default=40)
    p_all.add_argument("--mesh", type=int, default=48)
    p_all.add_argument("--sigma", type=float, default=0.05)
    p_all.set_defaults(func=cmd_run_all)

    p_exp = sub.add_parser("export", help="导出材料结构 (POSCAR/XYZ)")
    p_exp.add_argument("material", help="材料名 (graphene/MoS2/…)")
    p_exp.add_argument("--out", default="structures")
    p_exp.set_defaults(func=cmd_export)

    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
