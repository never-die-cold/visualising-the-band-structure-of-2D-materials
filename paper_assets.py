# -*- coding: utf-8 -*-
"""为课程论文生成真实计算结果与插图"""
import sys, os, time, tracemalloc, json
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

def setup_plot():
    """配置中文字体（优先使用运行时内置字体）"""
    rroot = os.environ.get("DAIMON_RUNTIME_ROOT", "")
    candidates = []
    if rroot:
        candidates += list(Path(rroot).rglob("NotoSansSC*.ttf")) + \
                      list(Path(rroot).rglob("NotoSansSC*.otf"))
    from matplotlib import font_manager
    for f in candidates:
        try:
            font_manager.fontManager.addfont(str(f))
        except Exception:
            pass
    available = {f.name for f in font_manager.fontManager.ttflist}
    for name in ["Noto Sans SC", "Microsoft YaHei", "SimHei"]:
        if name in available:
            plt.rcParams["font.sans-serif"] = [name, "DejaVu Sans"]
            break
    plt.rcParams["axes.unicode_minus"] = False
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

ROOT = Path(r"C:\Users\ASUS\visualising the band structure of 2D materials")
sys.path.insert(0, str(ROOT))
from core.parser import VASPEigenvalParser
from core.band_analyzer import BandAnalyzer
from core.dos_analyzer import DosAnalyzer

FIGDIR = ROOT / "paper_figures"
FIGDIR.mkdir(exist_ok=True)
setup_plot()

results = {}

def load(mat):
    p = VASPEigenvalParser(str(ROOT / "data" / "example" / mat / "EIGENVAL"))
    return p.parse()

# ---------- 1. 解析 + 分析 ----------
for mat, lat in [("graphene", 2.46), ("mos2", 3.16)]:
    data = load(mat)
    an = BandAnalyzer(data, lattice_constant_angstrom=lat)
    gap = an.get_band_gap()
    masses = an.estimate_effective_masses_at_gap()
    results[mat] = {
        "nk": int(data.energies.shape[0]),
        "nb": int(data.energies.shape[1]),
        "ne": int(data.num_electrons),
        "gap": gap, "masses": masses,
        "labels": data.kpoint_labels,
    }
    print(mat, json.dumps(results[mat], ensure_ascii=False, default=str))

# ---------- 2. 性能测试 ----------
def bench(path, sigma=0.1):
    tracemalloc.start()
    t0 = time.perf_counter()
    d = VASPEigenvalParser(str(path)).parse()
    t1 = time.perf_counter()
    dos = DosAnalyzer(d.energies)
    dos.calculate_dos(energy_range=(-8, 8), num_points=1000, sigma=sigma)
    t2 = time.perf_counter()
    _, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    size_mb = Path(path).stat().st_size / 1e6
    return dict(nk=d.energies.shape[0], nb=d.energies.shape[1],
                size_mb=round(size_mb, 3), parse_s=round(t1 - t0, 4),
                dos_s=round(t2 - t1, 4), peak_mb=round(peak / 1e6, 2))

perf = {}
perf["graphene"] = bench(ROOT / "data" / "example" / "graphene" / "EIGENVAL")
perf["mos2"] = bench(ROOT / "data" / "example" / "mos2" / "EIGENVAL")

# 合成扩展样例：把 mos2 的 k 点复制 4 份（测试解析规模扩展性）
_src = (ROOT / "data" / "example" / "mos2" / "EIGENVAL").read_text().splitlines()
_d = load("mos2")
out = list(_src[:6])  # 原文件头部 6 行
for rep in range(4):
    for ik in range(_d.energies.shape[0]):
        kx, ky, kz = _d.kpoints[ik]
        out.append(f"   {kx:.8f}   {ky:.8f}   {kz:.8f}   0.01724138")
        for ib in range(_d.energies.shape[1]):
            out.append(f"     {ib + 1}    {_d.energies[ik, ib]:.10f}    1.000000")
        out.append("")
big = ROOT / "data" / "example" / "mos2_x4_EIGENVAL"
big.write_text("\n".join(out))
perf["mos2_x4"] = bench(big)
print("PERF", json.dumps(perf, ensure_ascii=False))

# ---------- 3. 能带图 ----------
def plot_band(mat, fname, title, mark_gap=False):
    data = load(mat)
    e = data.energies  # 相对费米（示例数据费米=0）
    x = data.kdistances
    fig, ax = plt.subplots(figsize=(6.2, 4.6))
    for b in range(e.shape[1]):
        ax.plot(x, e[:, b], color="#1f5fa8", lw=1.2)
    ax.axhline(0, color="gray", ls="--", lw=0.9)
    ticks, ticklabels = [], []
    for idx, lab in data.kpoint_labels:
        ax.axvline(x[idx], color="lightgray", lw=0.8, zorder=0)
        ticks.append(x[idx]); ticklabels.append(lab)
    ax.set_xticks(ticks); ax.set_xticklabels(ticklabels)
    ax.set_xlim(x[0], x[-1])
    ax.set_ylabel("E − E$_F$ (eV)")
    ax.set_title(title)
    if mark_gap:
        g = results[mat]["gap"]
        vk, ck = g["vbm_k"], g["cbm_k"]
        ax.plot(x[vk], g["vbm"], "o", color="#d62728", ms=7, zorder=5)
        ax.plot(x[ck], g["cbm"], "s", color="#2ca02c", ms=7, zorder=5)
        ax.annotate(f"VBM = {g['vbm']:.2f} eV", (x[vk], g["vbm"]),
                    xytext=(x[vk] + 0.12, g["vbm"] - 1.0),
                    arrowprops=dict(arrowstyle="->", color="#d62728"), color="#d62728")
        ax.annotate(f"CBM = {g['cbm']:.2f} eV\n带隙 = {g['gap']:.2f} eV（{'直接' if g['direct'] else '间接'}）",
                    (x[ck], g["cbm"]), xytext=(x[ck] - 0.15, g["cbm"] + 1.6),
                    ha="right", va="center",
                    arrowprops=dict(arrowstyle="->", color="#2ca02c"), color="#2ca02c")
    fig.savefig(FIGDIR / fname, dpi=220, bbox_inches="tight")
    plt.close(fig)

plot_band("graphene", "fig_graphene_band.png", "石墨烯能带结构（Γ-M-K-Γ）")
plot_band("mos2", "fig_mos2_band.png", "单层 MoS$_2$ 能带结构（Γ-M-K-Γ）", mark_gap=True)

# ---------- 4. DOS 图 ----------
def get_dos(mat, sigma):
    d = load(mat)
    return DosAnalyzer(d.energies).calculate_dos(energy_range=(-8, 8), num_points=1200, sigma=sigma)

# 4a. 总/价带/导带 DOS
dos = get_dos("mos2", 0.1)
fig, ax = plt.subplots(figsize=(6.2, 4.4))
ax.plot(dos.energies, dos.total_dos, color="#333333", lw=1.4, label="总 DOS")
ax.fill_between(dos.energies, dos.vb_dos, color="#1f5fa8", alpha=0.55, label="价带 DOS")
ax.fill_between(dos.energies, dos.cb_dos, color="#d62728", alpha=0.45, label="导带 DOS")
ax.axvline(0, color="gray", ls="--", lw=0.9)
ax.set_xlabel("E − E$_F$ (eV)"); ax.set_ylabel("DOS (a.u.)")
ax.set_title("单层 MoS$_2$ 态密度（σ = 0.10 eV）")
ax.legend()
fig.savefig(FIGDIR / "fig_mos2_dos_vbcb.png", dpi=220, bbox_inches="tight")
plt.close(fig)

# 4b. 不同 σ 对比
fig, ax = plt.subplots(figsize=(6.2, 4.4))
for sigma, c in [(0.05, "#1f5fa8"), (0.15, "#e08600"), (0.30, "#d62728")]:
    d = get_dos("mos2", sigma)
    ax.plot(d.energies, d.total_dos, lw=1.3, color=c, label=f"σ = {sigma:.2f} eV")
ax.axvline(0, color="gray", ls="--", lw=0.9)
ax.set_xlabel("E − E$_F$ (eV)"); ax.set_ylabel("DOS (a.u.)")
ax.set_title("单层 MoS$_2$ 总态密度随展宽参数 σ 的变化")
ax.legend()
fig.savefig(FIGDIR / "fig_mos2_dos_sigma.png", dpi=220, bbox_inches="tight")
plt.close(fig)

# ---------- 5. 系统架构图 ----------
fig, ax = plt.subplots(figsize=(8.6, 5.2))
ax.axis("off"); ax.set_xlim(0, 10); ax.set_ylim(0, 6)

def box(x, y, w, h, text, fc="#eaf1fb", ec="#1f5fa8", fs=10):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.08",
                                fc=fc, ec=ec, lw=1.4))
    ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=fs)

def arrow(x1, y1, x2, y2):
    ax.add_patch(FancyArrowPatch((x1, y1), (x2, y2), arrowstyle="-|>",
                                 mutation_scale=16, color="#444444", lw=1.3))

box(0.2, 2.4, 1.9, 1.2, "VASP 输出文件\nEIGENVAL / KPOINTS", fc="#f5f5f5", ec="#666666")
box(2.9, 3.6, 2.0, 1.1, "数据解析层\nparser.py", fc="#eaf1fb")
box(2.9, 1.4, 2.0, 1.1, "持久化层\nSQLite / JSON", fc="#f0f7ee", ec="#2ca02c")
box(5.7, 3.6, 2.0, 1.1, "计算分析层\n带隙 / 有效质量 / DOS", fc="#eaf1fb")
box(8.0, 2.4, 1.8, 1.2, "界面层 PyQt5\n能带图 / DOS 图\n控制面板", fc="#fdf3e3", ec="#e08600")
box(5.7, 1.4, 2.0, 1.1, "导出与记录\nPNG / SVG / 历史", fc="#f0f7ee", ec="#2ca02c")
arrow(2.1, 3.3, 2.9, 4.1)
arrow(4.9, 4.15, 5.7, 4.15)
arrow(7.7, 4.0, 8.3, 3.6)
arrow(6.7, 3.6, 6.7, 2.5)
arrow(4.9, 1.95, 5.7, 1.95)
arrow(3.9, 3.6, 3.9, 2.5)
arrow(8.4, 2.4, 7.7, 2.2)
ax.set_title("系统总体架构与数据流", fontsize=13)
fig.savefig(FIGDIR / "fig_architecture.png", dpi=220, bbox_inches="tight")
plt.close(fig)

# ---------- 6. 解析流程图 ----------
fig, ax = plt.subplots(figsize=(5.6, 7.2))
ax.axis("off"); ax.set_xlim(0, 6); ax.set_ylim(0, 12)
steps = [
    "开始：读取 EIGENVAL 文本",
    "解析文件头部\n（电子数 / k 点数 / 能带数）",
    "逐 k 点读取坐标与权重",
    "读取该 k 点全部能带本征值",
    "是否还有未读 k 点？",
    "计算 k 点累积路径距离",
    "解析 KPOINTS 高对称点标签",
    "构造 BandData 对象，结束",
]
y = 11.2
coords = []
for i, s in enumerate(steps):
    diamond = (i == 4)
    if diamond:
        from matplotlib.patches import Polygon
        ax.add_patch(Polygon([(3, y + 0.62), (4.6, y), (3, y - 0.62), (1.4, y)],
                             closed=True, fc="#fdf3e3", ec="#e08600", lw=1.4))
        ax.text(3, y, s, ha="center", va="center", fontsize=9)
    else:
        ax.text(3, y, s, ha="center", va="center", fontsize=10,
                bbox=dict(boxstyle="round,pad=0.4", fc="#eaf1fb", ec="#1f5fa8"))
    coords.append(y)
    y -= 1.55
for i in range(len(steps) - 1):
    arrow(3, coords[i] - 0.42, 3, coords[i + 1] + 0.42)
# 循环回边
ax.add_patch(FancyArrowPatch((3.9, coords[4]), (5.3, coords[4]), arrowstyle="-", color="#444"))
ax.add_patch(FancyArrowPatch((5.3, coords[4]), (5.3, coords[2]), arrowstyle="-", color="#444"))
ax.add_patch(FancyArrowPatch((5.3, coords[2]), (4.35, coords[2]), arrowstyle="-|>",
                             mutation_scale=16, color="#444444", lw=1.3))
ax.text(5.35, (coords[4] + coords[2]) / 2, "是", fontsize=10, ha="left")
ax.text(3.15, coords[4] - 0.75, "否", fontsize=10)
ax.set_title("EIGENVAL / KPOINTS 解析流程", fontsize=13)
fig.savefig(FIGDIR / "fig_parser_flow.png", dpi=220, bbox_inches="tight")
plt.close(fig)

# ---------- 7. 汇总结果 ----------
summary = {"analysis": results, "performance": perf}
(ROOT / "paper_results.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2, default=str))
print("DONE")
