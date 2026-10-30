"""示例：多层堆叠引擎（大目标 D）——层间耦合、带边演化与诚实对账。

运行::

    python examples/run_multilayer_mos2.py

产出（保存到 ``examples/output/``）：

- (a) 2H-MoS₂ 单层/双层/三层沿 Γ–K–M 路径的能带（sp³d⁵ + SOC），
  可见层间耦合带来的带倍增与 Γ 点价带（p_z 型）劈裂；
- (b) 蜂窝族引擎的定量验证表：AB 双层谱与手写 BilayerGrapheneModel
  （McCann γ₁ 模型）逐位一致、ABA/ABC 三层 K 点精确谱、
  二聚位耦合的 Harrison d⁻² 标度；
- (c) 多层 MoS₂ 带边量随层数的表，与 Mak 2010 文献值对照——
  **带阻塞声明**（见下）。

**阻塞声明（诚实标注，不得跳过阅读）**：Zahid 2013 sp³d⁵ 表 + 当前
装配约定下，单层 MoS₂ 在一般 k 点**跨 k 带交叠**（金属化；该状态由
``tests/test_multilayer_model.py::TestDocumentedBlockage`` 固化）。
因此本示例的 MoS₂ 带边数值只用于展示引擎与机制，**不得作定量带隙
参考**；与 Mak 2010（1L 直接隙 1.90 eV、2L 间接隙 1.59 eV、体相
间接隙 1.29 eV）的定量对账被 T16-SK 装配约定问题阻塞——出路是
``vdw_studio.fitting`` 的 DFT→TB 自拟合（大目标 A 剩余项）或拿到
Nanoskif 用户手册。蜂窝族部分不受此影响，是该引擎的定量验证锚点。

模型与验证锚点见 ``engine/multilayer.py`` 与 ``structure/multilayer.py``。
"""

import matplotlib
matplotlib.use("Agg")

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np

from vdw_studio.engine.kpath import KPath
from vdw_studio.engine.models import BilayerGrapheneModel
from vdw_studio.engine.multilayer import HoneycombMultilayerModel
from vdw_studio.engine.multilayer import MoS2MultilayerModel
from vdw_studio.structure.builders import graphene, tmd
from vdw_studio.structure.multilayer import (
    stack_2h,
    stack_3r,
    stack_bernald,
)

OUT = Path(__file__).resolve().parent / "output"
OUT.mkdir(exist_ok=True)

K_POINT = np.array([1 / 3, 1 / 3])


def banner(title):
    print("\n" + "=" * 72)
    print(title)
    print("=" * 72)


# ----------------------------------------------------------------------
# (a) 2H-MoS₂ 1L/2L/3L 能带
# ----------------------------------------------------------------------
def mos2_path_bands():
    banner("(a) 2H-MoS₂ 多层能带（sp³d⁵ + SOC；阻塞声明见模块文档）")
    import matplotlib.pyplot as plt

    mono = tmd("MoS2")
    path = KPath.for_lattice(mono.lattice, ["G", "K", "M", "G"])
    n_seg = 60
    kfrac = path.sample_fractional(n_seg)
    _, x_axis, ticks = path.generate(mono.lattice, n_seg)
    from vdw_studio.engine.kpath import DISPLAY_SYMBOLS
    labels = [DISPLAY_SYMBOLS.get(nm, nm) for nm in path.path]

    fig, axes = plt.subplots(1, 3, figsize=(11, 3.6), sharey=True)
    window = (-2.0, 3.5)
    for ax, n_layers in zip(axes, (1, 2, 3)):
        stack = stack_2h(mono, n_layers, 6.15)
        model = MoS2MultilayerModel(stack)
        e = model.bands(kfrac)
        nv = model.n_valence
        # 只画带隙窗口内的带，价带顶对齐 0
        shift = e[:, nv - 1].max()
        for b in range(e.shape[1]):
            band = e[:, b] - shift
            if band.max() < window[0] or band.min() > window[1]:
                continue
            ax.plot(x_axis, band, "b-", lw=0.6)
        ax.axhline(0.0, color="k", lw=0.5, ls="--")
        for t in ticks[1:-1]:
            ax.axvline(t, color="k", lw=0.4)
        ax.set_xticks(ticks)
        ax.set_xticklabels(labels)
        ax.set_ylim(*window)
        ax.set_title(f"{n_layers}L (2H)")
        print(f"  {n_layers}L: n_interlayer_bonds={model.n_interlayer_bonds}, "
              f"网格VBM − K-VB = "
              f"{e[:, nv-1].max() - e[np.argmin(np.abs(kfrac[:, 0] - 1/3) + np.abs(kfrac[:, 1] - 1/3)), nv-1]:+.3f} eV"
              f"（跨 k 带交叠存在于金属化基模型，见阻塞声明）")
    axes[0].set_ylabel("E − E_VBM (eV)")
    fig.suptitle("2H-MoS₂ multilayer bands (sp3d5+SOC; engine demo, "
                 "quantitative gap reconciliation blocked by T16-SK)")
    fig.tight_layout()
    out = OUT / "multilayer_mos2_bands.png"
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"  -> {out.name}")


# ----------------------------------------------------------------------
# (b) 蜂窝族引擎的定量验证（引擎的解析锚点链）
# ----------------------------------------------------------------------
def honeycomb_validation():
    banner("(b) 蜂窝族多层引擎定量验证（不受 T16-SK 阻塞）")
    print("\n[1] AB 双层 vs 手写 BilayerGrapheneModel（取代目标对账）")
    m_gen = HoneycombMultilayerModel(stack_bernald(graphene(), 2, 3.35))
    m_ref = BilayerGrapheneModel()
    ks = np.array([[0, 0], K_POINT, [0.5, 0.0], [0.2, 0.1], [1 / 3, 0.0]])
    diff = np.abs(m_gen.bands(ks) - m_ref.bands(ks)).max()
    print(f"    max |ΔE| over 5 k = {diff:.2e} eV  "
          f"(n_interlayer_hoppings={m_gen.n_interlayer_hoppings}, 应为 1)")

    print("\n[2] 三层 K 点精确谱（f(K)=0 解耦 + γ₁ 链）")
    g1 = 0.4
    aba = HoneycombMultilayerModel(stack_bernald(graphene(), 3, 3.35))
    abc = HoneycombMultilayerModel(stack_3r(graphene(), 3, 3.35))
    e_aba = aba.bands([K_POINT])[0]
    e_abc = abc.bands([K_POINT])[0]
    print(f"    ABA  K: {np.round(e_aba, 4)}  精确: 0×4, ±√2γ₁ = ±{np.sqrt(2)*g1:.4f}")
    print(f"    ABC  K: {np.round(e_abc, 4)}  精确: 0×2, ±γ₁×2 = ±{g1}")
    print("    → 2H 与 3R 在 K 点谱不同：堆叠敏感性（耦合生成器读的是真实坐标）")

    print("\n[3] 二聚位耦合的 Harrison d⁻² 标度")
    for d in (3.35, 4.0, 5.0):
        m = HoneycombMultilayerModel(stack_bernald(graphene(), 2, d))
        e = m.bands([K_POINT])[0]
        expect = 0.4 * (3.35 / d) ** 2
        print(f"    d={d:.2f} Å: K max|E| = {np.abs(e).max():.6f} eV, "
              f"0.4·(3.35/d)² = {expect:.6f} eV")


# ----------------------------------------------------------------------
# (c) 多层 MoS₂ 带边量表 vs Mak 2010（带阻塞声明）
# ----------------------------------------------------------------------
def mos2_table():
    banner("(c) 多层 MoS₂ 带边量 vs Mak 2010（PRL 105, 136805）——阻塞状态")
    print("\n  文献锚点（Mak 2010 原文数值）:")
    print("    1L 直接隙（K，PL A 峰） 1.90 eV；2L 间接隙（PL I 峰）1.59 eV；")
    print("    bulk 间接隙 1.29 eV；间接→直接交叉由 Γ 价带层间杂化驱动。")

    print("\n  模型量（sp³d⁵ + SOC，builder 几何 a=3.1604/dxx=3.170）：")
    print("  N   K隙(eV)  Γ-VB−K-VB(eV)  网格间接隙(eV)  状态")
    mono = tmd("MoS2")
    ks = np.array([[0, 0], K_POINT, [0.25, 0.25], [0.15, 0.15],
                   [0.5, 0.0], [0.2, 0.0], [1 / 3, 0.0]])
    for n_layers in (1, 2, 3, 4):
        stack = stack_2h(mono, n_layers, 6.15)
        model = MoS2MultilayerModel(stack)
        e = model.bands(ks)
        nv = model.n_valence
        k_i = 1
        k_gap = e[k_i, nv] - e[k_i, nv - 1]
        gamma_vb_minus_k_vb = e[0, nv - 1] - e[k_i, nv - 1]
        grid_gap = e[:, nv].min() - e[:, nv - 1].max()
        status = "金属化（阻塞）" if grid_gap < 0.05 else "半导体"
        print(f"  {n_layers}   {k_gap:+.3f}    {gamma_vb_minus_k_vb:+.3f}"
              f"          {grid_gap:+.3f}        {status}")
    print("\n  对账结论：基模型在一般 k 点跨 k 带交叠（T16-SK 装配约定），")
    print("  gap-vs-N 序列 1.8→1.5→1.3→1.2 的定量复现**不可达**；")
    print("  定量对账需先解决装配约定（Nanoskif 手册）或走 vdw_studio.fitting。")
    print("  本示例的定量验证由 (b) 蜂窝族锚点承担。")


if __name__ == "__main__":
    mos2_path_bands()
    honeycomb_validation()
    mos2_table()
    print("\n完成。")
