"""多层堆叠引擎：层间耦合的 Harrison 标度自动生成 + 多层 TB 模型。

大目标 D 的引擎侧。从 :mod:`vdw_studio.structure.multilayer` 产出的
**实际原子坐标**出发，按"vdW 间距规则 + Harrison 键长标度"自动生成
层间 hopping，取代手写的 :class:`~vdw_studio.engine.models.BilayerGrapheneModel`
与 ``builders.tmd_bilayer`` 的离散对应物。

**Harrison 标度（REFERENCES.md [13]）**：Slater–Koster 双中心积分随键长
按幂律变化::

    V_ch(d) = V_ch(d₀) · (d₀/d)^η_ch

η = 2 是 Harrison 普适规则 V ∝ ħ²/(m d²) 的标度指数（对所有通道）；
相邻代码（QTCAD/NEMO-3D）同形式。参考积分 V_ch(d₀) 取单层自身的
SK 拟合表（d₀ = 该表在单层内被消费的键长），因此**层间耦合没有新的
自由拟合参数**——这是本生成器的核心声明。

两个多层模型：

- :class:`HoneycombMultilayerModel` —— 蜂窝单层（石墨烯/hBN）的 p_z
  位点模型，正交基。层间参考耦合用 McCann 双层石墨烯的 γ₁ = 0.4 eV
  （垂直二聚位、d₀ = 3.35 Å）标定；p-p 两通道按 Harrison 普适比值
  η_ppπ/η_ppσ = −0.81/3.24 分配。解析锚点（K 点层内 f(K)=0 解耦后的
  γ₁ 链精确谱）：AB 双层 {0, 0, ±γ₁}（与 BilayerGrapheneModel 逐位
  一致）；ABA 三层 {0×4, ±√2γ₁}；ABC 三层 {0×2, ±γ₁×2}——
  2H 与 3R 在 K 点谱不同，堆叠敏感性由此验证。
- :class:`MoS2MultilayerModel` —— sp³d⁵ 非正交 SK（Zahid 2013 表）
  的多层推广。层内键表与单层相同；层间键按原子对种类
  （S–Mo / S–S / Mo–Mo）从 Zahid 表出发做 Harrison 标度，层间重叠
  近似为 0（块对角保证 S 正定）。**继承 T16-SK 的已知问题**：
  单层 K 点带隙偏低（~0.98 eV vs 文献 1.805），多层结果只能作
  定性/机制用途——诚实标注，不得当定量带隙参考。
"""

from __future__ import annotations

import copy
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np
import scipy.linalg

from ..numerics import finite_scalar
from ..structure.crystal import Crystal
from ..structure.multilayer import MultilayerStack
from .models import Hopping, TightBindingModel
from .zahid_mos2 import (
    DEFAULT_MIRROR_IMAGES,
    DEFAULT_TABLES,
    ORBITALS,
    _ONSITE_KEY,
    onsite_soc,
    sk_element,
)

# Harrison 普适标度指数：V ∝ ħ²/(m d²)，对所有 SK 通道（REFERENCES.md [13]）。
HARRISON_EXPONENT = 2.0

# Harrison 普适矩阵元常数（无量纲，V = η·ħ²/(m d²)）——只使用其**比值**
# （p-p 两通道的相对权重），绝对值不进入本模块（参考积分由单层表标定）。
HARRISON_ETA = {"ppσ": 3.24, "ppπ": -0.81}

# 元素对（排序后）→ Zahid 异种原子积分表前缀
_PAIR_TABLE = {"MoS": "SMo"}
_KIND_OF_PAIR = {"SS": "SS", "MoMo": "MoMo", "MoS": "SMo"}


@dataclass(frozen=True)
class InterlayerBond:
    """一条层间近邻：原子 i 到原子 j 的镜像（两层不同）。"""

    i: int
    j: int
    dfrac: Tuple[float, float, float]   # j 镜像相对 i 的分数位移
    distance: float
    pair: str                           # "SS" / "MoS" / "MoMo"（排序元素对）
    layer_pair: Tuple[int, int]


def harrison_scale(v0: float, d0: float, d: float,
                   exponent: float = HARRISON_EXPONENT) -> float:
    """Harrison 键长标度：``v0·(d0/d)^exponent``。

    v0 为参考键长 d₀ 处的积分值；η = 2 为 Harrison 普适规则
    （V ∝ ħ²/(m d²)，REFERENCES.md [13]）。
    """
    finite_scalar(v0, 'v0')
    finite_scalar(d0, 'd0')
    finite_scalar(d, 'd')
    finite_scalar(exponent, 'exponent')
    if d0 <= 0 or d <= 0:
        raise ValueError("参考键长与目标键长必须为正")
    return float(v0 * (d0 / d) ** exponent)


def scaled_integrals(table: Dict[str, float], d0: float, d: float,
                     exponents: Optional[Dict[str, float]] = None
                     ) -> Dict[str, float]:
    """整张 SK 通道表按 Harrison 标度缩放（逐通道可指定指数）。"""
    exponents = exponents or {}
    return {name: harrison_scale(v0, d0, d, exponents.get(name,
                                                              HARRISON_EXPONENT))
            for name, v0 in table.items()}


def _layer_of_atom(stack: MultilayerStack) -> np.ndarray:
    layer_of = np.empty(stack.crystal.n_atoms, dtype=int)
    for li, (start, stop) in enumerate(stack.layer_slices):
        layer_of[start:stop] = li
    return layer_of


def _enumerate_interlayer_pairs(stack: MultilayerStack, cutoff: float
                                ) -> List[InterlayerBond]:
    """枚举跨层原子对（含周期镜像），距离 ≤ cutoff，每条无向边一次。

    去重约定：只从编号较小的原子一侧登记（跨层对 i≠j，故 i<j 唯一）。
    """
    crystal = stack.crystal
    symbols = crystal.symbols
    layer_of = _layer_of_atom(stack)
    bonds: List[InterlayerBond] = []
    for i in range(crystal.n_atoms):
        for nb in crystal.neighbors(crystal.frac_coords[i], cutoff):
            j = nb.index
            if layer_of[j] == layer_of[i] or j < i:
                continue
            dfrac = crystal.frac_coords[j] + np.asarray(nb.image, float) \
                - crystal.frac_coords[i]
            pair = "".join(sorted([symbols[i], symbols[j]]))
            bonds.append(InterlayerBond(
                i=int(i), j=int(j), dfrac=tuple(float(x) for x in dfrac),
                distance=nb.distance, pair=pair,
                layer_pair=(int(layer_of[i]), int(layer_of[j]))))
    return bonds


def _reference_distances(monolayer: Crystal) -> Dict[str, float]:
    """Zahid 表被消费的参考键长 d₀（排序元素对 → Å）。

    MoS：Mo–S 共价键长（SMo 表的消费距离）；SS / MoMo：**同亚层**
    六方近邻的面内距离 a（Zahid 的 SS/MoMo 镜像键表只在距离 a 处
    使用）。注意不是任意同元素对的最小距离——层内上下两个 S 相距
    dxx ≈ 3.13 Å，但单层模型在它们之间**没有**键，那张表不在该距离
    上被消费，不能作 d₀。
    """
    cart = monolayer.cart_coords
    symbols = monolayer.symbols
    a = float(np.linalg.norm(monolayer.lattice.a1[:2]))
    d0 = {"SS": a, "MoMo": a}
    best = None
    for i in range(monolayer.n_atoms):
        for j in range(i + 1, monolayer.n_atoms):
            pair = "".join(sorted([symbols[i], symbols[j]]))
            if pair != "MoS":
                continue
            dist = float(np.linalg.norm(cart[j] - cart[i]))
            if best is None or dist < best:
                best = dist
    if best is None:
        raise ValueError("单层中找不到 Mo–S 对")
    d0["MoS"] = best
    return d0


def _bond_rhat(crystal: Crystal, dfrac) -> np.ndarray:
    """键方向单位矢量（笛卡尔，沿分数位移 dfrac 方向）。"""
    vec = crystal.lattice.frac_to_cart(np.asarray(dfrac, float))
    return vec / np.linalg.norm(vec)


class HoneycombMultilayerModel(TightBindingModel):
    """蜂窝单层（石墨烯/hBN）N 层堆叠的 p_z 位点模型（正交基）。

    位点 = 堆叠晶体的全部原子（每层 2 个 p_z 轨道）；层内最近邻
    hopping ``t``（石墨烯 −2.7 eV，Reich 2002）；层间 hopping 由
    Harrison 标度从结构自动生成：

    - 参考：垂直二聚位耦合 γ₁ = ``gamma1``（默认 0.4 eV，McCann
      双层石墨烯标准值）在参考间距 ``d0 = 3.35 Å``（石墨层间距）；
    - p-p 两通道 SK 表取 V_ppσ(d₀) = γ₁（垂直对的 σ 通道），
      V_ppπ(d₀) = γ₁·η_ppπ/η_ppσ = −γ₁/4（Harrison 普适比值）；
    - 标度指数默认 η = 2（Harrison 普适 d⁻² 规则）。

    ``cutoff`` 限制层间近邻枚举半径。默认 ``interlayer + 0.25`` 只保留
    最近层间壳层——对 Bernal 双层即"仅二聚位"的 γ₁-only 最小模型，
    K 点谱 {0, 0, ±γ₁}（解析锚点，与 BilayerGrapheneModel 一致）；
    调大 cutoff 会纳入次近层间对（等效 γ₄ 等），锚点随之失效，
    属于有意的模型扩展而非默认行为。

    ``element_onsite``：按元素符号给在位能（hBN 用 B/N 差打开带隙；
    石墨烯全 0）。
    """

    def __init__(self, stack: MultilayerStack, t: float = -2.7,
                 gamma1: float = 0.4, d0: float = 3.35,
                 cutoff: Optional[float] = None,
                 exponent: float = HARRISON_EXPONENT,
                 element_onsite: Optional[Dict[str, float]] = None,
                 name: Optional[str] = None) -> None:
        crystal = stack.crystal
        if crystal.n_atoms != 2 * stack.n_layers:
            raise ValueError(
                f"蜂窝多层模型要求每层 2 个位点，堆叠给出 "
                f"{crystal.n_atoms} 原子 / {stack.n_layers} 层")
        finite_scalar(t, 't')
        finite_scalar(gamma1, 'gamma1')
        finite_scalar(d0, 'd0')
        finite_scalar(exponent, 'exponent')
        if cutoff is None:
            cutoff = stack.interlayer + 0.25
        finite_scalar(cutoff, 'cutoff')
        if cutoff <= 0:
            raise ValueError("cutoff 必须为正")

        onsite_map = dict(element_onsite or {})
        onsite = [onsite_map.get(sym, 0.0) for sym in crystal.symbols]

        hoppings: List[Hopping] = []
        hoppings.extend(_honeycomb_intralayer_hoppings(stack, t))
        table0 = {"ppσ": gamma1,
                  "ppπ": gamma1 * HARRISON_ETA["ppπ"] / HARRISON_ETA["ppσ"]}
        exponents = {"ppσ": exponent, "ppπ": exponent}
        n_inter = 0
        for bond in _enumerate_interlayer_pairs(stack, cutoff):
            rhat = _bond_rhat(crystal, bond.dfrac)
            table_d = scaled_integrals(table0, d0, bond.distance, exponents)
            value = sk_element("pz", "pz", rhat, table_d)
            if abs(value) < 1e-12:
                continue
            hoppings.append(Hopping(bond.i, bond.j, bond.dfrac, value))
            n_inter += 1

        super().__init__(crystal.lattice, n_sites=crystal.n_atoms,
                         site_symbols=list(crystal.symbols),
                         onsite=onsite, hoppings=hoppings,
                         name=name or f"honeycomb {stack.stacking} "
                                      f"({stack.n_layers}L, Harrison)")
        self.stack = stack
        self.t = t
        self.gamma1 = gamma1
        self.d0 = d0
        self.cutoff = float(cutoff)
        self.exponent = exponent
        self.n_interlayer_hoppings = n_inter


def _honeycomb_intralayer_hoppings(stack: MultilayerStack,
                                   t: float) -> List[Hopping]:
    """每层蜂窝最近邻 hopping（从结构枚举，每条无向边一次）。"""
    crystal = stack.crystal
    a = float(np.linalg.norm(crystal.lattice.a1[:2]))
    nn_cut = a / np.sqrt(3) * 1.1          # NN = a/√3；第二近邻 = a，安全间隔
    hoppings: List[Hopping] = []
    for li, (start, stop) in enumerate(stack.layer_slices):
        count = [0] * (stop - start)
        for i in range(start, stop):
            for nb in crystal.neighbors(crystal.frac_coords[i], nn_cut):
                j = nb.index
                if nb.distance < 1e-6:       # 自身像（image=0）不计配位
                    continue
                if not (start <= j < stop) or j < i:
                    continue
                dfrac = crystal.frac_coords[j] + np.asarray(nb.image, float) \
                    - crystal.frac_coords[i]
                hoppings.append(Hopping(i, j, tuple(float(x) for x in dfrac),
                                        t))
                count[i - start] += 1
                count[j - start] += 1
        if any(c != 3 for c in count):
            raise ValueError(
                f"层 {li} 的蜂窝最近邻配位数不是 3：{count}——结构或容差有误")
    return hoppings


def _private_tables(integrals: Optional[Dict[str, dict]]) -> Dict[str, dict]:
    """Zahid 表的私有深拷贝（拟合循环可就地改值不污染模块常量——踩坑 ⑭）。"""
    tables = copy.deepcopy(DEFAULT_TABLES)
    if integrals:
        unknown = set(integrals) - set(tables)
        if unknown:
            raise ValueError(f"未知积分表 {sorted(unknown)}，可选: {sorted(tables)}")
        for name, table in integrals.items():
            extra = set(table) - set(tables[name])
            if extra:
                raise ValueError(f"表 {name} 含未知通道 {sorted(extra)}")
            tables[name] = dict(table)
    return tables


class MoS2MultilayerModel:
    """MoS₂ sp³d⁵ 非正交 SK 多层模型（Zahid 2013 表 + Harrison 层间标度）。

    每层 = ZahidMoS2Model 的键表（三棱柱 6×Mo–S + 同亚层 S–S/Mo–Mo）；
    层间键按原子对种类从对应的 Zahid 积分表出发做 Harrison 标度::

        V_ch(d) = V_ch(d₀_kind) · (d₀_kind/d)^η

    d₀_kind 取该表在**单层内**被消费的键长（Mo–S 共价键 ≈2.41 Å；
    S–S / Mo–Mo：同亚层面内距离 a）——层间耦合不含新拟合参数。
    层间重叠积分近似为 0（vdW 间距下为小量；块对角保证 S(k) 正定）。
    on-site SOC 逐层保留。

    已知问题（诚实声明）：单层 Zahid 模型的 K 点带隙尚未与文献对账
    （T16-SK，见 ``zahid_mos2`` 模块文档）；本多层模型**继承**该偏差，
    带隙绝对值不得作定量参考。层间耦合的机制性结论（Γ 价带杂化、
    带隙随层数变化的方向）可用，定量对账待 T16-SK 解决或走
    ``vdw_studio.fitting`` 自拟合参数。

    参数:
        stack: ``structure.multilayer`` 的 1H-TMD 堆叠（每层 3 原子）。
        cutoff: 层间近邻枚举半径 (Å)，默认 interlayer + 0.5
            （含跨层 Mo–Mo ≈ 6.41 Å 与次近 S–S ≈ 4.8 Å，排除 S–Mo ≈ 7.7 Å）。
        exponents: 逐通道标度指数覆盖（默认全部 η = 2）。
        soc / mirror_images / conjugate_fill / integrals: 同 ZahidMoS2Model。
    """

    def __init__(self, stack: MultilayerStack, soc: bool = True,
                 cutoff: Optional[float] = None,
                 exponents: Optional[Dict[str, float]] = None,
                 mirror_images=None, conjugate_fill: bool = True,
                 integrals: Optional[Dict[str, dict]] = None,
                 name: Optional[str] = None) -> None:
        if stack.monolayer.n_atoms != 3:
            raise ValueError("MoS2 多层模型要求每层 3 原子（1H 单层）")
        self.stack = stack
        self.lattice = stack.crystal.lattice
        self.n_layers = stack.n_layers
        self.n_atoms = stack.crystal.n_atoms
        self.n_basis = 18 * self.n_atoms
        # 中性填充：每层 18 电子（Mo d⁵s¹ + 2×S s²p⁴）
        self.n_valence = 18 * self.n_layers
        self.soc = bool(soc)
        self.name = name or f"MoS2 sp3d5 {stack.stacking} ({self.n_layers}L)"
        self.exponents = dict(exponents or {})
        self.mirror_images = (tuple(DEFAULT_MIRROR_IMAGES) if mirror_images is None
                              else tuple((int(dx), int(dy)) for dx, dy in mirror_images))
        self.conjugate_fill = bool(conjugate_fill)
        self.tables = _private_tables(integrals)
        if cutoff is None:
            cutoff = stack.interlayer + 0.5
        finite_scalar(cutoff, 'cutoff')
        self.cutoff = float(cutoff)
        self.d0_by_kind = _reference_distances(stack.monolayer)
        self.bonds = self._build_bonds()
        self._h_entries, self._s_entries = self._prepare_entries()

    # ------------------------------------------------------------------
    def _build_bonds(self) -> List[Tuple[int, int, np.ndarray, str, bool]]:
        """键表: (i, j, dfrac, kind, interlayer)。

        kind 与 ZahidMoS2Model 同义："SMo"（i=S, j=Mo）、"SS"、"MoMo"。
        层间键的 (i, j) 顺序不保证 S 在前——装配时按元素角色重排。
        """
        crystal = self.stack.crystal
        mono = self.stack.monolayer
        symbols = crystal.symbols
        bonds: List[Tuple[int, int, np.ndarray, str, bool]] = []
        d0_bond = self.d0_by_kind["MoS"]

        # 层内：每层复制单层键表
        for li, (start, _) in enumerate(self.stack.layer_slices):
            # 该层若被 180° 翻转，其局部镜像规范随之转 C₂：同亚层镜像矢量
            # 必须取 (−dx, −dy)。沿用全局方向表会让翻转层的 H 不是下层
            # H 的 C₂ 规范拷贝——解耦极限下一般 k 点两层谱不同（实测差
            # 0.37 eV），Γ/K/M 对称点上恰好看不出来。
            rot180 = (self.stack.steps is not None
                      and bool(self.stack.steps[li].rot180))
            sgn = -1.0 if rot180 else 1.0
            s_local = [k for k in range(mono.n_atoms) if mono.symbols[k] == "S"]
            mo_local = [k for k in range(mono.n_atoms) if mono.symbols[k] == "Mo"]
            # S–Mo 三棱柱六键：对每个 S 枚举同层 Mo 的面内镜像
            for k_s in s_local:
                found = 0
                for nx in range(-2, 3):
                    for ny in range(-2, 3):
                        for k_mo in mo_local:
                            dvec = (crystal.frac_coords[start + k_mo]
                                    + np.array([nx, ny, 0.0])
                                    - crystal.frac_coords[start + k_s])
                            dist = float(np.linalg.norm(
                                crystal.lattice.frac_to_cart(dvec)))
                            if abs(dist - d0_bond) < 0.05 * d0_bond:
                                bonds.append((start + k_s, start + k_mo,
                                              tuple(float(x) for x in dvec),
                                              "SMo", False))
                                found += 1
                if found != 3:
                    raise ValueError(
                        f"S(层内 {k_s}) 的 Mo 近邻数 = {found}（应为 3）")
            # 同亚层镜像键：原子到自身镜像（距离 a，与单层同装配约定；
            # 镜像方向随层旋转，见上）
            for k in range(mono.n_atoms):
                pair = mono.symbols[k] * 2          # "SS" / "MoMo"
                for dx, dy in self.mirror_images:
                    dvec = np.array([sgn * float(dx), sgn * float(dy), 0.0])
                    dist = float(np.linalg.norm(
                        crystal.lattice.frac_to_cart(dvec)))
                    if abs(dist - self.d0_by_kind[pair]) < 0.05 * self.d0_by_kind[pair]:
                        bonds.append((start + k, start + k,
                                      tuple(float(x) for x in dvec), pair,
                                      False))

        # 层间：距离 ≤ cutoff 的跨层原子对（每条无向边一次）
        layer_of = _layer_of_atom(self.stack)
        n_inter = 0
        for i in range(crystal.n_atoms):
            for nb in crystal.neighbors(crystal.frac_coords[i], self.cutoff):
                j = nb.index
                if layer_of[j] == layer_of[i] or j < i:
                    continue
                dfrac = crystal.frac_coords[j] + np.asarray(nb.image, float) \
                    - crystal.frac_coords[i]
                pair = "".join(sorted([symbols[i], symbols[j]]))
                bonds.append((i, j, tuple(float(x) for x in dfrac),
                              _KIND_OF_PAIR[pair], True))
                n_inter += 1
        self.n_interlayer_bonds = n_inter
        return bonds

    # ------------------------------------------------------------------
    def _prepare_entries(self):
        """把键表转成 (行原子, 列原子, 9×9 块, 共轭标记) 的装配条目。

        SK 块只依赖键方向（与 k 无关），预计算一次；k 循环里只做相位与
        转置共轭——装配从逐元素 Python 循环变成两次切片加法。
        """
        h_entries: List[Tuple[int, int, np.ndarray, bool, np.ndarray]] = []
        s_entries: List[Tuple[int, int, np.ndarray, bool, np.ndarray]] = []
        symbols = self.stack.crystal.symbols
        for (i, j, dfrac, kind, inter) in self.bonds:
            dfrac = np.asarray(dfrac, float)
            r_vec = self.lattice.frac_to_cart(dfrac)
            dist = float(np.linalg.norm(r_vec))
            rhat = r_vec / dist
            if kind == "SMo":
                # 统一 (S, Mo) 角色：行名按 (S 轨道, Mo 轨道) 消费
                if symbols[i] == "S":
                    s_atom, m_atom = i, j
                else:
                    s_atom, m_atom = j, i
                    rhat = -rhat
                if inter:
                    table_h = scaled_integrals(
                        self.tables["SMo_H"], self.d0_by_kind["MoS"], dist,
                        self.exponents)
                else:
                    table_h = self.tables["SMo_H"]
                v = _sk_block(rhat, table_h)
                h_entries.append((s_atom, m_atom, v, False, dfrac))
                h_entries.append((m_atom, s_atom, v, True, dfrac))
                if not inter:
                    vs = _sk_block(rhat, self.tables["SMo_S"])
                    s_entries.append((s_atom, m_atom, vs, False, dfrac))
                    s_entries.append((m_atom, s_atom, vs, True, dfrac))
            else:
                pair = "SS" if kind == "SS" else "MoMo"
                if inter:
                    table_h = scaled_integrals(
                        self.tables[_pair_table_h(pair)],
                        self.d0_by_kind[pair], dist, self.exponents)
                else:
                    table_h = self.tables[_pair_table_h(pair)]
                v = _sk_block(rhat, table_h)
                h_entries.append((i, j, v, False, dfrac))
                if inter or self.conjugate_fill:
                    # 层内同亚层的共轭填充受 conjugate_fill 门控
                    # （负结果复现配置，见 zahid_mos2.TestKnownIssue）；
                    # 层间为真实异像键对，厄米性要求必填。
                    h_entries.append((j, i, v, True, dfrac))
                if not inter:
                    table_s = self.tables[f"{pair}_S"]
                    vs = _sk_block(rhat, table_s)
                    s_entries.append((i, j, vs, False, dfrac))
                    if self.conjugate_fill:
                        s_entries.append((j, i, vs, True, dfrac))
        return h_entries, s_entries

    # ------------------------------------------------------------------
    def hamiltonian_overlap(self, kfrac) -> Tuple[np.ndarray, np.ndarray]:
        """装配 k 点的 (H, S)。kfrac: 倒格子分数坐标 (2,) 或 (3,)。

        层间键只进 H（Harrison 标度缩放），S 通道近似 0（类文档）。
        """
        k = np.asarray(kfrac, float).reshape(-1)[:3]
        if k.size == 2:
            k = np.r_[k, 0.0]
        if not np.isfinite(k).all():
            raise ValueError("kfrac 必须有限")
        TWO_PI = 2 * np.pi
        n = self.n_basis
        H = np.zeros((n, n), dtype=complex)
        S = np.zeros((n, n), dtype=complex)

        for (row, col, block, conj, dfrac) in self._h_entries:
            phase = np.exp(1j * TWO_PI * float(np.dot(k, dfrac)))
            if conj:
                _add_orbital_block(H, row, col,
                                   block.conj().T * np.conj(phase))
            else:
                _add_orbital_block(H, row, col, block * phase)
        for (row, col, block, conj, dfrac) in self._s_entries:
            phase = np.exp(1j * TWO_PI * float(np.dot(k, dfrac)))
            if conj:
                _add_orbital_block(S, row, col,
                                   block.conj().T * np.conj(phase))
            else:
                _add_orbital_block(S, row, col, block * phase)

        # on-site 能量 + SOC（逐原子）
        symbols = self.stack.crystal.symbols
        onsite = self.tables["ONSITE"]
        for atom, sym in enumerate(symbols):
            off = atom * 18
            for oi, o in enumerate(ORBITALS):
                e = onsite[sym][_ONSITE_KEY[o]]
                H[off + oi * 2: off + oi * 2 + 2,
                  off + oi * 2: off + oi * 2 + 2] += e * np.eye(2)
            if self.soc:
                H[off:off + 18, off:off + 18] += onsite_soc(onsite[sym]["lambda"])
        S += np.eye(n, dtype=complex)
        return H, S

    # ------------------------------------------------------------------
    def bands(self, kfracs) -> np.ndarray:
        """广义本征值 E(k)。kfracs: (N, 2) 倒格分数坐标。返回 (N, n_basis) 升序。

        .. warning::
           继承 T16-SK 已知问题：单层 K 点带隙偏差（~0.98 vs 1.805 eV），
           绝对带隙不得作定量参考（见类文档）。
        """
        kfracs = np.asarray(kfracs, float)
        out = np.empty((len(kfracs), self.n_basis))
        for ik, kf in enumerate(kfracs):
            out[ik] = self._generalized_eigh(kf)
        return out

    def energies_at(self, kfrac) -> np.ndarray:
        return self._generalized_eigh(kfrac)

    def _generalized_eigh(self, kfrac) -> np.ndarray:
        H, S = self.hamiltonian_overlap(kfrac)
        wS = np.linalg.eigvalsh(S)
        if wS.min() < -1e-8:
            raise RuntimeError(
                f"S(k) 非正定 (min eig = {wS.min():.4g} @ k={kfrac})："
                "层间装配出错，结果无定义")
        return scipy.linalg.eigh(H, S, eigvals_only=True)

    def __repr__(self) -> str:
        return (f"MoS2MultilayerModel({self.stack.stacking}, "
                f"{self.n_layers}L, n_basis={self.n_basis}, "
                f"n_interlayer_bonds={self.n_interlayer_bonds})")


def _pair_table_h(pair: str) -> str:
    """元素对 → H 积分表名（异种对经映射，同种对直接 SS_/MoMo_H）。"""
    prefix = _PAIR_TABLE.get(pair)
    return f"{prefix}_H" if prefix else f"{pair}_H"


def _sk_block(rhat: np.ndarray, table: Dict[str, float]) -> np.ndarray:
    """9×9 SK 块 ⟨orb_a@A|H|orb_b@B⟩（R̂ = B−A），不含相位。

    必须用 ``zahid_mos2.sk_element``（带反向行名回退：同亚层表只存
    "spσ" 方向的行，(pz,s) 序的查找回退到它）——``slater_koster.sk_element``
    无回退，会把镜像键的共轭序贡献漏掉一半（对账 Zahid 装配时抓到）。
    """
    block = np.empty((9, 9), dtype=complex)
    for a_i, oa in enumerate(ORBITALS):
        for b_i, ob in enumerate(ORBITALS):
            block[a_i, b_i] = sk_element(oa, ob, rhat, table)
    return block


def _add_orbital_block(H: np.ndarray, row: int, col: int,
                       block: np.ndarray) -> None:
    """把 9×9 轨道块以 2×2 自旋单位结构加进 H 的 18×18 原子块。

    块结构 = block ⊗ I₂：H[2i:2i+2, 2j:2j+2] += block[i, j]·I₂。
    基本切片返回视图，两次跨步加法即可完成。
    """
    sl = H[row * 18: row * 18 + 18, col * 18: col * 18 + 18]
    sl[0::2, 0::2] += block
    sl[1::2, 1::2] += block
