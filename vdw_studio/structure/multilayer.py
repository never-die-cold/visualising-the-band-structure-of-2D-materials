"""通用多层堆叠构建器（大目标 D）。

把任意单层 :class:`~vdw_studio.structure.crystal.Crystal` 按指定的逐层
刚体变换堆成 N 层晶体。层间耦合模型（``vdw_studio.engine.multilayer``）
从本模块产出的**实际原子坐标**自动生成 hopping——堆叠规范错了耦合就错，
因此每个构建器都带可解析的结构锚点（见 ``tests/test_multilayer.py``）。

四种堆叠规范：

- :func:`stack_sequence` —— 显式层序：每层一个
  :class:`LayerStep`（面内分数位移 + 可选 180° 旋转）。是其余规范的
  底层原语，也是非六方单层（如黑磷烯）唯一可用的方式；
- :func:`stack_2h` —— **1H-TMD 的 2H 多型**（2H-MoS₂ 的 P6₃/mmc 堆叠）：
  奇数层取单层平移 (1/3, 2/3)、偶数层取其 180° 翻转再平移 (2/3, 1/3)。
  n=2 与 ``builders.tmd_bilayer`` 逐坐标等价（回归锚点）。
  注意：对**中心对称的蜂窝单层**（石墨烯/hBN）这不给出 Bernal AB——
  那些体系请用 :func:`stack_bernald`；
- :func:`stack_bernald` —— 蜂窝单层的 AB (Bernal) 堆叠（石墨 2H）：
  层间面内位移在 (0,0) 与 (2/3, 1/3) 间交替，使上层 A 位落在下层
  B 位正上方（二聚位，McCann 模型的 γ₁ 耦合位形）；
- :func:`stack_3r` —— 3R 多型（R3m，ABC 序列）：所有层**同向**，
  逐层累计平移 (1/3, 2/3)，三层一个周期。适用于 1H-TMD 与蜂窝单层
  （石墨 3R）；与 2H 的本质区别是"翻转/不翻转"（3R 无反演对称，
  有面外极化——这就是 3R-MoS₂ 任意层数都有 SHG 的原因）；
- :func:`stack_twisted` —— 转角双层：公度 (m, n) 转角 θ，
  cos θ = (m² + 4mn + n²) / (2(m² + mn + n²))，两层各转 ∓θ/2，
  胞取公度重合胞（每层 N = m² + mn + n² 个原胞）。
  转角 TB 引擎侧对蜂窝单层走大超胞位点模型（本构建器直接可用），
  对 TMD 走连续 moiré 框架（``engine.moire_tmd``）。

层间距约定：``interlayer`` 为**层中面到层中面**的距离（如 2H-MoS₂
的 Mo–Mo 跨层距离 6.15 Å = c/2），各层 z 居中对称放置。

文献：2H/3R 堆叠描述见 docs/REFERENCES.md [11]；2H-MoS₂ 结构数据
（P6₃/mmc，Mo 2c / S 4f）见 [12]（Acta Cryst. B39, 404 (1983)）。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from math import gcd
from typing import List, Optional, Sequence, Tuple

import numpy as np

from ..numerics import finite_scalar, integer
from .crystal import Crystal
from .lattice import Lattice


@dataclass(frozen=True)
class LayerStep:
    """单层的刚体面内变换：先绕 z 转 180°（若 :attr:`rot180`），再平移。

    :attr:`shift` 以**单层自身晶格**的分数坐标给出（堆叠不改变面内格子，
    故也是堆叠后晶格的分数坐标）；旋转绕胞原点进行，只翻面内坐标、
    不动 z。对中心对称单层，"rot180 + 位移"与某个纯位移等价；
    对 1H-TMD 这类极性层则不等价——翻转与否是 2H/3R 的本质区别。
    """

    shift: Tuple[float, float] = (0.0, 0.0)
    rot180: bool = False


@dataclass
class MultilayerStack:
    """多层堆叠结果：原子按层从下到上依次追加。

    :attr:`layer_slices` 给出每层在 :attr:`crystal` 中的原子索引区间
    ``[start, stop)``（层内原子顺序与单层一致）——层间耦合生成器据此
    区分层内/层间原子对。:attr:`steps` 记录每层的 :class:`LayerStep`
    （扭转堆叠为 ``None``：其层旋转不是 0°/180°，无法用 LayerStep 表达），
    引擎侧装配层内镜像键的局部规范时需要知道每层是否翻转。
    """

    monolayer: Crystal
    crystal: Crystal
    interlayer: float
    layer_slices: Tuple[Tuple[int, int], ...]
    stacking: str
    meta: dict = field(default_factory=dict)
    steps: Optional[Tuple[LayerStep, ...]] = None

    @property
    def n_layers(self) -> int:
        return len(self.layer_slices)

    @property
    def lattice(self) -> Lattice:
        return self.crystal.lattice

    def layer_atoms(self, index: int) -> List[int]:
        start, stop = self.layer_slices[index]
        return list(range(start, stop))


def _mono_vacuum(monolayer: Crystal) -> float:
    """单层真空厚度（slab 约定：a3 垂直于层面）。"""
    return float(abs(monolayer.lattice.a3[2]))


def _validate_common(monolayer: Crystal, interlayer: float, n_layers: int,
                     vacuum) -> float:
    if not isinstance(monolayer, Crystal) or monolayer.n_atoms == 0:
        raise ValueError("monolayer 必须是非空 Crystal")
    finite_scalar(interlayer, 'interlayer')
    if interlayer <= 0:
        raise ValueError("interlayer 必须为正（层中面到层中面的距离，Å）")
    integer(n_layers, 'n_layers', minimum=1)
    mono_vac = _mono_vacuum(monolayer)
    span = (n_layers - 1) * interlayer
    if vacuum is None:
        return mono_vac + span
    finite_scalar(vacuum, 'vacuum')
    if vacuum < mono_vac + span - 1e-9:
        raise ValueError(
            f"vacuum={vacuum:.3f} Å 不足以容纳 {n_layers} 层（层间共 {span:.3f} Å，"
            f"单层自身需要 {mono_vac:.3f} Å 的余量）")
    return float(vacuum)


def _stack_with_steps(monolayer: Crystal, steps: Sequence[LayerStep],
                      interlayer: float, vacuum, stacking: str,
                      meta: dict) -> MultilayerStack:
    """按显式逐层变换堆叠。原子 z：层 i 中面放在
    z_i = (i − (n−1)/2)·interlayer，单层内部 z 偏移原样保留。"""
    steps = tuple(steps)
    n = len(steps)
    vac_total = _validate_common(monolayer, interlayer, n, vacuum)
    mono_lat = monolayer.lattice
    mono_vac = _mono_vacuum(monolayer)
    lat = Lattice(np.array([mono_lat.matrix[0], mono_lat.matrix[1],
                            [0.0, 0.0, vac_total]]))
    out = Crystal(lat)
    slices = []
    for i, step in enumerate(steps):
        start = out.n_atoms
        dz = (i - (n - 1) / 2.0) * interlayer
        for atom in monolayer.atoms:
            f = np.array([atom.frac[0], atom.frac[1], atom.frac[2]])
            if step.rot180:
                f[0], f[1] = -f[0], -f[1]
            f[0] += step.shift[0]
            f[1] += step.shift[1]
            cart = mono_lat.frac_to_cart(f)
            new_cart = np.array([cart[0], cart[1],
                                 vac_total / 2 + (cart[2] - mono_vac / 2) + dz])
            nf = lat.cart_to_frac(new_cart)
            # 面内折回 [0,1)；z 保持居中不折
            out.add_atom(atom.symbol,
                         (nf[0] % 1.0, nf[1] % 1.0, nf[2]), atom.label)
        slices.append((start, out.n_atoms))
    return MultilayerStack(monolayer=monolayer, crystal=out,
                           interlayer=float(interlayer),
                           layer_slices=tuple(slices), stacking=stacking,
                           meta=dict(meta), steps=steps)


# 六方格子的三个高对称面内位置：A 位、B 位与两个空位（分数坐标）。
# 堆叠规范用它们表达"层与层的面内错位"。
_HOLLOW_1 = (1.0 / 3, 2.0 / 3)
_HOLLOW_2 = (2.0 / 3, 1.0 / 3)


def stack_sequence(monolayer: Crystal, steps: Sequence[LayerStep],
                   interlayer: float, *,
                   vacuum: Optional[float] = None) -> MultilayerStack:
    """显式层序堆叠：第 i 层取 :class:`LayerStep` 变换后的单层副本。

    这是通用原语；2H/3R/Bernal 等命名规范都是它的预设。非六方单层
    （黑磷烯等）没有命名规范可用，直接给 ``steps`` 即可。
    """
    steps = tuple(steps)
    if not steps:
        raise ValueError("steps 不能为空（至少要有一层）")
    for s in steps:
        if not isinstance(s, LayerStep):
            raise ValueError("steps 的元素必须是 LayerStep")
        finite_scalar(s.shift[0], 'shift x')
        finite_scalar(s.shift[1], 'shift y')
    return _stack_with_steps(monolayer, steps, interlayer, vacuum,
                             "sequence", {})


def stack_2h(monolayer: Crystal, n_layers: int, interlayer: float,
             *, vacuum: Optional[float] = None) -> MultilayerStack:
    """1H-TMD 的 2H 多型（逐层 180° 翻转），n=2 等价 ``builders.tmd_bilayer``。

    变换序列：偶数层（0 起）= 单层 + (1/3, 2/3)；奇数层 =
    单层绕 z 转 180° + (2/3, 1/3)。两层一个周期（AbA/B 类），
    相邻层的 Mo 互相落在对方空位上方——与 2H-MoS₂ 体相 P6₃/mmc 的
    2c/4f 坐标一致（REFERENCES.md [12]）。

    警告：**不要**对石墨烯/hBN 用它冒充 Bernal/AA' 堆叠——中心对称
    蜂窝层的 2H（石墨）是纯位移交替（:func:`stack_bernald`），
    "逐层翻转"对它们给出另一种（对 hBN 恰为体相 AA'）堆叠。
    """
    integer(n_layers, 'n_layers', minimum=1)
    steps = [LayerStep(shift=_HOLLOW_1 if i % 2 == 0 else _HOLLOW_2,
                       rot180=(i % 2 == 1)) for i in range(n_layers)]
    return _stack_with_steps(monolayer, steps, interlayer, vacuum,
                             "2h", {})


def stack_bernald(monolayer: Crystal, n_layers: int, interlayer: float,
                  *, vacuum: Optional[float] = None) -> MultilayerStack:
    """蜂窝单层的 AB (Bernal) 堆叠（石墨 2H）：位移在 (0,0)/(2/3,1/3) 交替。

    上层 A 位落在下层 B 位正上方（二聚位），另一个子格落在空位上方——
    与 ``engine.models.BilayerGrapheneModel`` 的 B1–A2 二聚位 γ₁ 耦合
    位形一致（回归锚点：n=2 时两者给出同一 K 点谱）。
    """
    integer(n_layers, 'n_layers', minimum=1)
    steps = [LayerStep(shift=(0.0, 0.0) if i % 2 == 0 else _HOLLOW_2)
             for i in range(n_layers)]
    return _stack_with_steps(monolayer, steps, interlayer, vacuum,
                             "bernald", {})


def stack_3r(monolayer: Crystal, n_layers: int, interlayer: float,
             *, vacuum: Optional[float] = None) -> MultilayerStack:
    """3R 多型（R3m，ABC 序列）：所有层同向，逐层平移 (1/3, 2/3)。

    三层一个周期：(0,0) → (1/3, 2/3) → (2/3, 1/3) → (0,0)。
    与 2H 的本质区别：**不翻转**（层偶极同向 → 无反演对称、面外极化，
    这是 3R-MoS₂ 任意层数都有 SHG 的结构根源，REFERENCES.md [11]）。
    对蜂窝单层即石墨 3R（ABC 堆叠）。
    """
    integer(n_layers, 'n_layers', minimum=1)
    steps = [LayerStep(shift=((i * _HOLLOW_1[0]) % 1.0,
                              (i * _HOLLOW_1[1]) % 1.0))
             for i in range(n_layers)]
    return _stack_with_steps(monolayer, steps, interlayer, vacuum,
                             "3r", {})


def _rot2d(angle_rad: float) -> np.ndarray:
    c, s = np.cos(angle_rad), np.sin(angle_rad)
    return np.array([[c, -s], [s, c]])


def stack_twisted(monolayer: Crystal, m: int, n: int, interlayer: float,
                  *, vacuum: Optional[float] = None) -> MultilayerStack:
    """公度转角双层：(m, n) 转角 θ，两层各转 ∓θ/2，胞取公度重合胞。

    转角公式（六方格子的标准结果，Bistritzer–MacDonald 记号系）::

        cos θ = (m² + 4mn + n²) / (2N)，  N = m² + mn + n²

    重合胞基矢（两层旋转后格子的公共超格）::

        T1 = R(+θ/2)·(m·a1 + n·a2)，  T2 = R(60°)·T1

    每层含 N 个单层原胞（要求 gcd(m, n) = 1），总原子数 =
    2 × n_atoms(单层) × N。底层转 +θ/2、顶层转 −θ/2（手性约定，
    影响镜像配对选择不影响谱）。(m, n) = (2, 1) → θ ≈ 21.79°、
    N = 7；(1, 0) → θ = 60°、N = 1。

    小转角（θ ≲ 5°）时重合胞原子数 ∝ N 增长极快（(31, 30) →
    N = 2791 胞/层）；位点模型只对蜂窝单层（2 位点/胞）可用，
    TMD 转角物理走连续 moiré 框架（``engine.moire_tmd``）。
    """
    m = integer(m, 'm', minimum=0)
    n = integer(n, 'n', minimum=0)
    if gcd(m, n) != 1:
        raise ValueError(f"gcd(m, n) 必须为 1（收到 {(m, n)}），"
                         "否则重合胞可约化")
    if m + n < 1:
        raise ValueError("m + n 必须 ≥ 1（(0, 0) 不是转角）")
    big_cells = m * m + m * n + n * n
    cos_theta = (m * m + 4 * m * n + n * n) / (2.0 * big_cells)
    theta = float(np.arccos(np.clip(cos_theta, -1.0, 1.0)))

    vac_total = _validate_common(monolayer, interlayer, 2, vacuum)
    mono_lat = monolayer.lattice
    mono_vac = _mono_vacuum(monolayer)

    # 重合胞。注意本仓库六方晶格的 a1/a2 夹角是 120°（gamma=120），
    # 而转角公式族默认 60° 基——故在 (b1, b2) = (a1, −a2) 的 60° 基中
    # 构造：T1 = R(−θ/2)(m b1 + n b2)，T2 = R(−60°) T1。底层转 −θ/2、
    # 顶层转 +θ/2 时，T1/T2 都是两层旋转格子的公共超格矢量
    # （层 1 坐标 (m,n) 与 (−n,m+n)；层 2 镜像 R(−θ)w = n b1 + m b2），
    # 超格指数恰为 N。
    b1, b2 = mono_lat.a1[:2], -mono_lat.a2[:2]
    t1 = _rot2d(-theta / 2) @ (m * b1 + n * b2)
    t2 = _rot2d(-np.pi / 3) @ t1
    lat = Lattice(np.array([[t1[0], t1[1], 0.0],
                            [t2[0], t2[1], 0.0],
                            [0.0, 0.0, vac_total]]))
    inv_t = np.linalg.inv(lat.matrix[:2, :2])   # 行向量约定：f = r @ inv(A)

    out = Crystal(lat)
    slices = []
    r_max = int(np.ceil(2.0 * np.sqrt(big_cells))) + 2
    for layer, sign in enumerate((-1, +1)):
        start = out.n_atoms
        rot = _rot2d(sign * theta / 2)
        # 该层旋转后的晶格矢量——像平移必须用它（60° 基：b1, b2）
        a1r, a2r = rot @ b1, rot @ b2
        z_center = (layer - 0.5) * interlayer   # 两层：∓interlayer/2
        for atom in monolayer.atoms:
            base = mono_lat.frac_to_cart(atom.frac)
            xy = rot @ base[:2]
            dz_cart = base[2] - mono_vac / 2
            images = []
            for p in range(-r_max, r_max + 1):
                for q in range(-r_max, r_max + 1):
                    xy_img = xy + p * a1r + q * a2r
                    f2 = xy_img @ inv_t
                    f2 = np.round(np.mod(f2, 1.0), 9) % 1.0
                    images.append((f2[0], f2[1]))
            unique = np.unique(np.array(images), axis=0)
            for fx, fy in unique:
                z = vac_total / 2 + z_center + dz_cart
                out.add_atom(atom.symbol, (fx, fy, z / vac_total), atom.label)
        slices.append((start, out.n_atoms))
    expected = 2 * monolayer.n_atoms * big_cells
    if out.n_atoms != expected:
        raise RuntimeError(
            f"重合胞原子数 {out.n_atoms} ≠ 预期 {expected}"
            f"（2 层 × {monolayer.n_atoms} 原子 × {big_cells} 胞）——"
            "转角构造或边界去重出错，请勿使用该结果")
    return MultilayerStack(
        monolayer=monolayer, crystal=out, interlayer=float(interlayer),
        layer_slices=tuple(slices), stacking=f"twisted({m},{n})",
        meta={"theta_deg": float(np.degrees(theta)),
              "cells_per_layer": big_cells})
