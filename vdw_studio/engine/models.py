"""正交基组位点紧束缚（Tight-Binding）模型。

实现方式为通用的"位点 + hopping 表"框架：
``H(k) = Σ_{(i,j,Δ,t)} t·e^{i k·Δ} |i⟩⟨j| + h.c.（自动补共轭）+ Σ_i ε_i |i⟩⟨i|``

**hopping 表约定**（重要）：每条无向键只列出一次（从任一端点出发），
引擎自动在对称位置补上共轭项；若 i == j（同位点不同镜像），
必须把 (Δ, t) 与 (−Δ, t) 成对列出以保证厄米性。

包含的模型（参数出处见 docs/REFERENCES.md）：

- :class:`HoneycombModel` —— 石墨烯 / h-BN 家族二带模型
  （t = −2.7 eV，Reich et al. PRB 66, 035412 (2002)）；
- :class:`BuckledHoneycombModel` —— 硅烯低翘曲模型 + 垂直电场调控
  （交错势 δ = e·E·Δz，Kane-Mele 型带隙开启）；
- :class:`PhosphoreneRudenko` —— 黑磷烯四带模型
  （Rudenko & Katsnelson, PRB 89, 201408(R) (2014), Table 1）。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional, Sequence, Tuple

import numpy as np

from ..structure.lattice import Lattice
from ..numerics import finite_scalar, finite_vector, hermitian_matrix, integer

# 里德伯常量换算：ħ²/(2m₀) = 3.80998 eV·Å²（用于有效质量）
HBAR2_OVER_2M0 = 3.809982444


@dataclass(frozen=True)
class Hopping:
    """一条有向 hopping：从位点 i 到位点 j 的镜像（分数位移 dfrac）。"""

    i: int
    j: int
    dfrac: Tuple[float, float, float]   # 相对 i 的分数位移（含跨胞）
    t: float                            # hopping 强度 (eV)


class TightBindingModel:
    """位点紧束缚模型基类。

    子类需设置：``lattice``、``n_sites``、``site_symbols``、``onsite``、
    ``hoppings``；本基类提供哈密顿量装配与能带求解。
    """

    def __init__(self, lattice: Lattice, n_sites: int,
                 site_symbols: Sequence[str],
                 onsite: Sequence[float],
                 hoppings: Sequence[Hopping],
                 name: str = "TB model") -> None:
        self.lattice = lattice
        self.n_sites = integer(n_sites, 'n_sites')
        self.site_symbols = list(site_symbols)
        self.onsite = finite_vector(onsite, self.n_sites, 'onsite').copy()
        self.hoppings = list(hoppings)
        self.name = name
        if len(self.site_symbols) != self.n_sites:
            raise ValueError('site_symbols length must equal n_sites')
        for h in self.hoppings:
            integer(h.i, 'hopping i', minimum=0)
            integer(h.j, 'hopping j', minimum=0)
            finite_vector(h.dfrac, 3, 'hopping displacement')
            finite_scalar(h.t, 'hopping strength')
            if not (0 <= h.i < self.n_sites and 0 <= h.j < self.n_sites):
                raise ValueError(f"hopping 端点越界: {h}")

    # ------------------------------------------------------------------
    def hamiltonian(self, kfrac) -> np.ndarray:
        """装配 k 点哈密顿量。

        参数:
            kfrac: 长度 2 或 3 的倒格子**分数**坐标 (k1, k2[, k3])。
        返回:
            (n_sites, n_sites) 厄米复矩阵 (eV)。

        相位约定：分数坐标的相位为 e^{2πi k·Δ}（k 为倒格分数坐标、
        Δ 为实格分数坐标时，k_cart·Δ_cart = 2π k·Δ）。
        """
        k = np.asarray(kfrac, dtype=float)
        if k.shape not in ((2,), (3,)) or not np.isfinite(k).all():
            raise ValueError('kfrac must contain 2 or 3 finite coordinates')
        if k.size == 2:
            k = np.r_[k, 0.0]
        n = self.n_sites
        H = np.diag(self.onsite).astype(complex)
        TWO_PI = 2.0 * np.pi
        for h in self.hoppings:
            phase = np.exp(1j * TWO_PI * float(np.dot(k, np.asarray(h.dfrac))))
            H[h.i, h.j] += h.t * phase
            if h.i != h.j:
                H[h.j, h.i] += np.conj(h.t * phase)
        return hermitian_matrix(H)

    def bands(self, kfracs) -> np.ndarray:
        """一组 k 点的本征能量。

        参数:
            kfracs: (N, 2 或 3) 倒格子分数坐标数组。
        返回:
            (N, n_sites) 能量数组 (eV)，每行升序。
        """
        kfracs = np.asarray(kfracs, dtype=float)
        if kfracs.ndim != 2 or kfracs.shape[1] not in (2, 3) or not np.isfinite(kfracs).all():
            raise ValueError('kfracs must be a finite (N, 2 or 3) array')
        out = np.empty((len(kfracs), self.n_sites))
        for ik, kf in enumerate(kfracs):
            out[ik] = np.linalg.eigvalsh(self.hamiltonian(kf))
        return out

    def energies_at(self, kfrac) -> np.ndarray:
        """单个 k 点的能量（升序）。"""
        return np.linalg.eigvalsh(self.hamiltonian(kfrac))

    # ------------------------------------------------------------------
    def __repr__(self) -> str:
        return (f"{type(self).__name__}(name={self.name!r}, "
                f"n_sites={self.n_sites}, n_hoppings={len(self.hoppings)})")


# ----------------------------------------------------------------------
# 蜂窝家族
# ----------------------------------------------------------------------
class HoneycombModel(TightBindingModel):
    """蜂窝双原子二带模型（石墨烯 / h-BN）。

    哈密顿量：H(k) = [[ε_A, t·f(k)], [t·f*(k), ε_B]]，
    f(k) = Σ_m e^{i k·δ_m}，δ_m 为三个最近邻矢量。

    参数:
        a: 面内晶格常数 (Å)。
        t: 最近邻 hopping (eV)，石墨烯取 −2.7 (Reich 2002)。
        delta_onsite: A/B 子格在位能差 ε_A − ε_B (eV)；hBN 用它打开能隙，
            K 点带隙严格等于 |delta_onsite|。
        name: 模型名称。
    """

    def __init__(self, a: float, t: float = -2.7,
                 delta_onsite: float = 0.0, name: str = "honeycomb") -> None:
        lattice = Lattice.hexagonal(a, vacuum=15.0)
        # 三个最近邻分数位移（A 在 (0,0)，B 在 (2/3, 1/3)），
        # 笛卡尔形式 (±a/2, a/(2√3)) 与 (0, −a/√3)
        deltas = [(2 / 3, 1 / 3, 0.0), (-1 / 3, 1 / 3, 0.0), (-1 / 3, -2 / 3, 0.0)]
        eps_a = +delta_onsite / 2
        eps_b = -delta_onsite / 2
        hoppings = [Hopping(0, 1, d, t) for d in deltas]
        super().__init__(lattice, n_sites=2, site_symbols=["A", "B"],
                         onsite=[eps_a, eps_b], hoppings=hoppings, name=name)
        self.a = a
        self.t = t
        self.delta_onsite = delta_onsite


class BuckledHoneycombModel(HoneycombModel):
    """低翘曲蜂窝模型（硅烯）：子格面外错开 + 垂直电场调控。

    垂直电场 E (V/Å) 在两个子格间产生势能差 U = e·E·Δz（Δz 为翘曲高度，
    e·E·Δz 的单位恰为 eV），作为交错在位能 ±U/2 加入。
    K 点带隙 = |delta_onsite + U|，因此石墨烯型零带隙体系
    可被电场连续打开（"our own" 电场调控功能的物理基础）。

    参数:
        a: 晶格常数 (Å)，硅烯 3.86。
        buckling: 翘曲高度 Δz (Å)，硅烯 0.44。
        t: 最近邻 hopping (eV)。
        electric_field: 垂直电场 (V/Å)。
    """

    def __init__(self, a: float = 3.86, buckling: float = 0.44,
                 t: float = -1.6, electric_field: float = 0.0,
                 name: str = "silicene") -> None:
        u_stagger = electric_field * buckling  # eV
        super().__init__(a=a, t=t, delta_onsite=u_stagger, name=name)
        self.buckling = buckling
        self.electric_field = electric_field


class BilayerGrapheneModel(TightBindingModel):
    """AB (Bernal) 堆叠双层石墨烯最小模型 + 垂直电场（McCann–Fal'ko）。

    位点：上层 (A1, B1)、下层 (A2, B2)；层内最近邻 t（石墨烯 −2.7 eV），
    层间二聚位耦合 B1–A2：γ₁ = 0.4 eV（文献标准值）。
    垂直电场 U (eV, 层间势差)：上层在位 +U/2，下层 −U/2。

    解析性质（本模型的精确结果，测试对账）：
    - U=0：K 点能量 {0, 0, ±γ₁}——两条低能带在 K 点抛物线触碰（无隙）；
    - U≠0：K 点能量 {±U/2, ±√(γ₁²+U²/4)}，带隙 = |U|
      （最小模型线性开隙；含 γ₃/γ₄ 的完整参数化给出文献常用的
       E_g = Uγ₁²/√(γ₁²+U²) 饱和形式——见 ROADMAP）；
    - 色散：U=0 时低能带 ~ k²（抛物线，区别于单层的线性 Dirac 锥）。

    参考：E. McCann, D. S. L. Abergel & V. I. Fal'ko,
    Eur. Phys. J. Special Topics 148, 41 (2007) 及其中引文；
    层间距 3.35 Å（石墨实验值）。
    """

    def __init__(self, a: float = 2.46, t: float = -2.7,
                 gamma1: float = 0.4, interlayer: float = 3.35,
                 electric_field: float = 0.0,
                 vacuum: float = 15.0) -> None:
        lattice = Lattice.hexagonal(a, vacuum=vacuum)
        h = interlayer / 2
        dz_frac = interlayer / vacuum
        u = electric_field
        # 层内三近邻（与单层相同的分数位移）
        deltas = [(2 / 3, 1 / 3, 0.0), (-1 / 3, 1 / 3, 0.0),
                  (-1 / 3, -2 / 3, 0.0)]
        hs = []
        # 层内: A1-B1 (位点 0-1), A2-B2 (位点 2-3)
        for d in deltas:
            hs.append(Hopping(0, 1, d, t))
            hs.append(Hopping(2, 3, d, t))
        # 层间: B1(1) - A2(2) 二聚位耦合 (z 分数位移不影响 k_z=0 相位)
        hs.append(Hopping(1, 2, (0.0, 0.0, -dz_frac), gamma1))

        # 在位: 上层 +U/2, 下层 -U/2
        onsite = [u / 2, u / 2, -u / 2, -u / 2]
        super().__init__(lattice, n_sites=4,
                         site_symbols=["C", "C", "C", "C"],
                         onsite=onsite, hoppings=hs,
                         name="bilayer graphene (AB, McCann)")
        self.a = a
        self.t = t
        self.gamma1 = gamma1
        self.electric_field = electric_field


# ----------------------------------------------------------------------
# 黑磷烯
# ----------------------------------------------------------------------
# Rudenko & Katsnelson PRB 89, 201408(R) (2014) Table 1 的五个 hopping (eV)。
# 距离/配位数: t1 2.22Å N=2, t2 2.24Å N=1, t3 3.34Å N=2,
#              t4 3.47Å N=4, t5 4.23Å N=1
PHOSPHORENE_T = (-1.220, 3.665, -0.205, -0.105, -0.055)


class PhosphoreneRudenko(TightBindingModel):
    """单层黑磷四带 TB 模型（Rudenko & Katsnelson 2014）。

    结构：链方向 a = 3.3136 Å (x)、褶皱方向 c = 4.3763 Å (y)，
    4 个 P 位点（上下亚层各两个），面外高度 ±1.0654 Å。
    四个位点由对称性等价，在位能统一取 0（能量零点）。

    hopping 表（分数位移以 (x=链方向, y=褶皱方向) 表示，
    与构建器 ``structure.builders.phosphorene`` 的位点编号一致：

    - 位点 0 = 上亚层 A (x=0, y=+0.3525)
    - 位点 1 = 上亚层 H (x=a/2, y=+1.8356)
    - 位点 2 = 下亚层 B (x=0, y=−0.3525)
    - 位点 3 = 下亚层 G (x=a/2, y=−1.8356)

    参数出处：docs/REFERENCES.md 文献 [4]。
    """

    def __init__(self, a: float = 3.3136, c: float = 4.3763,
                 height: float = 1.0654,
                 t: Optional[Sequence[float]] = None) -> None:
        t1, t2, t3, t4, t5 = tuple(t) if t is not None else PHOSPHORENE_T
        # 面内分数位移（不含面外高度差；面外分量不进入相位）
        dy1 = 1.4831 / c       # t1: 链内键的面内 y 分量
        dy2 = 0.7050 / c       # t2: 跨亚层近键的面内 y 分量
        dy3 = -2.8931 / c      # t3: 链内远像 (Rudenko d=3.34 Å)
        dy5 = 3.6713 / c       # t5: 跨亚层远像 (Rudenko d=4.23 Å)
        half = 0.5

        hs: List[Hopping] = []
        # (0,1) 同亚层链内: t1 ×2
        hs.append(Hopping(0, 1, (+half, +dy1, 0.0), t1))
        hs.append(Hopping(0, 1, (-half, +dy1, 0.0), t1))
        # (0,1) 同亚层远像: t3 ×2 (成对 ±)
        hs.append(Hopping(0, 1, (+half, +dy3, 0.0), t3))
        hs.append(Hopping(0, 1, (-half, +dy3, 0.0), t3))
        # (0,2) 跨亚层近像 t2 ×1 + 远像 t5 ×1
        hs.append(Hopping(0, 2, (0.0, -dy2, 0.0), t2))
        hs.append(Hopping(0, 2, (0.0, +dy5, 0.0), t5))
        # (0,3) 跨亚层对角 t4 ×4 (成对)
        for sx in (+1, -1):
            for sy in (+1, -1):
                hs.append(Hopping(0, 3, (sx * half, sy * half, 0.0), t4))
        # (2,3) 下亚层链内（与 (0,1) 关于中面反演，y 分量反号）
        hs.append(Hopping(2, 3, (+half, -dy1, 0.0), t1))
        hs.append(Hopping(2, 3, (-half, -dy1, 0.0), t1))
        hs.append(Hopping(2, 3, (+half, -dy3, 0.0), t3))
        hs.append(Hopping(2, 3, (-half, -dy3, 0.0), t3))
        # (1,3) 上亚层H-下亚层G：与 (0,2) 同类但 y 相位相反
        hs.append(Hopping(1, 3, (0.0, +dy2, 0.0), t2))
        hs.append(Hopping(1, 3, (0.0, -dy5, 0.0), t5))
        # (1,2) 跨亚层对角：与 (0,3) 同类
        for sx in (+1, -1):
            for sy in (+1, -1):
                hs.append(Hopping(1, 2, (sx * half, sy * half, 0.0), t4))

        lattice = Lattice.rectangular(a, c, vacuum=15.0)
        super().__init__(lattice, n_sites=4,
                         site_symbols=["P", "P", "P", "P"],
                         onsite=[0.0, 0.0, 0.0, 0.0],
                         hoppings=hs, name="phosphorene (Rudenko 2014)")
        self.t_params = (t1, t2, t3, t4, t5)
