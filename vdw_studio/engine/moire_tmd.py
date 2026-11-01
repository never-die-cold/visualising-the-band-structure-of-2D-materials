"""转角双层 TMD 的单带连续 moiré 模型（层间激子的带结构侧）。

物理模型（Wu–Lovorn–MacDonald 式连续近似的**最小**版本）：

- 每层取 K 谷**各向同性有效质量**色散 ε(k) = ±ħ²|k|²/(2m*)（TMD 谷
  附近的抛物近似；质量取 Kormányos 2015 Table 3/4 的 m*_c/m*_v），
  符号 ``band="cb"`` 取导带、``"vb"`` 取价带（能量翻转，空穴视角）；
- 层间耦合按 moiré 周期调制：T(r) = Σ_{j=1..3} T_j e^{i q_j·r}，
  C₃ 对称性固定三个动量转移的方向，这里取 |T_j| = T 且同相位
  （堆叠相位是唯象自由度，见下）；
- 平面波基：每层同一套 moiré 动量壳（与已验证的石墨烯 BM 引擎
  ``engine.moire`` 同一套构造），维度 = 2·N_G。

与 BM 引擎的关系：moiré 几何的构造方式一致（q_j 由 C₃ 生成、
k_θ = 2K_D sin(θ/2)），但晶格常数按 TMD 取值（MoS₂ a ≈ 3.184 Å），
且层块是抛物线带而非 Dirac 锥——因此完全独立实现，不动 T16 已验证
的 ``engine.moire``。

验证锚点（可解析）：

1. T = 0：两层解耦，谱 = 两套抛物线带的并集（对任意截断精确）；
2. 厄米性；
3. C₃：谱对 120° 旋转不变（限定低能窗口）；
4. 微扰极限：T ≪ 动能标度时，k=0 处两带劈裂 → 2|T|；
5. 平带化：θ 减小使动能标度 E_kin = ħ²k_θ²/2m* 低于 T，带宽饱和于
   O(T) 量级。
"""

from dataclasses import dataclass
from typing import List, Optional, Tuple

import numpy as np

from ..constants import HBAR2_OVER_2M0
from .kp_tmd import TMDKpModel, TMDKpParams

_C3_DIRS = np.array([[0.0, -1.0],
                     [np.sqrt(3) / 2, 0.5],
                     [-np.sqrt(3) / 2, 0.5]])


@dataclass
class TmdMoireGeometry:
    """moiré 倒空间几何（TMD 晶格常数版，构造方式与 BM 引擎一致）。"""

    theta_deg: float
    a_lattice: float
    k_theta: float          # |q_j| = 2 K_D sin(θ/2)，K_D = 4π/(3a)
    q_vectors: np.ndarray   # (3,2)
    b1: np.ndarray
    b2: np.ndarray

    @property
    def l_moire(self) -> float:
        return self.a_lattice / (2.0 * np.sin(np.radians(self.theta_deg) / 2))


def build_tmd_moire_geometry(theta_deg: float, a_lattice: float) -> TmdMoireGeometry:
    """转角 TMD 的 moiré 几何：q_j 由 C₃ 旋转生成，b₁ = q₂−q₁、b₂ = q₃−q₁。"""
    if theta_deg <= 0:
        raise ValueError("θ 必须为正")
    if a_lattice <= 0:
        raise ValueError("晶格常数必须为正")
    k_theta = 2.0 * (4 * np.pi / (3 * a_lattice)) * np.sin(np.radians(theta_deg) / 2)
    q = k_theta * _C3_DIRS
    return TmdMoireGeometry(theta_deg=float(theta_deg), a_lattice=float(a_lattice),
                            k_theta=k_theta, q_vectors=q,
                            b1=q[1] - q[0], b2=q[2] - q[0])


class TwistedTMDBilayer:
    """转角双层 TMD 的单带 moiré 平面波求解器。

    参数:
        theta_deg: 相对转角 (度)。
        a_lattice: 面内晶格常数 (Å)，MoS₂ 默认 3.184。
        effective_mass: 单带有效质量 (m₀)，Kormányos Table 3/4。
        t_interlayer: 层间耦合幅度 |T_j| (eV)，唯象参数。
        n_shells: 平面波壳层数（|G| ≤ n_shells·|b₁|）。
        band: "cb"（导带，能量 ≥ 0）或 "vb"（价带，能量 ≤ 0）。
    """

    def __init__(self, theta_deg: float = 2.0, a_lattice: float = 3.184,
                 effective_mass: float = 0.55, t_interlayer: float = 0.020,
                 n_shells: int = 4, band: str = "cb"):
        if theta_deg <= 0 or a_lattice <= 0 or effective_mass <= 0:
            raise ValueError("θ、晶格常数与有效质量必须为正")
        if t_interlayer < 0:
            raise ValueError("层间耦合幅度必须非负")
        if n_shells < 1:
            raise ValueError("n_shells ≥ 1")
        if band not in ("cb", "vb"):
            raise ValueError("band 必须为 'cb' 或 'vb'")
        self.theta_deg = float(theta_deg)
        self.a_lattice = float(a_lattice)
        self.effective_mass = float(effective_mass)
        self.t_interlayer = float(t_interlayer)
        self.band = band
        self.n_shells = int(n_shells)
        self.geom = build_tmd_moire_geometry(theta_deg, a_lattice)
        self.pw = self._build_plane_waves(n_shells)
        self._index = {tuple(g): i for i, g in enumerate(self.pw)}
        self._coupling = self._build_coupling()

    # ------------------------------------------------------------------
    def _build_plane_waves(self, n_shells: int) -> np.ndarray:
        """整数坐标集合（圆形截断，确定性排序；与 BM 引擎同一构造）。"""
        return _pw_shells(self.geom, n_shells)

    def _build_coupling(self) -> list:
        """层间耦合表 [(层1 指标, 层2 指标, T_j)]。

        动量 q₁+L 的层 2 态（L = 0, b₁, b₂ → 动量 q₁, q₂, q₃）依次携带
        T₁, T₂, T₃——与 BM 引擎的自然配对同构（q₂ = q₁+b₁、q₃ = q₁+b₂）。
        """
        offsets = [(np.array([0, 0]), 0.0 + 0.0j),
                   (np.array([1, 0]), 0.0 + 0.0j),
                   (np.array([0, 1]), 0.0 + 0.0j)]
        table = []
        for j2, g2 in enumerate(self.pw):
            for off, _phase in offsets:
                i1 = self._index.get(tuple(g2 - off))
                if i1 is not None:
                    table.append((i1, j2, self.t_interlayer))
        return table

    # ------------------------------------------------------------------
    @property
    def coupling_ratio(self) -> float:
        """无量纲耦合比 T/E_kin（E_kin = ħ²k_θ²/2m*，谷间动量处的动能）。"""
        kinetic = HBAR2_OVER_2M0 * self.geom.k_theta ** 2 / self.effective_mass
        return self.t_interlayer / kinetic

    def hamiltonian(self, k) -> np.ndarray:
        """Bloch 哈密顿量（eV，厄米；k 相对 moiré BZ 中心，Å⁻¹）。"""
        k = np.asarray(k, dtype=float).reshape(-1)[:2]
        n = len(self.pw)
        dim = 2 * n
        H = np.zeros((dim, dim), complex)
        sign = 1.0 if self.band == "cb" else -1.0
        q1 = self.geom.q_vectors[0]

        for i, g in enumerate(self.pw):
            p1 = k + g[0] * self.geom.b1 + g[1] * self.geom.b2
            H[i, i] = sign * HBAR2_OVER_2M0 * float(p1 @ p1) / self.effective_mass
            p2 = p1 + q1
            H[n + i, n + i] = sign * HBAR2_OVER_2M0 * float(p2 @ p2) / self.effective_mass
        for i1, j2, t_j in self._coupling:
            H[i1, n + j2] += t_j
            H[n + j2, i1] += t_j
        return H

    def bands(self, k_list) -> np.ndarray:
        """沿 k 列表求本征值（升序），返回 (n_k, 2·N_G)。"""
        k_list = np.asarray(k_list, dtype=float)
        out = np.empty((len(k_list), 2 * len(self.pw)))
        for ik, k in enumerate(k_list):
            out[ik] = np.linalg.eigvalsh(self.hamiltonian(k))
        return out

    def moire_effective_mass(self, band_index: int = 0, dk: float = 1e-4) -> float:
        """k=0 处第 ``band_index`` 条 moiré 带的有效质量 (m₀，二阶差分)。

        d²E/dk² = 2·(ħ²/2m₀)/m*，故 m* = 2·HBAR2_OVER_2M0/curvature
        （与 ``analysis.properties.effective_mass`` 的约定一致）。

        注意 ``band="vb"`` 的谱升序排列且能量 ≤ 0：价带**顶**在
        ``band_index=-1``（最高带），不是 0。
        """
        if dk <= 0:
            raise ValueError("dk 必须为正")
        e = self.bands([[0.0, 0.0], [dk, 0.0], [-dk, 0.0]])
        curvature = (e[1, band_index] - 2 * e[0, band_index] + e[2, band_index]) / dk ** 2
        if abs(curvature) < 1e-15:
            raise ValueError("带曲率为零，无法定义有效质量")
        return abs(2 * HBAR2_OVER_2M0 / curvature)


# ----------------------------------------------------------------------
# 完整 k·p 层哈密顿量版（T17 剩余项：替换单带近似）
# ----------------------------------------------------------------------
def _pw_shells(geom: TmdMoireGeometry, n_shells: int) -> np.ndarray:
    """整数坐标集合（圆形截断 |G| ≤ n_shells·|b₁|，确定性排序）。

    与 BM 引擎同一构造；单带与 k·p 两类求解器共用。
    """
    rng = np.arange(-n_shells, n_shells + 1)
    m, n = np.meshgrid(rng, rng, indexing="ij")
    cand = np.stack([m.ravel(), n.ravel()], axis=1)
    cart = cand[:, 0, None] * geom.b1 + cand[:, 1, None] * geom.b2
    bnorm = np.linalg.norm(geom.b1)
    keep = cand[np.linalg.norm(cart, axis=1) <= n_shells * bnorm + 1e-9]
    order = np.lexsort((keep[:, 1], keep[:, 0],
                        np.linalg.norm(
                            keep[:, 0, None] * geom.b1
                            + keep[:, 1, None] * geom.b2, axis=1)))
    return keep[order]


class TwistedTMDKp:
    """转角双层 TMD 的**完整 k·p** moiré 求解器（Kormányos 4×4 层块）。

    单带模型 :class:`TwistedTMDBilayer` 的升级（T17 剩余项）：每层的
    层块不再是各向同性抛物线带，而是 Kormányos 2015 的 4×4 k·p
    哈密顿量（``engine.kp_tmd.TMDKpModel``：vb/cb × ↑/↓，含 SOC 劈裂、
    电子-空穴不对称、三角翘曲与立方项），动量取 p = k+G（层 1）与
    p = k+G+q₁（层 2），与单带模型同一动量规范。

    层间耦合：同带同自旋的跃迁（自旋守恒、带对角），幅度
    γ_cb/γ_vb（默认与单带模型的唯象值 0.020 eV 一致），动量配对与
    单带模型同约定（偏移 0/b₁/b₂，堆叠相位取同号——唯象自由度）。
    层间 vb↔cb 跃迁在文献参数化中为小量，此处略去（诚实标注）。

    基矢顺序：层 1 的 (G, 态) 后接层 2 的 (G, 态)，态序沿用
    ``TMDKpModel`` 的 (vb↑, cb↑, vb↓, cb↓)——谱升序下导带底在
    :attr:`i_cb_bottom` = dim/2，价带顶在 :attr:`i_vb_top` = dim/2−1
    （moiré 窗口内带隙远大于带宽，排序不交叉）。

    验证锚点（``tests/test_moire_tmd.py::TestTwistedTMDKp``）：
    γ=0 解耦极限逐位还原 Kormányos 谱（含一般 k）、厄米性、C₃ 低能
    窗口对称性、二阶弱耦合 |ΔE| ∝ γ²、平带化、解耦 moiré 质量 =
    Kormányos 单层质量（衔接激子管线）。
    """

    def __init__(self, theta_deg: float = 2.0, a_lattice: Optional[float] = None,
                 material: str = "MoS2", source: str = "dft", valley: int = +1,
                 gamma_cb: float = 0.020, gamma_vb: float = 0.020,
                 n_shells: int = 4):
        if theta_deg <= 0:
            raise ValueError("θ 必须为正")
        if gamma_cb < 0 or gamma_vb < 0:
            raise ValueError("层间耦合幅度必须非负")
        if n_shells < 1:
            raise ValueError("n_shells ≥ 1")
        if valley not in (+1, -1):
            raise ValueError("valley 必须为 +1 或 -1")
        self.params = TMDKpParams.from_library(material, source)
        self.kp = TMDKpModel(self.params)
        self.theta_deg = float(theta_deg)
        self.a_lattice = float(a_lattice) if a_lattice is not None else self.params.a
        self.valley = int(valley)
        self.gamma_cb = float(gamma_cb)
        self.gamma_vb = float(gamma_vb)
        self.n_shells = int(n_shells)
        self.geom = build_tmd_moire_geometry(theta_deg, self.a_lattice)
        self.pw = _pw_shells(self.geom, n_shells)
        self._index = {tuple(g): i for i, g in enumerate(self.pw)}
        # 基矢 (vb↑, cb↑, vb↓, cb↓) 的通道幅度
        self._gamma = np.array([self.gamma_vb, self.gamma_cb,
                                self.gamma_vb, self.gamma_cb])
        self._coupling = self._build_coupling()

    # ------------------------------------------------------------------
    def _build_coupling(self) -> List[Tuple[int, int]]:
        """层间耦合表 [(层1 指标, 层2 指标)]，G' = G − {0, b₁, b₂}。

        与单带模型同一动量配对（q₁+L 的层 2 态 ↔ 对应层 1 态，
        L = 0, b₁, b₂ → q₁, q₂, q₃）；幅度按态通道取 γ_vb/γ_cb。
        """
        offsets = [np.array([0, 0]), np.array([1, 0]), np.array([0, 1])]
        table = []
        for j2, g2 in enumerate(self.pw):
            for off in offsets:
                i1 = self._index.get(tuple(g2 - off))
                if i1 is not None:
                    table.append((i1, j2))
        return table

    # ------------------------------------------------------------------
    @property
    def n_states_per_pw(self) -> int:
        return 4

    @property
    def dim(self) -> int:
        """哈密顿量维度 = 2 层 × N_G 平面波 × 4 态。"""
        return 8 * len(self.pw)

    @property
    def i_cb_bottom(self) -> int:
        """谱升序下导带底 moiré 带的指标（dim/2）。"""
        return self.dim // 2

    @property
    def i_vb_top(self) -> int:
        """谱升序下价带顶 moiré 带的指标（dim/2 − 1）。"""
        return self.dim // 2 - 1

    # ------------------------------------------------------------------
    def hamiltonian(self, k) -> np.ndarray:
        """Bloch 哈密顿量（eV，厄米；k 相对 moiré BZ 中心，Å⁻¹）。"""
        k = np.asarray(k, dtype=float).reshape(-1)[:2]
        n = len(self.pw)
        dim = 8 * n
        H = np.zeros((dim, dim), complex)
        b1, b2, q1 = self.geom.b1, self.geom.b2, self.geom.q_vectors[0]

        for i, g in enumerate(self.pw):
            p1 = k + g[0] * b1 + g[1] * b2
            H[4 * i:4 * i + 4, 4 * i:4 * i + 4] = \
                self.kp.hamiltonian(p1, self.valley)
            r = 4 * (n + i)
            H[r:r + 4, r:r + 4] = self.kp.hamiltonian(p1 + q1, self.valley)
        gamma = np.diag(self._gamma)
        for i1, j2 in self._coupling:
            r = 4 * (n + j2)
            H[4 * i1:4 * i1 + 4, r:r + 4] = gamma
            H[r:r + 4, 4 * i1:4 * i1 + 4] = gamma
        return H

    def bands(self, k_list, n_bands: Optional[int] = None) -> np.ndarray:
        """沿 k 列表求本征值（升序），返回 (n_k, dim)。"""
        k_list = np.asarray(k_list, dtype=float)
        out = np.empty((len(k_list), self.dim))
        for ik, k in enumerate(k_list):
            out[ik] = np.linalg.eigvalsh(self.hamiltonian(k))
        return out if n_bands is None else out[:, :n_bands]

    def moire_effective_mass(self, band_index: int, dk: float = 1e-4) -> float:
        """k=0 处第 ``band_index`` 条 moiré 带的有效质量 (m₀，二阶差分)。

        约定与 :meth:`TwistedTMDBilayer.moire_effective_mass` 相同
        （m* = 2·(ħ²/2m₀)/d²E）；导带底用 :attr:`i_cb_bottom`、
        价带顶用 :attr:`i_vb_top`。
        """
        if dk <= 0:
            raise ValueError("dk 必须为正")
        e = self.bands([[0.0, 0.0], [dk, 0.0], [-dk, 0.0]])
        curvature = (e[1, band_index] - 2 * e[0, band_index]
                     + e[2, band_index]) / dk ** 2
        if abs(curvature) < 1e-15:
            raise ValueError("带曲率为零，无法定义有效质量")
        return abs(2 * HBAR2_OVER_2M0 / curvature)

    def __repr__(self) -> str:
        return (f"TwistedTMDKp({self.params.name}, θ={self.theta_deg}°, "
                f"n_shells={self.n_shells}, dim={self.dim})")
