"""转角双层石墨烯连续模型（Phase 5：莫尔超晶格）。

模型（Bistritzer & MacDonald, PNAS→PRB 84, 035440 (2011)，
arXiv:1009.4203 源码在 papers/arxiv/bistritzer2011/）：

- **单层 Dirac 哈密顿量**（BM Eq. (1)，k 相对该层 Dirac 点）::

      h_k(θ) = −v·k · [[0, e^{i(θ_k−θ)}], [e^{−i(θ_k−θ)}, 0]]

  解耦双层 = h(+θ/2) ⊕ h(−θ/2)。

- **层间隧穿**（BM Eq. (2)-(5)，局域连续极限，d = 0）::

      T(r) = w · Σ_{j=1}^{3} e^{−i q_j·r} · T_j

      T₁ = [[1,1],[1,1]]
      T₂ = [[e^{−iφ}, 1], [e^{iφ}, e^{−iφ}]]
      T₃ = [[e^{iφ}, 1], [e^{−iφ}, e^{iφ}]]，  φ = 2π/3

  w ≈ 110 meV（与 AB 堆叠双层实验能带标定，BM 正文）。

- **moiré 几何**（BM Fig. 1）：动量转移 |q_j| = k_θ = 2K_D·sin(θ/2)，
  K_D = 4π/(3a₀)（a₀ = 2.46 Å 晶格常数）；方向
  q̂₁ = (0,−1)、q̂₂ = (√3/2, 1/2)、q̂₃ = (−√3/2, 1/2)。
  moiré 倒格矢 b₁ = q₂−q₁、b₂ = q₃−q₁（BM 的 K 空间蜂巢格）；
  moiré 晶格常数 L_m = a₀/(2 sin(θ/2))（BM 正文：√3·a_cc/(2 sin(θ/2))，
  与 a_cc = a₀/√3 等价）。

- **平面波展开**（BM：Bloch 定理对任意转角成立，无需公度）：
  单一动量系：层 1 Dirac 点取为原点，层 2 Dirac 点位于 q₁（间隔 k_θ）。
  层 1 分量动量 p₁ = k+G，层 2 分量 p₂ = k+q₁+G（G = moiré 倒格矢，
  整数坐标 (m,n)·b₁+(m',n')·b₂ 基）。层间矩阵元 (1,G')←(2,G)::

      G' = G − (q_j − q₁) = G − {0, b₁, b₂}

  ——严格局域的"三方向耦合"。**配对约定**（易错点！）：动量为
  q₁+L 的层 2 态（L = 0, b₁, b₂ → 动量 q₁, q₂, q₃）依次携带幅度
  **T₁, T₃, T₂**——该配对由 BM 恒等式 Σ_j T_j(σ·q̂_j)T_j† = 0
  （hopping_identity，Eq. (SI-4)）唯一固定（自然顺序 T₁,T₂,T₃ 不满足）。
  共振检验：层 2 的 {G=0,b₁,b₂} 态 p₂ = {q₁,q₂,q₃}（|p₂| = k_θ ✓，
  即 BM H8 的 Ψ_j，满足 SI 的 h_j² = ε_θ²I）全部耦合到层 1 G=0。
  维度 4·N_G（N_G = 平面波数），远小于微观 TB 的 ~10⁴·θ⁻²。

- **物理锚点**：
  - 无量纲耦合 α = w/(v·k_θ)；第一壳层（8 带）解析速度
    v*/v = (1−3α²)/(1+6α²)（BM Eq. (8)），第一魔角 α₁ = 1/√3；
  - 数值魔角 θ ≈ 1.05°（w = 110 meV；BM Fig. 2/3：魔角序列
    1.05°/0.5°/0.35°/0.24°/0.2°），魔角处最低 moiré 带极端平化；
  - 手征对称（d=0、无势项时谱 ± 成对），k=0（moiré BZ 中心，
    即 K₁）恰有两个零能态（BM Supplementary 解析结果）。

实现注记：默认 ħv 取 (√3/2)|t|a₀ = 5.755 eV·Å（Reich 2002 的
t = −2.7 eV、a₀ = 2.46 Å，与 engine.models.HoneycombModel 默认一致）。
圆形平面波截断 |G| ≤ n_shells·|b| 下 k → k+b 的周期性仅在截断尺度
上破缺，对低能带影响可忽略（标准做法）。
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Tuple

import numpy as np

# 单层石墨烯 ħv_F（Reich 2002: t = −2.7 eV, a₀ = 2.46 Å → (√3/2)|t|a₀）
HBAR_V_GRAPHENE = np.sqrt(3) / 2 * 2.7 * 2.46
# 石墨烯晶格常数（实验值，与 structure.builders 一致）
A_LATTICE = 2.46
PHI = 2 * np.pi / 3

# BM Eq. (3)-(5) 的三个隧穿矩阵（d = 0）
_T1 = np.array([[1.0, 1.0], [1.0, 1.0]], complex)
_T2 = np.array([[np.exp(-1j * PHI), 1.0],
                [np.exp(1j * PHI), np.exp(-1j * PHI)]], complex)
_T3 = np.array([[np.exp(1j * PHI), 1.0],
                [np.exp(-1j * PHI), np.exp(1j * PHI)]], complex)
_T_MATRICES = (_T1, _T2, _T3)


@dataclass
class MoireGeometry:
    """moiré 倒空间几何（BM Fig. 1 约定）。"""

    theta_deg: float
    k_theta: float          # |q_j| = 2 K_D sin(θ/2) (Å⁻¹)
    q_vectors: np.ndarray   # (3,2) 动量转移 q_j (Å⁻¹)
    b1: np.ndarray          # moiré 倒格矢 (Å⁻¹)
    b2: np.ndarray
    k1: np.ndarray          # 层 1 Dirac 点（本模型动量原点）
    k2: np.ndarray          # 层 2 Dirac 点 (K₁ − q₁)

    @property
    def l_moire(self) -> float:
        """moiré 晶格常数 a₀/(2 sin(θ/2)) (Å)。"""
        return A_LATTICE / (2.0 * np.sin(np.radians(self.theta_deg) / 2))

    @property
    def bz_corner(self) -> np.ndarray:
        """moiré BZ 角点 K_m = (b₁+b₂)/3（BM 约定：BZ 中心 = 层 1 Dirac 点）。"""
        return (self.b1 + self.b2) / 3.0

    @property
    def bz_edge_center(self) -> np.ndarray:
        """moiré BZ 边中点 M_m = b₁/2。"""
        return self.b1 / 2.0


class TwistedBilayerGraphene:
    """BM 连续模型平面波求解器。

    参数:
        theta_deg: 相对转角 (度)。
        w: 层间隧穿幅度 (eV)，默认 0.110（BM 标定值）。
        hbar_v: 单层 Dirac 速度 ħv (eV·Å)，默认石墨烯 5.755。
        n_shells: 平面波壳层数，|G| ≤ n_shells·|b₁|。
        first_shell: True 时用 BM 8 带截断（层 1 仅 G=0，
            层 2 用 {0, b₁, b₂}）——v*/v 解析式严格成立的极限。

    复杂度: 维度 4·N_G，N_G ≈ π·n_shells²/sin60°。
    """

    def __init__(self, theta_deg: float = 1.05, w: float = 0.110,
                 hbar_v: float = HBAR_V_GRAPHENE, n_shells: int = 4,
                 first_shell: bool = False):
        if theta_deg <= 0 or w <= 0 or hbar_v <= 0:
            raise ValueError("θ、w、ħv 必须为正")
        if not first_shell and n_shells < 1:
            raise ValueError("n_shells ≥ 1")
        self.theta_deg = float(theta_deg)
        self.w = float(w)
        self.hbar_v = float(hbar_v)
        self.geom = self._build_geometry(theta_deg)
        self.pw1, self.pw2 = self._build_plane_waves(n_shells, first_shell)
        self._pw1_index = {tuple(g): i for i, g in enumerate(self.pw1)}
        self._coupling = self._build_coupling()

    # ------------------------------------------------------------------
    # 几何与基组
    # ------------------------------------------------------------------
    @staticmethod
    def _build_geometry(theta_deg: float) -> MoireGeometry:
        th = np.radians(theta_deg)
        k_theta = 2.0 * (4 * np.pi / (3 * A_LATTICE)) * np.sin(th / 2)
        dirs = np.array([[0.0, -1.0],
                         [np.sqrt(3) / 2, 0.5],
                         [-np.sqrt(3) / 2, 0.5]])
        q = k_theta * dirs
        return MoireGeometry(theta_deg=theta_deg, k_theta=k_theta,
                             q_vectors=q, b1=q[1] - q[0], b2=q[2] - q[0],
                             k1=np.zeros(2), k2=q[0])

    def _build_plane_waves(self, n_shells: int, first_shell: bool
                           ) -> Tuple[np.ndarray, np.ndarray]:
        """平面波整数坐标集合（层 1 / 层 2）。

        对称截断: |G| ≤ n_shells·|b₁|（圆形）；first_shell=True 时取
        BM Eq. (6) 的 8 带截断: 层 1 仅 G=0，层 2 用 {0, b₁, b₂}。
        """
        if first_shell:
            g1 = np.array([[0, 0]])
            g2 = np.array([[0, 0], [1, 0], [0, 1]])
            return g1, g2
        rng = np.arange(-n_shells, n_shells + 1)
        m, n = np.meshgrid(rng, rng, indexing="ij")
        cand = np.stack([m.ravel(), n.ravel()], axis=1)
        cart = cand[:, 0, None] * self.geom.b1 + cand[:, 1, None] * self.geom.b2
        bnorm = np.linalg.norm(self.geom.b1)
        keep = cand[np.linalg.norm(cart, axis=1) <= n_shells * bnorm + 1e-9]
        # 确定性排序: 先按 |G| 升序、再按 (m, n) 字典序
        order = np.lexsort((keep[:, 1], keep[:, 0],
                            np.linalg.norm(
                                keep[:, 0, None] * self.geom.b1
                                + keep[:, 1, None] * self.geom.b2, axis=1)))
        kept = keep[order]
        return kept, kept.copy()

    def _build_coupling(self) -> List[Tuple[int, int, np.ndarray]]:
        """层间耦合表 [(层2 指标, 层1 指标, T_j)]，G' = G − {0, b₁, b₂}。

        自然配对：动量 q₁+L 的层 2 态（L = 0, b₁, b₂ → q₁, q₂, q₃）
        依次携带 T₁, T₂, T₃——已验证满足 BM 恒等式
        Σ_j T_j(σ·q̂_j)T_j† = 0（hopping_identity）并精确复现
        v*/v = (1−3α²)/(1+6α²)（见 tests/test_moire.py）。
        """
        offsets = [(np.array([0, 0]), _T1),
                   (np.array([1, 0]), _T2),
                   (np.array([0, 1]), _T3)]
        table = []
        for j2, g2 in enumerate(self.pw2):
            for off, T in offsets:
                i1 = self._pw1_index.get(tuple(g2 - off))
                if i1 is not None:
                    table.append((j2, i1, T))
        return table

    # ------------------------------------------------------------------
    # 物理量
    # ------------------------------------------------------------------
    @property
    def alpha(self) -> float:
        """无量纲耦合 α = w/(ħv·k_θ)（moiré 带对 α 近似普适）。"""
        return self.w / (self.hbar_v * self.geom.k_theta)

    def _dirac_block(self, p: np.ndarray) -> np.ndarray:
        """单层 Dirac 块 h(p) = −ħv|p|[[0, e^{iφ_p}], [e^{−iφ_p}, 0]]。

        相位约定（重要）：BM Eq. (1) 的层旋转相位 e^{i(θ_k∓θ/2)} 在
        θ→0 系综下省略（BM 自己的 8 带解析推导同样略去——"the
        dependence of h(θ) on angle is parametrically small"）。
        这保证 BM 恒等式 Σ_j T_j(σ·q̂_j)T_j† = 0 严格成立，从而
        数值复现 v*/v = (1−3α²)/(1+6α²) 与魔角 θ ≈ 1.05°；
        保留 ±θ/2 相位会以 O(θ) 破坏相消干涉（魔角最小值被抬高
        ~40 倍），与 BM Fig. 3 矛盾。
        """
        pm = float(np.hypot(p[0], p[1]))
        if pm < 1e-14:
            return np.zeros((2, 2), complex)
        phase = np.arctan2(p[1], p[0])
        amp = -self.hbar_v * pm
        return amp * np.array([[0.0, np.exp(1j * phase)],
                               [np.exp(-1j * phase), 0.0]], complex)

    def hamiltonian(self, k) -> np.ndarray:
        """Bloch 哈密顿量 H(k)（eV，厄米；k 相对 moiré BZ 中心，Å⁻¹）。"""
        k = np.asarray(k, dtype=float)
        n1, n2 = len(self.pw1), len(self.pw2)
        dim = 2 * (n1 + n2)
        H = np.zeros((dim, dim), complex)

        # 层内（对角块）
        for i, g in enumerate(self.pw1):
            p = k + g[0] * self.geom.b1 + g[1] * self.geom.b2
            H[2 * i:2 * i + 2, 2 * i:2 * i + 2] = self._dirac_block(p)
        for j, g in enumerate(self.pw2):
            p = k + g[0] * self.geom.b1 + g[1] * self.geom.b2 + self.geom.q_vectors[0]
            r = 2 * (n1 + j)
            H[r:r + 2, r:r + 2] = self._dirac_block(p)

        # 层间（1 ← 2 幅值 w·T_j；共轭转置反向）
        for j2, i1, T in self._coupling:
            r = 2 * (n1 + j2)
            H[2 * i1:2 * i1 + 2, r:r + 2] = self.w * T
            H[r:r + 2, 2 * i1:2 * i1 + 2] = self.w * T.conj().T
        return H

    def bands(self, k_list, n_bands: Optional[int] = None) -> np.ndarray:
        """沿 k 列表求本征值（升序），返回 (n_k, n_bands)。"""
        vals = []
        for k in k_list:
            vals.append(np.linalg.eigvalsh(self.hamiltonian(k)))
        out = np.array(vals)
        return out if n_bands is None else out[:, :n_bands]

    # ------------------------------------------------------------------
    # 魔角诊断量
    # ------------------------------------------------------------------
    def dirac_velocity(self, dk: Optional[float] = None,
                       n_k: int = 3) -> float:
        """Dirac 点速度 v* (eV·Å)：k=0 附近最低正带色散的线性拟合斜率。

        k=0 处两个零能态 = Dirac 锥的价带/导带两支；小 k 时
        E_c(k) ≈ +v*·k（沿 x̂），对最低正带做过原点最小二乘。
        dk 默认取 0.02·k_θ（抑制色散高阶项污染）。
        符号保留（穿过魔角时 v* 变号——BM Fig. 3 的物理）。
        """
        if dk is None:
            dk = 0.02 * self.geom.k_theta
        ks = [(i + 1) * dk * np.array([1.0, 0.0]) for i in range(n_k)]
        E = self.bands(ks)
        dim = E.shape[1]
        pos = E[:, dim // 2]                      # 最低正带（Dirac 锥导带支）
        kk = np.array([(i + 1) * dk for i in range(n_k)])
        return float(np.sum(pos * kk) / np.sum(kk * kk))   # 过原点最小二乘

    def flat_band_width(self, n_grid: int = 12) -> float:
        """最低正 moiré 带带宽 W = max−min（BZ 平行四边形采样, eV）。"""
        s = np.arange(n_grid) / n_grid
        ss, tt = np.meshgrid(s, s, indexing="ij")
        ks = [a * self.geom.b1 + b * self.geom.b2
              for a, b in zip(ss.ravel(), tt.ravel())]
        E = self.bands(ks)
        dim = E.shape[1]
        return float(E[:, dim // 2].max() - E[:, dim // 2].min())

    def moire_path(self, n_seg: int = 40) -> Tuple[np.ndarray, List[str], List[int]]:
        """Γ_m → M_m → K_m → Γ_m 高对称路径（BM 约定，BZ 中心 = k=0）。

        返回:
            (路径点 (n,2), 标签列表, 刻位置（路径点索引）)
        """
        G = self.geom
        labels = ["Γ", "M", "K", "Γ"]
        pts = [np.zeros(2), G.bz_edge_center, G.bz_corner, np.zeros(2)]
        path = []
        for a, b in zip(pts[:-1], pts[1:]):
            seg = np.linspace(0, 1, n_seg, endpoint=False)[:, None]
            path.append(a + seg * (b - a))
        path.append(np.zeros(2)[None, :])
        ticks = [i * n_seg for i in range(len(pts))]
        return np.vstack(path), labels, ticks
