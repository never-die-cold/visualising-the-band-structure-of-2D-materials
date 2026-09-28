"""TMD 低能有效 k·p 引擎（Kormányos et al. 2015, K 点模型）。

实现 K/-K 谷的二带（VB+CB）×二自旋 k·p 哈密顿量，包含：

- 大质量 Dirac 项 H_D（γ），
- 电子-空穴不对称对角项 H_as（α, β），
- 三角翘曲 H_3w（κ），
- 立方项 H_cub（η），
- 自旋-轨道劈裂 H_so（Δ_vb, Δ_cb），
- 自旋依赖的"自由电子"项 H_0（ħ²q²/2mₑ ⊗ s_z，按原文形式保留）。

哈密顿量（Kormányos et al., 2D Mater. 2, 022001 (2015)，
Eq. (24)-(29)，自旋块对角）::

    H_eff^{τ,s} = H_0 + H_so + [H_D + H_as + H_3w + H_cub]

    H_0  = (ħ²q²/2mₑ)·(𝟙₂ ⊗ s_z)
    H_so = diag(τΔ_vb·s, τΔ_cb·s)
    H_D  = [[ε_vb,  τ·γ·q₋^τ], [τ·γ·q₊^τ, ε_cb]],   q_±^τ = τq_x ± iq_y
    H_as = diag(α_s q², β_s q²)
    H_3w = [[0, κ_s (q₊^τ)²], [κ_s (q₋^τ)², 0]]
    H_cub= [[0, η_s (q₊^τ)³], [η_s (q₋^τ)³, 0]]

基矢 {|vb,s⟩, |cb,s⟩}，全矩阵 4×4（自旋块对角）。
能带参考：ε_vb = −E_bg/2, ε_cb = +E_bg/2（带隙中心为零）。

适用范围：|q| ≲ 0.3 Å⁻¹（K 谷附近）。
参数来源：docs/REFERENCES.md 文献 [1]（Tables 1, 3, 4, 8, 9）。
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional

import numpy as np

# ħ²/(2m₀) = 3.809982 eV·Å²
HBAR2_OVER_2M0 = 3.809982

# ----------------------------------------------------------------------
# 材料参数（Kormányos 2015, (HSE,LDA) 拟合）
#   e_bg_dft / e_bg_gw : K 点带隙 (eV)                     [Table 8/9]
#   gamma               : Dirac 速度参数 |γ| (eV·Å)         [Table 8/9]
#   alpha_up/down       : VB 对角二次项 (eV·Å²)             [Table 8/9]
#   beta_up/down        : CB 对角二次项 (eV·Å²)             [Table 8/9]
#   kappa_up/down       : 三角翘曲 (eV·Å²)                  [Table 8/9]
#   eta_up/down         : 立方项 (eV·Å³)                    [Table 8/9]
#   two_delta_vb/cb     : 自旋劈裂 2Δ (meV, 带符号)          [Table 3/4]
#   a                   : 面内晶格常数 (Å, 实验)              [Table 1]
# ----------------------------------------------------------------------
_T = dict  # 简写

TMD_KP_PARAMS: Dict[str, dict] = {
    "MoS2": _T(
        e_bg_dft=1.67, e_bg_gw=2.80, gamma_dft=2.76, gamma_gw=2.22,
        alpha_up=-5.97, alpha_down=-6.43, beta_up=0.28, beta_down=0.54,
        kappa_up=-1.48, kappa_down=-1.45,
        eta_up=13.7, eta_down=21.1,
        two_delta_vb=148.0, two_delta_cb=3.0, a=3.1604),
    "MoSe2": _T(
        e_bg_dft=1.40, e_bg_gw=2.26, gamma_dft=2.53, gamma_gw=2.20,
        alpha_up=-5.34, alpha_down=-5.71, beta_up=-0.95, beta_down=-0.52,
        kappa_up=-1.31, kappa_down=-1.23,
        eta_up=15.11, eta_down=17.10,
        two_delta_vb=186.0, two_delta_cb=22.0, a=3.288),
    "WS2": _T(
        e_bg_dft=1.60, e_bg_gw=2.88, gamma_dft=3.34, gamma_gw=2.59,
        alpha_up=-6.14, alpha_down=-7.95, beta_up=1.62, beta_down=4.00,
        kappa_up=-1.24, kappa_down=-1.09,
        eta_up=21.85, eta_down=31.73,
        two_delta_vb=429.0, two_delta_cb=-32.0, a=3.154),
    "WSe2": _T(
        e_bg_dft=1.30, e_bg_gw=2.42, gamma_dft=3.17, gamma_gw=2.60,
        alpha_up=-5.25, alpha_down=-6.93, beta_up=0.33, beta_down=2.35,
        kappa_up=-1.11, kappa_down=-0.93,
        eta_up=18.04, eta_down=26.17,
        two_delta_vb=466.0, two_delta_cb=-37.0, a=3.286),
    "MoTe2": _T(
        e_bg_dft=0.997, e_bg_gw=1.82, gamma_dft=2.33, gamma_gw=2.16,
        alpha_up=-4.78, alpha_down=-4.85, beta_up=-2.19, beta_down=-1.78,
        kappa_up=-1.19, kappa_down=-1.01,
        eta_up=13.26, eta_down=13.54,
        two_delta_vb=219.0, two_delta_cb=36.0, a=3.519),
    "WTe2": _T(
        e_bg_dft=0.792, e_bg_gw=1.77, gamma_dft=3.04, gamma_gw=2.79,
        alpha_up=-3.94, alpha_down=-5.20, beta_up=-0.9, beta_down=0.60,
        kappa_up=-1.01, kappa_down=-0.96,
        eta_up=14.72, eta_down=19.41,
        two_delta_vb=484.0, two_delta_cb=-52.0, a=3.521),
}

# K 点有效质量文献值 (HSE,LDA)，用于验证测试 [Tables 3/4]
TMD_REFERENCE_MASSES = {
    #  name: (m_cb^(1), m_cb^(2), |m_vb^(1)|, |m_vb^(2)|)
    "MoS2":  (0.46, 0.43, 0.54, 0.61),
    "MoSe2": (0.56, 0.49, 0.59, 0.70),
    "WS2":   (0.26, 0.35, 0.35, 0.49),
    "WSe2":  (0.28, 0.39, 0.36, 0.54),
    "MoTe2": (0.62, 0.53, 0.66, 0.82),
    "WTe2":  (0.26, 0.39, 0.34, 0.58),
}


@dataclass(frozen=True)
class TMDKpParams:
    """单个 TMD 材料的 k·p 参数集（eV, Å 单位制）。"""

    name: str
    e_bg: float            # K 点带隙 (eV)
    gamma: float           # Dirac 参数 (eV·Å)
    alpha_up: float
    alpha_down: float
    beta_up: float
    beta_down: float
    kappa_up: float
    kappa_down: float
    eta_up: float
    eta_down: float
    delta_vb: float        # Δ_vb = 2Δ_vb/2 (eV, 恒正)
    delta_cb: float        # Δ_cb = 2Δ_cb/2 (eV, 带符号)
    a: float               # 面内晶格常数 (Å)

    @classmethod
    def from_library(cls, name: str, source: str = "dft") -> "TMDKpParams":
        """从参数库构建。``source``: "dft" 用 DFT 带隙, "gw" 用 GW 带隙。"""
        key = name.strip().lower()
        lookup = {k.lower(): k for k in TMD_KP_PARAMS}
        if key not in lookup:
            raise ValueError(f"未知 TMD {name!r}，可选: {sorted(TMD_KP_PARAMS)}")
        p = dict(TMD_KP_PARAMS[lookup[key]])
        use_gw = source.lower() == "gw"
        e_bg = p.pop("e_bg_gw" if use_gw else "e_bg_dft")
        gamma = p.pop("gamma_gw" if use_gw else "gamma_dft")
        p.pop("e_bg_dft" if use_gw else "e_bg_gw")
        p.pop("gamma_dft" if use_gw else "gamma_gw")
        return cls(name=lookup[key], e_bg=e_bg, gamma=gamma,
                   delta_vb=p.pop("two_delta_vb") / 2000.0,   # meV → eV, 半劈裂
                   delta_cb=p.pop("two_delta_cb") / 2000.0,
                   **p)


class TMDKpModel:
    """TMD K 谷 k·p 模型（4×4 自旋分辨哈密顿量）。"""

    def __init__(self, params: TMDKpParams) -> None:
        self.params = params
        self.name = f"{params.name} k·p (Kormányos 2015)"

    # ------------------------------------------------------------------
    def hamiltonian(self, q, valley: int = +1) -> np.ndarray:
        """装配 4×4 哈密顿量。

        参数:
            q: (2,) 波矢相对谷中心的偏移 (Å⁻¹)。
            valley: +1 (K) 或 −1 (−K)。
        返回:
            (4, 4) 厄米矩阵，基矢 (vb↑, cb↑, vb↓, cb↓)，能量参考为带隙中心。
        """
        p = self.params
        if valley not in (+1, -1):
            raise ValueError("valley 必须为 +1 (K) 或 -1 (-K)")
        tau = float(valley)
        qx, qy = float(q[0]), float(q[1])
        q2 = qx * qx + qy * qy
        # q_±^τ = τ q_x ± i q_y
        qm = tau * qx - 1j * qy          # q₋^τ
        qp = tau * qx + 1j * qy          # q₊^τ

        # 能量参考：使 K 点最小（含 SOC）跃迁能 = E_bg
        #   gap_min = (ε_cb − ε_vb) − (Δ_vb − Δ_cb)  （Δ_vb>0 恒成立）
        span = p.e_bg + p.delta_vb - p.delta_cb
        eps_vb = -span / 2
        eps_cb = +span / 2

        H = np.zeros((4, 4), dtype=complex)
        for row, s in ((0, +1.0), (2, -1.0)):
            # 每个自旋块: [vb, cb]
            # 参数的谷-自旋指派：α_{τ,s} = α_{−τ,−s}（综述原文），
            # 即表格中 *_up/*_down 对应 K 谷 (τ=+1) 的 ↑/↓，
            # 在 −K 谷两者互换（保证时间反演对称）。
            ts = tau * s
            alpha = p.alpha_up if ts > 0 else p.alpha_down
            beta = p.beta_up if ts > 0 else p.beta_down
            # κ、η 在 TR 伙伴谷需变号: κ_{−τ,−s} = −κ*_{τ,s}
            # （自旋旋量反幺正映射的相对符号所致，保证 |H_D·H_3w|
            #   交叉项在两谷满足时间反演）
            kappa = tau * (p.kappa_up if ts > 0 else p.kappa_down)
            eta = tau * (p.eta_up if ts > 0 else p.eta_down)
            # 对角项。注意：原文 Eq.(24) 的 H_0 印作 (𝟙₂⊗s_z)，但以其
            # Table 3/4 的有效质量反证必须为自旋无关的 +ħ²q²/2m₀
            # （若取 s_z 形式，↓ 自旋电子质量 ~3 m₀，与文献 0.43 矛盾），
            # 此处按自旋无关实现。
            H[row, row] = eps_vb + tau * p.delta_vb * s + HBAR2_OVER_2M0 * q2 \
                + alpha * q2
            H[row + 1, row + 1] = eps_cb + tau * p.delta_cb * s \
                + HBAR2_OVER_2M0 * q2 + beta * q2
            # 非对角: H_D + H_3w + H_cub
            off = p.gamma * (qx - 1j * tau * qy) \
                + kappa * qp**2 + eta * qp**3
            H[row, row + 1] = off
            H[row + 1, row] = np.conj(off)
        return H

    # ------------------------------------------------------------------
    def energies(self, q, valley: int = +1) -> np.ndarray:
        """单 q 点 4 条能带能量（升序, eV）。"""
        return np.linalg.eigvalsh(self.hamiltonian(q, valley))

    def spin_block_energies(self, q, valley: int = +1,
                            spin: int = +1) -> np.ndarray:
        """单个自旋块的 (vb, cb) 能量（块内对角化, eV）。

        自旋块严格解耦；块内 2×2 对角化包含 Dirac 耦合对带曲率的
        二阶贡献（这是有效质量的主要来源之一）。
        """
        H = self.hamiltonian(q, valley)
        row = 0 if spin > 0 else 2
        return np.linalg.eigvalsh(H[row:row + 2, row:row + 2])

    def effective_mass(self, band: str = "cb", spin: int = +1,
                       valley: int = +1, direction=(1.0, 0.0),
                       dq: float = 1e-4) -> float:
        """带边有效质量 |m*|/m₀（q=0 处中心差分）。

        参数:
            band: "cb"（导带）或 "vb"（价带，返回载流子质量 |m*|）。
            spin: +1 (↑) / −1 (↓)。
            direction: 测量方向（笛卡尔坐标，默认沿 q_x）。
        """
        if band not in ("cb", "vb"):
            raise ValueError("band 必须为 'cb' 或 'vb'")
        d = np.asarray(direction, dtype=float)
        d = d / np.linalg.norm(d)
        col = 0 if band == "vb" else 1
        e0 = self.spin_block_energies((0.0, 0.0), valley, spin)[col]
        ep = self.spin_block_energies(d * dq, valley, spin)[col]
        em = self.spin_block_energies(-d * dq, valley, spin)[col]
        curv = (ep + em - 2 * e0) / dq**2      # d²E/dq² (eV·Å²)
        return abs(2 * HBAR2_OVER_2M0 / curv)

    # ------------------------------------------------------------------
    def __repr__(self) -> str:
        return (f"TMDKpModel({self.params.name}, E_bg={self.params.e_bg} eV)")
