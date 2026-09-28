"""MoS₂ sp³d⁵ 非正交 Slater-Koster 紧束缚模型（Zahid et al. 2013）。

实现 Zahid, Liu, Zhu, Wang, Guo, Phys. Rev. B **87**, 125302 (2013)
的 96 参数模型：sp³d⁵ 非正交基、最近邻、Slater-Koster 两点积分 +
on-site 自旋轨道（arXiv:1304.0074 Table 3，参数逐一转录）。

- 基组：每原子 9 轨道 (s, p×3, d×5) × 2 自旋 = 18 态；
  单层 MoS₂（3 原子）共 54×54 广义本征值问题 H(k)c = E·S(k)c；
- 近邻：Mo–S 六键（三棱柱）+ 同亚层 S–S、Mo–Mo（距离 a）；
- 自旋轨道：on-site λ·(LxSx+LySy+LzSz)，p/d 壳层用该元素的 λ。

验证目标（论文 Table 1/2）：单层 K_v1→K_c = 1.805 eV、
K_v2→K_c = 1.969 eV（VB 劈裂 0.164 eV）、m_e*(K) ≈ 0.43 m₀、
m_h*(K) ≈ 0.46 m₀。

SK 角因子的构造与验证见 ``slater_koster.py``。
"""

from __future__ import annotations

from typing import Dict, List, Tuple

import numpy as np
import scipy.linalg

from ..structure.lattice import Lattice
from .slater_koster import ORBITAL_SHELL, _P_HAT, D_TENSORS, _perp_basis

# ----------------------------------------------------------------------
# Zahid 2013 Table 3 参数（eV，逐一转录自 arXiv:1304.0074）
# ----------------------------------------------------------------------
ONSITE = {
    "S":  {"s": 7.6595, "p": -2.1537, "d": 8.7689, "lambda": 0.2129},
    "Mo": {"s": 5.5994, "p": 6.7128, "d": 2.6429, "lambda": 1.0675},
}

# 异种原子对 (S, Mo)：行名 = ⟨X_S|H|Y_Mo⟩ 的两点积分
INTEGRALS_SMO_H = {
    "ssσ": -0.0917, "spσ": 0.6656, "psσ": -1.6515,
    "ppσ": 1.4008, "ppπ": -0.4812,
    "sdσ": 0.2177, "dsσ": -1.0654,
    "pdσ": -2.8732, "dpσ": 2.1898, "pdπ": 0.7739, "dpπ": -1.9408,
    "ddσ": -3.1425, "ddπ": 2.4975, "ddδ": -0.3703,
}
INTEGRALS_SMO_S = {
    "ssσ": 0.0294, "spσ": 0.1042, "psσ": 0.1765,
    "ppσ": -0.1865, "ppπ": 0.0303,
    "sdσ": -0.0480, "dsσ": -0.1432,
    "pdσ": 0.0942, "dpσ": 0.2002, "pdπ": 0.0132, "dpπ": -0.2435,
    "ddσ": 0.0273, "ddπ": 0.1940, "ddδ": 0.1261,
}
INTEGRALS_SS_H = {
    "ssσ": 0.3093, "spσ": -0.9210, "ppσ": 0.7132, "ppπ": -0.1920,
    "sdσ": -0.2016, "pdσ": -0.5204, "pdπ": -0.1203,
    "ddσ": 0.8347, "ddπ": 0.7434, "ddδ": -0.1919,
}
INTEGRALS_SS_S = {
    "ssσ": -0.0532, "spσ": 0.0240, "ppσ": 0.0478, "ppπ": -0.0104,
    "sdσ": 0.0946, "pdσ": 0.0724, "pdπ": 0.0772,
    "ddσ": 0.1849, "ddπ": -0.0429, "ddδ": -0.0333,
}
INTEGRALS_MOMO_H = {
    "ssσ": 0.1768, "spσ": 1.0910, "ppσ": -0.3842, "ppπ": 0.5203,
    "sdσ": -0.5635, "pdσ": -0.2316, "pdπ": 0.0582,
    "ddσ": 0.3602, "ddπ": 0.0432, "ddδ": 0.1008,
}
INTEGRALS_MOMO_S = {
    "ssσ": -0.0575, "spσ": 0.0057, "ppσ": 0.0296, "ppπ": 0.0946,
    "sdσ": -0.1082, "pdσ": 0.0212, "pdπ": -0.0448,
    "ddσ": -0.0216, "ddπ": -0.0285, "ddδ": 0.0432,
}

ORBITALS = ("s", "px", "py", "pz", "dxy", "dyz", "dzx", "dx2-y2", "dz2")
SHELL = dict(ORBITAL_SHELL)
_SHELL_ORBS = {1: ["px", "py", "pz"], 2: ["dxy", "dyz", "dzx", "dx2-y2", "dz2"]}
_ONSITE_KEY = {"s": "s", "px": "p", "py": "p", "pz": "p", "dxy": "d",
               "dyz": "d", "dzx": "d", "dx2-y2": "d", "dz2": "d"}


# ----------------------------------------------------------------------
# on-site 自旋轨道：λ·(LxSx + LySy + LzSz)
# ----------------------------------------------------------------------
def _real_to_complex_l1() -> np.ndarray:
    """p 实基 (px,py,pz) 在复 m 基 (m=-1,0,1) 下的展开 U[m, real]。"""
    U = np.zeros((3, 3), dtype=complex)
    U[0, 0] = 1 / np.sqrt(2)          # px = (Y_{1,-1} - Y_{1,1})/√2
    U[2, 0] = -1 / np.sqrt(2)
    U[0, 1] = 1j / np.sqrt(2)         # py = i(Y_{1,-1} + Y_{1,1})/√2
    U[2, 1] = 1j / np.sqrt(2)
    U[1, 2] = 1.0                     # pz = Y_{1,0}
    return U


def _real_to_complex_l2() -> np.ndarray:
    """d 实基在复 m 基 (m=-2..2) 下的展开 U[m, real]。"""
    U = np.zeros((5, 5), dtype=complex)
    r = {nm: i for i, nm in enumerate(("dxy", "dyz", "dzx", "dx2-y2", "dz2"))}
    mi = {k: k + 2 for k in (-2, -1, 0, 1, 2)}
    U[mi[2], r["dxy"]] = -1j / np.sqrt(2)    # dxy = -i(Y22 - Y2,-2)/√2
    U[mi[-2], r["dxy"]] = 1j / np.sqrt(2)
    U[mi[2], r["dx2-y2"]] = 1 / np.sqrt(2)   # dx2-y2 = (Y22 + Y2,-2)/√2
    U[mi[-2], r["dx2-y2"]] = 1 / np.sqrt(2)
    U[mi[-1], r["dzx"]] = 1 / np.sqrt(2)     # dzx = (Y2,-1 - Y21)/√2
    U[mi[1], r["dzx"]] = -1 / np.sqrt(2)
    U[mi[1], r["dyz"]] = 1j / np.sqrt(2)     # dyz = i(Y21 + Y2,-1)/√2
    U[mi[-1], r["dyz"]] = 1j / np.sqrt(2)
    U[mi[0], r["dz2"]] = 1.0                 # dz2 = Y20
    return U


def angular_momentum_matrices(l: int) -> List[np.ndarray]:
    """实轨道基下的 (Lx, Ly, Lz)/ħ（无量纲厄米矩阵），l = 1 或 2。"""
    n = 2 * l + 1
    m = np.arange(-l, l + 1)
    Lz_c = np.diag(m.astype(complex))
    Lp_c = np.zeros((n, n), dtype=complex)
    for i in range(n - 1):
        Lp_c[i + 1, i] = np.sqrt(l * (l + 1) - m[i] * (m[i] + 1))
    U = _real_to_complex_l1() if l == 1 else _real_to_complex_l2()
    Uc = U.conj().T
    Lz = Uc @ Lz_c @ U
    Lp = Uc @ Lp_c @ U
    Lm = Lp.conj().T
    return [(Lp + Lm) / 2, (Lp - Lm) / 2j, Lz]


def onsite_soc(lambda_so: float) -> np.ndarray:
    """单原子 on-site SOC（18×18，9 轨道 × 2 自旋），p/d 壳层用 λ。"""
    n = 9
    H = np.zeros((2 * n, 2 * n), dtype=complex)
    S = [np.array([[0, 1], [1, 0]], complex) / 2,
         np.array([[0, -1j], [1j, 0]], complex) / 2,
         np.array([[1, 0], [0, -1]], complex) / 2]
    for shell in (1, 2):
        Ls = angular_momentum_matrices(shell)
        orbs = _SHELL_ORBS[shell]
        for a, nm in enumerate(orbs):
            for b, nm2 in enumerate(orbs):
                i, j = ORBITALS.index(nm), ORBITALS.index(nm2)
                for k in range(3):
                    H[i * 2:i * 2 + 2, j * 2:j * 2 + 2] += \
                        lambda_so * Ls[k][a, b] * S[k]
    return H


# ----------------------------------------------------------------------
# SK 通道系数与矩阵元（与 slater_koster 模块一致的紧凑实现）
# ----------------------------------------------------------------------
def _channel_coeffs(orbital: str, rhat: np.ndarray) -> Dict:
    rhat = np.asarray(rhat, float)
    rhat = rhat / np.linalg.norm(rhat)
    e1, e2 = _perp_basis(rhat)
    shell = SHELL[orbital]
    if shell == 0:
        return {"sigma": 1.0, "pi": (0.0, 0.0), "delta": (0.0, 0.0)}
    if shell == 1:
        p = _P_HAT[orbital]
        return {"sigma": float(p @ rhat),
                "pi": (float(p @ e1), float(p @ e2)),
                "delta": (0.0, 0.0)}
    T = D_TENSORS[orbital]
    T_sigma = (3.0 * np.outer(rhat, rhat) - np.eye(3)) / np.sqrt(6.0)
    t1 = 0.5 * (np.outer(rhat, e1) + np.outer(e1, rhat))
    t2 = 0.5 * (np.outer(rhat, e2) + np.outer(e2, rhat))
    t1 /= np.linalg.norm(t1)
    t2 /= np.linalg.norm(t2)
    t3 = (np.outer(e1, e1) - np.outer(e2, e2)) / np.sqrt(2)
    t4 = (np.outer(e1, e2) + np.outer(e2, e1)) / np.sqrt(2)
    return {"sigma": float(np.tensordot(T, T_sigma)),
            "pi": (float(np.tensordot(T, t1)), float(np.tensordot(T, t2))),
            "delta": (float(np.tensordot(T, t3)), float(np.tensordot(T, t4)))}


def _ch_name(l1: int, l2: int, m: int) -> str:
    letter = {0: "s", 1: "p", 2: "d"}
    suffix = {0: "σ", 1: "π", 2: "δ"}
    return f"{letter[l1]}{letter[l2]}{suffix[m]}"


def sk_element(orb_a: str, orb_b: str, rhat: np.ndarray,
               V: Dict[str, float]) -> float:
    """⟨orb_a@A|H|orb_b@B⟩，R̂ = B−A。异种对需提供成对行（spσ/psσ…）。

    单种原子对的积分表是对称的（只有 sdσ 无 dsσ 行）：反向查找时
    回退到交换壳层顺序的键名。
    """
    ca = _channel_coeffs(orb_a, rhat)
    cb = _channel_coeffs(orb_b, rhat)
    la, lb = SHELL[orb_a], SHELL[orb_b]

    def lookup(l1: int, l2: int, m: int) -> float:
        name = _ch_name(l1, l2, m)
        if name not in V and l1 > l2:
            name = _ch_name(l2, l1, m)
        return V.get(name, 0.0)

    total = lookup(la, lb, 0) * ca["sigma"] * cb["sigma"]
    if min(la, lb) >= 1:
        total += lookup(la, lb, 1) * (
            ca["pi"][0] * cb["pi"][0] + ca["pi"][1] * cb["pi"][1])
    if la == 2 and lb == 2:
        total += lookup(2, 2, 2) * (
            ca["delta"][0] * cb["delta"][0] + ca["delta"][1] * cb["delta"][1])
    return total


# ----------------------------------------------------------------------
# 模型
# ----------------------------------------------------------------------
class ZahidMoS2Model:
    """单层 MoS₂ sp³d⁵ 非正交 TB（Zahid 2013）。

    参数:
        a: 面内晶格常数 (Å)，论文 PBE-D2 优化值 3.179。
        dxx: 面外 S–S 距离 (Å)，论文值 3.135。
        soc: 是否开启自旋轨道耦合。
        unlike_sign: 异种原子对角因子的整体符号 ±1
            （SK 表键方向约定，由文献带隙值校验确定）。
    """

    VACUUM = 15.0

    def __init__(self, a: float = 3.179, dxx: float = 3.135,
                 soc: bool = True, unlike_sign: float = 1.0) -> None:
        self.lattice = Lattice.hexagonal(a, vacuum=self.VACUUM)
        self.a = a
        self.dxx = dxx
        self.soc = soc
        self.unlike_sign = float(unlike_sign)
        self.name = "MoS2 sp3d5 (Zahid 2013)"
        h = dxx / 2
        self.sites = [("Mo", np.array([0.0, 0.0, 0.5])),
                      ("S", np.array([1 / 3, 2 / 3, 0.5 + h / self.VACUUM])),
                      ("S", np.array([2 / 3, 1 / 3, 0.5 - h / self.VACUUM]))]
        self.n_atoms = 3
        self.n_basis = 54
        self.bonds = self._build_bonds()

    # ------------------------------------------------------------------
    def _build_bonds(self) -> List[Tuple[int, int, np.ndarray, str]]:
        """近邻键表: (atom_i, atom_j, dfrac, kind)。

        kind: "SMo"（i=S, j=Mo, 用 (S,Mo) 积分列）、"SS"、"MoMo"。
        """
        bonds: List[Tuple[int, int, np.ndarray, str]] = []
        bond_len = float(np.sqrt(self.a ** 2 / 3 + (self.dxx / 2) ** 2))
        # Mo–S 六键：对每个 S 枚举 Mo 镜像，保留距离 = Mo–S 键长者
        for i, s_frac in ((1, self.sites[1][1]), (2, self.sites[2][1])):
            count = 0
            for nx in range(-2, 3):
                for ny in range(-2, 3):
                    d = np.array([nx, ny, 0.5]) - s_frac   # Mo z 分数 = 0.5
                    cart = self.lattice.frac_to_cart(d)
                    dist = float(np.linalg.norm(cart))
                    if abs(dist - bond_len) < 0.05 * bond_len:
                        bonds.append((i, 0, d, "SMo"))
                        count += 1
            assert count == 3, f"S{i} 的 Mo 近邻数 = {count} (应为 3)"
        # 同亚层 S–S、Mo–Mo 六近邻（面内距离 = a；γ=120° 六方格子）。
        # 取 ± 对中的 3 个代表镜像，配合共轭填充即可覆盖全部 6 近邻
        # 且保证 H(k)/S(k) 厄米（SK 矩阵元对 R̂ 含奇宇称通道）。
        images = [(1, 0), (0, 1), (1, 1)]
        for i in (1, 2):
            for dx, dy in images:
                bonds.append((i, i, np.array([dx, dy, 0.0]), "SS"))
        for dx, dy in images:
            bonds.append((0, 0, np.array([dx, dy, 0.0]), "MoMo"))
        return bonds

    # ------------------------------------------------------------------
    def hamiltonian_overlap(self, kfrac) -> Tuple[np.ndarray, np.ndarray]:
        """装配 k 点的 (H, S)。kfrac: 倒格子分数坐标 (2,) 或 (3,)。"""
        k = np.asarray(kfrac, float).reshape(-1)[:3]
        if k.size == 2:
            k = np.r_[k, 0.0]
        TWO_PI = 2 * np.pi
        n = self.n_basis
        H = np.zeros((n, n), dtype=complex)
        S = np.zeros((n, n), dtype=complex)

        def add_pair(i: int, j: int, dfrac, Vh: Dict, Vs: Dict,
                     sign: float, symmetric: bool) -> None:
            phase = np.exp(1j * TWO_PI * float(np.dot(k, dfrac)))
            rhat = self.lattice.frac_to_cart(dfrac)
            rhat /= np.linalg.norm(rhat)
            for a_i, oa in enumerate(ORBITALS):
                for b_i, ob in enumerate(ORBITALS):
                    v = sign * sk_element(oa, ob, rhat, Vh) * phase
                    s_ = sign * sk_element(oa, ob, rhat, Vs) * phase
                    bi = (i * 18 + a_i * 2, i * 18 + a_i * 2 + 2)
                    bj = (j * 18 + b_i * 2, j * 18 + b_i * 2 + 2)
                    H[bi[0]:bi[1], bj[0]:bj[1]] += v * np.eye(2)
                    S[bi[0]:bi[1], bj[0]:bj[1]] += s_ * np.eye(2)
                    # 共轭填充：同种原子块也必须显式填充（SK 矩阵元
                    # 对 R̂ 有奇宇称通道，仅靠 ± 镜像求和不能保证厄米）
                    if symmetric:
                        H[bj[0]:bj[1], bi[0]:bi[1]] += np.conj(v) * np.eye(2)
                        S[bj[0]:bj[1], bi[0]:bi[1]] += np.conj(s_) * np.eye(2)

        for (i, j, dfrac, kind) in self.bonds:
            if kind == "SMo":
                # ⟨S|H|Mo⟩ 块 + 厄米共轭块 ⟨Mo|H|S⟩
                add_pair(i, j, dfrac, INTEGRALS_SMO_H, INTEGRALS_SMO_S,
                         self.unlike_sign, symmetric=True)
            elif kind == "SS":
                add_pair(i, j, dfrac, INTEGRALS_SS_H, INTEGRALS_SS_S,
                         1.0, symmetric=True)
            else:
                add_pair(i, j, dfrac, INTEGRALS_MOMO_H, INTEGRALS_MOMO_S,
                         1.0, symmetric=True)

        # on-site 能量
        for atom, (sp, _) in enumerate(self.sites):
            off = atom * 18
            for oi, o in enumerate(ORBITALS):
                e = ONSITE[sp][_ONSITE_KEY[o]]
                H[off + oi * 2: off + oi * 2 + 2,
                  off + oi * 2: off + oi * 2 + 2] += e * np.eye(2)
        # on-site 重叠（正交归一：S_ii = 1）
        S += np.eye(n, dtype=complex)
        # on-site SOC
        if self.soc:
            for atom, (sp, _) in enumerate(self.sites):
                off = atom * 18
                H[off:off + 18, off:off + 18] += onsite_soc(ONSITE[sp]["lambda"])
        return H, S

    # ------------------------------------------------------------------
    def bands(self, kfracs) -> np.ndarray:
        """广义本征值 E(k)。kfracs: (N, 2) 倒格分数坐标。返回 (N, 54) 升序。

        .. warning::
           已知问题：按论文转录的重叠积分在本文的 SK 角因子约定下，
           S(k) 在部分 k 点非正定（Mo–Mo 的 d-d π 元素符号与本文约定
           不一致）。Zahid 原文未写明其 SK 表的方向/相位约定（Nanoskif
           内部约定），在获得该约定参考前，本求解器对非正定 S(k)
           **显式抛错**而不输出未定义的结果。见 ``docs/REFERENCES.md``。
        """
        kfracs = np.asarray(kfracs, float)
        out = np.empty((len(kfracs), self.n_basis))
        for ik, kf in enumerate(kfracs):
            H, S = self.hamiltonian_overlap(kf)
            wS = np.linalg.eigvalsh(S)
            if wS.min() < -1e-8:
                raise RuntimeError(
                    f"S(k) 非正定 (min eig = {wS.min():.4g} @ k={kf})："
                    "SK 方向/相位约定与拟合参数不匹配，结果无定义。"
                    "见 docs/REFERENCES.md 中 'Nanoskif SK 约定' 待办。")
            out[ik] = scipy.linalg.eigh(H, S, eigvals_only=True)
        return out

    def energies_at(self, kfrac) -> np.ndarray:
        H, S = self.hamiltonian_overlap(kfrac)
        return scipy.linalg.eigh(H, S, eigvals_only=True)

    def __repr__(self) -> str:
        return (f"ZahidMoS2Model(a={self.a}, dxx={self.dxx}, "
                f"soc={self.soc}, n_basis={self.n_basis})")
