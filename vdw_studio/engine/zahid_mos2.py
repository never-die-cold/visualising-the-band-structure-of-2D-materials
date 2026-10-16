"""MoS₂ sp³d⁵ 非正交 Slater-Koster 紧束缚模型（Zahid et al. 2013）。

实现 Zahid, Liu, Zhu, Wang, Guo, AIP Advances **3**, 052111 (2013)
的 96 参数模型：sp³d⁵ 非正交基、最近邻、Slater-Koster 两点积分 +
on-site 自旋轨道（arXiv:1304.0074 Table 3，参数逐一转录）。

- 基组：每原子 9 轨道 (s, p×3, d×5) × 2 自旋 = 18 态；
  单层 MoS₂（3 原子）共 54×54 广义本征值问题 H(k)c = E·S(k)c；
- 近邻：Mo–S 六键（三棱柱）+ 同亚层 S–S、Mo–Mo（距离 a）；
- 自旋轨道：on-site λ·(LxSx+LySy+LzSz)，p/d 壳层用该元素的 λ。

**几何（2026-10-14 修正）**：两个 S 位于**同一面内空位 (1/3, 2/3)**、
面外 ±dxx/2——上下 S 三角形对齐的**三棱柱配位**（1H 相的定义特征；
2H-MoS₂ 体相 P6₃/mmc 的 2c/4f 坐标同构）。此前的实现把两个 S 放在
不同空位（60° 交错 = 八面体/1T 型），晶体场给反了 d 能级顺序。
键数、键长与结构构建器 ``structure.builders.tmd`` 同步修正。

**中性填充（2026-10-14 确认）**：Mo (4d⁵5s¹) + 2×S (3s²3p⁴) =
**18 电子/胞**。含 SOC 时 54 态中占据 18 个（不可用"半填 27"——
预设 n_valence 已同步修正为 18）。

**验证目标（论文 Table 1/2）**：单层 K_v1→K_c = 1.805 eV、
K_v2→K_c = 1.969 eV（VB 劈裂 0.164 eV）、m_e*(K) ≈ 0.43 m₀、
m_h*(K) ≈ 0.46 m₀。

**当前状态（防幻觉声明）**：几何与填充已独立核实；S(k) 正定、
H 厄米、广义本征值可解。但 **带隙/能级顺序尚未与文献对账**：
现实现下 K 点晶体场顺序反转（带顶为 e′ 型 d_yz/d_zx，正确应为
e 型 d_x²−y²/d_xy），修正几何下 E[18]−E[17] ≈ 0.98 eV（无 SOC，
目标 ~1.9）。已系统扫描的装配自由度（S–Mo 角色互换 ×2、
通道族符号翻转 2⁶ 组合、S–Mo 单/双通道符号 106 组合）均不能
同时满足"带顶 m=±2 主导 + 带隙接近目标"；π 族整体翻转可恢复
正确带顶字符（m=±2 权重 0.74）但带隙仍偏低（0.76 eV）——差异落在
文档未给出的装配细节上。2026-10-17 文献核查：参数由 Nanoskif
（NanoAcademic 的 SK 拟合工具，从未公开发布）生成，其自述"限于 SK
传统的双中心近似"，故不支持"旋转轨道框架"假设（此前记录的 η_ij
框架旋转在 QTCAD/NEMO-3D 文档中实为 Harrison 键长标度指数）；S–Mo
块的 8 个独立积分无法在一次标准 SK54 方向余弦装配中全部消费，最可能
的错配位置是逐壳层的方向余弦/参数分配。唯一可能给出答案的文档是
未公开的 Nanoskif 用户手册。详见 docs/REFERENCES.md。

SK 角因子的构造与验证见 ``slater_koster.py``。
"""

from __future__ import annotations

from typing import Dict, List, Tuple

import numpy as np
import scipy.linalg

from ..structure.lattice import Lattice
from .slater_koster import ORBITAL_SHELL, _P_HAT, D_TENSORS, _perp_basis

# 同亚层六方近邻（距离 a）的镜像选择。默认取每对 ±R 中的一个代表，
# 配合共轭填充覆盖全部 6 近邻；这是使**拟合表**厄米的必要装配选择
# （E_spσ = 0.6656 与 E_psσ = −1.6515 是独立参数而非相反数，物理宇称
# 交换关系不适用）。显式列全 6 个且关闭同亚层填充会破坏厄米性——该
# 负结果（K 点 max|H−H†| ≈ 1.72 eV）已固化为回归测试，配置见
# tests/test_zahid_mos2.py::TestKnownIssue。
DEFAULT_MIRROR_IMAGES = ((1, 0), (0, 1), (1, 1))
SIX_MIRROR_IMAGES = ((1, 0), (0, 1), (1, 1), (-1, 0), (0, -1), (-1, -1))

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
    """

    VACUUM = 15.0
    # 中性填充：Mo(4d⁵5s¹) + 2×S(3s²3p⁴) = 18 电子/胞，含 SOC 时占 54 态中的
    # 18 条。非半填，故必须显式声明——否则绘图与带隙分析会各自推断出不同
    # 的费米参考（半填 27 错误）。
    n_valence = 18

    def __init__(self, a: float = 3.179, dxx: float = 3.135,
                 soc: bool = True, *, mirror_images=None,
                 conjugate_fill: bool = True) -> None:
        """``mirror_images`` / ``conjugate_fill`` 只影响同亚层 S–S、Mo–Mo
        近邻的装配方式，用于复现 2026-10-14 的装配约定复核（见模块文档与
        ``docs/REFERENCES.md``）。S–Mo 的六个近邻本就是显式列出的真实键，
        其共轭块 ⟨Mo|H|S⟩ = ⟨S|H|Mo⟩† 是厄米性的定义要求而非约定选择，
        故不受这两个开关影响。
        """
        self.lattice = Lattice.hexagonal(a, vacuum=self.VACUUM)
        self.a = a
        self.dxx = dxx
        self.soc = soc
        self.mirror_images = (tuple(DEFAULT_MIRROR_IMAGES) if mirror_images is None
                              else tuple((int(dx), int(dy)) for dx, dy in mirror_images))
        self.conjugate_fill = bool(conjugate_fill)
        self.name = "MoS2 sp3d5 (Zahid 2013)"
        h = dxx / 2
        # 三棱柱配位：两个 S 同一面内空位 (1/3, 2/3)（上下对齐）
        self.sites = [("Mo", np.array([0.0, 0.0, 0.5])),
                      ("S", np.array([1 / 3, 2 / 3, 0.5 + h / self.VACUUM])),
                      ("S", np.array([1 / 3, 2 / 3, 0.5 - h / self.VACUUM]))]
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
        # 默认取 ± 对中的代表镜像，配合共轭填充覆盖全部 6 近邻并保证
        # H(k)/S(k) 厄米（SK 矩阵元对 R̂ 含奇宇称通道）。
        for i in (1, 2):
            for dx, dy in self.mirror_images:
                bonds.append((i, i, np.array([dx, dy, 0.0]), "SS"))
        for dx, dy in self.mirror_images:
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
                # ⟨S|H|Mo⟩ 块 + 厄米共轭块 ⟨Mo|H|S⟩（厄米性的定义要求）
                add_pair(i, j, dfrac, INTEGRALS_SMO_H, INTEGRALS_SMO_S,
                         1.0, symmetric=True)
            elif kind == "SS":
                add_pair(i, j, dfrac, INTEGRALS_SS_H, INTEGRALS_SS_S,
                         1.0, symmetric=self.conjugate_fill)
            else:
                add_pair(i, j, dfrac, INTEGRALS_MOMO_H, INTEGRALS_MOMO_S,
                         1.0, symmetric=self.conjugate_fill)

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
           已知问题（防幻觉声明）：S(k) 正定性与 H 厄米性已验证；
           但**带隙/能级顺序尚未与文献（K_v1→K_c = 1.805 eV）对账**——
           修正几何（三棱柱）后 K 点晶体场顺序反转，装配约定扫描
           未能同时满足字符与带隙判据。结果仅可作结构/定性用途，
           不得当作定量带隙参考。见模块文档与 ``docs/REFERENCES.md``。
        """
        kfracs = np.asarray(kfracs, float)
        out = np.empty((len(kfracs), self.n_basis))
        for ik, kf in enumerate(kfracs):
            out[ik] = self._generalized_eigh(kf)
        return out

    def _generalized_eigh(self, kfrac) -> np.ndarray:
        """单个 k 点的广义本征值，附 S(k) 正定诊断。

        ``bands`` 与 ``energies_at`` 共用——此前 ``energies_at`` 缺该检查，
        非正定 S(k) 会在 LAPACK 深处以无上下文的报错失败。
        """
        H, S = self.hamiltonian_overlap(kfrac)
        wS = np.linalg.eigvalsh(S)
        if wS.min() < -1e-8:
            raise RuntimeError(
                f"S(k) 非正定 (min eig = {wS.min():.4g} @ k={kfrac})："
                "SK 方向/相位约定与拟合参数不匹配，结果无定义。"
                "见 docs/REFERENCES.md 中 'Nanoskif SK 约定' 待办。")
        return scipy.linalg.eigh(H, S, eigvals_only=True)

    def energies_at(self, kfrac) -> np.ndarray:
        return self._generalized_eigh(kfrac)

    def __repr__(self) -> str:
        return (f"ZahidMoS2Model(a={self.a}, dxx={self.dxx}, "
                f"soc={self.soc}, n_basis={self.n_basis})")
