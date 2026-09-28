"""Slater-Koster 双中心积分的角因子（严格构造，不依赖背表）。

方法：把 p / d 实球谐函数表示为对称无迹张量（归一化使
``Tr(T1 @ T2) = δ_ij``），对给定键方向 R̂ 构造键轴坐标系下的
σ/π/δ 通道函数，再用与全局基的内积得到通道系数。

两点近似的矩阵元::

    H[α@A, β@B] = Σ_channels V_{l_α l_β, channel} · C_α,ch · C_β,ch

其中 C_α,ch 是轨道 α 在键轴系 (R̂, ê1, ê2) 下的通道展开系数。
该构造与 Slater-Koster (1954) 封闭表逐元素等价（见本模块测试：
沿 x/y/z 轴与闭式表逐一比对）。

通道约定：
- s-p / s-d / s-s：仅 σ；
- p-p：σ, π；p-d：σ, π；d-d：σ, π, δ。
"""

from __future__ import annotations

import numpy as np

# ----------------------------------------------------------------------
# 轨道定义
# ----------------------------------------------------------------------
# sp³d⁵ 壳层轨道顺序
SPD5_ORBITALS = ("s", "px", "py", "pz",
                 "dxy", "dyz", "dzx", "dx2-y2", "dz2")

ORBITAL_SHELL = {"s": 0, "px": 1, "py": 1, "pz": 1,
                 "dxy": 2, "dyz": 2, "dzx": 2, "dx2-y2": 2, "dz2": 2}

# p 轨道单位矢量
_P_HAT = {"px": np.array([1.0, 0, 0]),
          "py": np.array([0, 1.0, 0]),
          "pz": np.array([0, 0, 1.0])}


def _d_tensor(name: str) -> np.ndarray:
    """d 轨道的归一化对称无迹张量（Tr(T_i@T_j) = δ_ij）。"""
    if name == "dxy":
        T = np.zeros((3, 3)); T[0, 1] = T[1, 0] = 1.0 / np.sqrt(2)
    elif name == "dyz":
        T = np.zeros((3, 3)); T[1, 2] = T[2, 1] = 1.0 / np.sqrt(2)
    elif name == "dzx":
        T = np.zeros((3, 3)); T[0, 2] = T[2, 0] = 1.0 / np.sqrt(2)
    elif name == "dx2-y2":
        T = np.diag([1.0, -1.0, 0.0]) / np.sqrt(2)
    elif name == "dz2":
        T = np.diag([-1.0, -1.0, 2.0]) / np.sqrt(6)
    else:
        raise ValueError(name)
    return T


D_TENSORS = {name: _d_tensor(name) for name in
             ("dxy", "dyz", "dzx", "dx2-y2", "dz2")}


def _perp_basis(rhat: np.ndarray):
    """返回与 R̂ 正交的单位矢量对 (e1, e2)。"""
    ref = np.array([1.0, 0, 0]) if abs(rhat[0]) < 0.9 else np.array([0, 1.0, 0])
    e1 = np.cross(rhat, ref); e1 /= np.linalg.norm(e1)
    e2 = np.cross(rhat, e1)
    return e1, e2


def channel_coeffs(orbital: str, rhat: np.ndarray) -> dict:
    """轨道 ``orbital`` 在键轴系下的通道系数。

    返回 {"sigma": c0, "pi": (c1, c2), "delta": (c1, c2)}；
    s 轨道仅 σ=1，其余通道为 0。
    """
    rhat = np.asarray(rhat, dtype=float)
    rhat = rhat / np.linalg.norm(rhat)
    e1, e2 = _perp_basis(rhat)
    shell = ORBITAL_SHELL[orbital]

    if shell == 0:
        return {"sigma": 1.0, "pi": (0.0, 0.0), "delta": (0.0, 0.0)}

    if shell == 1:
        p = _P_HAT[orbital]
        return {"sigma": float(np.dot(p, rhat)),
                "pi": (float(np.dot(p, e1)), float(np.dot(p, e2))),
                "delta": (0.0, 0.0)}

    # d 轨道：与键轴系 d 函数的张量内积
    T = D_TENSORS[orbital]
    # σ: d0(R̂) 的张量 = (3 R̂R̂ᵀ − I)/√6
    T_sigma = (3.0 * np.outer(rhat, rhat) - np.eye(3)) / np.sqrt(6.0)
    # π: ½(R̂êᵀ + êR̂ᵀ) 归一化
    t1 = 0.5 * (np.outer(rhat, e1) + np.outer(e1, rhat))
    t2 = 0.5 * (np.outer(rhat, e2) + np.outer(e2, rhat))
    t1 /= np.linalg.norm(t1)   # = √(2/3)·√... 数值归一化, Tr(T²)=1
    t2 /= np.linalg.norm(t2)
    # δ: (ê1ê1ᵀ − ê2ê2ᵀ)/√2 与 (ê1ê2ᵀ + ê2ê1ᵀ)/√2
    t3 = (np.outer(e1, e1) - np.outer(e2, e2)) / np.sqrt(2)
    t4 = (np.outer(e1, e2) + np.outer(e2, e1)) / np.sqrt(2)

    return {"sigma": float(np.tensordot(T, T_sigma)),
            "pi": (float(np.tensordot(T, t1)), float(np.tensordot(T, t2))),
            "delta": (float(np.tensordot(T, t3)), float(np.tensordot(T, t4)))}


def sk_element(orb_a: str, orb_b: str, rhat: np.ndarray,
               integrals: dict) -> float:
    """两点 SK 矩阵元 ⟨orb_a@A | H | orb_b@B⟩，R̂ = (B−A)/|B−A|。

    参数:
        integrals: 积分名字典，键名如 "ssσ", "spσ", "ppσ", "ppπ",
                   "sdσ", "pdσ", "pdπ", "ddσ", "ddπ", "ddδ"。
                   （同壳层对使用该对全部通道；异壳层仅用存在的通道）
    """
    ca = channel_coeffs(orb_a, rhat)
    cb = channel_coeffs(orb_b, rhat)
    la, lb = ORBITAL_SHELL[orb_a], ORBITAL_SHELL[orb_b]

    def get(name: str) -> float:
        return float(integrals.get(name, 0.0))

    total = 0.0
    # σ 通道 (m=0) 总是存在
    total += get(_ch_name(la, lb, 0)) * ca["sigma"] * cb["sigma"]
    if min(la, lb) >= 1:
        total += get(_ch_name(la, lb, 1)) * (
            ca["pi"][0] * cb["pi"][0] + ca["pi"][1] * cb["pi"][1])
    if la == 2 and lb == 2:
        total += get(_ch_name(2, 2, 2)) * (
            ca["delta"][0] * cb["delta"][0] + ca["delta"][1] * cb["delta"][1])
    return total


def _ch_name(l1: int, l2: int, m: int) -> str:
    """通道积分命名：ssσ / spσ / ppσ / ppπ / sdσ / pdσ / pdπ / ddσ / ddπ / ddδ。"""
    letter = {0: "s", 1: "p", 2: "d"}
    suffix = {0: "σ", 1: "π", 2: "δ"}
    return f"{letter[l1]}{letter[l2]}{suffix[m]}"
