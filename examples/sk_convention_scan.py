"""SK 装配约定判据表：拿到 Nanoskif 约定后即可对账的可重跑入口。

背景见 docs/REFERENCES.md 「待确认的约定」：Zahid 2013 的 sp³d⁵ 非正交
参数表已逐项转录、几何与中性填充（18 电子/胞）已独立核实，但带隙与能级
顺序仍未与文献对账。T16 验收门槛：

- K_v1→K_c = 1.805 eV、K_v2→K_c = 1.969 eV（VB 自旋劈裂 0.164 eV）
- 带顶字符应以 m=±2（e 型 d_x²−y²/d_xy）为主导；现实现给出 e′ 型
- 前置条件：H(k) 厄米、S(k) 正定——不满足则本征值无定义

本脚本对**定义明确**的装配变体计算这些判据，便于约定到手后逐条对账。

未包含：通道族符号的 2⁶ 扫描。原扫描所用的族划分（"π/δ/奇宇称/pσ/dσ/dd"）
未在文档或代码中定义，重建它需要先确定分类口径，否则只会产出不可比的
数字——那比缺少脚本更坏。相关负结果与线索见 docs/REFERENCES.md。

运行::

    python examples/sk_convention_scan.py
"""

from __future__ import annotations

import sys
from pathlib import Path
from unittest import mock

import numpy as np
import scipy.linalg

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import vdw_studio.engine.zahid_mos2 as zahid  # noqa: E402
from vdw_studio.engine.zahid_mos2 import (  # noqa: E402
    SIX_MIRROR_IMAGES,
    ZahidMoS2Model,
)

K_POINT = np.array([1 / 3, 1 / 3, 0.0])
N_VALENCE = 18                     # 中性填充，Mo 4d⁵5s¹ + 2×S 3s²3p⁴
# m=±2 子空间 = span{d_xy, d_x²−y²}（实基），在 ORBITALS 中的位置
_M2_ORBITAL_INDICES = (4, 7)
ORBITALS_PER_ATOM = 18             # 9 轨道 × 2 自旋

# 文献目标（docs/REFERENCES.md 验收门槛）
TARGET_MAIN_GAP = 1.805
TARGET_SECOND_GAP = 1.969
TARGET_SPLITTING = 0.164


def swap_sigma_roles(integrals: dict) -> dict:
    """角色互换变体：把异种对 σ/π 通道的两种角色用法对调。

    Zahid 表的 E_spσ 与 E_psσ（以及 E_pdσ/E_dpσ 等）是**独立拟合参数**，
    所以"⟨s|H|p⟩ 该用 spσ 还是 psσ"是一个真实的装配自由度，不是恒等式。
    """
    out = dict(integrals)
    for left, right in (("spσ", "psσ"), ("pdσ", "dpσ"), ("pdπ", "dpπ")):
        if left in out and right in out:
            out[left], out[right] = out[right], out[left]
    return out


def d_m2_weight(model, k=K_POINT, n_valence=N_VALENCE, atom=0) -> float:
    """价带顶态在 m=±2（d_x²−y² / d_xy）子空间上的 S-正交投影权重。

    广义本征问题 Hψ = E·Sψ 以 S 为度规，基矢彼此不正交（S_ab ≠ δ），
    子空间 A 上的权重必须含子空间重叠矩阵的逆：w = c† S_A⁻¹ c，
    其中 c = (Sψ)_A。此前版本漏掉 S_A⁻¹（得到的是 ‖Sψ‖²，非归一），
    模块文档与 REFERENCES 记录的"m=±2 权重 0.74"即该错误公式下的产物，
    已在 2026-10-17 更正并标注待复核——见 ``vdw_studio.fitting.projection``。
    """
    H, S = model.hamiltonian_overlap(k)
    _, v = scipy.linalg.eigh(H, S)
    s_psi = (S @ v[:, n_valence - 1])
    indices = np.array([atom * ORBITALS_PER_ATOM + orbital * 2 + spin
                        for orbital in _M2_ORBITAL_INDICES for spin in (0, 1)])
    sub = s_psi[indices]
    S_sub = S[np.ix_(indices, indices)]
    return float(np.real(sub.conj() @ np.linalg.inv(S_sub) @ sub))


def evaluate(model, k=K_POINT, n_valence=N_VALENCE) -> dict:
    """单个变体的判据。"""
    H, S = model.hamiltonian_overlap(k)
    report = {
        "hermiticity_error": float(np.abs(H - H.conj().T).max()),
        "overlap_min_eig": float(np.linalg.eigvalsh(S).min()),
    }
    if report["hermiticity_error"] > 1e-10 or report["overlap_min_eig"] <= 0:
        report["note"] = "H 非厄米或 S 非正定 → 本征值无定义，不再继续"
        return report
    energies = scipy.linalg.eigh(H, S, eigvals_only=True)
    report["main_gap"] = float(energies[n_valence] - energies[n_valence - 1])
    report["second_gap"] = float(energies[n_valence] - energies[n_valence - 2])
    report["splitting"] = report["second_gap"] - report["main_gap"]
    report["m2_weight"] = d_m2_weight(model, k, n_valence)
    return report


def variants():
    """(名称, 模型构造器, 装配表替换) 列表。

    替换以 (模块属性名, 替换表) 给出，在 evaluate 期间生效——积分表是在
    ``hamiltonian_overlap`` 调用时才读取的，只在构造期替换不会生效。
    """
    return [
        ("default (3 images + fill)",
         lambda: ZahidMoS2Model(), None),
        ("six-mirror, no sublayer fill",
         lambda: ZahidMoS2Model(mirror_images=SIX_MIRROR_IMAGES, conjugate_fill=False),
         None),
        ("swapped sigma roles (sp<->ps)",
         lambda: ZahidMoS2Model(),
         ("INTEGRALS_SMO_H", swap_sigma_roles(zahid.INTEGRALS_SMO_H))),
    ]


def main() -> int:
    print(f"K point = {K_POINT[:2]}, n_valence = {N_VALENCE}")
    print(f"targets: main gap {TARGET_MAIN_GAP} eV, second {TARGET_SECOND_GAP} eV, "
          f"splitting {TARGET_SPLITTING} eV, m=±2 weight dominant\n")
    header = (f"{'variant':30s} {'|H-H+|':>9s} {'minS':>7s} "
              f"{'gap18':>8s} {'gap19':>8s} {'split':>8s} {'m=±2':>7s}")
    print(header)
    print("-" * len(header))
    for name, build, override in variants():
        if override is None:
            report = evaluate(build())
        else:
            attribute, table = override
            with mock.patch.object(zahid, attribute, table):
                report = evaluate(build())
        if "note" in report:
            print(f"{name:30s} {report['hermiticity_error']:9.3f} "
                  f"{report['overlap_min_eig']:7.3f}   {report['note']}")
            continue
        print(f"{name:30s} {report['hermiticity_error']:9.2e} "
              f"{report['overlap_min_eig']:7.3f} {report['main_gap']:8.3f} "
              f"{report['second_gap']:8.3f} {report['splitting']:8.3f} "
              f"{report['m2_weight']:7.3f}")
    print("\n未包含通道族符号的 2⁶ 扫描：原始族划分未定义，见模块文档。")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
