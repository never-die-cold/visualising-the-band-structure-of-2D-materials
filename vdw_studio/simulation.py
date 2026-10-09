"""GUI 与 CLI 共用的仿真流程，区分周期模型与局部谷模型。"""

from __future__ import annotations

from typing import Callable, Optional

import numpy as np

from .analysis.properties import GapResult, analyze_gap, analyze_path_gap
from .engine.kpath import KPath
from .engine.kp_tmd import TMDKpModel
from .engine.models import TightBindingModel
from .engine.solver import BandStructure, solve_bands, solve_dos
from .engine.zahid_mos2 import ZahidMoS2Model
from .structure.lattice import Lattice
from .numerics import integer

KP_QMAX = 0.25  # Å⁻¹，相对 K 谷中心的局部路径范围
KP_DOS_REASON = "k·p 模型仅描述 K 谷附近，未提供全布里渊区态密度。"


def _solve_valley(model: TMDKpModel, lattice: Lattice,
                  n_per_segment: int, cancel_check=None) -> dict:
    """局部 Γ 方向端点 → K → M 方向端点，端点不冒充 Γ/M。"""
    B = lattice.reciprocal_matrix[:2, :2]
    K = np.array([1 / 3, 1 / 3]) @ B
    M = np.array([1 / 2, 0.0]) @ B
    u_g = -K / np.linalg.norm(K)
    u_m = (M - K) / np.linalg.norm(M - K)
    radii = np.linspace(0.0, KP_QMAX, n_per_segment + 1)
    qpath = np.vstack((radii[::-1, None] * u_g,
                       radii[1:, None] * u_m))
    kcart = np.column_stack((K + qpath, np.zeros(len(qpath))))
    x = np.concatenate(([0.0], np.cumsum(np.linalg.norm(
        np.diff(qpath, axis=0), axis=1))))
    kfrac = (K + qpath[[0, n_per_segment, -1]]) @ np.linalg.inv(B)
    path = ["→Γ", "K", "→M"]
    kpath = KPath(points=dict(zip(path, map(tuple, kfrac))), path=path)
    sampled = []
    for q in qpath:
        if cancel_check is not None:
            cancel_check()
        sampled.append(model.energies(q, +1))
    band = BandStructure(
        model_name=model.name, kpoints=kcart, x_axis=x,
        ticks=x[[0, n_per_segment, -1]],
        tick_labels=[f"→Γ (|q|={KP_QMAX} Å$^{{-1}}$)", "K",
                     f"→M (|q|={KP_QMAX} Å$^{{-1}}$)"],
        path=path, energies=np.array(sampled),
    )
    e0 = model.energies((0.0, 0.0), +1)
    gaps = [np.diff(model.spin_block_energies((0, 0), +1, s))[0]
            for s in (+1, -1)]
    gap = GapResult(
        gap=float(min(gaps)), direct=True, vbm=float(e0[1]), cbm=float(e0[2]),
        vbm_k=np.array([1 / 3, 1 / 3]),
        cbm_k=np.array([1 / 3, 1 / 3]),
        vbm_label="K", cbm_label="K", n_valence=2,
        scope="valley-local", status="insulator", raw_gap=float(min(gaps)),
    )
    masses = {f"m_{carrier}_{spin_name}": model.effective_mass(band_name, spin)
              for carrier, band_name in (("e", "cb"), ("h", "vb"))
              for spin_name, spin in (("up", +1), ("down", -1))}
    return {"band": band, "gap": gap, "path_gap": None, "kpath": kpath,
            "dos": None, "dos_reason": KP_DOS_REASON,
            "effective_masses_m0": masses, "qmax": KP_QMAX,
            "domain": "valley-local", "valley": None}


def simulate(model, n_per_segment: int = 40,
             dos_mesh: tuple = (48, 48), dos_sigma: float = 0.05,
             n_valence: Optional[int] = None, *,
             lattice: Optional[Lattice] = None,
             gap_mesh: tuple = (24, 24),
             include_valley: bool = False,
             on_progress: Optional[Callable[[str], None]] = None,
             cancel_check=None) -> dict:
    """执行受支持模型的能带、性质及可用 DOS 分析。

    周期 TB/SK 使用模型晶格；局部 k·p 需要调用方传入材料晶格，
    仅用于确定谷位置与采样方向，不将模型扩展到全 BZ。
    返回 ``dos=None`` 与原因表示模型不支持全 BZ DOS。
    """
    n_per_segment = integer(n_per_segment, 'n_per_segment', minimum=2)
    def log(message):
        if cancel_check is not None:
            cancel_check()
        if on_progress is not None:
            on_progress(message)
    if isinstance(model, TMDKpModel):
        if lattice is None:
            raise ValueError("k·p 仿真需要材料晶格以确定 K 谷采样方向")
        if n_valence not in (None, 2):
            raise ValueError("当前 TMD k·p 模型使用 2 条价带")
        log("求解 K 谷附近能带与有效质量…")
        result = _solve_valley(model, lattice, n_per_segment, cancel_check)
        log(KP_DOS_REASON)
        if include_valley:
            from .analysis.berry import valley_report
            log("计算谷物理 (Berry 曲率/圆二色性)…")
            result["valley"] = valley_report(
                lambda q: model.hamiltonian(q, +1),
                lambda q: model.hamiltonian(q, -1),
                band_v=1, band_c=3, qmax=KP_QMAX, n_grid=41, cancel_check=cancel_check)
    elif isinstance(model, (TightBindingModel, ZahidMoS2Model)):
        lattice = model.lattice
        log("生成 k 路径…")
        kpath = KPath.for_lattice(lattice)
        log("求解能带…")
        band = solve_bands(model, kpath, n_per_segment, cancel_check=cancel_check)
        log("计算态密度…")
        dos = solve_dos(model, mesh=dos_mesh, sigma=dos_sigma, cancel_check=cancel_check)
        log("分析路径带隙…")
        path_gap = analyze_path_gap(model, kpath, n_per_segment=n_per_segment,
                                   n_valence=n_valence, band=band)
        if isinstance(model, ZahidMoS2Model):
            log("实验性 SK：仅报告路径带隙，尚未完成全区模型验证")
            gap = path_gap
        else:
            log("搜索全布里渊区带边…")
            gap = analyze_gap(model, kpath, n_valence=n_valence, mesh=gap_mesh, cancel_check=cancel_check)
        result = {"band": band, "dos": dos, "gap": gap, "path_gap": path_gap, "kpath": kpath,
                  "dos_reason": None, "domain": "periodic",
                  "valley": None, "effective_masses_m0": {}}
    else:
        raise TypeError(f"模型 {type(model).__name__} 尚未接入仿真流程")
    result["lattice"] = lattice
    result["experimental"] = isinstance(model, ZahidMoS2Model)
    return result
