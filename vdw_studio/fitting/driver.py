"""残差构造与最小二乘拟合驱动。

残差 = 能带残差（eV）+ 投影权重残差（无量纲），按 ``energy_weight`` /
``weight_weight`` 加权。拟合通过 :func:`scipy.optimize.least_squares` 的
数值雅可比进行——每次残差评估要对全部 k 点做 54×54 广义本征分解，因此
自由参数的数量与 k 网格规模直接决定成本（见 :func:`fit` 的说明）。
"""

from dataclasses import dataclass
from typing import List, Optional, Sequence

import numpy as np
from scipy.optimize import least_squares

from .parameters import apply_overrides, get_vector, parameter_names
from .projection import group_weights


@dataclass
class FitResult:
    """拟合结果。``parameters`` 是 {参数名: 终值}（全部参数，不只自由项）。"""

    cost: float
    nfev: int
    success: bool
    message: str
    free_names: List[str]
    x: np.ndarray          # 自由参数终值（按 free_names 顺序）
    x0: np.ndarray         # 自由参数初值
    parameters: dict


def band_residuals(model, reference, *, energy_weight: float = 1.0,
                   weight_weight: float = 0.5) -> np.ndarray:
    """当前参数下的残差向量（能带 + 投影，均已展平）。"""
    if energy_weight < 0 or weight_weight < 0:
        raise ValueError("残差权重必须非负")
    start, end = reference.band_window
    energies, weights = group_weights(model, reference.kpoints)
    e_tb = energies[:, start:end]
    if e_tb.shape != reference.energies.shape:
        raise ValueError(f"TB 能带形状 {e_tb.shape} 与参考 {reference.energies.shape} 不一致")
    parts = [(e_tb - reference.energies).ravel() * energy_weight]
    if reference.weights:
        for name, w_ref in reference.weights.items():
            if name not in weights:
                raise ValueError(f"参考数据含未知投影组 {name!r}")
            parts.append((weights[name][:, start:end] - w_ref).ravel() * weight_weight)
    return np.concatenate(parts)


def fit(model, reference, *, free_names: Optional[Sequence[str]] = None,
        energy_weight: float = 1.0, weight_weight: float = 0.5,
        max_nfev: Optional[int] = None, x_tol: float = 1e-12,
        g_tol: float = 1e-10) -> FitResult:
    """以 ``model`` 为初值，把 ``free_names`` 列出的参数拟合到 ``reference``。

    模型在返回后持有拟合终值（就地更新）。``free_names`` 缺省为全部参数
    ——但每次残差评估的成本是 O(nk) 次 54×54 广义本征分解，数值雅可比
    每迭代还要乘 (自由参数数 + 1)，因此大网格/全参数拟合请显式挑选关键
    参数（例如 Mo 的 d 在位能与 dd/pd 系列通道）并控制 k 点数量。
    """
    names = parameter_names(model)
    free = list(names) if free_names is None else list(free_names)
    unknown = [name for name in free if name not in names]
    if unknown:
        raise ValueError(f"未知自由参数 {unknown}；可选前缀示例 {names[:4]}...")
    if not free:
        raise ValueError("至少需要一个自由参数")

    x0_full = get_vector(model)
    x0 = x0_full[[names.index(name) for name in free]]

    def residual(x_free):
        apply_overrides(model, dict(zip(free, x_free)))
        return band_residuals(model, reference, energy_weight=energy_weight,
                              weight_weight=weight_weight)

    result = least_squares(residual, x0, xtol=x_tol, gtol=g_tol, max_nfev=max_nfev)
    apply_overrides(model, dict(zip(free, result.x)))
    return FitResult(
        cost=float(result.cost), nfev=int(result.nfev), success=bool(result.success),
        message=str(result.message), free_names=free,
        x=np.asarray(result.x, dtype=float), x0=x0,
        parameters=dict(zip(parameter_names(model), get_vector(model))),
    )
