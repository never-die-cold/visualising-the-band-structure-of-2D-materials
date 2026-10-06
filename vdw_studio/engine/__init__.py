"""仿真引擎子包：k 路径、紧束缚模型、k·p 模型、moiré 连续模型、能带 / DOS 求解。"""

from .kpath import KPath, high_symmetry_points
from .models import (
    BilayerGrapheneModel,
    BuckledHoneycombModel,
    HoneycombModel,
    PhosphoreneRudenko,
    TightBindingModel,
)
from .solver import BandStructure, DOSResult, solve_bands, solve_dos
from .kp_tmd import TMD_KP_PARAMS, TMDKpModel, TMDKpParams
from .zahid_mos2 import ZahidMoS2Model
from .moire import TwistedBilayerGraphene

__all__ = [
    "KPath", "high_symmetry_points",
    "TightBindingModel", "HoneycombModel", "BuckledHoneycombModel",
    "PhosphoreneRudenko", "BilayerGrapheneModel",
    "BandStructure", "DOSResult", "solve_bands", "solve_dos",
    "TMDKpModel", "TMDKpParams", "TMD_KP_PARAMS",
    "ZahidMoS2Model", "TwistedBilayerGraphene",
]
