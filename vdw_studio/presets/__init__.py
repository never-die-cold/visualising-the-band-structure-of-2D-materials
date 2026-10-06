"""材料预设子包：一键从材料名到"结构 + 引擎 + 参考性质"。"""

from .materials import (
    PRESETS,
    ExcitonParams,
    MaterialPreset,
    get_exciton_params,
    get_preset,
    list_presets,
    run_preset,
    solve_preset_exciton,
)

__all__ = [
    "PRESETS", "MaterialPreset", "ExcitonParams",
    "get_preset", "list_presets", "run_preset",
    "get_exciton_params", "solve_preset_exciton",
]
