"""材料预设子包：一键从材料名到"结构 + 引擎 + 参考性质"。"""

from .materials import PRESETS, MaterialPreset, get_preset, list_presets, run_preset

__all__ = ["PRESETS", "MaterialPreset", "get_preset", "list_presets", "run_preset"]
