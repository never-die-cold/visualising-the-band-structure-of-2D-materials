"""模型积分表 ↔ 命名参数向量。

键名编码：``ONSITE:<物种>.<壳层>``（如 ``ONSITE:S.p``）与
``<表名>:<通道>``（如 ``SMo_H:spσ``）。向量顺序固定为
ONSITE（按物种、壳层）后跟各积分表（按通道的表内顺序），
:func:`parameter_names` 是唯一的顺序定义处——残差、雅可比与
结果报告都必须经由它取参数名，避免多处各自排序后错位。
"""

from typing import List

import numpy as np


def parameter_names(model) -> List[str]:
    """按固定顺序返回全部命名参数。"""
    names: List[str] = []
    for species, shell_map in model.tables["ONSITE"].items():
        for key in shell_map:
            names.append(f"ONSITE:{species}.{key}")
    for table in list(model.tables)[1:]:
        for channel in model.tables[table]:
            names.append(f"{table}:{channel}")
    return names


def _get_key(model, name: str) -> float:
    table, key = name.split(":", 1)
    if table == "ONSITE":
        species, shell = key.split(".", 1)
        return float(model.tables["ONSITE"][species][shell])
    return float(model.tables[table][key])


def _set_key(model, name: str, value: float) -> None:
    table, key = name.split(":", 1)
    if table == "ONSITE":
        species, shell = key.split(".", 1)
        model.tables["ONSITE"][species][shell] = float(value)
    else:
        model.tables[table][key] = float(value)


def get_vector(model) -> np.ndarray:
    """读出当前参数向量。"""
    return np.array([_get_key(model, name) for name in parameter_names(model)], dtype=float)


def set_vector(model, x) -> None:
    """就地写入参数向量（长度与顺序必须与 :func:`parameter_names` 一致）。"""
    x = np.asarray(x, dtype=float)
    names = parameter_names(model)
    if x.shape != (len(names),):
        raise ValueError(f"参数向量长度 {len(x)} 与模型参数数 {len(names)} 不一致")
    for name, value in zip(names, x):
        _set_key(model, name, value)


def apply_overrides(model, overrides: dict) -> None:
    """按命名键就地覆盖部分参数，如 ``{"SMo_H:ddσ": -3.0}``。"""
    valid = parameter_names(model)
    for name, value in overrides.items():
        if name not in valid:
            raise ValueError(f"未知参数 {name!r}")
        _set_key(model, name, float(value))


def get_parameter(model, name: str) -> float:
    """按命名键读取单个参数。"""
    if name not in parameter_names(model):
        raise ValueError(f"未知参数 {name!r}")
    return _get_key(model, name)
