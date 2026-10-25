"""DFT→TB 拟合：从能带与轨道投影数据辨识 Slater-Koster 积分参数。

模块结构：

- :mod:`.parameters` — 模型积分表 ↔ 命名参数向量；
- :mod:`.projection` — 广义本征矢的（原子, 壳层）投影权重；
- :mod:`.reference`  — 参考数据容器（合成生成；PROCAR 读取留作接口）；
- :mod:`.driver`     — 残差构造与 :func:`scipy.optimize.least_squares` 驱动。

动机（docs/REFERENCES.md 技术债第二条路）：Zahid 2013 的参数表转录无误但
带隙与带顶字符与文献目标不符，装配约定受阻于未公开的 Nanoskif 手册。
与其等文献，不如用 DFT 数据（能带 + 轨道投影）自行辨识参数——本模块提供
这条管线的拟合侧，参考数据侧由 ``reference`` 抽象（合成数据用于自验证，
真实数据接 ``from_procar``，后续交付）。

用法::

    from vdw_studio.fitting import BandReference, fit, parameter_names

    ref = BandReference.from_model(truth_model, kpoints, n_bands=(14, 22))
    result = fit(model, ref, free_names=[...])
    assert abs(model.tables["SMo_H"]["ddσ"] - truth) < 1e-3
"""

from .parameters import (
    apply_overrides,
    get_parameter,
    get_vector,
    parameter_names,
    set_vector,
)
from .projection import DEFAULT_GROUPS, group_weights
from .reference import BandReference
from .driver import FitResult, fit

__all__ = [
    "parameter_names", "get_vector", "set_vector", "apply_overrides",
    "get_parameter", "group_weights", "DEFAULT_GROUPS",
    "BandReference", "fit", "FitResult",
]
