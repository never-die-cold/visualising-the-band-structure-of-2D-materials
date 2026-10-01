"""应变工程模块。

对面位点紧束缚模型施加面内应变（双轴/单轴），返回应变后的新模型：

1. **晶格**：``Lattice.scaled_xy(1+εx, 1+εy)``（真空层方向不变）；
2. **hopping**：按键长标度律 ``t' = t·(|d|/|d'|)^n`` 重整——n 为标度
   指数（Harrison 型键合标度，pπ–pπ 键常用 n ≈ 2）；
3. **在位能**：不变（一阶近似；特定材料的形变势可后续挂载）；
4. **hopping 的分数位移不变**（拓扑/配位关系不变，只有键长变化）。

物理预期（可解析推导的自检锚点，见 tests/test_strain.py）：
- 石墨烯双轴应变 ε：K 点保持无隙（C₃ 对称性不变），费米速度
  v_F(ε) = v_F(0)·(1+ε)^{1−n}；n=2 时 v_F(ε) = v_F(0)/(1+ε)；
- 石墨烯单轴应变（NN 模型）：无隙但 Dirac 锥位置移动
  （开隙需要三阶近邻 hopping，超出 NN 模型范围——如实标注）；
- 黑磷烯：带隙对应变敏感（DFT 文献共识：应变可显著调节磷烯带隙），
  本模块验证其灵敏度而非文献符号细节（四带 TB 无形变势参数，
  符号定量结论需文献标定，见 ROADMAP Phase 2）。

限制说明（诚实标注）：
- 单一指数标度对"多类 hopping"的模型（如黑磷烯 t₁…t₅、sp³d⁵ 的
  σ/π/δ 积分）是粗略近似；定量使用前应按材料标定各类键的指数或
  改用文献的应变-参数插值表；
- k·p 引擎的参数不含应变依赖，本模块不适用于 TMDKpModel。
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

from ..structure.lattice import Lattice
from .models import TightBindingModel

if TYPE_CHECKING:  # pragma: no cover
    from .models import TightBindingModel as _Model


def apply_strain(model: "TightBindingModel",
                 ex: float = 0.0, ey: float = 0.0,
                 exponent: float = 2.0) -> TightBindingModel:
    """返回施加面内应变后的新模型（原模型不变）。

    参数:
        model: 位点紧束缚模型（``engine.models`` 家族）。
        ex / ey: x / y 方向工程应变（0.05 = 拉伸 5%，负值为压缩）。
        exponent: 键长标度指数 n（t' = t·(d/d')^n）。

    返回:
        新的 :class:`TightBindingModel`（基类实例，晶格与 hopping 已更新）。
    """
    if not hasattr(model, "hoppings") or not hasattr(model, "lattice"):
        raise TypeError("apply_strain 仅支持位点型紧束缚模型 "
                        "(engine.models 家族)")
    if ex == 0.0 and ey == 0.0:
        return model

    old_lat = model.lattice
    new_lat = old_lat.scaled_xy(1.0 + ex, 1.0 + ey)

    new_hoppings = []
    for h in model.hoppings:
        d_old = np.asarray(h.dfrac, dtype=float) @ old_lat.matrix
        d_new = np.asarray(h.dfrac, dtype=float) @ new_lat.matrix
        d0 = float(np.linalg.norm(d_old))
        d1 = float(np.linalg.norm(d_new))
        if d0 < 1e-12:
            t_new = h.t
        else:
            t_new = h.t * (d0 / d1) ** exponent
        new_hoppings.append(
            type(h)(h.i, h.j, tuple(np.asarray(h.dfrac, dtype=float)), t_new))

    strained = TightBindingModel(
        lattice=new_lat,
        n_sites=model.n_sites,
        site_symbols=list(model.site_symbols),
        onsite=np.array(model.onsite, dtype=float, copy=True),
        hoppings=new_hoppings,
        name=f"{model.name} (εx={ex:+.1%}, εy={ey:+.1%})",
    )
    # 保留原模型类信息，便于界面显示
    strained.parent_name = getattr(model, "name", "")
    return strained
