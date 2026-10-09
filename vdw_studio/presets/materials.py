"""材料预设库：把"结构构建器 + 仿真引擎 + 文献参考值"打包成一键预设。

每个 :class:`MaterialPreset` 绑定：

- 结构构建函数（``structure.builders``）；
- 默认仿真模型工厂（``engine`` 各模型）；
- 占据带数与 k 路径类型；
- 文献参考性质（带隙/有效质量/费米速度），来源标注在 ``source``，
  供 GUI 展示与测试自检（"仿真结果 vs 文献"即时对账）。

新增材料的推荐流程：先在 ``structure.builders`` 加构建器（结构参数
标注文献出处）→ 在 ``engine`` 加/复用模型 → 在此注册预设并填参考值
→ 在 ``tests/test_presets.py`` 加端到端验证。
"""

from __future__ import annotations

import numpy as np
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional

from ..analysis.properties import GapResult, analyze_gap, analyze_path_gap
from ..engine.kp_tmd import TMDKpModel, TMDKpParams, TMD_REFERENCE_MASSES
from ..engine.models import (
    BilayerGrapheneModel,
    BuckledHoneycombModel,
    HoneycombModel,
    PhosphoreneRudenko,
)
from ..engine.zahid_mos2 import ZahidMoS2Model
from ..structure.builders import build
from ..structure.crystal import Crystal


@dataclass
class ExcitonParams:
    """激子模块材料默认参数（文献溯源，防幻觉）。

    - ``mu_over_m0``：电子–空穴约化质量 (m₀ 单位)；
    - ``r0_angstrom``：Keldysh 二维极化长度 r₀ = 2πχ₂D (Å)
      （Cudazzo et al., PRB 84, 085406 (2011) 的 χ₂D→r₀ 关系，
       与 Berkelbach et al., PRB 88, 045318 (2013) Eq. (1) 的 ρ₀ 等价）；
    - ``eb_ref``：(文献 1s 束缚能 eV, 相对容差)。文献值为**变分法**结果
      （试探波函数上界 → 束缚能下界），精确求解器应给出 ≥ 且接近的值。
    """

    mu_over_m0: float
    r0_angstrom: float
    eb_ref: Optional[tuple] = None
    source: str = ""


@dataclass
class MaterialPreset:
    """单个材料的预设配置。"""

    key: str
    name: str                                   # 显示名
    formula: str
    category: str                               # 蜂窝家族 / TMD / 黑磷烯 / …
    engine: str                                 # "tb" | "kp" | "sp3d5"
    structure_key: str                          # structure.builders 的材料名
    make_structure: Callable[..., Crystal]
    make_model: Callable                        # -> 默认仿真模型
    n_valence: int                              # 占据带数
    gap_ref: Optional[tuple] = None             # (文献值 eV, 容差)；None = 无隙/待验证
    gap_note: str = ""                          # 带隙备注（零带隙/待验证等）
    mass_ref: Optional[tuple] = None            # (m1, m2) 参考 |m*| (m₀)
    v_fermi_ref: Optional[float] = None         # 费米速度参考 (m/s)
    exciton: Optional[ExcitonParams] = None     # 激子默认参数（文献标定）
    source: str = ""                            # 参数与参考值出处
    tags: List[str] = field(default_factory=list)


# ----------------------------------------------------------------------
# 预设注册表
# ----------------------------------------------------------------------
PRESETS: Dict[str, MaterialPreset] = {}


def _register(p: MaterialPreset) -> None:
    PRESETS[p.key] = p


# --- 蜂窝家族 ----------------------------------------------------------
_register(MaterialPreset(
    key="graphene", name="石墨烯", formula="C2", category="蜂窝家族",
    engine="tb", structure_key="graphene",
    make_structure=build,  # build("graphene")
    make_model=lambda: HoneycombModel(a=2.46, t=-2.7),
    n_valence=1,
    gap_ref=None, gap_note="零带隙半金属（K 点 Dirac 锥）",
    v_fermi_ref=8.74e5,
    source="t = −2.7 eV（Reich 2002 / Castro Neto RMP 2009 教科书值）；"
           "a = 2.46 Å（实验）",
    tags=["Dirac锥", "半金属", "柔性电子"],
))

_register(MaterialPreset(
    key="hbn", name="六方氮化硼", formula="BN", category="蜂窝家族",
    engine="tb", structure_key="hbn",
    make_structure=build,
    make_model=lambda: HoneycombModel(a=2.504, t=-2.7, delta_onsite=3.5),
    n_valence=1,
    gap_ref=(3.5, 1e-6), gap_note="简单二带参数化：K 点带隙 = 子格势差 Δ；"
    "严格 hBN（实验带隙 ~6 eV）需多带模型（ROADMAP）",
    source="a = 2.504 Å（实验）；t = −2.7 eV、Δ 为简单参数化占位值",
    tags=["绝缘体", "介电衬底", "UV发光"],
))

_register(MaterialPreset(
    key="silicene", name="硅烯", formula="Si2", category="蜂窝家族",
    engine="tb", structure_key="silicene",
    make_structure=build,
    make_model=lambda electric_field=0.0: BuckledHoneycombModel(
        a=3.86, buckling=0.44, t=-1.6, electric_field=electric_field),
    n_valence=1,
    gap_ref=None, gap_note="零场下无隙；垂直电场开启带隙 "
    "（K 点带隙 = |E·Δz|，可做电场调控演示）",
    source="a = 3.86 Å、翘曲 0.44 Å（标准 DFT 值）；t = −1.6 eV（文献常用）",
    tags=["电场调控", "拓扑", "自旋轨道"],
))

_register(MaterialPreset(
    key="bilayer_graphene", name="双层石墨烯 (AB)", formula="C4",
    category="蜂窝家族", engine="tb", structure_key="bilayer_graphene",
    make_structure=build,
    make_model=lambda electric_field=0.0: BilayerGrapheneModel(
        electric_field=electric_field),
    n_valence=2,
    gap_ref=None, gap_note="零场：K 点抛物线触碰无隙；垂直电场线性开隙"
    "（最小模型 E_g ≈ |U|；含 γ₃/γ₄ 的饱和形式见 ROADMAP）",
    source="McCann–Fal'ko (2006/2007) 最小模型：γ₁ = 0.4 eV, t = −2.7 eV, "
    "层间距 3.35 Å（实验）",
    tags=["电场调控", "抛物线带", "双层"],
))

_register(MaterialPreset(
    key="phosphorene", name="黑磷烯", formula="P4", category="黑磷烯",
    engine="tb", structure_key="phosphorene",
    make_structure=build,
    make_model=lambda: PhosphoreneRudenko(),
    n_valence=2,
    gap_ref=(1.52, 0.02), gap_note="Γ 点直接带隙（四带模型值；GW 参考 1.60 eV）",
    mass_ref=(0.167, 0.849),   # 引擎算得：轻方向(扶手椅)/重方向(锯齿)
    source="Rudenko & Katsnelson, PRB 89, 201408(R) (2014) Table 1",
    tags=["各向异性", "高迁移率", "光电"],
))

# --- TMD（k·p 引擎，Kormányos 2015 (HSE,LDA) 参数）---------------------
_TMD_GAP = {k: v["e_bg_dft"] for k, v in {
    "MoS2": dict(e_bg_dft=1.67), "MoSe2": dict(e_bg_dft=1.40),
    "WS2": dict(e_bg_dft=1.60), "WSe2": dict(e_bg_dft=1.30),
    "MoTe2": dict(e_bg_dft=0.997), "WTe2": dict(e_bg_dft=0.792)}.items()}
_TMD_TAGS = {
    "MoS2": ["谷物理", "光伏", "经典"],
    "MoSe2": ["谷物理"],
    "WS2": ["强自旋轨道", "谷物理"],
    "WSe2": ["强自旋轨道", "谷激子"],
    "MoTe2": ["相变", "拓扑"],
    "WTe2": ["量子自旋液体候选"],
}

# --- 激子材料默认值（Berkelbach 2013 Table）-----------------------------
# Berkelbach, Hybertsen & Reichman, PRB 88, 045318 (2013) Table（tab:binding）:
#   μ (m₀)、χ₂D (Å)、激子 1s 束缚能（变分法, eV）；r₀ = 2πχ₂D (Cudazzo 2011)。
_BERKELBACH_EXCITON = {
    "MoS2":  (0.25, 6.60, 0.54),
    "MoSe2": (0.27, 8.23, 0.47),
    "WS2":   (0.16, 6.03, 0.50),
    "WSe2":  (0.17, 7.18, 0.45),
}
_BERKELBACH_SOURCE = (
    "Berkelbach, Hybertsen & Reichman, PRB 88, 045318 (2013) Table "
    "(μ, χ₂D, E_b 变分)；r₀ = 2πχ₂D (Cudazzo, PRB 84, 085406 (2011))"
)

for _tmd in ("MoS2", "MoSe2", "WS2", "WSe2", "MoTe2", "WTe2"):
    _ref = TMD_REFERENCE_MASSES[_tmd]
    _exciton = None
    if _tmd in _BERKELBACH_EXCITON:
        _mu, _chi, _eb = _BERKELBACH_EXCITON[_tmd]
        _exciton = ExcitonParams(
            mu_over_m0=_mu, r0_angstrom=2.0 * np.pi * _chi,
            eb_ref=(_eb, 0.10), source=_BERKELBACH_SOURCE)
    _register(MaterialPreset(
        key=f"{_tmd.lower()}_kp", name=f"单层 {_tmd}（k·p）",
        formula=_tmd, category="TMD (k·p)", engine="kp",
        structure_key=_tmd,
        make_structure=build,
        make_model=(lambda t=_tmd: TMDKpModel(TMDKpParams.from_library(t, "dft"))),
        n_valence=2,
        gap_ref=(_TMD_GAP[_tmd], 1e-6),
        gap_note="K 点最小（含 SOC）带隙 = E_bg（DFT 拟合值）",
        mass_ref=(min(_ref[:2]), max(_ref[2:])),
        exciton=_exciton,
        source="Kormányos et al., 2D Mater. 2, 022001 (2015) (HSE,LDA) 参数",
        tags=_TMD_TAGS[_tmd],
    ))

# --- TMD（sp³d⁵ 全 BZ 引擎，Zahid 2013）---------------------------------
_mu, _chi, _eb = _BERKELBACH_EXCITON["MoS2"]
_register(MaterialPreset(
    key="mos2_sp3d5", name="单层 MoS₂（sp³d⁵ 全 BZ）",
    formula="MoS2", category="TMD (sp³d⁵)", engine="sp3d5",
    structure_key="MoS2",
    make_structure=build,
    make_model=lambda: ZahidMoS2Model(soc=True),
    n_valence=27,
    gap_ref=None,
    gap_note="sp³d⁵ 引擎：H/S 厄米与 S(k) 正定已验证；带隙与文献吻合"
    "尚待 Nanoskif 相位约定确认（见 docs/REFERENCES.md）",
    exciton=ExcitonParams(
        mu_over_m0=_mu, r0_angstrom=2.0 * np.pi * _chi,
        eb_ref=(_eb, 0.10), source=_BERKELBACH_SOURCE),
    source="Zahid et al., AIP Advances 3, 052111 (2013) Table 3（96 参数）",
    tags=["全BZ", "量子输运级"],
))


# ----------------------------------------------------------------------
# 访问接口
# ----------------------------------------------------------------------
def get_preset(key: str) -> MaterialPreset:
    """按名称取预设（大小写不敏感，'mos2_kp' / 'MoS2_KP' 均可）。"""
    lookup = {k.lower(): k for k in PRESETS}
    if key.lower() not in lookup:
        raise ValueError(f"未知预设 {key!r}，可选: {sorted(PRESETS)}")
    return PRESETS[lookup[key.lower()]]


def list_presets() -> List[str]:
    """全部预设键。"""
    return sorted(PRESETS)


def get_exciton_params(key: str) -> ExcitonParams:
    """材料的激子默认参数（r₀/μ/文献束缚能）；未标定则抛 KeyError。"""
    p = get_preset(key)
    if p.exciton is None:
        raise KeyError(f"预设 {p.key} 尚无文献标定的激子参数")
    return p.exciton


def solve_preset_exciton(key: str, eps_env: float = 1.0,
                         **solve_kw) -> "ExcitonResult":
    """用预设的文献默认参数求解 2D 激子束缚能。

    参数来自 :class:`ExcitonParams`（μ、r₀ 均有文献出处）；
    ``eps_env`` 为环境介电常数（介电工程调谐），其余关键字
    透传 :func:`vdw_studio.analysis.exciton.solve_exciton`。
    """
    from ..analysis.exciton import ExcitonResult, solve_exciton

    ex = get_exciton_params(key)
    return solve_exciton(mu_over_m0=ex.mu_over_m0, eps_env=eps_env,
                         r0=ex.r0_angstrom, **solve_kw)


def run_preset(key: str, n_per_segment: int = 40) -> Dict:
    """一键运行预设：结构 → 模型 → k 路径 → 带隙分析。

    返回字典包含结构、模型、GapResult 与与文献参考的偏差标记，
    供 GUI/CLI 与测试复用。
    """
    p = get_preset(key)
    structure = p.make_structure(p.structure_key)
    model = p.make_model()
    result: Dict = {
        "preset": p, "structure": structure, "model": model,
    }
    if p.engine in ("tb", "sp3d5"):
        from ..engine.kpath import KPath
        analyze = analyze_path_gap if p.engine == "sp3d5" else analyze_gap
        gap = analyze(model, n_per_segment=n_per_segment, n_valence=p.n_valence)
        result["gap"] = gap
        if p.gap_ref is not None:
            ref, tol = p.gap_ref
            result["gap_matches_ref"] = bool(
                gap.gap is not None and abs(gap.gap - ref) <= tol)
    elif p.engine == "kp":
        kp = model
        e0 = kp.energies((0.0, 0.0), +1)
        gaps = [kp.spin_block_energies((0, 0), +1, s)[1]
                - kp.spin_block_energies((0, 0), +1, s)[0] for s in (+1, -1)]
        gap_min = float(min(gaps))
        result["gap"] = GapResult(
            gap=gap_min, direct=True, vbm=float(e0[1]), cbm=float(e0[2]),
            vbm_k=np.zeros(2), cbm_k=np.zeros(2),
            vbm_label="K", cbm_label="K", n_valence=p.n_valence,
            scope="valley-local", status="insulator", raw_gap=gap_min)
        if p.gap_ref is not None:
            ref, tol = p.gap_ref
            result["gap_matches_ref"] = bool(abs(gap_min - ref) <= tol)
    return result
