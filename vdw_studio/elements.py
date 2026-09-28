"""元素周期表数据。

收录与二维材料体系相关的元素（III-V 族、IV 族、过渡金属硫族化合物、
主族半导体等），每个元素提供：

- 原子序数 ``Z`` 与符号
- 中英文名称
- 标准原子质量 (u)
- 共价半径（Cordero et al., *Dalton Trans.* 2008，单位 Å），
  用于可视化成键判定与球棍模型半径缩放
- Jmol/CPK 配色（十六进制 RGB），用于结构可视化

数据来源：
- 共价半径：B. Cordero et al., "Covalent radii revisited",
  Dalton Trans., 2832-2838 (2008)。
- 颜色：Jmol 颜色表（CPK 方案的通行扩展）。
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Tuple


@dataclass(frozen=True)
class Element:
    """单个元素的静态数据。"""

    symbol: str          # 元素符号, 如 "Mo"
    z: int               # 原子序数
    name_en: str         # 英文名称
    name_zh: str         # 中文名称
    mass: float          # 标准原子质量 (u)
    covalent_radius: float  # 共价半径 (Å), Cordero 2008
    color: str           # Jmol/CPK 颜色, "#RRGGBB"

    @property
    def color_rgb(self) -> Tuple[float, float, float]:
        """颜色转为 0-1 浮点 RGB 三元组，便于 matplotlib/pyqt 使用。"""
        h = self.color.lstrip("#")
        return tuple(int(h[i : i + 2], 16) / 255.0 for i in (0, 2, 4))


# symbol -> (Z, 英文名, 中文名, 原子质量, 共价半径 Å, Jmol 颜色)
_RAW: Dict[str, tuple] = {
    "H":  (1,   "Hydrogen",   "氢",   1.008,  0.31, "#FFFFFF"),
    "Li": (3,   "Lithium",    "锂",   6.94,   1.28, "#CC80FF"),
    "B":  (5,   "Boron",      "硼",   10.81,  0.84, "#FFB5B5"),
    "C":  (6,   "Carbon",     "碳",   12.011, 0.76, "#909090"),
    "N":  (7,   "Nitrogen",   "氮",   14.007, 0.71, "#3050F8"),
    "O":  (8,   "Oxygen",     "氧",   15.999, 0.66, "#FF0D0D"),
    "F":  (9,   "Fluorine",   "氟",   18.998, 0.57, "#90E050"),
    "Na": (11,  "Sodium",     "钠",   22.990, 1.66, "#AB5CF2"),
    "Mg": (12,  "Magnesium",  "镁",   24.305, 1.41, "#8AFF00"),
    "Al": (13,  "Aluminium",  "铝",   26.982, 1.21, "#BFA6A6"),
    "Si": (14,  "Silicon",    "硅",   28.085, 1.11, "#F0C8A0"),
    "P":  (15,  "Phosphorus", "磷",   30.974, 1.07, "#FF8000"),
    "S":  (16,  "Sulfur",     "硫",   32.06,  1.05, "#FFFF30"),
    "Cl": (17,  "Chlorine",   "氯",   35.45,  1.02, "#1FF01F"),
    "K":  (19,  "Potassium",  "钾",   39.098, 2.03, "#8F40D4"),
    "Ca": (20,  "Calcium",    "钙",   40.078, 1.76, "#3DFF00"),
    "Ti": (22,  "Titanium",   "钛",   47.867, 1.60, "#BFC2C7"),
    "V":  (23,  "Vanadium",   "钒",   50.942, 1.53, "#A6A6AB"),
    "Cr": (24,  "Chromium",   "铬",   51.996, 1.39, "#8A99C7"),
    "Mn": (25,  "Manganese",  "锰",   54.938, 1.39, "#9C7AC7"),
    "Fe": (26,  "Iron",       "铁",   55.845, 1.32, "#E06633"),
    "Co": (27,  "Cobalt",     "钴",   58.933, 1.26, "#F090A0"),
    "Ni": (28,  "Nickel",     "镍",   58.693, 1.24, "#50D050"),
    "Cu": (29,  "Copper",     "铜",   63.546, 1.32, "#C88033"),
    "Zn": (30,  "Zinc",       "锌",   65.38,  1.22, "#7D80B0"),
    "Ga": (31,  "Gallium",    "镓",   69.723, 1.22, "#C28F8F"),
    "Ge": (32,  "Germanium",  "锗",   72.630, 1.20, "#668F8F"),
    "As": (33,  "Arsenic",    "砷",   74.922, 1.19, "#BD80E3"),
    "Se": (34,  "Selenium",   "硒",   78.971, 1.20, "#FFA100"),
    "Br": (35,  "Bromine",    "溴",   79.904, 1.20, "#A62929"),
    "Zr": (40,  "Zirconium",  "锆",   91.224, 1.75, "#7DB4E5"),
    "Nb": (41,  "Niobium",    "铌",   92.906, 1.64, "#A0A0A0"),
    "Mo": (42,  "Molybdenum", "钼",   95.95,  1.54, "#54B5B5"),
    "Tc": (43,  "Technetium", "锝",   98.0,   1.47, "#74B3B3"),
    "Ru": (44,  "Ruthenium",  "钌",   101.07, 1.46, "#6885BB"),
    "Rh": (45,  "Rhodium",    "铑",   102.91, 1.42, "#986DB4"),
    "Pd": (46,  "Palladium",  "钯",   106.42, 1.39, "#C0A0C0"),
    "Ag": (47,  "Silver",     "银",   107.87, 1.45, "#C0C0C0"),
    "Cd": (48,  "Cadmium",    "镉",   112.41, 1.44, "#FFD98F"),
    "In": (49,  "Indium",     "铟",   114.82, 1.42, "#A67573"),
    "Sn": (50,  "Tin",        "锡",   118.71, 1.39, "#668080"),
    "Sb": (51,  "Antimony",   "锑",   121.76, 1.39, "#9E63B5"),
    "Te": (52,  "Tellurium",  "碲",   127.60, 1.38, "#D47A00"),
    "I":  (53,  "Iodine",     "碘",   126.90, 1.39, "#940094"),
    "Hf": (72,  "Hafnium",    "铪",   178.49, 1.75, "#77ABD5"),
    "Ta": (73,  "Tantalum",   "钽",   180.95, 1.70, "#54A5A5"),
    "W":  (74,  "Tungsten",   "钨",   183.84, 1.62, "#2194D6"),
    "Re": (75,  "Rhenium",    "铼",   186.21, 1.51, "#76B4B4"),
    "Os": (76,  "Osmium",     "锇",   190.23, 1.44, "#7D63B4"),
    "Ir": (77,  "Iridium",    "铱",   192.22, 1.41, "#575AB1"),
    "Pt": (78,  "Platinum",   "铂",   195.08, 1.36, "#D0D0E0"),
    "Au": (79,  "Gold",       "金",   196.97, 1.36, "#FFD123"),
    "Tl": (81,  "Thallium",   "铊",   204.38, 1.45, "#A6544D"),
    "Pb": (82,  "Lead",       "铅",   207.2,  1.46, "#575961"),
    "Bi": (83,  "Bismuth",    "铋",   208.98, 1.48, "#9E4BDB"),
}

ELEMENTS: Dict[str, Element] = {
    sym: Element(sym, *vals) for sym, vals in _RAW.items()
}

# 供按原子序数查询
_BY_Z: Dict[int, Element] = {el.z: el for el in ELEMENTS.values()}


def get_element(symbol: str) -> Element:
    """按元素符号查询，未知符号抛出 ``ValueError``。

    参数:
        symbol: 元素符号，如 ``"Mo"``；大小写不敏感。

    返回:
        对应的 :class:`Element`。
    """
    sym = symbol.strip().capitalize()
    if sym not in ELEMENTS:
        raise ValueError(
            f"未知元素符号: {symbol!r}。"
            f"当前收录元素: {', '.join(sorted(ELEMENTS, key=lambda s: ELEMENTS[s].z))}"
        )
    return ELEMENTS[sym]


def get_element_by_z(z: int) -> Element:
    """按原子序数查询元素。"""
    if z not in _BY_Z:
        raise ValueError(f"未知原子序数: {z}")
    return _BY_Z[z]


def covalent_radius(symbol: str) -> float:
    """元素共价半径 (Å)。"""
    return get_element(symbol).covalent_radius


def validate_symbols(symbols) -> None:
    """批量校验元素符号，任一未知即抛出 ``ValueError``。"""
    for s in symbols:
        get_element(s)
