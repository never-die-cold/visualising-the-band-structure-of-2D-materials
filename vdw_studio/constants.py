"""共享物理常数（CODATA 2018）。

这些量此前在 properties / kp_tmd / exciton / orbital_moment / transport /
models 中各自定义，其中 models.py 还用了不同的舍入值（3.809982444）。
集中后只有单一来源，更新时不会漂移。
"""

# ħ²/(2m₀)，eV·Å²
HBAR2_OVER_2M0 = 3.809982

# 基本电荷 (C)
ELEMENTARY_CHARGE = 1.602176634e-19

# 约化普朗克常数 (J·s)
PLANCK_REDUCED = 1.054571817e-34

# (eV·Å) 斜率 → 速度 (m/s)，由 CODATA 常数推导
EV_ANGSTROM_TO_V = ELEMENTARY_CHARGE * 1e-10 / PLANCK_REDUCED

# 同一物理量的文献舍入写法（与 EV_ANGSTROM_TO_V 相差 < 5e-6）。
# 保留它使既有数值不因集中化而漂移；新代码优先用 EV_ANGSTROM_TO_V。
SLOPE_TO_VELOCITY = 1.51927e5

# 电阻量子 h/e² (Ω)
H_OVER_E2 = 25812.80745
