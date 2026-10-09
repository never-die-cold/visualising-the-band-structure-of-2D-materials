# 坐标、速度与有效质量

## 统一约定

晶格 `A` 与倒格 `B` 均按行存放基矢，`A` 单位为 Å：

```text
B = 2π inv(A).T
k_cart = k_frac @ B                # Å⁻¹
delta_frac = delta_cart @ inv(B)   # 行向量逆变换
m*/m0 = 7.619964 / curvature       # curvature: eV·Å²
```

二维模型使用前两支倒格矢张成的平面。导数计算保留三维矢量的完整长度与内积，支持倾斜 slab，不能截掉 z 分量后再换算。

## vdW Studio 导数接口

`effective_mass(model, k0, band, direction=(1,0), dk=1e-4)` 和 `fermi_velocity(..., dk=1e-5)` 的 `k0` 均为二维分数坐标。为保持已有调用含义，默认 `direction_space="fractional"`：方向是倒格基矢的组合，`dk` 是归一化分数方向的步长。因此 `(1,0)` 表示沿 **b₁**，不必等于 Cartesian x。

指定 `direction_space="cartesian"` 时，方向是全局 Cartesian 的两/三个分量，两个分量补 z=0；`dk` 的单位改为 Å⁻¹。方向必须位于倒空间平面内。零方向、非有限坐标/步长、非正步长和无效带索引明确报错。

`effective_mass()` 返回曲率质量的绝对值，平坦曲率返回无穷。`principal_masses()` 在正交平面坐标中构造 Hessian，返回两支绝对质量和第一本征轴角度。其 `dk` 始终为 Å⁻¹；按带符号的曲率本征值排序，调用者需要轻/重质量时自行按质量排序。角度以投影到倒空间平面的全局 x 为基准，第二轴为平面法向叉乘第一轴；x 投影为零时改用全局 y。通常 xy slab 保持 x/y 定义，简并曲率的轴角没有唯一物理意义。

速度采用单侧 `|E(k0+delta)-E(k0)| / |delta_cart| / ħ`，适用于 Dirac 尖点，并且不随常数能量平移变化。它是指定方向的有限差分速度；步长收敛仍需按具体模型检查。

## BandViz 晶格导入

解析 EIGENVAL 时自动查找同目录 `POSCAR`，只读取晶格头部。支持正标量尺度、负值目标体积和三个正 Cartesian 分量尺度；三尺度乘晶格矩阵的列。也可显式指定 `poscar_path`（包括选定的 CONTCAR），或者提供 Å 单位的 `lattice_matrix`。这两个参数不能同时提供。用户需提供与该 EIGENVAL 一致的晶格。

`BandData.lattice_matrix / reciprocal_matrix / kcartesian / lattice_source / distance_unit` 保存物理坐标及来源。存在晶格时 `kdistances` 为 Å⁻¹，否则为未校准分数距离；路径段间的跳跃始终不累计。横轴明确标注单位或缺晶格状态。单一 `lattice_constant_angstrom` 旧参数保留调用兼容，但不能校准一般晶格，不参与定量质量计算。

正标量/负体积 POSCAR 自动提供 Cartesian KPOINTS 的转换：对于未缩放行晶格 `A_raw`，VASP 的坐标以 `2π/s` 为单位，因此 `f = x @ A_raw.T`。三尺度 POSCAR 可以校准分数 k 点的物理距离，但本接口不自动解释其 Cartesian KPOINTS 尺度；需要显式的 `cartesian_to_fractional`，无矩阵会报错。接口提供物理晶格矩阵但没有 POSCAR 尺度时，也不能据此猜测 Cartesian KPOINTS 的归一化。

## BandViz 路径质量诊断

`fit_effective_mass(band_index, k_index, num_points=5, side=None)` 返回诊断字典；`effective_mass()` 保留浮点/None 兼容接口。`mass_fit_reports_at_gap()` 分别报告 VBM/CBM，GUI 在数值旁标注 path，悬浮提示显示方向、实际采样索引、跨度、残差和窗口检查；无法拟合时提示原因。带边必须属于采样绝缘状态。

- 拟合 `E = a*s²+b*s+c`，`s` 为同一直线分支上的物理距离，允许带边落在采样点之间。返回有符号 `m*/m0 = 3.80998/a`；VBM 通常为负曲率质量，空穴正质量取其绝对值。
- 窗口不会越过 line-mode 段边界、方向转折或反向点；重复点不增加拟合样本。未分段路径的拐角必须显式选择 `side="left"/"right"`，默认不给出一个混合方向的质量。
- 至少五个不同 k 点。检查同一自旋通道的相邻带；窗口内带序交换或带间隔 ≤10⁻⁴ eV 时拒绝。不同通道简并不作为同通道交叉。
- 拟合 RMS 必须 ≤ `max(10⁻⁵ eV, 2% 能量跨度)`；由残差估算的曲率相对标准误差必须 ≤20%。至少七点时再缩小窗口，曲率变化须 ≤10%。平坦/无法分辨曲率不给出定量质量。

这些阈值是拒绝不稳定拟合的数值检查，不代表测量误差，也无法排除采样点之间未被解析的交叉。路径质量仅对应所报告方向，不是完整质量张量。高精度材料结论仍需更密带边采样、窗口收敛与实际模型验证。

## 独立验证

非正交六方晶格中指定 Cartesian 能量 `E=E0+kx²+5ky²`，Hessian 主值为 2/10 eV·Å²，主质量应为 **3.809982/0.7619964 m₀**。回归覆盖 xy 与倾斜平面旋转、明确 Cartesian 方向、Dirac 模型能量平移、POSCAR 多种尺度、实际后台导入与质量提示，以及转折/交叉/非抛物线拒绝。基准由 Cartesian 解析式给出，不复用待测分数曲率公式。

格式依据：[VASP POSCAR 文档](https://vasp.at/wiki/POSCAR)、[VASP KPOINTS 坐标说明](https://vasp.at/wiki/KPOINTS#Coordinate_system)。
