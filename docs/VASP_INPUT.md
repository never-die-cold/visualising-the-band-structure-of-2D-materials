# BandViz VASP 输入说明

## EIGENVAL

默认读取标准 VASP 格式：首行最后一项为 `ISPIN`，第 6 行依次为 `NELECT NKPTS NBANDS`。支持非自旋/自旋极化的 3/5 列能带行；保留能量、占据数、分数 k 坐标和原始 k 权重。电子数允许非整数，支持 UTF-8/BOM 和 Fortran D 指数。

解析会核对声明的块数、每块能带数、从 1 开始的带序号、自旋列数以及有限数值。截断、额外块或格式错误会报告文件路径和行号，不返回部分结果。未实现的格式明确报错。

`BandData` 的兼容契约：

| 字段 | 含义 |
|---|---|
| `nkpoints / nbands / nelect / ispin` | NKPTS、每个自旋通道的 NBANDS、NELECT、ISPIN |
| `energies / occupations` | `(nk, nbands * ispin)`，先全部自旋向上带，再全部自旋向下带 |
| `num_bands` | 上述二维数组的列数；与旧绘图/DOS 接口一致 |
| `spin_energies / spin_occupations` | `(nk, nbands, ispin)` 视图，最后一维依次 up/down |
| `weights` | 文件中的原始权重；不在解析阶段归一化 |
| `segments` | line-mode 的分段 `(start, end)`，含两个端点，索引从 0 开始 |
| `kdistances / distance_unit` | 有晶格时为 Å⁻¹，否则为未校准分数距离；不连续段的跳跃不累计 |
| `lattice_matrix / reciprocal_matrix / kcartesian` | Å 单位行晶格、含 2π 的行倒格矩阵、Å⁻¹ 单位 Cartesian k 点；缺晶格时为 None |
| `lattice_source` | 实际 POSCAR 路径或显式矩阵来源 |

GUI 绘制全部自旋能带，用蓝色/橙色区分两个通道。分析与展宽谱输入也包含两个通道；采样带隙与金属判据见 [带隙分析说明](GAP_ANALYSIS.md)。DOS 的范围、权重与自旋态数见 [DOS 契约](DOS_CONTRACT.md)。

历史仓库样例使用 `Generated example` 和自定义四字段头部。默认拒绝此格式；确需读取旧文件时显式使用 `VASPEigenvalParser(path, allow_legacy=True)`。该分支发出警告、标记 `input_format="legacy-teaching"`，忽略旧 KPOINTS 标签，因为旧格式没有可靠的点数声明与标准端点约定。当前仓库样例已转换为标准格式。

## KPOINTS

默认在 EIGENVAL 同目录寻找 KPOINTS；缺省文件可只读取能量。显式传入不存在的 KPOINTS 路径会报错。

line-mode 按顺序每两个非空/非注释端点构成一段，段间空行可省略。每段 N 个点包含两个端点：两段、每段 3 点的标签索引为 `0, 2, 3, 5`。支持直接行末标签、`!`/`#` 标签及 Γ/GAMMA。显式模式按实际坐标行编号，忽略空行和纯注释；自动网格不生成路径标签，显式列表后的四面体信息不用于能带路径。

导入时验证展开后的全部 k 坐标与 EIGENVAL 一致，容许输出舍入误差和倒格等价坐标。错配的 KPOINTS 会报错。不连续段分别绘制，共用横轴位置的端点合并为 `X|M`；连续段的重复 `X` 只显示一次。带隙直接性以倒格等价坐标判断，重复端点索引不同也可对应直接带隙。

坐标模式遵循 VASP 的首字符规则：`C/c/K/k` 为 Cartesian，其他为分数坐标。GUI/解析器自动读取同目录 POSCAR；对于单标量尺度（含负值目标体积）和未缩放行晶格矩阵 A，VASP 的 Cartesian 坐标以 `2π/s` 为单位，转换矩阵为 `A.T`。Python 接口也可提供 `cartesian_to_fractional` 的可逆 3×3 行向量变换矩阵，即 `fractional = cartesian @ matrix`。缺 POSCAR 或使用三个分量尺度时不自动猜测该转换；没有显式矩阵则明确报错。

可显式指定 `poscar_path` 或 `lattice_matrix`，物理晶格和 KPOINTS 必须对应同一 EIGENVAL。定量质量只在晶格完整、同一直线分支样本充足且拟合诊断通过时给出；缺晶格可绘图，但质量显示不可用。坐标、方向、尺度支持边界和拟合阈值见 [坐标与导数说明](DERIVATIVE_COORDINATES.md)。

## 数据与验证来源

`data/example/graphene` 和 `mos2` 的能量是合成教学数据，58 个采样点保留原数值；占据数按非自旋整带填充写为 2/0，权重为 `1/58`。这些曲线没有对应的计算晶格，横轴为分数距离，不报告定量质量。扩展文件 `mos2_x4_EIGENVAL` 仅重复原能量用于解析规模测试，声明 232 个点、权重为 `1/232`，不用于新材料物理结论。`paper_assets.py` 的扩展生成逻辑同步使用标准头部。

真实来源夹具和全部通道的独立参考结果位于 [测试夹具说明](../tests/fixtures/vasp/README.md)，来自固定版本 pymatgen 的公开测试数据；常规回归可离线运行，不新增 pymatgen 运行依赖。

格式依据：[pymatgen 官方 Eigenval 实现](https://github.com/materialsproject/pymatgen/blob/v2025.6.14/src/pymatgen/io/vasp/outputs.py) 和 [VASP KPOINTS 文档](https://vasp.at/wiki/KPOINTS)。
