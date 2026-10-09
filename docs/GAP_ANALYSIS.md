# 带隙分析范围与判据

## 周期 TB 模型

`vdw_studio.analysis.analyze_gap()` 现在搜索完整二维倒格原胞，计算价带顶 `max E[n_valence-1]` 和导带底 `min E[n_valence]`，再以两者之差作为带隙。高对称路径用于展示，由单独的 `analyze_path_gap()` 分析。

搜索过程：

1. 默认使用 24×24 周期网格，包括 Γ 点并按周期边界寻找局部带边。
2. 每条带选最多 8 个网格局部极值起点，额外加入高对称路径顶点，用 Nelder–Mead 分别优化 VBM 和 CBM；优化时按倒格周期折回坐标。
3. 使用 48×48 网格独立重复搜索。两次 VBM/CBM 能量差均不超过 10⁻⁵ eV、局部优化正常结束，才标记数值收敛。
4. 返回候选点中最优的带边、分数坐标与计算域。只在坐标确实等价于高对称点时写 Γ/M/K，否则标记 `off-path`，界面显示带边实际坐标。

两次网格结果一致是数值收敛检查，不能证明任意复杂模型不存在遗漏的极窄带边。可通过 API 的 `mesh`、`max_starts`、`convergence_tol` 加密和检查；CLI 的 `--gap-mesh` 设置初始边长，自动另检查两倍边长的网格，独立于绘图 `--npoints` 和 DOS `--mesh`。局部优化选项见 [SciPy 官方文档](https://docs.scipy.org/doc/scipy/reference/optimize.minimize-neldermead.html)。

`GapResult` 增加：

| 字段 | 含义 |
|---|---|
| `scope` | `brillouin-zone`、`path` 或 `valley-local` |
| `status` | `insulator`、`zero-gap` 或 `metal` |
| `raw_gap` | CBM−VBM 的未截断值；负值表示带边重叠 |
| `search_metadata` | 方法、两级网格、容差、边值差及收敛状态 |

默认零隙容差为 1 meV：`raw_gap < -tol` 为带边重叠；`-tol <= raw_gap <= tol` 为零隙；正值超过容差才报告绝缘带隙。为兼容已有接口，金属/零隙的 `gap` 和 `direct` 保持 `None`，应使用 `status/raw_gap` 区分。

直接性检查考虑全部简并带边候选，避免随意选取一个 VBM/CBM 导致错误结论。分数坐标按倒格周期等价处理。高对称路径不保证经过移位 Dirac 点或实际带边：NN 石墨烯施加 `ex=0.02, ey=0` 时，原路径带隙约 **0.158808 eV**，移位 Dirac 点约为 **(0.32773621, 0.33613189)**（或时间反演等价点），全区搜索的剩余隙小于 **10⁻⁷ eV**。测试用哈密顿量非对角元的独立二维求根对账。这里的 ex/ey 现按 Cartesian x/y 变形定义；T03 初始记录的 0.181551 eV 来自修正前按晶格基矢缩放的实现，历史记录保留。

GUI 与 CLI 共用仿真服务，周期 TB 结果同时带有 `gap` 和 `path_gap`。CLI JSON 保存两者的范围、状态、带边坐标、填充数与搜索元数据；GUI 工程树分别显示全区结果与路径结果。预设验证也使用全区搜索。

局部 k·p 仍只报告 `valley-local` 结果。实验性 SK 的方向/相位与全区适用性尚待 T16 验证，默认仿真和预设流程保留 `path` 结果并明确范围，不升级为可靠的材料全区结论。

## BandViz 导入数据

EIGENVAL 只提供已有 k 点，BandViz 无法补算路径之外的能量。结果明确标记 `scope="sampled-kpoints"`，界面显示 `Sampled Gap`，不能据此保证全区材料带隙。

- 有占据数时优先使用物理填充，不随绘图能量零点变化。完整占据/空带确定价带与导带；占据分类随 k 改变或带边重叠时报告金属。完整占据带数还应与 NELECT 一致。
- 无占据数时，非自旋整填充由 NELECT 确定价/导带，并检查同一带是否跨越费米能级；自旋极化缺少通道填充数时，利用各通道对费米能级的位置分类。非整数填充但无穿越证据时不猜测带隙。
- 确定的金属/零隙返回 `gap=0`、`direct=None`；正隙报告直接/间接类型；缺带、填充冲突或证据不足返回 `gap=None, status="unknown"` 和原因。金属、零隙和未确定情况不进行带边质量估算。
- 显著部分占据而无明确穿越时保留未确定状态。部分占据可能由展宽或固定占据造成，不能一律当作金属；尤其 Methfessel–Paxton 占据可能不符合简单的物理范围，见 [VASP ISMEAR 文档](https://vasp.at/wiki/ISMEAR)。默认占据容差为 10⁻³，微小展宽尾部不触发金属判定。
- 默认每态容量为 ISPIN=1 时 2、ISPIN=2 时 1。SOC/非共线自旋器等单态容量为 1 的数据需 Python 接口显式传入 `BandAnalyzer(data, state_capacity=1)`；ISPIN 单独无法确定这一约定，GUI 自动读取相关设置仍未实现。
- 默认零隙容差为 1 meV，可用 `zero_gap_tol` 设置。分析保留小正隙精度，界面仅格式化显示；直接性考虑简并带边 和倒格等价点。

回归包含：应变 Dirac 点独立求根、具有解析已知路径外极值的周期余弦能带、周期边界、能量平移、金属重叠、零/小隙、不同自旋填充、展宽尾部、未确定占据、实际 GUI 按钮/后台完成和 CLI 导出。一般晶格导数与质量见 [坐标说明](DERIVATIVE_COORDINATES.md)，DOS 积分见 [DOS 契约](DOS_CONTRACT.md)；实验模型验证仍待 T16。
