# DOS 与采样谱契约

BandViz 优先读取与 EIGENVAL 同目录的 DOSCAR；没有 DOSCAR 时展宽导入的本征值。路径谱不能代替全布里渊区积分。只有 KPOINTS 明确声明 Gamma/Monkhorst 自动积分网格，才将其权重视为 BZ 积分权重；显式列表的积分含义未知，保守显示采样谱。line-mode 显示路径谱。DOSCAR 与 EIGENVAL 是否来自同一计算由输入提供者保证。

`DosAnalyzer.from_band_data(data)` 使用全部自旋列和原始权重，归一化后 `sum(w)=1`。ISPIN=1 默认每列容量 2，ISPIN=2 每列容量 1；SOC/自旋子模型须明确指定 `state_capacity=1`。单独数组接口默认每列 1 态，不猜测自旋。高斯谱为 `sum_k,b capacity*w_k*g(E-E_kb)`，全部能量范围下期望积分为实际能带列数乘容量。重复整个采样不改变强度，拆分同一点时应拆分其权重。零权重点不贡献积分；全零路径权重明确退回均匀采样谱，积分网格全零权重报错。

total、occupied、unoccupied 使用同一套权重，满足 `total≈occupied+unoccupied`。占据数有效时按占据数/容量拆分，部分占据也可分配谱；这不意味着带隙判定已确定。缺占据数或展宽占据超出容量时退回按本征值相对显示零点的正负拆分，附状态说明。分量保留高斯尾部。`vb_dos/cb_dos` 字段保留兼容名称，实际分量含义由 `component_labels` 给出。

有限能量窗口丢失高斯尾部或窗口外的态，积分可能低于 `expected_states`；程序记录窗口积分并保留这一差别，不能通过再次缩放伪造守恒。粗能量网格可能漏掉窄高斯峰；高精度结果需要能量步长和 k 网格收敛。默认 Gaussian 工作数组预算 32 MiB，单个能量列至少占 `8*num_points` 字节；预算不包含输入能带、权重与输出数组。

DOSCAR 读取头部 `EMAX EMIN NEDOS EFERMI weight`，支持总 DOS 的非自旋 3 列和自旋 5 列，保留 up/down 和积分 DOS；自旋总 DOS 直接相加，不再次乘容量。它已由 VASP 处理采样积分，不重新 Gaussian 展宽。显示零点调整只平移能量轴，不改变原谱；界面禁用 σ 调节。存在但格式损坏的文件报错，缺文件才退回本征值谱。投影 DOS 块尚未实现。依据：[VASP DOSCAR 官方说明](https://vasp.at/wiki/DOSCAR)。

DOSCAR 某些展宽方法可产生小幅负值，读入时保留原值；不根据有限窗口的积分猜测全部模型态数。积分 DOS 与数值积分的 DOS 也可能因 VASP 展宽/能量网格不同而有差别。

vdW Studio 的 `solve_dos()` 在倒格原胞的均匀网格上积分，默认每个返回本征态计 1 次。未显式包含自旋的模型可选择 `spin_degeneracy=2`，含显式自旋的模型保持 1。结果和 CLI JSON 记录 `scope/spin_degeneracy/n_bands/expected_states`、网格、σ 和窗口积分。局部 k·p 仍不提供全区 DOS；实验性 SK 的适用性等级仍由模型验证计划约束。

回归用独立 Gaussian 公式和明确态数检查不等权重、分量可加、自旋/SOC、重复采样、占据与能量零点、分块一致性、DOSCAR 导入和错误，以及真实后台/界面范围显示。DOS 阈值推断带边只是依赖 σ 与网格的显示启发式，不能取代本征值带边分析。
