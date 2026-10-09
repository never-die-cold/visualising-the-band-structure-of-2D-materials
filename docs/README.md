# 文档索引

本次工程改进完成至 T15；T16 科研模型验证与 T17 后续扩展保留待办。两个应用的入口分别见 [BandViz](../README.md) 与 [vdW Studio](../vdw_studio/README.md)。

| 使用场景 | 文档 | 内容 |
|---|---|---|
| 安装与启动 | [安装说明](INSTALLATION.md) | Python/依赖、三个入口、实际安装验证与平台限制 |
| 导入 VASP 数据 | [输入契约](VASP_INPUT.md) | EIGENVAL 自旋/占据/权重、KPOINTS 分段、晶格与参考夹具 |
| 解读带隙 | [带隙分析](GAP_ANALYSIS.md) | 采样/路径/全区/局部谷的范围与数值收敛 |
| 解读质量与速度 | [导数坐标](DERIVATIVE_COORDINATES.md) | 行晶格、单位、方向质量、主质量与拟合诊断 |
| 解读 DOS | [DOS 契约](DOS_CONTRACT.md) | 态数归一化、占据分量、DOSCAR、路径谱与工作区预算 |
| 校验计算参数 | [数值输入](NUMERICAL_INPUTS.md) | 公共边界、简并、激子束缚态与错误行为 |
| 管理仿真状态 | [任务快照](TASK_SNAPSHOTS.md) | 参数冻结、模型隔离、几何一致性与过期结果拒绝 |
| 取消与关闭 | [任务生命周期](TASK_LIFECYCLE.md) | 协作取消、线程回收、后台 DOS 与交互响应 |
| 配置与数据库 | [应用状态](APPLICATION_STATE.md) | 默认数据目录、旧状态迁移、原子写入与失败恢复 |
| 恢复与复现 | [结果复现](REPRODUCIBILITY.md) | provenance、resume/replay、产物完整性、CSV 与限制 |
| 衡量资源使用 | [性能测量](PERFORMANCE.md) | 合成基准、原始测量、分块策略与内存限制 |
| 核对物理出处 | [参考文献](REFERENCES.md) | 模型/材料参数的来源与验证状态 |
| 追踪工程改进 | [检查计划](PROJECT_REVIEW_PLAN.md) | T01–T15 验收、实施与测试记录、T16/T17 待办 |
| 规划研究扩展 | [路线图](ROADMAP.md) | 中长期物理模型与独立基准要求 |
| 定位版本改动 | [提交索引](COMMIT_INDEX.md) | 本次按主题拆分的实际提交与最终验证 |

本地最终验证为 Windows / Python 3.12 的 641 项测试通过。CI 已配置 Windows/Linux × Python 3.12/3.13；远端矩阵结果需查看实际运行。教学输入、历史论文插图与 SK/BM 科研验证的限制分别在对应文档中说明。
