# T01–T15 提交索引

日期：2026-10-09。基线：`7779d6d`。交付分支：`main`。本批包含 **44 条有实际改动的提交**：下列 43 条主题提交，以及最后一条整理本索引的文档提交。使用现有作者身份与实际提交时间。

功能范围止于 T15，T16/T17 保留待办。按模块、测试与说明拆分，便于审阅、定位和组合回退；部分模块存在依赖，单独回退时需检查对应契约与调用方。

## 最终验证

- 收束后的完整测试：641 项通过，121.58 秒，无 warning。
- 独立环境安装 wheel 后：依赖检查通过；三个入口从安装位置加载，两个真实 GUI 默认任务完成。
- 安装位置的 CLI 批处理：11 材料完成、0 失败、1 实验性 SK 跳过；结果写入隔离数据库。
- 本地验证环境为 Windows / Python 3.12；CI 配置覆盖 Windows/Linux × Python 3.12/3.13，远端结果不在本地验证声明中。

个人未跟踪脚本、研究文件、运行数据库、缓存与构建产物未纳入提交。详细验收见 [项目检查计划](PROJECT_REVIEW_PLAN.md)，全部说明见 [文档索引](README.md)。

## 主题提交

| 序号 | 提交 | 改动主题 |
|---|---|---|
| 01 | [`3e89d8c0e6`](https://github.com/never-die-cold/visualising-the-band-structure-of-2D-materials/commit/3e89d8c0e6) | feat(utils): add atomic writes and application data paths |
| 02 | [`891f61f93b`](https://github.com/never-die-cold/visualising-the-band-structure-of-2D-materials/commit/891f61f93b) | fix(numerics): validate finite inputs and Hermitian TB matrices |
| 03 | [`a7070bfe46`](https://github.com/never-die-cold/visualising-the-band-structure-of-2D-materials/commit/a7070bfe46) | fix(exciton): validate solver inputs and report bound-state availability |
| 04 | [`414657c6c8`](https://github.com/never-die-cold/visualising-the-band-structure-of-2D-materials/commit/414657c6c8) | fix(analysis): reject undefined single-band Berry and orbital results |
| 05 | [`00e6bd90f1`](https://github.com/never-die-cold/visualising-the-band-structure-of-2D-materials/commit/00e6bd90f1) | fix(strain): apply Cartesian deformation consistently to lattice geometry |
| 06 | [`8a213683e0`](https://github.com/never-die-cold/visualising-the-band-structure-of-2D-materials/commit/8a213683e0) | fix(kpath): preserve three-dimensional reciprocal-space distances |
| 07 | [`bde497214b`](https://github.com/never-die-cold/visualising-the-band-structure-of-2D-materials/commit/bde497214b) | fix(properties): separate full-zone gaps and correct derivative coordinates |
| 08 | [`c86b0b5def`](https://github.com/never-die-cold/visualising-the-band-structure-of-2D-materials/commit/c86b0b5def) | perf(solver): bound DOS workspace and support cooperative cancellation |
| 09 | [`670b248b1c`](https://github.com/never-die-cold/visualising-the-band-structure-of-2D-materials/commit/670b248b1c) | feat(simulation): share engine-aware execution between GUI and CLI |
| 10 | [`ff552dd82c`](https://github.com/never-die-cold/visualising-the-band-structure-of-2D-materials/commit/ff552dd82c) | feat(tasks): freeze simulation inputs and reject mismatched model state |
| 11 | [`1f950eaaa3`](https://github.com/never-die-cold/visualising-the-band-structure-of-2D-materials/commit/1f950eaaa3) | fix(vasp): stream standard spin-aware input with independent fixtures |
| 12 | [`5edf2e2c30`](https://github.com/never-die-cold/visualising-the-band-structure-of-2D-materials/commit/5edf2e2c30) | fix(examples): convert graphene teaching input to standard VASP format |
| 13 | [`d8761e601f`](https://github.com/never-die-cold/visualising-the-band-structure-of-2D-materials/commit/d8761e601f) | fix(examples): convert MoS2 teaching input to standard VASP format |
| 14 | [`16c2548ac2`](https://github.com/never-die-cold/visualising-the-band-structure-of-2D-materials/commit/16c2548ac2) | fix(examples): preserve occupations in repeated MoS2 benchmark input |
| 15 | [`59f3f3b59d`](https://github.com/never-die-cold/visualising-the-band-structure-of-2D-materials/commit/59f3f3b59d) | fix(bandviz): classify sampled gaps and report calibrated mass fits |
| 16 | [`b389d30d82`](https://github.com/never-die-cold/visualising-the-band-structure-of-2D-materials/commit/b389d30d82) | fix(dos): normalize weighted components and prefer validated DOSCAR data |
| 17 | [`e0121d0c15`](https://github.com/never-die-cold/visualising-the-band-structure-of-2D-materials/commit/e0121d0c15) | fix(config): isolate defaults and atomically persist validated state |
| 18 | [`f8eaf26531`](https://github.com/never-die-cold/visualising-the-band-structure-of-2D-materials/commit/f8eaf26531) | fix(storage): migrate BandViz data safely and close SQLite connections |
| 19 | [`f7a4e39660`](https://github.com/never-die-cold/visualising-the-band-structure-of-2D-materials/commit/f7a4e39660) | feat(replay): record provenance and restore saved numeric model state |
| 20 | [`30bddcdef9`](https://github.com/never-die-cold/visualising-the-band-structure-of-2D-materials/commit/30bddcdef9) | feat(storage): migrate result records and persist failed task status |
| 21 | [`0bf89e18f2`](https://github.com/never-die-cold/visualising-the-band-structure-of-2D-materials/commit/0bf89e18f2) | fix(cli): complete batch execution with resume replay and status reporting |
| 22 | [`5ca2eb2f39`](https://github.com/never-die-cold/visualising-the-band-structure-of-2D-materials/commit/5ca2eb2f39) | fix(screening): persist actual gap directness and exciton solver settings |
| 23 | [`71a3e57279`](https://github.com/never-die-cold/visualising-the-band-structure-of-2D-materials/commit/71a3e57279) | fix(workers): execute frozen simulation tasks with cancellation signals |
| 24 | [`cbadaab940`](https://github.com/never-die-cold/visualising-the-band-structure-of-2D-materials/commit/cbadaab940) | fix(studio): keep GUI results geometry and task controls consistent |
| 25 | [`7b7ddcb38b`](https://github.com/never-die-cold/visualising-the-band-structure-of-2D-materials/commit/7b7ddcb38b) | feat(bandviz): add cancellable parsing and background DOS workers |
| 26 | [`873b416661`](https://github.com/never-die-cold/visualising-the-band-structure-of-2D-materials/commit/873b416661) | fix(bandviz): debounce DOS refresh and safely retire running tasks |
| 27 | [`de5c85346d`](https://github.com/never-die-cold/visualising-the-band-structure-of-2D-materials/commit/de5c85346d) | fix(plots): label BandViz path units and sampled-gap diagnostics |
| 28 | [`46411293ac`](https://github.com/never-die-cold/visualising-the-band-structure-of-2D-materials/commit/46411293ac) | fix(plots): display DOS scope broadening and window state counts |
| 29 | [`3dace226d9`](https://github.com/never-die-cold/visualising-the-band-structure-of-2D-materials/commit/3dace226d9) | test(lifecycle): cover cancellation closing stale results and GUI responsiveness |
| 30 | [`e32943d689`](https://github.com/never-die-cold/visualising-the-band-structure-of-2D-materials/commit/e32943d689) | fix(fonts): fall back to readable labels when CJK fonts are unavailable |
| 31 | [`7ae99fbff0`](https://github.com/never-die-cold/visualising-the-band-structure-of-2D-materials/commit/7ae99fbff0) | perf(benchmarks): measure streaming parsing and bounded DOS allocations |
| 32 | [`2efc8f8111`](https://github.com/never-die-cold/visualising-the-band-structure-of-2D-materials/commit/2efc8f8111) | fix(logging): store BandViz logs in the application data directory |
| 33 | [`0b1f3b6f91`](https://github.com/never-die-cold/visualising-the-band-structure-of-2D-materials/commit/0b1f3b6f91) | build(packaging): declare supported dependencies and installed app entries |
| 34 | [`c1a1cfa663`](https://github.com/never-die-cold/visualising-the-band-structure-of-2D-materials/commit/c1a1cfa663) | ci: validate installed GUI workflows on Windows and Linux |
| 35 | [`3ff88585ad`](https://github.com/never-die-cold/visualising-the-band-structure-of-2D-materials/commit/3ff88585ad) | docs(bandviz): align usage and teaching-data claims with verified behavior |
| 36 | [`36b9dcebf6`](https://github.com/never-die-cold/visualising-the-band-structure-of-2D-materials/commit/36b9dcebf6) | docs(studio): clarify engine scopes commands and platform limitations |
| 37 | [`590a00d3d8`](https://github.com/never-die-cold/visualising-the-band-structure-of-2D-materials/commit/590a00d3d8) | fix(presets): distinguish experimental path gaps from local valley results |
| 38 | [`bc0b3378c7`](https://github.com/never-die-cold/visualising-the-band-structure-of-2D-materials/commit/bc0b3378c7) | docs(models): correct BM and Zahid citations and approximation notes |
| 39 | [`f6642541b3`](https://github.com/never-die-cold/visualising-the-band-structure-of-2D-materials/commit/f6642541b3) | fix(paper-assets): analyze teaching spectra without unsupported calibration |
| 40 | [`b12c44a7bd`](https://github.com/never-die-cold/visualising-the-band-structure-of-2D-materials/commit/b12c44a7bd) | chore(gitignore): exclude installation checks and generated runtime outputs |
| 41 | [`68adaa6b39`](https://github.com/never-die-cold/visualising-the-band-structure-of-2D-materials/commit/68adaa6b39) | docs(roadmap): reconcile implemented features and pending research checks |
| 42 | [`20fa71abab`](https://github.com/never-die-cold/visualising-the-band-structure-of-2D-materials/commit/20fa71abab) | docs(review): record T01 through T15 acceptance and final verification |
| 43 | [`08109f196e`](https://github.com/never-die-cold/visualising-the-band-structure-of-2D-materials/commit/08109f196e) | docs: organize application and numerical contracts in a shared index |
| 44 | 最后一条文档提交（包含本文件） | docs(delivery): index T15 topic commits and final verification |

## 2026-10-17 改进轮次（复核后五批推进）

T15 之后的科研推进（T16/T17 门槛轮次）与本轮复核不在上表内，其状态见 [项目检查计划](PROJECT_REVIEW_PLAN.md)。本轮按主题拆分为 5 条提交，范围不含科研模型参数调整；以下 hash 为本地提交，未推送前不含远端链接。

| 序号 | 提交 | 改动主题 |
|---|---|---|
| 45 | `dbb7e38` | fix(engine): derive the Fermi reference from declared valence filling |
| 46 | `45ffca9` | fix(analysis): reject non-finite and unattainable inputs in solver paths |
| 47 | `68e0eba` | perf(bandviz): reuse plot artists, coalesce config writes and cache the parsed DOSCAR |
| 48 | `ebabdbf` | docs: reconcile T16 status and local paths; collect the transport reference and drop stale artifacts |
| 49 | `93ce315` | test(sk): freeze the assembly-convention negative result and add a criterion table |
| 50 | `ab750a3` | docs(studio): register the SK convention criterion script in the example list |
| 51 | `620313f` | docs(sk): record the literature check — the Nanoskif manual is the only path |

本轮验收：本机 Windows / Python 3.12 完整套件最终 **729 项通过**（286.71 s；T15 收束时为 641 项）。复核发现的问题、文件处置与未处理项见 [项目检查计划](PROJECT_REVIEW_PLAN.md) 的「2026-10-17 改进轮次」节。

### 后续迭代（按单项细粒度拆分）

在上表 45–51 之后继续按单个改动提交（共 52 条），按迭代分组：

| 迭代 | 主题 | 代表提交 |
|---|---|---|
| 1 | vdW Studio 结构与晶格正确性：邻居搜索上界、超胞校验、晶格一致性、SK 断言、应变保留子类 | `90df5d6` … `1451408` |
| 2 | GUI 硬编码与 moiré 后台化：qmax / n_grid / ylim、MoireWorker、关闭等待 | `fa0f4e5` … `901e057` |
| 3 | BandViz 健壮性：信号改名、traceback、DOS/数据库错误可见、关闭超时、金属标签 | `86f497f` … `adddc8b` |
| 4 | 去重与常量统一：共享物理常数、KPath 采样、TMD 参考带隙、原子写、CLI 错误、示例目录回退 | `82d58ec` … `4fe5efe` |
| 5 | lint 与 CI 门禁：ruff（E9 + pyflakes）、CI 步骤、测试适配 | `a1aefe0` … `47e9b8e` |
| 6 | 解析性能：浮点解析与有限性检查热路径、基准重测 | `7f79536` … `d92f9b7` |
| 收尾 | 共享导出 mixin、默认值单点化、文档与索引 | `1b2775e` … |
| 7 | 状态与测试基建：Clear Recent 持久化、最近文件单源、VASP/Qt 共享测试助手、取消基类、错误格式统一、pip 缓存、ISPIN=2 说明、画布重建 helper（含一次 revert） | `34bc494` … `19c0341` |
| 8 | **大目标 A**：DFT→TB 自拟合管线——SK 模型参数化（深拷贝隔离）、`fitting/` 模块（参数向量/投影权重/残差/最小二乘）、合成往返验证；投影公式修正波及 0.74 数字更正 | `b9eba5e` … |
| 9 | **大目标 B**：Wannier90 互操作——`_hr.dat` 读写、周期插值模型、solver/带隙/DOS 全链接入、合成往返验证 | `62315bd` |

完整列表见 `git log --oneline 6cf5dbe..HEAD`（共 88 条，含 1 条 revert）。
