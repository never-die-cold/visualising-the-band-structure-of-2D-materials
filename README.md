# 2D Material Band Structure Visualizer

> 基于 Raspberry Pi 4B 的二维材料能带结构可视化工具

直接从 VASP 的 `EIGENVAL` 与 `KPOINTS` 文件读取数据，实时绘制能带图与态密度图（DOS），并自动计算带隙、有效质量等关键参数。

![软件主界面](paper_figures/fig_gui_main.png)

---

## 功能特性

- **能带结构可视化** — 自动读取 EIGENVAL + KPOINTS，绘制带高对称点标注的能带图
- **态密度与采样谱** — 优先读取 DOSCAR；否则按归一化 k 权重与自旋容量计算高斯展宽谱，区分占据/未占据分量与全区 DOS
- **带隙分析** — 根据已有采样点的占据数/填充识别金属、零隙和正隙，计算 VBM / CBM；部分占据证据不足时显示未确定
- **有效质量估算** — 自动读取匹配的 POSCAR，用实际倒格矩阵拟合同一直线路径上的带边质量；显示方向与拟合诊断，缺晶格或拟合不稳定时说明不可用原因
- **后台解析** — 使用 QThread 异步解析，大文件不卡界面
- **配置持久化** — JSON 保存用户设置，SQLite 记录计算历史
- **文件树管理** — 左侧目录浏览器 + 最近文件列表，双击即加载
- **导出图片** — 支持 PNG / SVG 高清导出

---

## 快速开始

```bash
# 1. Python 3.12+，安装两个应用及依赖
python -m pip install .

# 2. 运行
bandviz
```

程序启动后，左侧文件树会自动扫描 `data/example/` 下的示例数据。双击 **graphene** 或 **mos2** 即可加载并查看能带图与 DOS 图。

### 基本操作

| 操作 | 说明 |
|------|------|
| 加载文件 | 点击 **Load EIGENVAL** 或双击文件树中的 `EIGENVAL` |
| 切换视图 | 中央 Tab 切换 **Band Structure** / **DOS** |
| 调节参数 | 右侧面板设置费米能级、能量范围、DOS 展宽 σ |
| 导出图片 | 点击 **Save PNG** / **Save SVG** |

---

## 示例结果

使用 `data/example/` 中自带的石墨烯与单层 MoS₂ **合成教学数据**（均为 58 k 点、Γ-M-K-Γ 路径，并非真实 VASP 材料计算结果）：

样例采用标准 EIGENVAL 头部与显式 KPOINTS，保留原有 58 个去重后的采样点。真实 VASP line-mode 每段包含两个端点，连续段连接处会重复采样；解析器保留这些点，并检查 KPOINTS 与 EIGENVAL 的数量和坐标是否一致。更多格式、兼容与验证说明见 [VASP 输入说明](docs/VASP_INPUT.md)。

加载真实计算时，将对应 POSCAR 放在 EIGENVAL 同目录即可校准 Å⁻¹ 横轴。教学曲线没有计算晶格，保留分数距离且不报告定量质量；方向约定与拟合限制见 [坐标与导数说明](docs/DERIVATIVE_COORDINATES.md)。

| 材料 | VBM (eV) | CBM (eV) | 带隙 (eV) | 带隙类型 |
|------|---------|----------|----------|----------|
| 石墨烯教学曲线 | 0.00 | 0.01 | ≈0.01 | 合成近零隙曲线；其小正隙不代表真实石墨烯 |
| 单层 MoS₂ | −0.90 | 0.90 | 1.80 | 直接带隙（K 点） |

| 石墨烯能带 | 单层 MoS₂ 能带 |
|-----------|---------------|
| ![石墨烯能带](paper_figures/fig_graphene_band.png) | ![MoS2 能带](paper_figures/fig_mos2_band.png) |

教学采样谱支持高斯展宽参数调节（下图为历史 σ = 0.05 / 0.15 / 0.30 eV 对比，曲线不代表全 BZ DOS；归一化与范围见 [DOS 契约](docs/DOS_CONTRACT.md)）：

BandViz 的带隙仅描述导入的 k 点。vdW Studio 的周期 TB 材料另使用全区网格与局部带边搜索，区分材料带隙与路径带隙；范围、判据、收敛和部分占据限制见 [带隙分析说明](docs/GAP_ANALYSIS.md)。

![DOS 展宽对比](paper_figures/fig_mos2_dos_sigma.png)

### 性能参考

当前流式解析/分块谱的可复现基准见 [性能测量](docs/PERFORMANCE.md)：4096 点 × 64 带合成输入，解析峰值约 4.38 MiB；401 能量点、4 MiB 工作区的 DOS 总峰值约 10.23 MiB。包含 NumPy/Python 追踪开销，不能代替真实大 VASP 文件或树莓派测量。

`paper_results.json` 与旧插图保留为历史教学产物；`paper_assets.py` 导入即生成全部产物，正常运行应用不会调用。新基准使用独立脚本 `python -m examples.benchmark_bandviz`。

---

## 技术栈

| 层级 | 技术 | 说明 |
|------|------|------|
| 数据解析 | Python 标准库 + NumPy | 自定义 EIGENVAL / KPOINTS 解析器 |
| 数值计算 | NumPy + SciPy | 数组运算、高斯展宽、抛物线拟合 |
| 可视化 | Matplotlib + PyQt5 | `Qt5Agg` 后端，能带图 + DOS 图 |
| 持久化 | SQLite + JSON | SQLite 存计算历史，JSON 存用户配置 |
| 日志 | Python `logging` | 控制台 + 文件双输出 |
| 并发 | PyQt5 `QThread` | 后台解析 worker，不阻塞 GUI |

---

## 项目结构

```
.
├── main.py                      # 程序入口
├── requirements.txt             # 依赖列表
├── config/
│   └── settings.json            # 旧配置位置；首次启动只读迁移
├── core/                        # 核心计算
│   ├── parser.py                # EIGENVAL / KPOINTS 解析
│   ├── band_analyzer.py         # 能带分析（带隙、有效质量）
│   └── dos_analyzer.py          # DOS 计算
├── gui/                         # 前端界面
│   ├── main_window.py           # 主窗口
│   ├── band_widget.py           # 能带图组件
│   ├── dos_widget.py            # DOS 图组件
│   ├── control_panel.py         # 控制面板
│   ├── file_tree.py             # 文件树
│   └── workers/
│       └── parse_worker.py      # 后台解析线程
├── storage/                     # 持久化
│   ├── database.py              # SQLite 封装
│   └── config_manager.py        # JSON 配置管理
├── utils/
│   └── logger.py                # 日志配置
├── paper_assets.py              # 生成示例结果与插图的脚本
├── screenshot_gui.py            # 离屏截取主界面的工具脚本
├── paper_figures/               # README 与文档用图
└── data/
    └── example/                 # 示例数据（graphene / mos2 / mos2_x4）
```

---

## 运行环境

- **已纳入 CI 的环境**：Windows / Linux，Python 3.12 / 3.13
- **最低 Python**：3.12（当前 NumPy/SciPy 依赖要求）；开发测试安装使用 `python -m pip install ".[test]"`
- **树莓派**：最初设计目标，当前固定依赖未做 ARM / Raspberry Pi OS 真机安装验证，暂不承诺直接安装成功
- **显示**：HDMI 直连 或 VNC 远程

源码开发也可使用 `python main.py`。`requirements.txt` 保留与包声明相同的运行版本；完整安装验证与平台限制见 [安装说明](docs/INSTALLATION.md)。

---

## 许可证

见 [MIT License](LICENSE)。第三方测试夹具保留其独立许可证。

> 📌 **本仓库还有新版本：[vdW Studio — 二维半导体仿真平台](vdw_studio/README.md)**
> （结构建模 → 仿真计算 → 性质分析 → 可视化 一体化，与树莓派版相互独立，详见其 README）

应用配置/数据库/日志保存到系统应用数据目录，已有旧配置与数据库只读迁移；详情见 [数据生命周期](docs/APPLICATION_STATE.md)。取消、关闭与后台 DOS 调参见 [任务生命周期](docs/TASK_LIFECYCLE.md)。

完整说明见 [文档索引](docs/README.md)；T01–T15 的验收与后续待办见 [项目检查计划](docs/PROJECT_REVIEW_PLAN.md)，本次改动的版本定位见 [提交索引](docs/COMMIT_INDEX.md)。
