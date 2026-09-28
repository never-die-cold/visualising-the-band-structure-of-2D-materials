"""vdW Studio —— 二维半导体材料仿真模拟软件（Simulator Edition）。

vdW Studio 是本仓库的第二个版本：面向二维（范德华）半导体材料的
"结构建模 → 仿真计算 → 性质分析 → 结果可视化" 一体化软件。

设计上仿照 Materials Studio / VESTA / QuantumATK 等主流仿真建模软件的
工作流（工程树 + 3D 结构视图 + 参数面板 + 多标签结果页），并结合二维材料
科研前沿规划特色功能（应变工程、介电环境/激子物理、谷自旋物理、
转角莫尔超晶格等，见 ROADMAP）。

与仓库根目录的树莓派轻量可视化版（BandViz, ``main.py`` + ``core/`` + ``gui/``）
相互独立、互不依赖：

- **Pi 版（BandViz）**：解析 VASP ``EIGENVAL``/``KPOINTS``，轻量能带可视化，
  适配 Raspberry Pi 4B；
- **仿真版（vdW Studio，``vdw_studio/`` 包）**：内置二维材料结构库与紧束缚
  仿真引擎，不依赖外部第一性原理软件即可完成全流程仿真。

模块总览::

    vdw_studio/
    ├── elements.py       # 元素周期表数据（质量 / 共价半径 / CPK 颜色）
    ├── structure/        # 结构建模：晶格、晶体、二维材料构建器
    ├── engine/           # 仿真引擎：k 路径、紧束缚哈密顿量、能带 / DOS 求解
    ├── analysis/         # 性质分析：带隙、有效质量、费米速度
    ├── presets/          # 材料预设参数库
    ├── visualization/    # 可视化：结构 / 能带 / DOS / 布里渊区
    ├── gui/              # PyQt5 图形界面
    ├── io/               # 结构导入导出（POSCAR / XYZ / 工程文件）
    └── cli.py            # 命令行接口
"""

__version__ = "0.1.0"

__all__ = ["__version__"]
