# 安装与平台验证

最低 Python 3.12。当前固定 NumPy 2.5.1 / SciPy 1.18.0 的安装元数据均要求 Python ≥3.12；Matplotlib 3.11.0 要求 ≥3.11，PyQt5 5.15.11 要求 ≥3.8。统一以最严格要求作为项目下限，移除原 3.9/3.10 支持宣称。

在仓库根目录使用独立环境：

```sh
python -m venv .venv
# Windows PowerShell: .venv\Scripts\Activate.ps1
# Linux: source .venv/bin/activate
python -m pip install .
python -m pip check
bandviz
vdw-studio
vdw-studio-cli list
```

开发测试：`python -m pip install ".[test]"` 后 `python -m pytest tests -q`。源码入口 `python main.py`、`python -m vdw_studio.gui`、`python -m vdw_studio.cli` 同样可用。requirements.txt 保留相同的四个运行依赖版本，pyproject.toml 是安装与 CI 的统一声明。

CI 矩阵为 Windows/Linux × Python 3.12/3.13。按包声明安装并检查依赖；从安装位置启动 CLI；`-I` 隔离模式检查三个命令入口加载，以及实际 BandViz 导入和 vdW Studio 默认 MoS₂ 后台任务。完整测试还覆盖真实 run-all 出图/入库/失败继续；独立解析和物理基准不依赖在线数据。

CI 配置已覆盖这些环境，远端执行结果需以实际工作流为准。本地验证记录见 PROJECT_REVIEW_PLAN.md。Linux 离屏 Qt 仍需工作流中列出的系统图形库；Raspberry Pi/ARM 未验证，暂不宣称支持。示例数据和论文图片留在源码仓库，不作为 wheel 的运行依赖；安装后可从文件选择器加载自己的 VASP 文件。

独立安装验证使用项目内 `.venv-install-check` 和 `.install-check` 临时目录，不修改系统环境。这些目录与测试缓存、构建产物及 results 输出已忽略。BandViz 状态已迁到系统应用目录，首次只读迁移旧数据；详见 APPLICATION_STATE.md。
