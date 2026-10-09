# 结果复现、恢复与筛选

CLI summary 使用 schema_version=1，保存 immutable task 的材料/来源、全部最终模型参数、结构、实际路径、填充和采样参数；provenance 保存 Git revision、实际 Python/依赖版本、每个 vdW Studio Python 文件及组合 SHA-256。源文件统一 LF 后哈希，未提交修改也参与身份；仅 commit SHA 不能代表本次实现。

task_key 排除每次调用的 UUID，包含输入、源代码、依赖与 Python 版本。重复输入且软件一致时身份相同；模型、σ、网格或软件改变时不同。结果保存真实带隙域/直接性、带边坐标、网格、极值方法、容差和收敛状态；DOS 保存能量窗口/点数、自旋容量与积分。PNG/npz/POSCAR 保存内容摘要。

```sh
vdw-studio-cli run-all --out results/batch --db results/runs.db
vdw-studio-cli run-all --out results/batch --db results/runs.db --resume
vdw-studio-cli replay results/batch/mos2_kp/summary.json --out results/replayed
vdw-studio-cli export-records --db results/runs.db --out results/records.csv
python -m examples.run_screening --db results/screening.db --resume --csv results/screening.csv
```

恢复只复用相同身份、完整所需产物且文件摘要一致的成功任务；缺失/损坏输出重新计算。逐材料写入批次汇总，失败不终止后续材料。SQLite 记录 completed/failed、错误与任务键；同键重试更新该任务当前状态，不重复增加成功行。构建模型前失败按请求键保留记录。历史尝试日志尚未单独保存，不把当前状态表当作不可变审计日志。

replay 从记录重建最终 TB 在位能/跃迁/晶格或 TMD k·p 参数，不调用当前预设工厂。只允许明确列出的已支持模型；不会按记录中的类名动态导入/执行代码。重建输出 bands.npz 与 gap 对比、当前 provenance，并明确源代码是否一致。TB/k·p 谱可重建；未验证 SK/BM 不承诺重建，BandViz 外部 VASP 输入仍需用户保留原文件。

筛选的 gap_direct 使用 GapResult.direct，零隙/金属为 null，间接绝缘隙为 false；不再根据“存在正隙”猜测直接性。各应变任务即时入库、失败后继续、已完成键可恢复。激子示例保存 μ/r₀/介电环境、径向盒、基组/求积/返回态数及实际结果状态；示例参数不代表材料标定。CSV 包含完整 JSON payload，不只导出带隙数值。

旧 runs 表自动添加键/状态/error/schema_version，保留原 ID 与数据；未来未知 schema 拒绝降级。独立回归使用带边分居 (0,0)/(1/2,0)、精确隙 0.9 eV 的周期 TB，验证间接性；另检查应变 TB/k·p 数值状态重建、失败重试、产物损坏、任务变化和旧库迁移。
