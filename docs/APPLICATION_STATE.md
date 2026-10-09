# 配置与数据库生命周期

BandViz 配置、数据库和日志保存到应用数据目录，用户导出仍通过文件选择器决定位置。Windows 默认 `%LOCALAPPDATA%/BandViz`，Linux 默认 `$XDG_DATA_HOME/BandViz`（未设置时 `~/.local/share/BandViz`），macOS 默认 `~/Library/Application Support/BandViz`；macOS 未纳入当前 CI。`BANDVIZ_DATA_DIR` 可显式指定目录，测试使用项目内临时路径。

首次启动且目标不存在时，读取运行目录下旧 config/settings.json，校验后保存新配置；旧 data/bandviz.db 使用 SQLite read-only backup 导入。旧文件不修改，已有应用目录不会被旧文件覆盖。显式传入 ConfigManager/Database 路径的 Python 接口继续尊重该路径；vdW Studio 的 CLI 数据库与输出使用显式 --db/--out。

默认配置、各实例和返回字典/列表均独立深拷贝。部分嵌套配置合并默认值，已知字段校验类型、有限数值、正 σ/窗口大小、能量区间等；损坏 JSON/非对象或错误字段恢复默认并记录 diagnostics。读取损坏现有文件时不立即覆盖，保留用户检查机会。未知扩展键可保留，但必须能序列化为有限 JSON。

配置更新先校验完整候选，再写同目录临时文件、flush/fsync，最后原子替换。失败保留原文件与内存配置并清理临时文件。两个能量边界一起更新，避免中间状态倒置。GUI 捕获设置写入失败、记录日志并继续处理事件/关闭；不把失败传播为 Qt slot 未捕获异常。

两个 SQLite 封装均为每次操作独立连接，事务结束后 finally 显式 close，而非仅依赖 connection 的提交/回滚上下文。PRAGMA user_version=1 标记当前 schema，未知未来版本拒绝修改。ResultsDB 旧表按 T13 添加复现字段，BandViz 保留现有表布局；当前单机规模不引入连接池或分布式数据库。

回归包括独立实例、嵌套部分配置/坏字段、无效更新、原子替换中断、旧库只读迁移与摘要保持、连续读写后的连接关闭、未来 schema 拒绝以及真实 GUI 磁盘失败后继续响应与关闭。
