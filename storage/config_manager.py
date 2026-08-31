"""
JSON configuration manager for user settings and application state.
"""

import json
from pathlib import Path
from typing import Any, Optional


DEFAULT_CONFIG = {
    "theme": "default",
    "energy_range": {"min": -5.0, "max": 5.0},
    "fermi_level": 0.0,
    "dos_sigma": 0.05,
    "dos_show_total": True,
    "dos_show_vb": True,
    "dos_show_cb": True,
    "export_format": "png",
    "export_dpi": 300,
    "recent_files": [],
    "last_open_dir": ".",
    "window_geometry": {"x": 100, "y": 100, "width": 1400, "height": 900},
}


class ConfigManager:
    """
    管理应用全局配置的 JSON 持久化。
    读写 config/settings.json，提供类型安全的 getter/setter。
    """

    def __init__(self, config_path: str = "config/settings.json"):
        self.config_path = Path(config_path)
        self.config_path.parent.mkdir(parents=True, exist_ok=True)
        self._config: dict = {}
        self.load()

    def load(self):
        """从 JSON 文件加载配置，不存在则使用默认配置"""
        if self.config_path.exists():
            try:
                with open(self.config_path, 'r', encoding='utf-8') as f:
                    loaded = json.load(f)
                # 合并默认配置，确保新字段存在
                self._config = {**DEFAULT_CONFIG, **loaded}
            except (json.JSONDecodeError, IOError):
                self._config = DEFAULT_CONFIG.copy()
        else:
            self._config = DEFAULT_CONFIG.copy()
            self.save()

    def save(self):
        """保存当前配置到 JSON 文件"""
        try:
            with open(self.config_path, 'w', encoding='utf-8') as f:
                json.dump(self._config, f, indent=2, ensure_ascii=False)
        except IOError as e:
            print(f"[ConfigManager] Failed to save config: {e}")

    def get(self, key: str, default: Any = None) -> Any:
        """获取配置项，支持点号分隔的嵌套键，如 'energy_range.min'"""
        keys = key.split('.')
        value = self._config
        for k in keys:
            if isinstance(value, dict) and k in value:
                value = value[k]
            else:
                return default
        return value

    def set(self, key: str, value: Any):
        """设置配置项，支持点号分隔的嵌套键"""
        keys = key.split('.')
        target = self._config
        for k in keys[:-1]:
            if k not in target or not isinstance(target[k], dict):
                target[k] = {}
            target = target[k]
        target[keys[-1]] = value
        self.save()

    def update(self, updates: dict):
        """批量更新配置"""
        self._config.update(updates)
        self.save()

    def get_energy_range(self) -> tuple:
        """获取默认能量范围"""
        return (
            self.get('energy_range.min', -5.0),
            self.get('energy_range.max', 5.0)
        )

    def set_energy_range(self, emin: float, emax: float):
        self.set('energy_range.min', emin)
        self.set('energy_range.max', emax)

    def add_recent_file(self, path: str, max_count: int = 10):
        """添加最近文件，保持列表长度限制"""
        recent = self.get('recent_files', [])
        path = str(Path(path).resolve())
        if path in recent:
            recent.remove(path)
        recent.insert(0, path)
        recent = recent[:max_count]
        self.set('recent_files', recent)

    def get_recent_files(self) -> list:
        return self.get('recent_files', [])

    def get_all(self) -> dict:
        """返回完整配置字典的副本"""
        return self._config.copy()

    def reset_to_default(self):
        """重置为默认配置"""
        self._config = DEFAULT_CONFIG.copy()
        self.save()
