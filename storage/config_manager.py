"""
JSON configuration manager for user settings and application state.
"""

import json
import math
from copy import deepcopy
from utils.atomic_io import write_json_atomic
from utils.paths import app_data_dir
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

    def __init__(self, config_path: Optional[str] = None):
        self.config_path = Path(config_path) if config_path is not None else app_data_dir() / "settings.json"
        self.diagnostics = []
        self._legacy_path = Path("config/settings.json") if config_path is None else None
        self.config_path.parent.mkdir(parents=True, exist_ok=True)
        self._config: dict = {}
        self.load()

    def load(self):
        """Merge partial configuration and repair malformed known fields."""
        source = self.config_path
        if not source.exists() and self._legacy_path is not None and self._legacy_path.exists():
            source = self._legacy_path
        try:
            loaded = json.loads(source.read_text(encoding='utf-8')) if source.exists() else {}
            if not isinstance(loaded, dict):
                raise ValueError('configuration root must be an object')
            self._config = self._validated(self._merge(deepcopy(DEFAULT_CONFIG), loaded), repair=True)
        except (OSError, ValueError, TypeError) as exc:
            self.diagnostics.append(str(exc))
            self._config = deepcopy(DEFAULT_CONFIG)
        if not self.config_path.exists():
            self.save()

    @staticmethod
    def _merge(base, updates):
        for key, value in updates.items():
            if isinstance(base.get(key), dict) and isinstance(value, dict):
                ConfigManager._merge(base[key], value)
            else:
                base[key] = deepcopy(value)
        return base

    def _validated(self, config, repair=False):
        def finite(value):
            return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value)
        def validate(mapping, defaults, prefix=''):
            for key, default in defaults.items():
                value = mapping.get(key)
                location = prefix + key
                if isinstance(default, dict):
                    valid = isinstance(value, dict)
                elif key == 'recent_files':
                    valid = isinstance(value, list) and all(isinstance(p, str) for p in value)
                elif isinstance(default, bool):
                    valid = isinstance(value, bool)
                elif isinstance(default, (int, float)):
                    valid = finite(value)
                    if key in ('width', 'height', 'export_dpi'):
                        valid = valid and isinstance(value, int) and value > 0
                    if key in ('x', 'y'):
                        valid = valid and isinstance(value, int)
                    if key == 'dos_sigma':
                        valid = valid and value > 0
                else:
                    valid = isinstance(value, str)
                    if key == 'export_format':
                        valid = valid and value in ('png', 'svg')
                if not valid:
                    if not repair:
                        raise ValueError(f'Invalid configuration field: {location}')
                    self.diagnostics.append(f'Reset invalid configuration field: {location}')
                    mapping[key] = deepcopy(default)
                if isinstance(default, dict):
                    validate(mapping[key], default, location + '.')
        validate(config, DEFAULT_CONFIG)
        bounds = config['energy_range']
        if bounds['min'] >= bounds['max']:
            if not repair:
                raise ValueError('energy_range must have increasing bounds')
            config['energy_range'] = deepcopy(DEFAULT_CONFIG['energy_range'])
            self.diagnostics.append('Reset invalid energy_range')
        # Unknown extension keys remain supported but must still serialize as finite JSON.
        json.dumps(config, allow_nan=False)
        return config

    def save(self):
        write_json_atomic(self.config_path, self._validated(deepcopy(self._config)))

    def get(self, key: str, default: Any = None) -> Any:
        """获取配置项，支持点号分隔的嵌套键，如 'energy_range.min'"""
        keys = key.split('.')
        value = self._config
        for k in keys:
            if isinstance(value, dict) and k in value:
                value = value[k]
            else:
                return default
        return deepcopy(value)

    def set(self, key: str, value: Any):
        candidate = deepcopy(self._config)
        keys = key.split('.')
        target = candidate
        for component in keys[:-1]:
            if not isinstance(target.get(component), dict):
                target[component] = {}
            target = target[component]
        target[keys[-1]] = deepcopy(value)
        self._replace(candidate)

    def _replace(self, candidate):
        candidate = self._validated(candidate)
        write_json_atomic(self.config_path, candidate)
        self._config = candidate

    def update(self, updates: dict):
        if not isinstance(updates, dict):
            raise ValueError('updates must be an object')
        self._replace(self._merge(deepcopy(self._config), updates))

    def get_energy_range(self) -> tuple:
        """获取默认能量范围"""
        return (
            self.get('energy_range.min', -5.0),
            self.get('energy_range.max', 5.0)
        )

    def set_energy_range(self, emin: float, emax: float):
        self.update({'energy_range': {'min': emin, 'max': emax}})

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
        return deepcopy(self._config)

    def reset_to_default(self):
        """重置为默认配置"""
        self._replace(deepcopy(DEFAULT_CONFIG))
