"""仿真结果数据库（SQLite，Phase 6 平台化）。

每次仿真落一条记录（材料/引擎/参数/结果 JSON），支持按材料检索——
高通量批量筛选（ROADMAP Phase 6）的数据底座。

用法::

    db = ResultsDB("results.db")
    db.record(material="MoS2", engine="kp", formula="MoS2",
              gap_eV=1.67, payload={"effective_masses": {...}})
    rows = db.history(material="MoS2")
"""

from __future__ import annotations

import json
import sqlite3
import csv
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, List, Optional

_SCHEMA = """
CREATE TABLE IF NOT EXISTS runs (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    timestamp TEXT NOT NULL,
    material TEXT NOT NULL,
    engine TEXT NOT NULL,
    formula TEXT,
    gap_eV REAL,
    gap_direct INTEGER,
    payload TEXT
);
CREATE INDEX IF NOT EXISTS idx_runs_material ON runs(material);
"""


class ResultsDB:
    """仿真结果 SQLite 封装（线程安全：每次操作独立连接）。"""

    def __init__(self, path: str = "vdw_results.db") -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self._connect() as conn:
            version = conn.execute('PRAGMA user_version').fetchone()[0]
            if version > 1:
                raise ValueError(f'Unsupported results database schema version: {version}')
            conn.executescript(_SCHEMA)
            columns = {row['name'] for row in conn.execute('PRAGMA table_info(runs)')}
            for name, declaration in [('task_key', 'TEXT'), ('status', "TEXT NOT NULL DEFAULT 'completed'"),
                                      ('error', 'TEXT'), ('schema_version', 'INTEGER NOT NULL DEFAULT 1')]:
                if name not in columns:
                    conn.execute(f'ALTER TABLE runs ADD COLUMN {name} {declaration}')
            conn.execute('CREATE UNIQUE INDEX IF NOT EXISTS idx_runs_task_key ON runs(task_key)')
            conn.execute('PRAGMA user_version=1')

    @contextmanager
    def _connect(self):
        conn = sqlite3.connect(self.path)
        conn.row_factory = sqlite3.Row
        try:
            with conn:
                yield conn
        finally:
            conn.close()

    # ------------------------------------------------------------------
    def record(self, material: str, engine: str,
               formula: str = "", gap_eV: Optional[float] = None,
               gap_direct: Optional[bool] = None,
               payload: Optional[dict[str, Any]] = None, *, task_key=None,
               status='completed', error=None) -> int:
        """写入一条仿真记录，返回记录 id。"""
        if status not in ('completed', 'failed'):
            raise ValueError('status must be completed or failed')
        with self._connect() as conn:
            cur = conn.execute(
                "INSERT INTO runs (timestamp, material, engine, formula,"
                " gap_eV, gap_direct, payload, task_key, status, error, schema_version) VALUES (?,?,?,?,?,?,?,?,?,?,1)"
                " ON CONFLICT(task_key) DO UPDATE SET timestamp=excluded.timestamp,"
                " material=excluded.material, engine=excluded.engine, formula=excluded.formula,"
                " gap_eV=excluded.gap_eV, gap_direct=excluded.gap_direct, payload=excluded.payload,"
                " status=excluded.status, error=excluded.error, schema_version=excluded.schema_version",
                (datetime.now(timezone.utc).isoformat(timespec="seconds"),
                 material, engine, formula,
                 gap_eV, None if gap_direct is None else int(gap_direct),
                 json.dumps(payload or {}, ensure_ascii=False, allow_nan=False), task_key, status, error))
            return int(cur.lastrowid) if task_key is None else int(conn.execute(
                'SELECT id FROM runs WHERE task_key=?', (task_key,)).fetchone()['id'])

    def history(self, material: Optional[str] = None,
                limit: int = 50) -> List[dict]:
        """按时间倒序检索记录（可按材料过滤）。"""
        sql = "SELECT * FROM runs"
        args: list = []
        if material is not None:
            sql += " WHERE material = ?"
            args.append(material)
        sql += " ORDER BY id DESC LIMIT ?"
        args.append(limit)
        with self._connect() as conn:
            rows = conn.execute(sql, args).fetchall()
        out = []
        for row in rows:
            d = dict(row)
            d["gap_direct"] = None if d["gap_direct"] is None \
                else bool(d["gap_direct"])
            d["payload"] = json.loads(d["payload"] or "{}")
            out.append(d)
        return out

    def export_csv(self, path, material=None):
        rows = self.history(material=material, limit=-1)
        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        fields = ['id', 'timestamp', 'material', 'engine', 'formula', 'gap_eV',
                  'gap_direct', 'status', 'error', 'task_key', 'schema_version', 'payload']
        with target.open('w', encoding='utf-8-sig', newline='') as stream:
            writer = csv.DictWriter(stream, fieldnames=fields)
            writer.writeheader()
            for row in rows:
                row['payload'] = json.dumps(row['payload'], ensure_ascii=False, allow_nan=False)
                writer.writerow({key: row.get(key) for key in fields})
        return len(rows)

    def find_task(self, task_key):
        with self._connect() as conn:
            row = conn.execute('SELECT * FROM runs WHERE task_key=?', (task_key,)).fetchone()
        if row is None:
            return None
        result = dict(row)
        result['payload'] = json.loads(result['payload'])
        result['gap_direct'] = None if result['gap_direct'] is None else bool(result['gap_direct'])
        return result
