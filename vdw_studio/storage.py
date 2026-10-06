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
            conn.executescript(_SCHEMA)

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.path)
        conn.row_factory = sqlite3.Row
        return conn

    # ------------------------------------------------------------------
    def record(self, material: str, engine: str,
               formula: str = "", gap_eV: Optional[float] = None,
               gap_direct: Optional[bool] = None,
               payload: Optional[dict[str, Any]] = None) -> int:
        """写入一条仿真记录，返回记录 id。"""
        with self._connect() as conn:
            cur = conn.execute(
                "INSERT INTO runs (timestamp, material, engine, formula,"
                " gap_eV, gap_direct, payload) VALUES (?,?,?,?,?,?,?)",
                (datetime.now(timezone.utc).isoformat(timespec="seconds"),
                 material, engine, formula,
                 gap_eV, None if gap_direct is None else int(gap_direct),
                 json.dumps(payload or {}, ensure_ascii=False)))
            return int(cur.lastrowid)

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
