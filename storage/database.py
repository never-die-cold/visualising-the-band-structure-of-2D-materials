"""
SQLite persistence layer for calculation metadata and task history.
"""

import sqlite3
import json
from pathlib import Path
from datetime import datetime
from typing import List, Optional, Dict, Any


class Database:
    """
    计算任务元数据与历史记录的 SQLite 持久化。
    自动创建 tasks 表和 settings 表。
    """

    def __init__(self, db_path: str = "data/bandviz.db"):
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._init_tables()

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(str(self.db_path))
        conn.row_factory = sqlite3.Row
        return conn

    def _init_tables(self):
        """初始化数据库表结构"""
        with self._connect() as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS tasks (
                    id          INTEGER PRIMARY KEY AUTOINCREMENT,
                    name        TEXT NOT NULL,
                    path        TEXT NOT NULL UNIQUE,
                    system      TEXT,
                    lattice_constant REAL,
                    band_gap    REAL,
                    is_direct   INTEGER,
                    vbm         REAL,
                    cbm         REAL,
                    num_bands   INTEGER,
                    num_electrons INTEGER,
                    num_kpoints INTEGER,
                    kpoint_labels TEXT,
                    created_at  TEXT NOT NULL
                )
            """)

            conn.execute("""
                CREATE TABLE IF NOT EXISTS history (
                    id          INTEGER PRIMARY KEY AUTOINCREMENT,
                    task_id     INTEGER,
                    action      TEXT NOT NULL,
                    details     TEXT,
                    created_at  TEXT NOT NULL,
                    FOREIGN KEY (task_id) REFERENCES tasks(id)
                )
            """)

            conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_tasks_path ON tasks(path);
            """)
            conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_tasks_gap ON tasks(band_gap);
            """)
            conn.commit()

    # ------------------------------------------------------------------
    # Task CRUD
    # ------------------------------------------------------------------
    def add_task(
        self,
        name: str,
        path: str,
        system: str = "",
        lattice_constant: Optional[float] = None,
        band_gap: Optional[float] = None,
        is_direct: Optional[bool] = None,
        vbm: Optional[float] = None,
        cbm: Optional[float] = None,
        num_bands: Optional[int] = None,
        num_electrons: Optional[int] = None,
        num_kpoints: Optional[int] = None,
        kpoint_labels: Optional[List[tuple]] = None,
    ) -> int:
        """添加或更新任务记录，返回 task_id"""
        created_at = datetime.now().isoformat()
        klabels_json = json.dumps(kpoint_labels) if kpoint_labels else "[]"

        with self._connect() as conn:
            # 如果路径已存在则更新
            existing = conn.execute(
                "SELECT id FROM tasks WHERE path = ?", (path,)
            ).fetchone()

            if existing:
                task_id = existing["id"]
                conn.execute("""
                    UPDATE tasks SET
                        name = ?, system = ?, lattice_constant = ?,
                        band_gap = ?, is_direct = ?, vbm = ?, cbm = ?,
                        num_bands = ?, num_electrons = ?, num_kpoints = ?,
                        kpoint_labels = ?, created_at = ?
                    WHERE id = ?
                """, (
                    name, system, lattice_constant,
                    band_gap, int(is_direct) if is_direct is not None else None,
                    vbm, cbm, num_bands, num_electrons, num_kpoints,
                    klabels_json, created_at, task_id
                ))
            else:
                cursor = conn.execute("""
                    INSERT INTO tasks (
                        name, path, system, lattice_constant,
                        band_gap, is_direct, vbm, cbm,
                        num_bands, num_electrons, num_kpoints,
                        kpoint_labels, created_at
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, (
                    name, path, system, lattice_constant,
                    band_gap, int(is_direct) if is_direct is not None else None,
                    vbm, cbm, num_bands, num_electrons, num_kpoints,
                    klabels_json, created_at
                ))
                task_id = cursor.lastrowid

            conn.commit()
            return task_id

    def get_task(self, task_id: int) -> Optional[Dict[str, Any]]:
        """根据 ID 获取任务详情"""
        with self._connect() as conn:
            row = conn.execute(
                "SELECT * FROM tasks WHERE id = ?", (task_id,)
            ).fetchone()
            return dict(row) if row else None

    def get_task_by_path(self, path: str) -> Optional[Dict[str, Any]]:
        """根据路径获取任务详情"""
        with self._connect() as conn:
            row = conn.execute(
                "SELECT * FROM tasks WHERE path = ?", (path,)
            ).fetchone()
            return dict(row) if row else None

    def list_tasks(
        self,
        system: Optional[str] = None,
        min_gap: Optional[float] = None,
        max_gap: Optional[float] = None,
        limit: int = 100
    ) -> List[Dict[str, Any]]:
        """按条件查询任务列表"""
        query = "SELECT * FROM tasks WHERE 1=1"
        params = []

        if system:
            query += " AND system LIKE ?"
            params.append(f"%{system}%")
        if min_gap is not None:
            query += " AND band_gap >= ?"
            params.append(min_gap)
        if max_gap is not None:
            query += " AND band_gap <= ?"
            params.append(max_gap)

        query += " ORDER BY created_at DESC LIMIT ?"
        params.append(limit)

        with self._connect() as conn:
            rows = conn.execute(query, params).fetchall()
            return [dict(r) for r in rows]

    def delete_task(self, task_id: int) -> bool:
        """删除任务及其历史记录"""
        with self._connect() as conn:
            conn.execute("DELETE FROM history WHERE task_id = ?", (task_id,))
            conn.execute("DELETE FROM tasks WHERE id = ?", (task_id,))
            conn.commit()
            return True

    # ------------------------------------------------------------------
    # History
    # ------------------------------------------------------------------
    def add_history(self, task_id: Optional[int], action: str, details: str = ""):
        """记录操作历史"""
        created_at = datetime.now().isoformat()
        with self._connect() as conn:
            conn.execute("""
                INSERT INTO history (task_id, action, details, created_at)
                VALUES (?, ?, ?, ?)
            """, (task_id, action, details, created_at))
            conn.commit()

    def get_history(self, task_id: Optional[int] = None, limit: int = 50) -> List[Dict[str, Any]]:
        """获取操作历史"""
        query = "SELECT * FROM history"
        params = []
        if task_id is not None:
            query += " WHERE task_id = ?"
            params.append(task_id)
        query += " ORDER BY created_at DESC LIMIT ?"
        params.append(limit)

        with self._connect() as conn:
            rows = conn.execute(query, params).fetchall()
            return [dict(r) for r in rows]

    # ------------------------------------------------------------------
    # Statistics
    # ------------------------------------------------------------------
    def get_stats(self) -> Dict[str, Any]:
        """获取数据库统计信息"""
        with self._connect() as conn:
            total_tasks = conn.execute(
                "SELECT COUNT(*) as count FROM tasks"
            ).fetchone()["count"]

            gap_stats = conn.execute("""
                SELECT AVG(band_gap) as avg_gap,
                       MIN(band_gap) as min_gap,
                       MAX(band_gap) as max_gap
                FROM tasks WHERE band_gap IS NOT NULL
            """).fetchone()

            direct_count = conn.execute("""
                SELECT COUNT(*) as count FROM tasks WHERE is_direct = 1
            """).fetchone()["count"]

            return {
                'total_tasks': total_tasks,
                'avg_gap': gap_stats['avg_gap'],
                'min_gap': gap_stats['min_gap'],
                'max_gap': gap_stats['max_gap'],
                'direct_count': direct_count,
            }
