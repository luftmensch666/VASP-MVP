from __future__ import annotations

import sqlite3
from datetime import datetime
from pathlib import Path

from .models import Metrics, TaskRecord, TaskStatus, TaskType


DB_NAME = "vasp_mvp.db"

TASKS_SCHEMA = """
CREATE TABLE IF NOT EXISTS tasks (
    task_id TEXT PRIMARY KEY,
    project TEXT NOT NULL,
    task_type TEXT NOT NULL,
    status TEXT NOT NULL,
    task_root TEXT NOT NULL,
    pid INTEGER,
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL
);
"""

METRICS_SCHEMA = """
CREATE TABLE IF NOT EXISTS metrics (
    task_id TEXT PRIMARY KEY,
    toten REAL,
    loop_avg REAL,
    converged INTEGER,
    updated_at TEXT NOT NULL,
    FOREIGN KEY(task_id) REFERENCES tasks(task_id)
);
"""


def db_path(workspace: Path) -> Path:
    workspace = Path(workspace)
    workspace.mkdir(parents=True, exist_ok=True)
    return workspace / DB_NAME


def init_db(workspace: Path) -> sqlite3.Connection:
    conn = sqlite3.connect(db_path(workspace))
    conn.row_factory = sqlite3.Row
    conn.execute(TASKS_SCHEMA)
    conn.execute(METRICS_SCHEMA)
    conn.commit()
    return conn


def create_task(
    conn: sqlite3.Connection,
    *,
    task_id: str,
    project: str,
    task_type: TaskType,
    task_root: Path,
    status: TaskStatus = "ready",
    pid: int | None = None,
) -> None:
    now = _now()
    conn.execute(
        """
        INSERT INTO tasks (task_id, project, task_type, status, task_root, pid, created_at, updated_at)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        ON CONFLICT(task_id) DO UPDATE SET
            project = excluded.project,
            task_type = excluded.task_type,
            status = excluded.status,
            task_root = excluded.task_root,
            pid = excluded.pid,
            updated_at = excluded.updated_at
        """,
        (task_id, project, task_type, status, str(Path(task_root)), pid, now, now),
    )
    conn.commit()


def update_status(
    conn: sqlite3.Connection,
    task_id: str,
    status: TaskStatus,
    pid: int | None = None,
) -> None:
    conn.execute(
        """
        UPDATE tasks
        SET status = ?, pid = COALESCE(?, pid), updated_at = ?
        WHERE task_id = ?
        """,
        (status, pid, _now(), task_id),
    )
    conn.commit()


def list_tasks(conn: sqlite3.Connection) -> list[TaskRecord]:
    rows = conn.execute("SELECT * FROM tasks ORDER BY updated_at DESC").fetchall()
    return [_row_to_task(row) for row in rows]


def save_metrics(
    conn: sqlite3.Connection,
    task_id: str,
    *,
    toten: float | None = None,
    loop_avg: float | None = None,
    converged: bool | None = None,
) -> None:
    conn.execute(
        """
        INSERT INTO metrics (task_id, toten, loop_avg, converged, updated_at)
        VALUES (?, ?, ?, ?, ?)
        ON CONFLICT(task_id) DO UPDATE SET
            toten = excluded.toten,
            loop_avg = excluded.loop_avg,
            converged = excluded.converged,
            updated_at = excluded.updated_at
        """,
        (
            task_id,
            toten,
            loop_avg,
            None if converged is None else int(converged),
            _now(),
        ),
    )
    conn.commit()


def get_metrics(conn: sqlite3.Connection, task_id: str) -> Metrics | None:
    row = conn.execute("SELECT * FROM metrics WHERE task_id = ?", (task_id,)).fetchone()
    if row is None:
        return None
    return Metrics(
        toten_ev=row["toten"],
        loop_avg_seconds=row["loop_avg"],
        ionic_converged=None if row["converged"] is None else bool(row["converged"]),
    )


def connect(workspace: Path) -> sqlite3.Connection:
    return init_db(workspace)


def upsert_task(
    conn: sqlite3.Connection,
    *,
    task_id: str,
    status: TaskStatus,
    path: Path,
    task_type: TaskType,
    pid: int | None = None,
    project: str = "default",
) -> None:
    create_task(
        conn,
        task_id=task_id,
        project=project,
        task_type=task_type,
        task_root=path,
        status=status,
        pid=pid,
    )


def _row_to_task(row: sqlite3.Row) -> TaskRecord:
    return TaskRecord(
        task_id=row["task_id"],
        project=row["project"],
        task_type=row["task_type"],
        status=row["status"],
        task_root=Path(row["task_root"]),
        pid=row["pid"],
        created_at=datetime.fromisoformat(row["created_at"]),
        updated_at=datetime.fromisoformat(row["updated_at"]),
    )


def _now() -> str:
    return datetime.utcnow().isoformat(timespec="seconds")
