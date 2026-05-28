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
    updated_at TEXT NOT NULL,
    start_time TEXT,
    end_time TEXT,
    return_code INTEGER
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
    _migrate_tasks_table(conn)
    conn.commit()
    return conn


def create_task(
    conn: sqlite3.Connection,
    *,
    task_id: str,
    project: str,
    task_type: TaskType,
    task_root: Path,
    status: TaskStatus = "committed",
    pid: int | None = None,
    start_time: datetime | None = None,
    end_time: datetime | None = None,
    return_code: int | None = None,
) -> None:
    now = _now()
    conn.execute(
        """
        INSERT INTO tasks (
            task_id, project, task_type, status, task_root, pid,
            created_at, updated_at, start_time, end_time, return_code
        )
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        ON CONFLICT(task_id) DO UPDATE SET
            project = excluded.project,
            task_type = excluded.task_type,
            status = excluded.status,
            task_root = excluded.task_root,
            pid = excluded.pid,
            start_time = excluded.start_time,
            end_time = excluded.end_time,
            return_code = excluded.return_code,
            updated_at = excluded.updated_at
        """,
        (
            task_id,
            project,
            task_type,
            status,
            str(Path(task_root)),
            pid,
            now,
            now,
            _dt_to_text(start_time),
            _dt_to_text(end_time),
            return_code,
        ),
    )
    conn.commit()


def update_task_status(
    conn: sqlite3.Connection,
    task_id: str,
    status: TaskStatus,
    pid: int | None = None,
    start_time: datetime | None = None,
    end_time: datetime | None = None,
    return_code: int | None = None,
) -> None:
    conn.execute(
        """
        UPDATE tasks
        SET status = ?,
            pid = COALESCE(?, pid),
            start_time = COALESCE(?, start_time),
            end_time = COALESCE(?, end_time),
            return_code = COALESCE(?, return_code),
            updated_at = ?
        WHERE task_id = ?
        """,
        (status, pid, _dt_to_text(start_time), _dt_to_text(end_time), return_code, _now(), task_id),
    )
    conn.commit()


def update_status(
    conn: sqlite3.Connection,
    task_id: str,
    status: TaskStatus,
    pid: int | None = None,
) -> None:
    update_task_status(conn, task_id, status, pid=pid)


def get_task(conn: sqlite3.Connection, task_id: str) -> TaskRecord | None:
    row = conn.execute("SELECT * FROM tasks WHERE task_id = ?", (task_id,)).fetchone()
    return _row_to_task(row) if row else None


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
        start_time=_text_to_dt(row["start_time"]),
        end_time=_text_to_dt(row["end_time"]),
        return_code=row["return_code"],
    )


def _now() -> str:
    return datetime.utcnow().isoformat(timespec="seconds")


def _dt_to_text(value: datetime | None) -> str | None:
    return value.isoformat(timespec="seconds") if value else None


def _text_to_dt(value: str | None) -> datetime | None:
    return datetime.fromisoformat(value) if value else None


def _migrate_tasks_table(conn: sqlite3.Connection) -> None:
    columns = {row["name"] for row in conn.execute("PRAGMA table_info(tasks)").fetchall()}
    migrations = {
        "start_time": "ALTER TABLE tasks ADD COLUMN start_time TEXT",
        "end_time": "ALTER TABLE tasks ADD COLUMN end_time TEXT",
        "return_code": "ALTER TABLE tasks ADD COLUMN return_code INTEGER",
    }
    for column, statement in migrations.items():
        if column not in columns:
            conn.execute(statement)
    conn.execute("UPDATE tasks SET status = 'committed' WHERE status = 'ready'")
