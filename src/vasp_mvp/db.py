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
    return_code INTEGER,
    name TEXT,
    notes TEXT
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

INPUT_SETS_SCHEMA = """
CREATE TABLE IF NOT EXISTS input_sets (
    input_set_id TEXT PRIMARY KEY,
    name TEXT,
    source TEXT,
    status TEXT,
    usable_for_vasp INTEGER,
    root_dir TEXT,
    incar_path TEXT,
    poscar_path TEXT,
    kpoints_path TEXT,
    potcar_path TEXT,
    created_at TEXT,
    updated_at TEXT,
    notes TEXT
);
"""

TASK_INPUT_SETS_SCHEMA = """
CREATE TABLE IF NOT EXISTS task_input_sets (
    task_id TEXT NOT NULL,
    role TEXT NOT NULL,
    input_set_id TEXT NOT NULL,
    created_at TEXT NOT NULL,
    PRIMARY KEY (task_id, role),
    FOREIGN KEY(task_id) REFERENCES tasks(task_id),
    FOREIGN KEY(input_set_id) REFERENCES input_sets(input_set_id)
);
"""

JOBS_SCHEMA = """
CREATE TABLE IF NOT EXISTS jobs (
    job_id TEXT PRIMARY KEY,
    calculation_type TEXT,
    status TEXT NOT NULL,
    run_dir TEXT NOT NULL,
    input_set_id TEXT,
    pid INTEGER,
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL,
    start_time TEXT,
    end_time TEXT,
    return_code INTEGER,
    mpi_ranks INTEGER,
    vasp_bin TEXT,
    name TEXT,
    notes TEXT
);
"""

JOB_METRICS_SCHEMA = """
CREATE TABLE IF NOT EXISTS job_metrics (
    job_id TEXT PRIMARY KEY,
    toten REAL,
    loop_avg REAL,
    loop_count INTEGER,
    ionic_converged INTEGER,
    electronic_converged INTEGER,
    oszicar_steps_json TEXT,
    errors_json TEXT,
    energy_source TEXT DEFAULT 'OUTCAR',
    energy_label TEXT DEFAULT 'final TOTEN',
    updated_at TEXT NOT NULL,
    FOREIGN KEY(job_id) REFERENCES jobs(job_id)
);
"""

WORKFLOWS_SCHEMA = """
CREATE TABLE IF NOT EXISTS workflows (
    workflow_id TEXT PRIMARY KEY,
    workflow_type TEXT NOT NULL,
    name TEXT NOT NULL,
    status TEXT NOT NULL,
    root_dir TEXT NOT NULL,
    method_family TEXT,
    functional TEXT,
    method_notes TEXT,
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL,
    notes TEXT
);
"""

WORKFLOW_JOBS_SCHEMA = """
CREATE TABLE IF NOT EXISTS workflow_jobs (
    workflow_job_id TEXT PRIMARY KEY,
    workflow_id TEXT NOT NULL,
    job_id TEXT NOT NULL,
    role TEXT NOT NULL,
    step_order INTEGER,
    required INTEGER DEFAULT 1,
    created_at TEXT NOT NULL,
    notes TEXT,
    FOREIGN KEY(workflow_id) REFERENCES workflows(workflow_id),
    FOREIGN KEY(job_id) REFERENCES jobs(job_id)
);
"""

WORKFLOW_JOBS_UNIQUE_INDEX = """
CREATE UNIQUE INDEX IF NOT EXISTS idx_workflow_jobs_unique_binding
ON workflow_jobs (workflow_id, job_id, role);
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
    conn.execute(INPUT_SETS_SCHEMA)
    conn.execute(TASK_INPUT_SETS_SCHEMA)
    conn.execute(JOBS_SCHEMA)
    conn.execute(JOB_METRICS_SCHEMA)
    conn.execute(WORKFLOWS_SCHEMA)
    conn.execute(WORKFLOW_JOBS_SCHEMA)
    conn.execute(WORKFLOW_JOBS_UNIQUE_INDEX)
    _migrate_tasks_table(conn)
    _migrate_jobs_table(conn)
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
    name: str | None = None,
    notes: str | None = None,
) -> None:
    now = _now()
    conn.execute(
        """
        INSERT INTO tasks (
            task_id, project, task_type, status, task_root, pid,
            created_at, updated_at, start_time, end_time, return_code,
            name, notes
        )
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        ON CONFLICT(task_id) DO UPDATE SET
            project = excluded.project,
            task_type = excluded.task_type,
            status = excluded.status,
            task_root = excluded.task_root,
            pid = excluded.pid,
            start_time = excluded.start_time,
            end_time = excluded.end_time,
            return_code = excluded.return_code,
            name = COALESCE(excluded.name, name),
            notes = COALESCE(excluded.notes, notes),
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
            name,
            notes,
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
        name=row["name"] if "name" in row.keys() else None,
        notes=row["notes"] if "notes" in row.keys() else None,
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
        "name": "ALTER TABLE tasks ADD COLUMN name TEXT",
        "notes": "ALTER TABLE tasks ADD COLUMN notes TEXT",
    }
    for column, statement in migrations.items():
        if column not in columns:
            conn.execute(statement)
    conn.execute("UPDATE tasks SET status = 'committed' WHERE status = 'ready'")


def _migrate_jobs_table(conn: sqlite3.Connection) -> None:
    columns = {row["name"] for row in conn.execute("PRAGMA table_info(jobs)").fetchall()}
    migrations = {
        "name": "ALTER TABLE jobs ADD COLUMN name TEXT",
        "notes": "ALTER TABLE jobs ADD COLUMN notes TEXT",
    }
    for column, statement in migrations.items():
        if column not in columns:
            conn.execute(statement)
