from __future__ import annotations

import sqlite3
from datetime import datetime
from pathlib import Path

from .jobs import get_job
from .models import JobRecord, WorkflowJobRecord, WorkflowRecord, WorkflowRole, WorkflowStatus, WorkflowType


WORKFLOW_TYPES = {"adsorption", "neb", "dos", "bader", "custom"}
WORKFLOW_STATUSES = {"draft", "committed", "running", "finished", "failed", "stopped"}
WORKFLOW_ROLES = {
    "clean_slab",
    "molecule_ref",
    "adsorbed_system",
    "initial_state",
    "final_state",
    "neb_image",
    "dos",
    "bader",
    "primary",
}


def create_workflow(
    db_path: Path,
    *,
    workflow_id: str,
    workflow_type: WorkflowType,
    name: str,
    root_dir: Path,
    status: WorkflowStatus = "draft",
    method_family: str | None = None,
    functional: str | None = None,
    method_notes: str | None = None,
    notes: str = "",
) -> WorkflowRecord:
    """创建或更新一个 workflow 元数据记录。

    这里只描述多步计算的组织关系和方法摘要，不启动任何 VASP 计算。
    """

    _validate_workflow_type(workflow_type)
    _validate_workflow_status(status)
    normalized_name = _normalize_required_name(name)
    now = _now()
    with _connect(db_path) as conn:
        _ensure_unique_name(conn, normalized_name, exclude_workflow_id=workflow_id)
        conn.execute(
            """
            INSERT INTO workflows (
                workflow_id, workflow_type, name, status, root_dir,
                method_family, functional, method_notes,
                created_at, updated_at, notes
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(workflow_id) DO UPDATE SET
                workflow_type = excluded.workflow_type,
                name = excluded.name,
                status = excluded.status,
                root_dir = excluded.root_dir,
                method_family = excluded.method_family,
                functional = excluded.functional,
                method_notes = excluded.method_notes,
                updated_at = excluded.updated_at,
                notes = excluded.notes
            """,
            (
                workflow_id,
                workflow_type,
                normalized_name,
                status,
                str(Path(root_dir)),
                method_family,
                functional,
                method_notes,
                now,
                now,
                notes,
            ),
        )
        row = conn.execute("SELECT * FROM workflows WHERE workflow_id = ?", (workflow_id,)).fetchone()
    if row is None:
        raise RuntimeError(f"Failed to create workflow: {workflow_id}")
    return _row_to_workflow(row)


def get_workflow(db_path: Path, workflow_id: str) -> WorkflowRecord | None:
    with _connect(db_path) as conn:
        row = conn.execute("SELECT * FROM workflows WHERE workflow_id = ?", (workflow_id,)).fetchone()
    return _row_to_workflow(row) if row else None


def list_workflows(
    db_path: Path,
    *,
    workflow_type: str | None = None,
    status: str | None = None,
) -> list[WorkflowRecord]:
    where: list[str] = []
    values: list[object] = []
    if workflow_type is not None:
        _validate_workflow_type(workflow_type)
        where.append("workflow_type = ?")
        values.append(workflow_type)
    if status is not None:
        _validate_workflow_status(status)
        where.append("status = ?")
        values.append(status)
    sql = "SELECT * FROM workflows"
    if where:
        sql += " WHERE " + " AND ".join(where)
    sql += " ORDER BY updated_at DESC"
    with _connect(db_path) as conn:
        rows = conn.execute(sql, tuple(values)).fetchall()
    return [_row_to_workflow(row) for row in rows]


def update_workflow_status(
    db_path: Path,
    workflow_id: str,
    status: WorkflowStatus,
    *,
    notes: str | None = None,
) -> None:
    _validate_workflow_status(status)
    fields = ["status = ?", "updated_at = ?"]
    values: list[object] = [status, _now()]
    if notes is not None:
        fields.append("notes = ?")
        values.append(notes)
    values.append(workflow_id)
    with _connect(db_path) as conn:
        conn.execute(
            f"UPDATE workflows SET {', '.join(fields)} WHERE workflow_id = ?",
            tuple(values),
        )


def bind_job_to_workflow(
    db_path: Path,
    *,
    workflow_id: str,
    job_id: str,
    role: WorkflowRole,
    workflow_job_id: str | None = None,
    step_order: int | None = None,
    required: bool = True,
    notes: str = "",
) -> WorkflowJobRecord:
    """把 job 绑定到 workflow 中的某个角色。

    同一个 workflow_id + job_id + role 只允许出现一次；重复调用返回已有记录。
    """

    _validate_workflow_role(role)
    now = _now()
    with _connect(db_path) as conn:
        existing = conn.execute(
            """
            SELECT * FROM workflow_jobs
            WHERE workflow_id = ? AND job_id = ? AND role = ?
            """,
            (workflow_id, job_id, role),
        ).fetchone()
        if existing is not None:
            return _row_to_workflow_job(existing)

        record_id = workflow_job_id or _new_workflow_job_id(workflow_id, job_id, role)
        conn.execute(
            """
            INSERT INTO workflow_jobs (
                workflow_job_id, workflow_id, job_id, role,
                step_order, required, created_at, notes
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                record_id,
                workflow_id,
                job_id,
                role,
                step_order,
                int(required),
                now,
                notes,
            ),
        )
        row = conn.execute(
            "SELECT * FROM workflow_jobs WHERE workflow_job_id = ?",
            (record_id,),
        ).fetchone()
    if row is None:
        raise RuntimeError(f"Failed to bind job to workflow: {workflow_id}, {job_id}, {role}")
    return _row_to_workflow_job(row)


def list_workflow_jobs(db_path: Path, workflow_id: str) -> list[WorkflowJobRecord]:
    with _connect(db_path) as conn:
        rows = conn.execute(
            """
            SELECT * FROM workflow_jobs
            WHERE workflow_id = ?
            ORDER BY step_order IS NULL, step_order, created_at
            """,
            (workflow_id,),
        ).fetchall()
    return [_row_to_workflow_job(row) for row in rows]


def list_jobs_for_workflow(db_path: Path, workflow_id: str) -> list[tuple[WorkflowJobRecord, JobRecord]]:
    with _connect(db_path) as conn:
        rows = conn.execute(
            """
            SELECT
                wj.workflow_job_id AS wj_workflow_job_id,
                wj.workflow_id AS wj_workflow_id,
                wj.job_id AS wj_job_id,
                wj.role AS wj_role,
                wj.step_order AS wj_step_order,
                wj.required AS wj_required,
                wj.created_at AS wj_created_at,
                wj.notes AS wj_notes,
                j.*
            FROM workflow_jobs wj
            JOIN jobs j ON j.job_id = wj.job_id
            WHERE wj.workflow_id = ?
            ORDER BY wj.step_order IS NULL, wj.step_order, wj.created_at
            """,
            (workflow_id,),
        ).fetchall()
    return [(_row_to_workflow_job_prefixed(row), _row_to_joined_job(row)) for row in rows]


def get_workflow_with_jobs(
    db_path: Path,
    workflow_id: str,
) -> tuple[WorkflowRecord, list[tuple[WorkflowJobRecord, JobRecord]]] | None:
    workflow = get_workflow(db_path, workflow_id)
    if workflow is None:
        return None
    return workflow, list_jobs_for_workflow(db_path, workflow_id)


def _connect(db_path: Path) -> sqlite3.Connection:
    conn = sqlite3.connect(Path(db_path))
    conn.row_factory = sqlite3.Row
    return conn


def _row_to_workflow(row: sqlite3.Row) -> WorkflowRecord:
    return WorkflowRecord(
        workflow_id=row["workflow_id"],
        workflow_type=row["workflow_type"],
        name=row["name"],
        status=row["status"],
        root_dir=Path(row["root_dir"]),
        method_family=row["method_family"],
        functional=row["functional"],
        method_notes=row["method_notes"],
        created_at=_text_to_dt(row["created_at"]),
        updated_at=_text_to_dt(row["updated_at"]),
        notes=row["notes"] or "",
    )


def _row_to_workflow_job(row: sqlite3.Row) -> WorkflowJobRecord:
    return WorkflowJobRecord(
        workflow_job_id=row["workflow_job_id"],
        workflow_id=row["workflow_id"],
        job_id=row["job_id"],
        role=row["role"],
        step_order=row["step_order"],
        required=bool(row["required"]),
        created_at=_text_to_dt(row["created_at"]),
        notes=row["notes"] or "",
    )


def _row_to_workflow_job_prefixed(row: sqlite3.Row) -> WorkflowJobRecord:
    return WorkflowJobRecord(
        workflow_job_id=row["wj_workflow_job_id"],
        workflow_id=row["wj_workflow_id"],
        job_id=row["wj_job_id"],
        role=row["wj_role"],
        step_order=row["wj_step_order"],
        required=bool(row["wj_required"]),
        created_at=_text_to_dt(row["wj_created_at"]),
        notes=row["wj_notes"] or "",
    )


def _row_to_joined_job(row: sqlite3.Row) -> JobRecord:
    return JobRecord(
        job_id=row["job_id"],
        calculation_type=row["calculation_type"] or "unknown",
        status=row["status"],
        run_dir=Path(row["run_dir"]),
        input_set_id=row["input_set_id"],
        pid=row["pid"],
        created_at=_text_to_dt(row["created_at"]),
        updated_at=_text_to_dt(row["updated_at"]),
        start_time=_text_to_dt_or_none(row["start_time"]),
        end_time=_text_to_dt_or_none(row["end_time"]),
        return_code=row["return_code"],
        mpi_ranks=row["mpi_ranks"],
        vasp_bin=row["vasp_bin"],
        name=row["name"] if "name" in row.keys() else None,
        notes=row["notes"] if "notes" in row.keys() else None,
    )


def _new_workflow_job_id(workflow_id: str, job_id: str, role: str) -> str:
    return f"{workflow_id}__{job_id}__{role}"


def _validate_workflow_type(value: str) -> None:
    if value not in WORKFLOW_TYPES:
        raise ValueError(f"Unknown workflow_type: {value}")


def _validate_workflow_status(value: str) -> None:
    if value not in WORKFLOW_STATUSES:
        raise ValueError(f"Unknown workflow status: {value}")


def _validate_workflow_role(value: str) -> None:
    if value not in WORKFLOW_ROLES:
        raise ValueError(f"Unknown workflow role: {value}")


def _normalize_required_name(name: str) -> str:
    normalized = (name or "").strip()
    if not normalized:
        raise ValueError("workflow.name_required")
    return normalized


def _ensure_unique_name(conn: sqlite3.Connection, name: str, *, exclude_workflow_id: str | None = None) -> None:
    """按 workflow.name.strip().lower() 做应用层唯一性检查。"""

    rows = conn.execute("SELECT workflow_id, name FROM workflows").fetchall()
    target = name.strip().lower()
    for row in rows:
        if exclude_workflow_id is not None and row["workflow_id"] == exclude_workflow_id:
            continue
        if (row["name"] or "").strip().lower() == target:
            raise ValueError("workflow.name_duplicate")


def _now() -> str:
    return datetime.utcnow().isoformat(timespec="seconds")


def _text_to_dt(value: str) -> datetime:
    return datetime.fromisoformat(value)


def _text_to_dt_or_none(value: str | None) -> datetime | None:
    return datetime.fromisoformat(value) if value else None
