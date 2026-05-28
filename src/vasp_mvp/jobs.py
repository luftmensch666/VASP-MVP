from __future__ import annotations

import json
import sqlite3
from datetime import datetime
from pathlib import Path

from .models import CalculationType, JobMetricsRecord, JobRecord, JobStatus, Metrics
from .parser import parse_metrics


# 这些枚举用于在 service 层尽早拒绝未知类型，避免后续 workflow 绑定难以诊断。
CALCULATION_TYPES = {
    "relax",
    "static",
    "molecule_relax",
    "molecule_static",
    "slab_static",
    "adsorbed_static",
    "dos",
    "bader",
    "neb",
    "unknown",
}
JOB_STATUSES = {"draft", "committed", "running", "finished", "failed", "stopped"}


def create_job(
    db_path: Path,
    *,
    job_id: str,
    run_dir: Path,
    calculation_type: CalculationType = "unknown",
    status: JobStatus = "committed",
    input_set_id: str | None = None,
    mpi_ranks: int | None = None,
    vasp_bin: str | Path | None = None,
) -> JobRecord:
    """创建或更新一次独立 VASP 计算记录。

    Job 自身不保存 workflow_id 或 role；这些关系只属于 workflow_jobs。
    本函数只写数据库元数据，不启动 VASP，也不读取 POTCAR。
    """

    _validate_calculation_type(calculation_type)
    _validate_job_status(status)
    now = _now()
    with _connect(db_path) as conn:
        conn.execute(
            """
            INSERT INTO jobs (
                job_id, calculation_type, status, run_dir, input_set_id, pid,
                created_at, updated_at, start_time, end_time, return_code,
                mpi_ranks, vasp_bin
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(job_id) DO UPDATE SET
                calculation_type = excluded.calculation_type,
                status = excluded.status,
                run_dir = excluded.run_dir,
                input_set_id = excluded.input_set_id,
                updated_at = excluded.updated_at,
                mpi_ranks = excluded.mpi_ranks,
                vasp_bin = excluded.vasp_bin
            """,
            (
                job_id,
                calculation_type,
                status,
                str(Path(run_dir)),
                input_set_id,
                None,
                now,
                now,
                None,
                None,
                None,
                mpi_ranks,
                None if vasp_bin is None else str(vasp_bin),
            ),
        )
        row = conn.execute("SELECT * FROM jobs WHERE job_id = ?", (job_id,)).fetchone()
    if row is None:
        raise RuntimeError(f"Failed to create job: {job_id}")
    return _row_to_job(row)


def get_job(db_path: Path, job_id: str) -> JobRecord | None:
    with _connect(db_path) as conn:
        row = conn.execute("SELECT * FROM jobs WHERE job_id = ?", (job_id,)).fetchone()
    return _row_to_job(row) if row else None


def list_jobs(
    db_path: Path,
    *,
    calculation_type: str | None = None,
    status: str | None = None,
) -> list[JobRecord]:
    where: list[str] = []
    values: list[object] = []
    if calculation_type is not None:
        _validate_calculation_type(calculation_type)
        where.append("calculation_type = ?")
        values.append(calculation_type)
    if status is not None:
        _validate_job_status(status)
        where.append("status = ?")
        values.append(status)
    sql = "SELECT * FROM jobs"
    if where:
        sql += " WHERE " + " AND ".join(where)
    sql += " ORDER BY updated_at DESC"
    with _connect(db_path) as conn:
        rows = conn.execute(sql, tuple(values)).fetchall()
    return [_row_to_job(row) for row in rows]


def update_job_status(
    db_path: Path,
    job_id: str,
    status: JobStatus,
    *,
    pid: int | None = None,
    start_time: datetime | None = None,
    end_time: datetime | None = None,
    return_code: int | None = None,
) -> None:
    _validate_job_status(status)
    with _connect(db_path) as conn:
        conn.execute(
            """
            UPDATE jobs
            SET status = ?,
                pid = COALESCE(?, pid),
                start_time = COALESCE(?, start_time),
                end_time = COALESCE(?, end_time),
                return_code = COALESCE(?, return_code),
                updated_at = ?
            WHERE job_id = ?
            """,
            (
                status,
                pid,
                _dt_to_text(start_time),
                _dt_to_text(end_time),
                return_code,
                _now(),
                job_id,
            ),
        )


def save_job_metrics(
    db_path: Path,
    job_id: str,
    metrics: Metrics,
    *,
    energy_source: str = "OUTCAR",
    energy_label: str = "final TOTEN",
) -> None:
    """保存 Job 结果指标。

    energy_source/energy_label 明确记录吸附能后续使用 OUTCAR final TOTEN，
    避免把 OSZICAR 中间能量或手动输入能量误当作最终能量。
    """

    with _connect(db_path) as conn:
        conn.execute(
            """
            INSERT INTO job_metrics (
                job_id, toten, loop_avg, loop_count, ionic_converged,
                electronic_converged, oszicar_steps_json, errors_json,
                energy_source, energy_label, updated_at
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(job_id) DO UPDATE SET
                toten = excluded.toten,
                loop_avg = excluded.loop_avg,
                loop_count = excluded.loop_count,
                ionic_converged = excluded.ionic_converged,
                electronic_converged = excluded.electronic_converged,
                oszicar_steps_json = excluded.oszicar_steps_json,
                errors_json = excluded.errors_json,
                energy_source = excluded.energy_source,
                energy_label = excluded.energy_label,
                updated_at = excluded.updated_at
            """,
            (
                job_id,
                getattr(metrics, "toten_ev", None),
                getattr(metrics, "loop_avg_seconds", None),
                int(getattr(metrics, "loop_count", 0) or 0),
                _bool_to_db(getattr(metrics, "ionic_converged", None)),
                _bool_to_db(getattr(metrics, "electronic_converged", None)),
                json.dumps(list(getattr(metrics, "oszicar_steps", ()) or [])),
                json.dumps(list(getattr(metrics, "errors", ()) or [])),
                energy_source,
                energy_label,
                _now(),
            ),
        )


def get_job_metrics(db_path: Path, job_id: str) -> JobMetricsRecord | None:
    with _connect(db_path) as conn:
        row = conn.execute("SELECT * FROM job_metrics WHERE job_id = ?", (job_id,)).fetchone()
    return _row_to_job_metrics(row) if row else None


def parse_and_save_job_metrics(
    db_path: Path,
    job_id: str,
    run_dir: Path,
    *,
    energy_source: str = "OUTCAR",
    energy_label: str = "final TOTEN",
) -> Metrics:
    """解析 run_dir 中的结果并保存到 job_metrics。

    本阶段复用现有 parser.parse_metrics，不重写解析逻辑；缺字段时 save_job_metrics 会容错。
    """

    metrics = parse_metrics(run_dir)
    save_job_metrics(db_path, job_id, metrics, energy_source=energy_source, energy_label=energy_label)
    return metrics


def _connect(db_path: Path) -> sqlite3.Connection:
    conn = sqlite3.connect(Path(db_path))
    conn.row_factory = sqlite3.Row
    return conn


def _row_to_job(row: sqlite3.Row) -> JobRecord:
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
    )


def _row_to_job_metrics(row: sqlite3.Row) -> JobMetricsRecord:
    return JobMetricsRecord(
        job_id=row["job_id"],
        toten_ev=row["toten"],
        loop_avg_seconds=row["loop_avg"],
        loop_count=int(row["loop_count"] or 0),
        ionic_converged=_db_to_bool(row["ionic_converged"]),
        electronic_converged=_db_to_bool(row["electronic_converged"]),
        oszicar_steps=tuple(float(value) for value in _json_loads(row["oszicar_steps_json"], [])),
        errors=tuple(str(value) for value in _json_loads(row["errors_json"], [])),
        energy_source=row["energy_source"] or "OUTCAR",
        energy_label=row["energy_label"] or "final TOTEN",
        updated_at=_text_to_dt(row["updated_at"]),
    )


def _validate_calculation_type(value: str) -> None:
    if value not in CALCULATION_TYPES:
        raise ValueError(f"Unknown calculation_type: {value}")


def _validate_job_status(value: str) -> None:
    if value not in JOB_STATUSES:
        raise ValueError(f"Unknown job status: {value}")


def _now() -> str:
    return datetime.utcnow().isoformat(timespec="seconds")


def _dt_to_text(value: datetime | None) -> str | None:
    return value.isoformat(timespec="seconds") if value else None


def _text_to_dt(value: str) -> datetime:
    return datetime.fromisoformat(value)


def _text_to_dt_or_none(value: str | None) -> datetime | None:
    return datetime.fromisoformat(value) if value else None


def _bool_to_db(value: bool | None) -> int | None:
    return None if value is None else int(value)


def _db_to_bool(value: int | None) -> bool | None:
    return None if value is None else bool(value)


def _json_loads(value: str | None, default):
    if not value:
        return default
    try:
        return json.loads(value)
    except json.JSONDecodeError:
        return default
