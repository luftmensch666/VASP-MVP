from __future__ import annotations

import shutil
import sqlite3
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path

from .workflow_runner import find_processes_by_run_dir, is_process_alive


BUSINESS_TABLES = (
    "workflow_jobs",
    "job_metrics",
    "jobs",
    "workflows",
    "task_input_sets",
    "input_sets",
    "metrics",
    "tasks",
)
RESET_DIRECTORIES = (
    "workflows",
    "input_sets",
    "jobs",
    ".trash",
    "failed_input_sets",
    "drafts",
    "tmp",
    "temp",
    "old_tasks",
)
REQUIRED_WORKSPACE_DIRECTORIES = ("workflows", "input_sets", "jobs", "backups")


@dataclass
class DeleteResult:
    ok: bool
    entity_type: str
    entity_id: str
    deleted_db_records: list[str] = field(default_factory=list)
    deleted_paths: list[str] = field(default_factory=list)
    skipped_paths: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    errors: list[str] = field(default_factory=list)
    backup_path: Path | None = None
    dry_run: bool = False


@dataclass
class FactoryResetResult(DeleteResult):
    pass


def ensure_path_inside_workspace(path: Path, workspace: Path) -> Path:
    """确认待操作路径位于 workspace 内。

    删除本地文件前统一调用该函数，避免误删 VASP/VASPKIT/POTCAR 库、
    项目源码或用户主目录中的其他文件。
    """

    root = Path(workspace).resolve()
    candidate = Path(path).resolve(strict=False)
    if candidate != root and root not in candidate.parents:
        raise ValueError(f"Path is outside workspace: {candidate}")
    return candidate


def preview_delete_workflow(db_path: Path, workflow_id: str, *, workspace: Path) -> DeleteResult:
    return delete_workflow(db_path, workflow_id, workspace=workspace, dry_run=True)


def delete_workflow(
    db_path: Path,
    workflow_id: str,
    *,
    workspace: Path,
    delete_files: bool = True,
    require_not_running: bool = True,
    dry_run: bool = False,
) -> DeleteResult:
    """删除一个 workflow 及只属于它的 job。

    删除前会检查数据库状态、pid 是否存活以及 run_dir 下是否存在 VASP/MPI
    相关进程。发现仍在运行时只返回错误，不自动杀进程。
    """

    result = _empty_result("workflow", workflow_id, dry_run=dry_run)
    root = Path(workspace)
    with _connect(db_path) as conn:
        workflow = conn.execute("SELECT * FROM workflows WHERE workflow_id = ?", (workflow_id,)).fetchone()
        if workflow is None:
            result.errors.append(f"Workflow not found: {workflow_id}")
            return _finalize(result)
        bindings = conn.execute(
            """
            SELECT wj.*, j.*
            FROM workflow_jobs wj
            JOIN jobs j ON j.job_id = wj.job_id
            WHERE wj.workflow_id = ?
            """,
            (workflow_id,),
        ).fetchall()
        if require_not_running:
            running = [_job_running_reason(row) for row in bindings]
            running = [item for item in running if item]
            if running:
                result.errors.extend(running)
                return _finalize(result)

        orphan_job_ids: list[str] = []
        reused_job_run_dirs: list[Path] = []
        for row in bindings:
            refs = conn.execute("SELECT COUNT(*) FROM workflow_jobs WHERE job_id = ?", (row["job_id"],)).fetchone()[0]
            if int(refs) <= 1:
                orphan_job_ids.append(row["job_id"])
                _try_append_path(result, row["run_dir"], root)
            else:
                reused_job_run_dirs.append(Path(row["run_dir"]))
                result.warnings.append(f"Job is used by another workflow and will only be unbound: {row['job_id']}")

        workflow_root = Path(workflow["root_dir"])
        if delete_files:
            if any(_path_is_inside(child, workflow_root) for child in reused_job_run_dirs):
                result.skipped_paths.append(str(workflow_root))
                result.warnings.append("Workflow root directory contains a reused job run_dir; root directory deletion skipped.")
            else:
                _try_append_path(result, workflow_root, root)

        result.deleted_db_records.append(f"workflow_jobs:{workflow_id}")
        result.deleted_db_records.append(f"workflows:{workflow_id}")
        for job_id in orphan_job_ids:
            result.deleted_db_records.append(f"job_metrics:{job_id}")
            result.deleted_db_records.append(f"jobs:{job_id}")

        if dry_run or result.errors:
            return _finalize(result)

        with conn:
            for job_id in orphan_job_ids:
                conn.execute("DELETE FROM job_metrics WHERE job_id = ?", (job_id,))
                conn.execute("DELETE FROM jobs WHERE job_id = ?", (job_id,))
            conn.execute("DELETE FROM workflow_jobs WHERE workflow_id = ?", (workflow_id,))
            conn.execute("DELETE FROM workflows WHERE workflow_id = ?", (workflow_id,))

    if delete_files:
        _delete_paths(result, root)
    return _finalize(result)


def preview_delete_input_set(db_path: Path, input_set_id: str, *, workspace: Path) -> DeleteResult:
    return delete_input_set(db_path, input_set_id, workspace=workspace, dry_run=True)


def delete_input_set(
    db_path: Path,
    input_set_id: str,
    *,
    workspace: Path,
    delete_files: bool = True,
    force: bool = False,
    dry_run: bool = False,
) -> DeleteResult:
    """删除未被任何 job 或旧 task 绑定的 Input Set。

    第一版不开放 force delete；force 参数仅保留给后续危险模式扩展。
    """

    result = _empty_result("input_set", input_set_id, dry_run=dry_run)
    root = Path(workspace)
    if force:
        result.errors.append("Force delete is not enabled in this version.")
        return _finalize(result)
    with _connect(db_path) as conn:
        record = conn.execute("SELECT * FROM input_sets WHERE input_set_id = ?", (input_set_id,)).fetchone()
        if record is None:
            result.errors.append(f"Input Set not found: {input_set_id}")
            return _finalize(result)
        job_refs = conn.execute("SELECT job_id FROM jobs WHERE input_set_id = ?", (input_set_id,)).fetchall()
        task_refs = conn.execute("SELECT task_id, role FROM task_input_sets WHERE input_set_id = ?", (input_set_id,)).fetchall()
        if job_refs or task_refs:
            for row in job_refs:
                result.errors.append(f"Input Set is used by job: {row['job_id']}")
            for row in task_refs:
                result.errors.append(f"Input Set is used by legacy task: {row['task_id']} ({row['role']})")
            return _finalize(result)

        result.deleted_db_records.append(f"task_input_sets:{input_set_id}")
        result.deleted_db_records.append(f"input_sets:{input_set_id}")
        if delete_files:
            _try_append_path(result, record["root_dir"], root)

        if dry_run or result.errors:
            return _finalize(result)

        with conn:
            conn.execute("DELETE FROM task_input_sets WHERE input_set_id = ?", (input_set_id,))
            conn.execute("DELETE FROM input_sets WHERE input_set_id = ?", (input_set_id,))

    if delete_files:
        _delete_paths(result, root)
    return _finalize(result)


def preview_delete_legacy_task(db_path: Path, task_id: str, *, workspace: Path) -> DeleteResult:
    return delete_legacy_task(db_path, task_id, workspace=workspace, dry_run=True)


def delete_legacy_task(
    db_path: Path,
    task_id: str,
    *,
    workspace: Path,
    delete_files: bool = True,
    dry_run: bool = False,
) -> DeleteResult:
    """删除旧 tasks 表中的单个任务，不影响新的 jobs/workflows。"""

    result = _empty_result("legacy_task", task_id, dry_run=dry_run)
    root = Path(workspace)
    with _connect(db_path) as conn:
        task = conn.execute("SELECT * FROM tasks WHERE task_id = ?", (task_id,)).fetchone()
        if task is None:
            result.errors.append(f"Legacy task not found: {task_id}")
            return _finalize(result)
        running_reason = _legacy_task_running_reason(task)
        if running_reason:
            result.errors.append(running_reason)
            return _finalize(result)
        result.deleted_db_records.extend([f"metrics:{task_id}", f"task_input_sets:{task_id}", f"tasks:{task_id}"])
        if delete_files:
            _try_append_path(result, task["task_root"], root)
        if dry_run or result.errors:
            return _finalize(result)
        with conn:
            conn.execute("DELETE FROM metrics WHERE task_id = ?", (task_id,))
            conn.execute("DELETE FROM task_input_sets WHERE task_id = ?", (task_id,))
            conn.execute("DELETE FROM tasks WHERE task_id = ?", (task_id,))

    if delete_files:
        _delete_paths(result, root)
    return _finalize(result)


def update_workflow_metadata(db_path: Path, workflow_id: str, *, name: str | None = None, notes: str | None = None) -> None:
    _update_metadata(db_path, "workflows", "workflow_id", workflow_id, name=name, notes=notes)


def update_input_set_metadata(db_path: Path, input_set_id: str, *, name: str | None = None, notes: str | None = None) -> None:
    _update_metadata(db_path, "input_sets", "input_set_id", input_set_id, name=name, notes=notes)


def update_job_metadata(db_path: Path, job_id: str, *, name: str | None = None, notes: str | None = None) -> None:
    _update_metadata(db_path, "jobs", "job_id", job_id, name=name, notes=notes)


def update_legacy_task_metadata(db_path: Path, task_id: str, *, name: str | None = None, notes: str | None = None) -> None:
    _update_metadata(db_path, "tasks", "task_id", task_id, name=name, notes=notes)


def preview_factory_reset(db_path: Path, *, workspace: Path) -> FactoryResetResult:
    return factory_reset(db_path, workspace=workspace, dry_run=True)


def factory_reset(db_path: Path, *, workspace: Path, dry_run: bool = False) -> FactoryResetResult:
    """清空业务数据并删除 workspace 内项目生成目录。

    执行前会备份数据库，且 backups 目录不会在本次 reset 中清空。该函数不删除
    项目源码、.venv、config、tests，也不会触碰 workspace 之外的任何路径。
    """

    result = FactoryResetResult(ok=True, entity_type="factory_reset", entity_id="workspace", dry_run=dry_run)
    root = Path(workspace)
    root.mkdir(parents=True, exist_ok=True)
    backup_path = _factory_reset_backup_path(root)
    result = _replace_result_backup(result, backup_path)

    with _connect(db_path) as conn:
        running_errors = _running_business_processes(conn)
        if running_errors:
            result.errors.extend(running_errors)
            return _finalize(result)
        table_names = _table_names(conn)
        for table in BUSINESS_TABLES:
            if table in table_names:
                count = conn.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0]
                result.deleted_db_records.append(f"{table}:{count}")
        for table in sorted(name for name in table_names if name.startswith("adsorption")):
            count = conn.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0]
            result.deleted_db_records.append(f"{table}:{count}")

        for dirname in RESET_DIRECTORIES:
            path = root / dirname
            if path.exists():
                _try_append_path(result, path, root)
            else:
                result.skipped_paths.append(str(path))

        for dirname in REQUIRED_WORKSPACE_DIRECTORIES:
            result.warnings.append(f"Directory will be recreated: {root / dirname}")

        if dry_run or result.errors:
            return _finalize(result)

        backup_path.parent.mkdir(parents=True, exist_ok=True)
        if Path(db_path).exists():
            shutil.copy2(db_path, backup_path)
        else:
            backup_path.write_bytes(b"")

        with conn:
            for table in BUSINESS_TABLES:
                if table in table_names:
                    conn.execute(f"DELETE FROM {table}")
            for table in sorted(name for name in table_names if name.startswith("adsorption")):
                conn.execute(f"DELETE FROM {table}")

    _delete_paths(result, root)
    for dirname in REQUIRED_WORKSPACE_DIRECTORIES:
        (root / dirname).mkdir(parents=True, exist_ok=True)
    return _finalize(result)


def _connect(db_path: Path) -> sqlite3.Connection:
    conn = sqlite3.connect(Path(db_path))
    conn.row_factory = sqlite3.Row
    return conn


def _empty_result(entity_type: str, entity_id: str, *, dry_run: bool = False) -> DeleteResult:
    return DeleteResult(ok=True, entity_type=entity_type, entity_id=entity_id, dry_run=dry_run)


def _finalize(result):
    result.ok = not result.errors
    return result


def _append_path(paths: list[str], path: str | Path | None, workspace: Path) -> None:
    if path is None:
        return
    try:
        safe = ensure_path_inside_workspace(Path(path), workspace)
    except ValueError as exc:
        raise ValueError(str(exc)) from exc
    if safe == Path(workspace).resolve():
        raise ValueError(f"Refuse to delete workspace root directly: {safe}")
    text = str(safe)
    if text not in paths:
        paths.append(text)


def _try_append_path(result: DeleteResult, path: str | Path | None, workspace: Path) -> None:
    try:
        _append_path(result.deleted_paths, path, workspace)
    except Exception as exc:
        result.errors.append(str(exc))


def _delete_paths(result: DeleteResult, workspace: Path) -> None:
    for raw in list(result.deleted_paths):
        try:
            path = ensure_path_inside_workspace(Path(raw), workspace)
            if path == Path(workspace).resolve():
                result.errors.append(f"Refuse to delete workspace root directly: {path}")
                continue
            if not path.exists():
                result.skipped_paths.append(str(path))
                continue
            if path.is_file() or path.is_symlink():
                path.unlink()
            elif path.is_dir():
                shutil.rmtree(path)
        except Exception as exc:
            result.errors.append(str(exc))


def _path_is_inside(path: Path, parent: Path) -> bool:
    candidate = Path(path).resolve(strict=False)
    root = Path(parent).resolve(strict=False)
    return candidate == root or root in candidate.parents


def _job_running_reason(row: sqlite3.Row) -> str | None:
    job_id = row["job_id"]
    status = row["status"]
    pid = row["pid"]
    run_dir = Path(row["run_dir"])
    if status == "running":
        return f"Job is marked running: {job_id}"
    if pid is not None and is_process_alive(int(pid)):
        return f"Job pid is still alive: {job_id} pid={pid}"
    if find_processes_by_run_dir(run_dir):
        return f"VASP/MPI process found under job run_dir: {job_id}"
    return None


def _legacy_task_running_reason(row: sqlite3.Row) -> str | None:
    task_id = row["task_id"]
    status = row["status"]
    pid = row["pid"]
    task_root = Path(row["task_root"])
    if status == "running":
        return f"Legacy task is marked running: {task_id}"
    if pid is not None and is_process_alive(int(pid)):
        return f"Legacy task pid is still alive: {task_id} pid={pid}"
    for path in (task_root, task_root / "run"):
        if path.exists() and find_processes_by_run_dir(path):
            return f"VASP/MPI process found under legacy task path: {task_id}"
    return None


def _running_business_processes(conn: sqlite3.Connection) -> list[str]:
    errors: list[str] = []
    for row in conn.execute("SELECT * FROM jobs").fetchall():
        reason = _job_running_reason(row)
        if reason:
            errors.append(reason)
    for row in conn.execute("SELECT * FROM tasks").fetchall():
        reason = _legacy_task_running_reason(row)
        if reason:
            errors.append(reason)
    return errors


def _update_metadata(
    db_path: Path,
    table: str,
    id_column: str,
    entity_id: str,
    *,
    name: str | None,
    notes: str | None,
) -> None:
    if table not in {"workflows", "input_sets", "jobs", "tasks"}:
        raise ValueError(f"Unsupported metadata table: {table}")
    fields = ["updated_at = ?"]
    values: list[object] = [_now()]
    if name is not None:
        normalized_name = name.strip()
        if not normalized_name:
            if table == "input_sets":
                raise ValueError("input_set.name_required")
            if table == "workflows":
                raise ValueError("workflow.name_required")
        if table in {"input_sets", "workflows"}:
            _ensure_metadata_name_unique(db_path, table, id_column, entity_id, normalized_name)
        fields.append("name = ?")
        values.append(normalized_name)
    if notes is not None:
        fields.append("notes = ?")
        values.append(notes)
    values.append(entity_id)
    with _connect(db_path) as conn:
        with conn:
            conn.execute(
                f"UPDATE {table} SET {', '.join(fields)} WHERE {id_column} = ?",
                tuple(values),
            )


def _ensure_metadata_name_unique(db_path: Path, table: str, id_column: str, entity_id: str, name: str) -> None:
    duplicate_error = "input_set.name_duplicate" if table == "input_sets" else "workflow.name_duplicate"
    with _connect(db_path) as conn:
        rows = conn.execute(f"SELECT {id_column}, name FROM {table}").fetchall()
    target = name.strip().lower()
    for row in rows:
        if row[id_column] == entity_id:
            continue
        if (row["name"] or "").strip().lower() == target:
            raise ValueError(duplicate_error)


def _table_names(conn: sqlite3.Connection) -> set[str]:
    rows = conn.execute("SELECT name FROM sqlite_master WHERE type = 'table'").fetchall()
    return {row["name"] for row in rows}


def _factory_reset_backup_path(workspace: Path) -> Path:
    timestamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S_%f")
    return Path(workspace) / "backups" / f"vasp_mvp_before_factory_reset_{timestamp}.db"


def _replace_result_backup(result: FactoryResetResult, backup_path: Path) -> FactoryResetResult:
    return FactoryResetResult(
        ok=result.ok,
        entity_type=result.entity_type,
        entity_id=result.entity_id,
        deleted_db_records=result.deleted_db_records,
        deleted_paths=result.deleted_paths,
        skipped_paths=result.skipped_paths,
        warnings=result.warnings,
        errors=result.errors,
        backup_path=backup_path,
        dry_run=result.dry_run,
    )


def _now() -> str:
    return datetime.utcnow().isoformat(timespec="seconds")
