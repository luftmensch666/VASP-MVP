from __future__ import annotations

import sqlite3
from datetime import datetime
from pathlib import Path

from .models import InputSet, InputSetRole, InputSetSource, InputSetStatus, TaskInputSet


# 这些集合用于在写入数据库前做轻量校验，避免后续 UI 或运行模块绑定到未知状态。
INPUT_SET_SOURCES = {"vaspkit", "manual", "imported"}
INPUT_SET_STATUSES = {"dry_run", "generated", "edited", "validated", "committed", "invalid"}
INPUT_SET_ROLES = {"primary", "adsorbed", "clean_slab", "molecule_ref"}


def create_input_set(
    conn: sqlite3.Connection,
    *,
    input_set_id: str,
    name: str,
    source: InputSetSource,
    status: InputSetStatus,
    usable_for_vasp: bool,
    root_dir: Path,
    incar_path: Path,
    poscar_path: Path,
    kpoints_path: Path,
    potcar_path: Path,
    notes: str = "",
) -> InputSet:
    """创建或更新一组 VASP 输入文件记录。

    这里只登记文件路径和元数据，不读取 POTCAR 全文，也不触发 VASP/VASPKIT。
    """

    _validate_source(source)
    _validate_status(status)
    now = _now()
    conn.execute(
        """
        INSERT INTO input_sets (
            input_set_id, name, source, status, usable_for_vasp,
            root_dir, incar_path, poscar_path, kpoints_path, potcar_path,
            created_at, updated_at, notes
        )
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        ON CONFLICT(input_set_id) DO UPDATE SET
            name = excluded.name,
            source = excluded.source,
            status = excluded.status,
            usable_for_vasp = excluded.usable_for_vasp,
            root_dir = excluded.root_dir,
            incar_path = excluded.incar_path,
            poscar_path = excluded.poscar_path,
            kpoints_path = excluded.kpoints_path,
            potcar_path = excluded.potcar_path,
            updated_at = excluded.updated_at,
            notes = excluded.notes
        """,
        (
            input_set_id,
            name,
            source,
            status,
            int(usable_for_vasp),
            str(Path(root_dir)),
            str(Path(incar_path)),
            str(Path(poscar_path)),
            str(Path(kpoints_path)),
            str(Path(potcar_path)),
            now,
            now,
            notes,
        ),
    )
    conn.commit()
    created = get_input_set(conn, input_set_id)
    if created is None:
        raise RuntimeError(f"Failed to create input set: {input_set_id}")
    return created


def list_input_sets(conn: sqlite3.Connection) -> list[InputSet]:
    rows = conn.execute("SELECT * FROM input_sets ORDER BY updated_at DESC").fetchall()
    return [_row_to_input_set(row) for row in rows]


def get_input_set(conn: sqlite3.Connection, input_set_id: str) -> InputSet | None:
    row = conn.execute(
        "SELECT * FROM input_sets WHERE input_set_id = ?",
        (input_set_id,),
    ).fetchone()
    return _row_to_input_set(row) if row else None


def update_input_set_status(
    conn: sqlite3.Connection,
    input_set_id: str,
    status: InputSetStatus,
    usable_for_vasp: bool | None = None,
    notes: str | None = None,
) -> None:
    """更新输入文件组状态。

    usable_for_vasp 使用 None 表示保持原值；False 会被明确写入数据库。
    """

    _validate_status(status)
    fields = ["status = ?", "updated_at = ?"]
    values: list[object] = [status, _now()]
    if usable_for_vasp is not None:
        fields.append("usable_for_vasp = ?")
        values.append(int(usable_for_vasp))
    if notes is not None:
        fields.append("notes = ?")
        values.append(notes)
    values.append(input_set_id)
    conn.execute(
        f"UPDATE input_sets SET {', '.join(fields)} WHERE input_set_id = ?",
        tuple(values),
    )
    conn.commit()


def rename_input_set(conn: sqlite3.Connection, input_set_id: str, name: str) -> None:
    """重命名输入文件组。

    只更新数据库中的显示名称，不移动目录，避免影响已有文件路径和后续任务绑定。
    """

    conn.execute(
        """
        UPDATE input_sets
        SET name = ?, updated_at = ?
        WHERE input_set_id = ?
        """,
        (name.strip(), _now(), input_set_id),
    )
    conn.commit()


def bind_input_set_to_task(
    conn: sqlite3.Connection,
    task_id: str,
    role: InputSetRole,
    input_set_id: str,
) -> TaskInputSet:
    """把输入文件组绑定到任务角色。

    普通任务使用 primary；吸附能任务后续会使用 adsorbed/clean_slab/molecule_ref。
    """

    _validate_role(role)
    now = _now()
    conn.execute(
        """
        INSERT INTO task_input_sets (task_id, role, input_set_id, created_at)
        VALUES (?, ?, ?, ?)
        ON CONFLICT(task_id, role) DO UPDATE SET
            input_set_id = excluded.input_set_id,
            created_at = excluded.created_at
        """,
        (task_id, role, input_set_id, now),
    )
    conn.commit()
    return TaskInputSet(
        task_id=task_id,
        role=role,
        input_set_id=input_set_id,
        created_at=datetime.fromisoformat(now),
    )


def list_task_input_sets(conn: sqlite3.Connection, task_id: str) -> list[TaskInputSet]:
    rows = conn.execute(
        """
        SELECT * FROM task_input_sets
        WHERE task_id = ?
        ORDER BY created_at ASC
        """,
        (task_id,),
    ).fetchall()
    return [_row_to_task_input_set(row) for row in rows]


def _row_to_input_set(row: sqlite3.Row) -> InputSet:
    return InputSet(
        input_set_id=row["input_set_id"],
        name=row["name"],
        source=row["source"],
        status=row["status"],
        usable_for_vasp=bool(row["usable_for_vasp"]),
        root_dir=Path(row["root_dir"]),
        incar_path=Path(row["incar_path"]),
        poscar_path=Path(row["poscar_path"]),
        kpoints_path=Path(row["kpoints_path"]),
        potcar_path=Path(row["potcar_path"]),
        created_at=datetime.fromisoformat(row["created_at"]),
        updated_at=datetime.fromisoformat(row["updated_at"]),
        notes=row["notes"] or "",
    )


def _row_to_task_input_set(row: sqlite3.Row) -> TaskInputSet:
    return TaskInputSet(
        task_id=row["task_id"],
        role=row["role"],
        input_set_id=row["input_set_id"],
        created_at=datetime.fromisoformat(row["created_at"]),
    )


def _validate_source(source: str) -> None:
    if source not in INPUT_SET_SOURCES:
        raise ValueError(f"Unsupported input set source: {source}")


def _validate_status(status: str) -> None:
    if status not in INPUT_SET_STATUSES:
        raise ValueError(f"Unsupported input set status: {status}")


def _validate_role(role: str) -> None:
    if role not in INPUT_SET_ROLES:
        raise ValueError(f"Unsupported input set role: {role}")


def _now() -> str:
    return datetime.utcnow().isoformat(timespec="seconds")
