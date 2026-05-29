from __future__ import annotations

import hashlib
import json
import shutil
import sqlite3
import uuid
from datetime import datetime
from pathlib import Path

from .models import InputSet, InputSetRole, InputSetSource, InputSetStatus, TaskInputSet


# 这些集合用于在写入数据库前做轻量校验，避免后续 UI 或运行模块绑定到未知状态。
INPUT_SET_SOURCES = {"vaspkit", "manual", "imported"}
INPUT_SET_STATUSES = {"dry_run", "generated", "edited", "validated", "committed", "invalid"}
INPUT_SET_ROLES = {"primary", "adsorbed", "clean_slab", "molecule_ref"}
CORE_INPUT_FILES = ("INCAR", "POSCAR", "KPOINTS", "POTCAR")
EDITABLE_INPUT_FILES = ("INCAR", "POSCAR", "KPOINTS")


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
    normalized_name = _normalize_required_name(name)
    _ensure_unique_name(conn, normalized_name, exclude_input_set_id=input_set_id)
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
            normalized_name,
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


def create_auto_input_set(
    conn: sqlite3.Connection,
    *,
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
    id_prefix: str = "is",
) -> InputSet:
    """创建后台自动编号的 Input Set。

    UI 不再让用户编辑 input_set_id；用户只提供 name/notes。这里生成短 UUID，
    并复用 create_input_set 的 name 必填和归一化唯一性检查。
    """

    input_set_id = f"{id_prefix}-{uuid.uuid4().hex[:12]}"
    return create_input_set(
        conn,
        input_set_id=input_set_id,
        name=name,
        source=source,
        status=status,
        usable_for_vasp=usable_for_vasp,
        root_dir=root_dir,
        incar_path=incar_path,
        poscar_path=poscar_path,
        kpoints_path=kpoints_path,
        potcar_path=potcar_path,
        notes=notes,
    )


def list_input_sets(conn: sqlite3.Connection) -> list[InputSet]:
    rows = conn.execute("SELECT * FROM input_sets ORDER BY updated_at DESC").fetchall()
    return [_row_to_input_set(row) for row in rows]


def list_usable_input_sets(db_path: Path) -> list[InputSet]:
    """用短连接读取可用于真实 VASP 计算的输入文件组。

    新 workflow UI 使用该入口，避免在 Streamlit rerun 或切换语言时复用长期 sqlite connection。
    """

    with sqlite3.connect(Path(db_path)) as conn:
        conn.row_factory = sqlite3.Row
        rows = conn.execute(
            """
            SELECT * FROM input_sets
            WHERE usable_for_vasp = 1
            ORDER BY updated_at DESC
            """
        ).fetchall()
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

    normalized_name = _normalize_required_name(name)
    _ensure_unique_name(conn, normalized_name, exclude_input_set_id=input_set_id)
    conn.execute(
        """
        UPDATE input_sets
        SET name = ?, updated_at = ?
        WHERE input_set_id = ?
        """,
        (normalized_name, _now(), input_set_id),
    )
    conn.commit()


def save_editable_input_file(
    input_set: InputSet,
    filename: str,
    content: str,
    user_action: str = "manual_edit",
) -> dict:
    """保存可编辑输入文件，并记录备份、hash 和编辑历史。

    POTCAR 不在允许列表内。这里从函数入口就拒绝 POTCAR，避免 UI 以外的调用路径绕过限制。
    """

    if filename not in EDITABLE_INPUT_FILES:
        raise ValueError(f"File is not editable in normal UI mode: {filename}")
    paths = input_set_file_paths(input_set)
    target = paths[filename]
    target.parent.mkdir(parents=True, exist_ok=True)
    backup_path = backup_input_file(input_set, filename)
    old_hash = sha256_file(target) if target.exists() else None
    target.write_text(content, encoding="utf-8")
    new_hash = sha256_file(target)
    file_hashes = build_input_file_hashes(input_set)
    write_json_file(input_set.root_dir / "file_hashes.json", file_hashes)
    history = {
        "timestamp": _now(),
        "filename": filename,
        "old_hash": old_hash,
        "new_hash": new_hash,
        "user_action": user_action,
    }
    append_edit_history(input_set, history)
    return {
        "filename": filename,
        "backup_path": backup_path,
        "old_hash": old_hash,
        "new_hash": new_hash,
        "history_path": input_set.root_dir / "edit_history.jsonl",
    }


def backup_input_file(input_set: InputSet, filename: str) -> Path | None:
    """把旧文件复制到 backups/，文件不存在时不创建空备份。"""

    source = input_set_file_paths(input_set)[filename]
    if not source.exists():
        return None
    backups_dir = input_set.root_dir / "backups"
    backups_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
    backup_path = backups_dir / f"{filename}.{timestamp}.bak"
    shutil.copy2(source, backup_path)
    return backup_path


def append_edit_history(input_set: InputSet, record: dict) -> None:
    history_path = input_set.root_dir / "edit_history.jsonl"
    history_path.parent.mkdir(parents=True, exist_ok=True)
    with history_path.open("a", encoding="utf-8") as fh:
        fh.write(json.dumps(record, ensure_ascii=False) + "\n")


def build_input_file_hashes(input_set: InputSet) -> dict:
    hashes: dict[str, dict] = {}
    for filename, path in input_set_file_paths(input_set).items():
        hashes[filename] = {
            "exists": path.exists(),
            "size_bytes": path.stat().st_size if path.exists() else 0,
            "sha256": sha256_file(path) if path.exists() else None,
        }
    return hashes


def input_set_file_paths(input_set: InputSet) -> dict[str, Path]:
    return {
        "INCAR": input_set.incar_path,
        "POSCAR": input_set.poscar_path,
        "KPOINTS": input_set.kpoints_path,
        "POTCAR": input_set.potcar_path,
    }


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_json_file(path: Path, data: dict) -> None:
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


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


def _normalize_required_name(name: str) -> str:
    normalized = (name or "").strip()
    if not normalized:
        raise ValueError("input_set.name_required")
    return normalized


def _ensure_unique_name(conn: sqlite3.Connection, name: str, *, exclude_input_set_id: str | None = None) -> None:
    """按 name.strip().lower() 做应用层唯一性检查。

    第一版不加数据库 UNIQUE index，避免历史重复数据直接迁移失败。
    """

    rows = conn.execute("SELECT input_set_id, name FROM input_sets").fetchall()
    target = name.strip().lower()
    for row in rows:
        if exclude_input_set_id is not None and row["input_set_id"] == exclude_input_set_id:
            continue
        if (row["name"] or "").strip().lower() == target:
            raise ValueError("input_set.name_duplicate")


def _now() -> str:
    return datetime.utcnow().isoformat(timespec="seconds")
