from __future__ import annotations

import shutil
import sqlite3
from dataclasses import dataclass
from pathlib import Path

from .input_sets import get_input_set, input_set_file_paths
from .jobs import create_job
from .models import InputSet, WorkflowRecord
from .workflows import bind_job_to_workflow, create_workflow


CORE_INPUT_FILES = ("INCAR", "POSCAR", "KPOINTS", "POTCAR")


@dataclass(frozen=True)
class AdsorptionRoleSpec:
    role: str
    input_set_id: str
    calculation_type: str
    step_order: int


def create_adsorption_workflow(
    db_path: Path,
    *,
    workflow_id: str,
    name: str,
    root_dir: Path,
    clean_slab_input_set_id: str,
    molecule_ref_input_set_id: str,
    adsorbed_input_set_id: str,
    method_family: str = "DFT",
    functional: str | None = None,
    method_notes: str | None = None,
    mpi_ranks: int | None = None,
    vasp_bin: str | Path | None = None,
    notes: str = "",
) -> WorkflowRecord:
    """创建吸附能 workflow 的三步 VASP job 后端骨架。

    本函数只做输入文件组校验、文件复制和数据库记录创建；不会启动 VASP，
    不解析 OUTCAR，也不会读取或展示 POTCAR 全文。
    """

    specs = (
        AdsorptionRoleSpec("clean_slab", clean_slab_input_set_id, "slab_static", 1),
        AdsorptionRoleSpec("molecule_ref", molecule_ref_input_set_id, "molecule_static", 2),
        AdsorptionRoleSpec("adsorbed_system", adsorbed_input_set_id, "adsorbed_static", 3),
    )
    input_sets = _load_and_validate_input_sets(db_path, specs)
    root_dir = Path(root_dir)

    # 先完成全部文件复制，再写数据库，避免复制失败后留下 workflow/job 记录。
    for spec in specs:
        _copy_input_set_to_role_run(input_sets[spec.role], _role_run_dir(root_dir, spec.role))

    workflow = create_workflow(
        db_path,
        workflow_id=workflow_id,
        workflow_type="adsorption",
        name=name,
        root_dir=root_dir,
        status="committed",
        method_family=method_family,
        functional=functional,
        method_notes=method_notes,
        notes=notes,
    )
    for spec in specs:
        job_id = _job_id(workflow_id, spec.role)
        create_job(
            db_path,
            job_id=job_id,
            calculation_type=spec.calculation_type,
            status="committed",
            run_dir=_role_run_dir(root_dir, spec.role),
            input_set_id=spec.input_set_id,
            mpi_ranks=mpi_ranks,
            vasp_bin=vasp_bin,
        )
        bind_job_to_workflow(
            db_path,
            workflow_id=workflow_id,
            job_id=job_id,
            role=spec.role,
            step_order=spec.step_order,
            required=True,
            notes="adsorption workflow required static calculation",
        )
    return workflow


def _load_and_validate_input_sets(
    db_path: Path,
    specs: tuple[AdsorptionRoleSpec, ...],
) -> dict[str, InputSet]:
    with _connect(db_path) as conn:
        loaded = {spec.role: get_input_set(conn, spec.input_set_id) for spec in specs}
    input_sets: dict[str, InputSet] = {}
    for spec in specs:
        input_set = loaded[spec.role]
        if input_set is None:
            raise ValueError(f"Input Set for role {spec.role} does not exist: {spec.input_set_id}")
        _validate_usable_input_set(spec.role, input_set)
        input_sets[spec.role] = input_set
    return input_sets


def _validate_usable_input_set(role: str, input_set: InputSet) -> None:
    if not input_set.usable_for_vasp:
        raise ValueError(f"Input Set for role {role} is not usable for VASP: {input_set.input_set_id}")
    paths = input_set_file_paths(input_set)
    missing_paths = [filename for filename in CORE_INPUT_FILES if filename not in paths]
    missing_files = [
        filename
        for filename in CORE_INPUT_FILES
        if filename in paths and (not paths[filename].exists() or not paths[filename].is_file())
    ]
    if missing_paths or missing_files:
        missing = ", ".join(missing_paths + missing_files)
        raise FileNotFoundError(f"Input Set for role {role} is missing required files: {missing}")


def _copy_input_set_to_role_run(input_set: InputSet, run_dir: Path) -> None:
    run_dir.mkdir(parents=True, exist_ok=True)
    paths = input_set_file_paths(input_set)
    for filename in CORE_INPUT_FILES:
        shutil.copy2(paths[filename], run_dir / filename)


def _role_run_dir(root_dir: Path, role: str) -> Path:
    return Path(root_dir) / role / "run"


def _job_id(workflow_id: str, role: str) -> str:
    return f"{workflow_id}-{role}"


def _connect(db_path: Path) -> sqlite3.Connection:
    conn = sqlite3.connect(Path(db_path))
    conn.row_factory = sqlite3.Row
    return conn
