from __future__ import annotations

import os
import signal
import sqlite3
import subprocess
from datetime import datetime
from pathlib import Path

from . import db
from .models import AppConfig, PotcarConfig, TaskDraft, TaskRecord
from .security import potcar_paths, task_dir, validate_mpi_ranks, validate_vasp_bin


def run_dir(task_root: Path) -> Path:
    return task_root / "run"


def mpirun_args(ranks: int, vasp_bin: Path) -> list[str]:
    return [
        "mpirun",
        "-np",
        str(ranks),
        "--map-by",
        "core",
        "--bind-to",
        "core",
        str(vasp_bin),
    ]


def write_confirmed_task(config: AppConfig, potcars: PotcarConfig, draft: TaskDraft, conn: sqlite3.Connection) -> Path:
    request = draft.request
    validate_mpi_ranks(config, request.mpi_ranks)
    task_root = task_dir(config, request.task_id)
    workdir = run_dir(task_root)
    workdir.mkdir(parents=True, exist_ok=True)

    (workdir / "POSCAR").write_text(request.structure.poscar_text, encoding="utf-8")
    (workdir / "INCAR").write_text(draft.incar_text, encoding="utf-8")
    (workdir / "KPOINTS").write_text(draft.kpoints_text, encoding="utf-8")
    (workdir / "run.sh").write_text(draft.run_sh_text, encoding="utf-8")
    (workdir / "run.sh").chmod(0o750)
    _write_potcar(config, potcars, request.structure.elements, workdir / "POTCAR")

    db.upsert_task(
        conn,
        task_id=request.task_id,
        status="committed",
        path=task_root,
        task_type=request.task_type,
    )
    return task_root


def start_vasp(
    config: AppConfig,
    draft: TaskDraft,
    conn: sqlite3.Connection,
    *,
    dry_run: bool = False,
) -> subprocess.Popen | None:
    request = draft.request
    ranks = validate_mpi_ranks(config, request.mpi_ranks)
    task_root = task_dir(config, request.task_id)
    workdir = run_dir(task_root)
    missing = validate_run_inputs(task_root)
    if missing:
        raise FileNotFoundError("Missing required run inputs: " + ", ".join(missing))

    out_path = workdir / "vasp.out"
    args = mpirun_args(ranks, config.vasp_bin)
    start_time = datetime.utcnow()
    if dry_run:
        db.update_task_status(conn, request.task_id, "running", start_time=start_time)
        _write_dry_run_outputs(workdir, args)
        db.update_task_status(
            conn,
            request.task_id,
            "finished",
            end_time=datetime.utcnow(),
            return_code=0,
        )
        return None

    vasp_bin = validate_vasp_bin(config)
    env = os.environ.copy()
    env["OMP_NUM_THREADS"] = "1"
    env["OPENBLAS_NUM_THREADS"] = "1"
    env["OMP_STACKSIZE"] = "512m"
    args = mpirun_args(ranks, vasp_bin)
    with out_path.open("ab") as out:
        process = subprocess.Popen(
            args,
            cwd=workdir,
            stdout=out,
            stderr=subprocess.STDOUT,
            env=env,
            shell=False,
            start_new_session=True,
        )
    db.update_task_status(conn, request.task_id, "running", pid=process.pid, start_time=start_time)
    return process


def validate_run_inputs(task_root: Path) -> list[str]:
    """检查任务 run/ 目录中启动 VASP 所需的四个输入文件。

    这里只检查存在性和非空，具体 POSCAR/POTCAR 元素顺序由 Input Set 验证流程负责。
    """

    workdir = run_dir(task_root)
    missing: list[str] = []
    for filename in ("INCAR", "POSCAR", "KPOINTS", "POTCAR"):
        path = workdir / filename
        if not path.exists() or path.stat().st_size == 0:
            missing.append(filename)
    return missing


def start_task_record(
    config: AppConfig,
    task: TaskRecord,
    conn: sqlite3.Connection,
    *,
    ranks: int | None = None,
    dry_run: bool = False,
) -> subprocess.Popen | None:
    """从已提交任务的 run/ 目录启动 VASP。

    该入口用于 Input Set 绑定创建的任务，不依赖 TaskDraft；仍使用参数列表和 shell=False。
    """

    selected_ranks = validate_mpi_ranks(config, ranks or config.default_mpi_ranks)
    missing = validate_run_inputs(task.task_root)
    if missing:
        raise FileNotFoundError("Missing required run inputs: " + ", ".join(missing))

    workdir = run_dir(task.task_root)
    out_path = workdir / "vasp.out"
    args = mpirun_args(selected_ranks, config.vasp_bin)
    start_time = datetime.utcnow()
    if dry_run:
        db.update_task_status(conn, task.task_id, "running", start_time=start_time)
        _write_dry_run_outputs(workdir, args)
        db.update_task_status(
            conn,
            task.task_id,
            "finished",
            end_time=datetime.utcnow(),
            return_code=0,
        )
        return None

    vasp_bin = validate_vasp_bin(config)
    env = os.environ.copy()
    env["OMP_NUM_THREADS"] = "1"
    env["OPENBLAS_NUM_THREADS"] = "1"
    env["OMP_STACKSIZE"] = "512m"
    args = mpirun_args(selected_ranks, vasp_bin)
    with out_path.open("ab") as out:
        process = subprocess.Popen(
            args,
            cwd=workdir,
            stdout=out,
            stderr=subprocess.STDOUT,
            env=env,
            shell=False,
            start_new_session=True,
        )
    db.update_task_status(conn, task.task_id, "running", pid=process.pid, start_time=start_time)
    return process


def stop_task(pid: int) -> None:
    if pid <= 0:
        raise ValueError("PID must be a positive integer.")
    try:
        pgid = os.getpgid(pid)
    except ProcessLookupError as exc:
        raise ProcessLookupError(f"Process {pid} no longer exists.") from exc
    os.killpg(pgid, signal.SIGTERM)


def tail_file(path: Path, max_bytes: int = 20000) -> str:
    if not path.exists():
        return ""
    size = path.stat().st_size
    with path.open("rb") as fh:
        if size > max_bytes:
            fh.seek(-max_bytes, os.SEEK_END)
        data = fh.read()
    return data.decode("utf-8", errors="replace")


def _write_potcar(config: AppConfig, potcars: PotcarConfig, elements: tuple[str, ...], target: Path) -> None:
    workdir = target.parent
    if workdir.resolve().is_relative_to(config.potpaw_pbe.resolve()):
        raise ValueError("Refusing to write POTCAR inside the POTCAR source tree.")
    paths = potcar_paths(config, potcars, elements)
    with target.open("wb") as out:
        for path in paths:
            with path.open("rb") as src:
                out.write(src.read())


def _write_dry_run_outputs(workdir: Path, args: list[str]) -> None:
    (workdir / "vasp.out").write_text(
        "DRY RUN: VASP was not started.\n"
        f"Would run: {' '.join(args)}\n"
        "dry run completed successfully\n",
        encoding="utf-8",
    )
    (workdir / "OSZICAR").write_text(
        " 1 F= -.10000000E+02 E0= -.10000000E+02 d E =0\n"
        " 2 F= -.10500000E+02 E0= -.10500000E+02 d E =-.5\n",
        encoding="utf-8",
    )
    (workdir / "OUTCAR").write_text(
        " free  energy   TOTEN  =       -10.500000 eV\n"
        " LOOP:  cpu time   1.00: real time   2.00\n"
        " reached required accuracy - stopping structural energy minimisation\n",
        encoding="utf-8",
    )
