from __future__ import annotations

import os
from datetime import datetime
from pathlib import Path

from .config import load_app_config
from .jobs import get_job, update_job_status
from .models import JobRecord
from .runner import launch_vasp_process, stop_task


CORE_INPUT_FILES = ("INCAR", "POSCAR", "KPOINTS", "POTCAR")
TAIL_FILE_WHITELIST = {"vasp.out", "OSZICAR"}


def start_workflow_job(
    db_path: Path,
    *,
    job_id: str,
    dry_run: bool = True,
) -> JobRecord:
    """启动 workflow job。

    dry-run 不启动真实 VASP，只写入模拟日志并将 job 标记为 finished。
    """

    job = _require_job(db_path, job_id)
    if job.status == "running":
        raise RuntimeError(f"Workflow job is already running: {job_id}")
    missing = _missing_vasp_inputs(job.run_dir)
    if missing:
        raise FileNotFoundError("Missing VASP input files: " + ", ".join(missing))

    now = datetime.utcnow()
    if dry_run:
        _write_dry_run_outputs(job)
        update_job_status(
            db_path,
            job_id,
            "finished",
            start_time=now,
            end_time=now,
            return_code=0,
        )
        return _require_job(db_path, job_id)

    vasp_bin, mpi_ranks = _resolve_launch_settings(job)
    _backup_existing_logs(job.run_dir)
    pid = launch_vasp_process(job.run_dir, vasp_bin, mpi_ranks)
    update_job_status(db_path, job_id, "running", pid=pid, start_time=now)
    return _require_job(db_path, job_id)


def stop_workflow_job(
    db_path: Path,
    *,
    job_id: str,
) -> JobRecord:
    """安全停止 workflow job。

    非 running job 不抛错；dry-run finished job 会保持 finished 状态。
    """

    job = _require_job(db_path, job_id)
    if job.status != "running":
        return job
    if job.pid is not None:
        try:
            stop_task(int(job.pid))
        except ProcessLookupError:
            pass
    update_job_status(db_path, job_id, "stopped", end_time=datetime.utcnow())
    return _require_job(db_path, job_id)


def get_workflow_job_log_paths(
    db_path: Path,
    job_id: str,
) -> dict:
    job = _require_job(db_path, job_id)
    return {
        "vasp.out": _file_info(job.run_dir / "vasp.out"),
        "OSZICAR": _file_info(job.run_dir / "OSZICAR"),
        "OUTCAR": _file_info(job.run_dir / "OUTCAR"),
    }


def tail_workflow_job_file(
    db_path: Path,
    job_id: str,
    filename: str,
    max_chars: int = 20000,
) -> str:
    if filename not in TAIL_FILE_WHITELIST:
        raise ValueError(f"Tail file is not allowed for workflow job logs: {filename}")
    job = _require_job(db_path, job_id)
    path = job.run_dir / filename
    if not path.exists():
        return ""
    text = path.read_text(encoding="utf-8", errors="replace")
    return text[-max_chars:]


def refresh_workflow_job_status(
    db_path: Path,
    job_id: str,
) -> JobRecord:
    """刷新进程层面的状态，不解析 OUTCAR。"""

    job = _require_job(db_path, job_id)
    if job.status != "running":
        return job
    if job.pid is not None and _process_exists(int(job.pid)):
        return job
    update_job_status(db_path, job_id, "finished", end_time=datetime.utcnow())
    return _require_job(db_path, job_id)


def _require_job(db_path: Path, job_id: str) -> JobRecord:
    job = get_job(db_path, job_id)
    if job is None:
        raise ValueError(f"Workflow job not found: {job_id}")
    return job


def _missing_vasp_inputs(run_dir: Path) -> list[str]:
    workdir = Path(run_dir)
    if not workdir.exists() or not workdir.is_dir():
        return list(CORE_INPUT_FILES)
    return [
        filename
        for filename in CORE_INPUT_FILES
        if not (workdir / filename).exists() or (workdir / filename).stat().st_size == 0
    ]


def _resolve_launch_settings(job: JobRecord) -> tuple[str | Path, int]:
    """解析真实 VASP 启动参数。

    优先使用 job 自己保存的 vasp_bin/mpi_ranks；为空时才回退到默认配置。
    这样后续同一 workflow 内不同 job 可以使用不同并行规模。
    """

    config = None
    vasp_bin: str | Path | None = job.vasp_bin
    mpi_ranks: int | None = job.mpi_ranks
    if not vasp_bin or mpi_ranks is None:
        config = load_app_config()
    if not vasp_bin:
        vasp_bin = config.vasp_bin if config is not None else None
    if mpi_ranks is None:
        mpi_ranks = config.default_mpi_ranks if config is not None else None
    if not vasp_bin:
        raise ValueError(f"Workflow job is missing vasp_bin: {job.job_id}")
    if mpi_ranks is None:
        raise ValueError(f"Workflow job is missing mpi_ranks: {job.job_id}")
    return vasp_bin, int(mpi_ranks)


def _backup_existing_logs(run_dir: Path) -> list[Path]:
    """真实 VASP 启动前备份旧日志，避免静默覆盖用户已有输出。"""

    timestamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S_%f")
    backups: list[Path] = []
    for filename in ("vasp.out", "OSZICAR"):
        path = Path(run_dir) / filename
        if path.exists() and path.is_file():
            backup = path.with_name(f"{filename}.{timestamp}.bak")
            path.rename(backup)
            backups.append(backup)
    return backups


def _write_dry_run_outputs(job: JobRecord) -> None:
    workdir = Path(job.run_dir)
    workdir.mkdir(parents=True, exist_ok=True)
    (workdir / "vasp.out").write_text(
        "DRY-RUN VASP JOB\n"
        f"job_id: {job.job_id}\n"
        f"run_dir: {workdir}\n"
        "This is not a real VASP calculation. No mpirun or vasp_std process was started.\n"
        "dry-run completed successfully\n",
        encoding="utf-8",
    )
    (workdir / "OSZICAR").write_text(
        " 1 F= -.10000000E+02 E0= -.10000000E+02 d E =0\n"
        " 2 F= -.10500000E+02 E0= -.10500000E+02 d E =-.5\n"
        " 3 F= -.10550000E+02 E0= -.10550000E+02 d E =-.05\n",
        encoding="utf-8",
    )


def _file_info(path: Path) -> dict:
    return {
        "path": str(path),
        "exists": path.exists(),
        "size_bytes": path.stat().st_size if path.exists() else 0,
    }


def _process_exists(pid: int) -> bool:
    if pid <= 0:
        return False
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    return True
