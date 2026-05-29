from __future__ import annotations

import os
import signal
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

from .config import load_app_config
from .jobs import get_job, update_job_status
from .models import JobRecord
from .runner import launch_vasp_process


CORE_INPUT_FILES = ("INCAR", "POSCAR", "KPOINTS", "POTCAR")
TAIL_FILE_WHITELIST = {"vasp.out", "OSZICAR"}
PROCESS_NAME_WHITELIST = {"vasp_std", "vasp_gam", "vasp_ncl", "mpirun", "orterun", "prterun", "orted", "pmix"}


@dataclass(frozen=True)
class ProcessInfo:
    pid: int
    pgid: int | None
    cwd: Path
    command: str
    cmdline: str


@dataclass(frozen=True)
class WorkflowJobActionResult:
    job: JobRecord
    message_key: str
    warnings: tuple[str, ...] = ()
    process_alive: bool | None = None
    processes: tuple[ProcessInfo, ...] = ()

    def __getattr__(self, name: str):
        return getattr(self.job, name)


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
    update_job_status(db_path, job_id, "running", pid=pid, start_time=now, clear_return_code=True)
    return _require_job(db_path, job_id)


def stop_workflow_job(
    db_path: Path,
    *,
    job_id: str,
) -> WorkflowJobActionResult:
    """安全停止 workflow job。

    不只依赖数据库 status：如果 pid 失效，会按 run_dir 精确查找 VASP/MPI 进程组。
    """

    job = _require_job(db_path, job_id)
    stopped = False
    attempted = False
    warnings: list[str] = []
    if job.pid is not None:
        attempted = True
        try:
            stopped = terminate_process_group(int(job.pid))
        except ProcessLookupError:
            warnings.append("process_not_found")

    processes = find_processes_by_run_dir(job.run_dir)
    for process in processes:
        if job.pid is not None and process.pid == int(job.pid):
            continue
        attempted = True
        try:
            stopped = terminate_process_group(process.pid) or stopped
        except ProcessLookupError:
            warnings.append("process_not_found")

    should_mark_stopped = job.status == "running" or attempted
    if should_mark_stopped:
        update_job_status(db_path, job_id, "stopped", end_time=datetime.utcnow())
    loaded = _require_job(db_path, job_id)
    if stopped:
        message_key = "workflow_job.stop_success"
    elif attempted:
        message_key = "workflow_job.stop_no_process_found"
    else:
        message_key = "workflow_job.stop_no_process_found"
    return WorkflowJobActionResult(
        job=loaded,
        message_key=message_key,
        warnings=tuple(warnings),
        process_alive=bool(find_processes_by_run_dir(loaded.run_dir)),
    )


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
) -> WorkflowJobActionResult:
    """刷新进程层面的状态，不解析 OUTCAR。"""

    job = _require_job(db_path, job_id)
    warnings: list[str] = []
    pid_alive = is_process_alive(int(job.pid)) if job.pid is not None else False
    run_dir_processes = find_processes_by_run_dir(job.run_dir)
    run_dir_alive = bool(run_dir_processes)

    if job.status != "running" and (pid_alive or run_dir_alive):
        warnings.append("workflow_job.status_mismatch")
        update_job_status(db_path, job_id, "running")
        loaded = _require_job(db_path, job_id)
        return WorkflowJobActionResult(loaded, "workflow_job.status_mismatch", tuple(warnings), True, tuple(run_dir_processes))

    if job.status == "running" and pid_alive:
        return WorkflowJobActionResult(job, "workflow_job.refresh", process_alive=True, processes=tuple(run_dir_processes))

    if job.status == "running" and not pid_alive and run_dir_alive:
        warnings.append("workflow_job.pid_stale")
        return WorkflowJobActionResult(job, "workflow_job.pid_stale", tuple(warnings), True, tuple(run_dir_processes))

    if job.status == "running" and not pid_alive and not run_dir_alive:
        warnings.append("workflow_job.database_status_updated")
        update_job_status(db_path, job_id, "stopped", end_time=datetime.utcnow())
        loaded = _require_job(db_path, job_id)
        return WorkflowJobActionResult(loaded, "workflow_job.stop_no_process_found", tuple(warnings), False)

    return WorkflowJobActionResult(job, "workflow_job.refresh", process_alive=pid_alive or run_dir_alive, processes=tuple(run_dir_processes))


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


def is_process_alive(pid: int) -> bool:
    if pid <= 0:
        return False
    proc_stat = Path("/proc") / str(pid) / "stat"
    if proc_stat.exists():
        try:
            parts = proc_stat.read_text(encoding="utf-8", errors="replace").split()
            if len(parts) > 2 and parts[2] == "Z":
                return False
        except OSError:
            pass
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    return True


def find_processes_by_run_dir(run_dir: Path, proc_root: Path = Path("/proc")) -> list[ProcessInfo]:
    """按 run_dir 精确查找 VASP/MPI 相关进程。

    只匹配 cwd 等于 run_dir 或位于 run_dir 内的进程，并叠加命令白名单。
    不按全局命令名杀进程，避免影响其他工作目录中的 VASP 任务。
    """

    root = Path(run_dir).resolve()
    if not proc_root.exists():
        return []
    matches: list[ProcessInfo] = []
    for entry in proc_root.iterdir():
        if not entry.name.isdigit():
            continue
        pid = int(entry.name)
        try:
            cwd = Path(os.readlink(entry / "cwd")).resolve()
        except OSError:
            continue
        if cwd != root and root not in cwd.parents:
            continue
        command = _read_proc_text(entry / "comm").strip()
        cmdline = _read_cmdline(entry / "cmdline")
        if not _is_allowed_process(command, cmdline):
            continue
        try:
            pgid = os.getpgid(pid)
        except ProcessLookupError:
            pgid = None
        matches.append(ProcessInfo(pid=pid, pgid=pgid, cwd=cwd, command=command, cmdline=cmdline))
    return matches


def terminate_process_group(pid: int, timeout: float = 5.0) -> bool:
    """终止 pid 所在进程组，先 TERM，超时后只对同一进程组 KILL。"""

    if pid <= 0:
        return False
    try:
        pgid = os.getpgid(pid)
    except ProcessLookupError:
        return False
    try:
        os.killpg(pgid, signal.SIGTERM)
    except ProcessLookupError:
        return False
    deadline = time.time() + timeout
    while time.time() < deadline:
        if not _pgid_has_alive_processes(pgid):
            return True
        time.sleep(0.1)
    try:
        os.killpg(pgid, signal.SIGKILL)
    except ProcessLookupError:
        return True
    deadline = time.time() + 1.0
    while time.time() < deadline:
        if not _pgid_has_alive_processes(pgid):
            return True
        time.sleep(0.1)
    return not _pgid_has_alive_processes(pgid)


def get_workflow_job_process_state(db_path: Path, job_id: str) -> dict:
    job = _require_job(db_path, job_id)
    pid_alive = is_process_alive(int(job.pid)) if job.pid is not None else None
    processes = find_processes_by_run_dir(job.run_dir)
    return {
        "job_id": job.job_id,
        "pid": job.pid,
        "pid_alive": pid_alive,
        "run_dir_processes": processes,
        "process_alive": bool(pid_alive or processes),
    }


def _pgid_has_alive_processes(pgid: int) -> bool:
    proc_root = Path("/proc")
    if not proc_root.exists():
        return False
    for entry in proc_root.iterdir():
        if not entry.name.isdigit():
            continue
        stat = _read_proc_text(entry / "stat")
        if not stat:
            continue
        parts = stat.split()
        if len(parts) < 5:
            continue
        try:
            process_pgid = int(parts[4])
        except ValueError:
            continue
        if process_pgid == pgid and len(parts) > 2 and parts[2] != "Z":
            return True
    return False


def _read_proc_text(path: Path) -> str:
    try:
        return path.read_text(encoding="utf-8", errors="replace")
    except OSError:
        return ""


def _read_cmdline(path: Path) -> str:
    try:
        data = path.read_bytes()
    except OSError:
        return ""
    return " ".join(part.decode("utf-8", errors="replace") for part in data.split(b"\0") if part)


def _is_allowed_process(command: str, cmdline: str) -> bool:
    command_name = Path(command.strip()).name
    if command_name in PROCESS_NAME_WHITELIST:
        return True
    for token in cmdline.split():
        if Path(token).name in PROCESS_NAME_WHITELIST:
            return True
    return any(name in cmdline for name in PROCESS_NAME_WHITELIST)
