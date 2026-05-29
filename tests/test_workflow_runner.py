from __future__ import annotations

import sys
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from vasp_mvp.db import db_path, init_db
from vasp_mvp.jobs import create_job, get_job, update_job_status
from vasp_mvp.workflow_runner import (
    ProcessInfo,
    find_processes_by_run_dir,
    get_workflow_job_log_paths,
    is_process_alive,
    refresh_workflow_job_status,
    start_workflow_job,
    stop_workflow_job,
    tail_workflow_job_file,
    terminate_process_group,
)


class WorkflowRunnerTest(unittest.TestCase):
    def test_dry_run_start_writes_logs_and_finishes_job(self) -> None:
        with TemporaryDirectory() as tmp:
            database, run_dir = _create_job_with_inputs(Path(tmp))

            job = start_workflow_job(database, job_id="job-1", dry_run=True)

            self.assertEqual(job.status, "finished")
            self.assertEqual(job.return_code, 0)
            self.assertIsNotNone(job.start_time)
            self.assertIsNotNone(job.end_time)
            self.assertTrue((run_dir / "vasp.out").exists())
            self.assertTrue((run_dir / "OSZICAR").exists())
            vasp_out = (run_dir / "vasp.out").read_text(encoding="utf-8")
            self.assertIn("DRY-RUN VASP JOB", vasp_out)
            self.assertIn("job_id: job-1", vasp_out)
            self.assertIn(str(run_dir), vasp_out)
            self.assertIn("not a real VASP calculation", vasp_out)

    def test_start_rejects_missing_vasp_input_file(self) -> None:
        with TemporaryDirectory() as tmp:
            database, run_dir = _create_job_with_inputs(Path(tmp))
            (run_dir / "POTCAR").unlink()

            with self.assertRaisesRegex(FileNotFoundError, "POTCAR"):
                start_workflow_job(database, job_id="job-1", dry_run=True)

    def test_start_rejects_already_running_job(self) -> None:
        with TemporaryDirectory() as tmp:
            database, _run_dir = _create_job_with_inputs(Path(tmp), status="running")

            with self.assertRaisesRegex(RuntimeError, "already running"):
                start_workflow_job(database, job_id="job-1", dry_run=True)

    def test_tail_log_uses_filename_whitelist(self) -> None:
        with TemporaryDirectory() as tmp:
            database, run_dir = _create_job_with_inputs(Path(tmp))
            (run_dir / "vasp.out").write_text("line 1\nline 2\n", encoding="utf-8")

            self.assertEqual(tail_workflow_job_file(database, "job-1", "vasp.out"), "line 1\nline 2\n")
            with self.assertRaisesRegex(ValueError, "not allowed"):
                tail_workflow_job_file(database, "job-1", "POTCAR")
            with self.assertRaisesRegex(ValueError, "not allowed"):
                tail_workflow_job_file(database, "job-1", "../POTCAR")

    def test_stop_dry_run_finished_job_does_not_crash_or_change_status(self) -> None:
        with TemporaryDirectory() as tmp:
            database, _run_dir = _create_job_with_inputs(Path(tmp))
            start_workflow_job(database, job_id="job-1", dry_run=True)

            stopped = stop_workflow_job(database, job_id="job-1")
            loaded = get_job(database, "job-1")

            self.assertEqual(stopped.status, "finished")
            self.assertIsNotNone(loaded)
            self.assertEqual(loaded.status, "finished")

    def test_log_paths_include_outcar_existence_without_parsing(self) -> None:
        with TemporaryDirectory() as tmp:
            database, run_dir = _create_job_with_inputs(Path(tmp))
            (run_dir / "OUTCAR").write_text("not parsed in this test\n", encoding="utf-8")

            paths = get_workflow_job_log_paths(database, "job-1")

            self.assertTrue(paths["OUTCAR"]["exists"])
            self.assertIn("OUTCAR", paths["OUTCAR"]["path"])

    def test_real_start_uses_job_launch_settings_and_backs_up_existing_logs(self) -> None:
        with TemporaryDirectory() as tmp:
            database, run_dir = _create_job_with_inputs(Path(tmp), mpi_ranks=6, vasp_bin="/job/vasp_std")
            (run_dir / "vasp.out").write_text("old out\n", encoding="utf-8")
            (run_dir / "OSZICAR").write_text("old oszicar\n", encoding="utf-8")

            with patch("vasp_mvp.workflow_runner.launch_vasp_process", return_value=4321) as launch:
                job = start_workflow_job(database, job_id="job-1", dry_run=False)

            self.assertEqual(job.status, "running")
            self.assertEqual(job.pid, 4321)
            self.assertIsNone(job.return_code)
            launch.assert_called_once_with(run_dir, "/job/vasp_std", 6)
            self.assertFalse((run_dir / "vasp.out").exists())
            self.assertFalse((run_dir / "OSZICAR").exists())
            self.assertEqual(len(list(run_dir.glob("vasp.out.*.bak"))), 1)
            self.assertEqual(len(list(run_dir.glob("OSZICAR.*.bak"))), 1)

    def test_real_start_falls_back_to_default_config_when_job_settings_are_empty(self) -> None:
        with TemporaryDirectory() as tmp:
            database, run_dir = _create_job_with_inputs(Path(tmp), mpi_ranks=None, vasp_bin=None)
            fake_config = SimpleNamespace(vasp_bin=Path("/config/vasp_std"), default_mpi_ranks=12)

            with patch("vasp_mvp.workflow_runner.load_app_config", return_value=fake_config):
                with patch("vasp_mvp.workflow_runner.launch_vasp_process", return_value=9876) as launch:
                    job = start_workflow_job(database, job_id="job-1", dry_run=False)

            self.assertEqual(job.status, "running")
            self.assertEqual(job.pid, 9876)
            launch.assert_called_once_with(run_dir, Path("/config/vasp_std"), 12)

    def test_update_job_status_can_clear_return_code_without_breaking_old_calls(self) -> None:
        with TemporaryDirectory() as tmp:
            database, _run_dir = _create_job_with_inputs(Path(tmp))
            update_job_status(database, "job-1", "finished", return_code=0)
            self.assertEqual(get_job(database, "job-1").return_code, 0)

            update_job_status(database, "job-1", "running", pid=2222, clear_return_code=True)
            loaded = get_job(database, "job-1")

            self.assertEqual(loaded.status, "running")
            self.assertEqual(loaded.pid, 2222)
            self.assertIsNone(loaded.return_code)

    def test_stop_running_job_terminates_process_group_and_marks_stopped(self) -> None:
        with TemporaryDirectory() as tmp:
            database, _run_dir = _create_job_with_inputs(Path(tmp), status="running")
            update_job_status(database, "job-1", "running", pid=3333)

            with patch("vasp_mvp.workflow_runner.terminate_process_group", return_value=True) as terminate:
                with patch("vasp_mvp.workflow_runner.find_processes_by_run_dir", return_value=[]):
                    result = stop_workflow_job(database, job_id="job-1")

            self.assertEqual(result.status, "stopped")
            self.assertEqual(result.message_key, "workflow_job.stop_success")
            terminate.assert_called_once_with(3333)

    def test_stop_uses_run_dir_fallback_when_pid_is_missing(self) -> None:
        with TemporaryDirectory() as tmp:
            database, run_dir = _create_job_with_inputs(Path(tmp), status="running")
            fake_process = ProcessInfo(pid=4444, pgid=4444, cwd=run_dir, command="vasp_std", cmdline="vasp_std")

            with patch("vasp_mvp.workflow_runner.find_processes_by_run_dir", return_value=[fake_process]):
                with patch("vasp_mvp.workflow_runner.terminate_process_group", return_value=True) as terminate:
                    result = stop_workflow_job(database, job_id="job-1")

            self.assertEqual(result.status, "stopped")
            self.assertEqual(result.message_key, "workflow_job.stop_success")
            terminate.assert_called_once_with(4444)

    def test_refresh_running_pid_alive_keeps_running(self) -> None:
        with TemporaryDirectory() as tmp:
            database, _run_dir = _create_job_with_inputs(Path(tmp), status="running")
            update_job_status(database, "job-1", "running", pid=5555)

            with patch("vasp_mvp.workflow_runner.is_process_alive", return_value=True):
                with patch("vasp_mvp.workflow_runner.find_processes_by_run_dir", return_value=[]):
                    result = refresh_workflow_job_status(database, "job-1")

            self.assertEqual(result.status, "running")
            self.assertEqual(result.warnings, ())

    def test_refresh_running_pid_dead_without_run_dir_process_marks_stopped(self) -> None:
        with TemporaryDirectory() as tmp:
            database, _run_dir = _create_job_with_inputs(Path(tmp), status="running")
            update_job_status(database, "job-1", "running", pid=6666)

            with patch("vasp_mvp.workflow_runner.is_process_alive", return_value=False):
                with patch("vasp_mvp.workflow_runner.find_processes_by_run_dir", return_value=[]):
                    result = refresh_workflow_job_status(database, "job-1")

            self.assertEqual(result.status, "stopped")
            self.assertIn("workflow_job.database_status_updated", result.warnings)

    def test_refresh_non_running_pid_alive_corrects_status_to_running(self) -> None:
        with TemporaryDirectory() as tmp:
            database, _run_dir = _create_job_with_inputs(Path(tmp), status="committed")
            update_job_status(database, "job-1", "committed", pid=7777)

            with patch("vasp_mvp.workflow_runner.is_process_alive", return_value=True):
                with patch("vasp_mvp.workflow_runner.find_processes_by_run_dir", return_value=[]):
                    result = refresh_workflow_job_status(database, "job-1")

            self.assertEqual(result.status, "running")
            self.assertIn("workflow_job.status_mismatch", result.warnings)

    def test_refresh_running_pid_dead_with_run_dir_process_keeps_running_and_warns(self) -> None:
        with TemporaryDirectory() as tmp:
            database, run_dir = _create_job_with_inputs(Path(tmp), status="running")
            update_job_status(database, "job-1", "running", pid=8888)
            fake_process = ProcessInfo(pid=9999, pgid=9999, cwd=run_dir, command="mpirun", cmdline="mpirun vasp_std")

            with patch("vasp_mvp.workflow_runner.is_process_alive", return_value=False):
                with patch("vasp_mvp.workflow_runner.find_processes_by_run_dir", return_value=[fake_process]):
                    result = refresh_workflow_job_status(database, "job-1")

            self.assertEqual(result.status, "running")
            self.assertIn("workflow_job.pid_stale", result.warnings)

    def test_find_processes_by_run_dir_rejects_matching_command_outside_run_dir(self) -> None:
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            run_dir = root / "workspace" / "job" / "run"
            outside = root / "other" / "run"
            fake_proc = root / "proc"
            run_dir.mkdir(parents=True)
            outside.mkdir(parents=True)
            pid_dir = fake_proc / "12345"
            pid_dir.mkdir(parents=True)
            (pid_dir / "cwd").symlink_to(outside)
            (pid_dir / "comm").write_text("vasp_std\n", encoding="utf-8")
            (pid_dir / "cmdline").write_bytes(b"vasp_std\0")

            self.assertEqual(find_processes_by_run_dir(run_dir, proc_root=fake_proc), [])

    def test_find_processes_by_run_dir_matches_only_whitelisted_command_inside_run_dir(self) -> None:
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            run_dir = root / "workspace" / "job" / "run"
            fake_proc = root / "proc"
            run_dir.mkdir(parents=True)
            pid_dir = fake_proc / "12345"
            pid_dir.mkdir(parents=True)
            (pid_dir / "cwd").symlink_to(run_dir)
            (pid_dir / "comm").write_text("python\n", encoding="utf-8")
            (pid_dir / "cmdline").write_bytes(b"python\0script.py\0")
            self.assertEqual(find_processes_by_run_dir(run_dir, proc_root=fake_proc), [])

            (pid_dir / "comm").write_text("mpirun\n", encoding="utf-8")
            matches = find_processes_by_run_dir(run_dir, proc_root=fake_proc)
            self.assertEqual([item.pid for item in matches], [12345])

    def test_tail_and_stop_do_not_touch_potcar(self) -> None:
        with TemporaryDirectory() as tmp:
            database, run_dir = _create_job_with_inputs(Path(tmp), status="running")
            potcar = run_dir / "POTCAR"
            original = potcar.read_text(encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "not allowed"):
                tail_workflow_job_file(database, "job-1", "POTCAR")
            with patch("vasp_mvp.workflow_runner.find_processes_by_run_dir", return_value=[]):
                stop_workflow_job(database, job_id="job-1")
            self.assertEqual(potcar.read_text(encoding="utf-8"), original)


def _create_job_with_inputs(
    tmp: Path,
    *,
    status: str = "committed",
    mpi_ranks: int | None = 20,
    vasp_bin: str | None = "/bin/false",
) -> tuple[Path, Path]:
    workspace = tmp / "workspace"
    init_db(workspace).close()
    database = db_path(workspace)
    run_dir = workspace / "workflow" / "job-1" / "run"
    run_dir.mkdir(parents=True)
    for filename in ("INCAR", "POSCAR", "KPOINTS", "POTCAR"):
        (run_dir / filename).write_text(f"{filename}\n", encoding="utf-8")
    create_job(
        database,
        job_id="job-1",
        calculation_type="static",
        status=status,
        run_dir=run_dir,
        mpi_ranks=mpi_ranks,
        vasp_bin=vasp_bin,
    )
    return database, run_dir


if __name__ == "__main__":
    unittest.main()
