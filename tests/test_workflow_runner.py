from __future__ import annotations

import sys
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from vasp_mvp.db import db_path, init_db
from vasp_mvp.jobs import create_job, get_job
from vasp_mvp.workflow_runner import (
    get_workflow_job_log_paths,
    start_workflow_job,
    stop_workflow_job,
    tail_workflow_job_file,
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
            self.assertIn("DRY RUN", (run_dir / "vasp.out").read_text(encoding="utf-8"))

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


def _create_job_with_inputs(tmp: Path, *, status: str = "committed") -> tuple[Path, Path]:
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
        mpi_ranks=20,
        vasp_bin="/bin/false",
    )
    return database, run_dir


if __name__ == "__main__":
    unittest.main()
