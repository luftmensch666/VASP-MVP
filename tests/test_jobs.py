from __future__ import annotations

import sys
import unittest
from datetime import datetime
from pathlib import Path
from tempfile import TemporaryDirectory

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from vasp_mvp.db import db_path, init_db
from vasp_mvp.jobs import (
    create_job,
    get_job,
    get_job_metrics,
    list_jobs,
    parse_and_save_job_metrics,
    update_job_status,
)


class JobsTest(unittest.TestCase):
    def test_create_get_list_and_update_job(self) -> None:
        with TemporaryDirectory() as tmp:
            workspace = Path(tmp) / "workspace"
            init_db(workspace).close()
            database = db_path(workspace)
            run_dir = workspace / "jobs" / "job-1" / "run"

            created = create_job(
                database,
                job_id="job-1",
                calculation_type="static",
                status="committed",
                run_dir=run_dir,
                input_set_id="input-set-1",
                mpi_ranks=24,
                vasp_bin="/opt/vasp/vasp_std",
            )

            self.assertEqual(created.job_id, "job-1")
            self.assertEqual(created.calculation_type, "static")
            self.assertEqual(created.status, "committed")
            self.assertEqual(created.run_dir, run_dir)
            self.assertEqual(created.input_set_id, "input-set-1")
            self.assertEqual(created.mpi_ranks, 24)
            self.assertEqual(created.vasp_bin, "/opt/vasp/vasp_std")

            start_time = datetime(2026, 5, 28, 12, 0, 0)
            end_time = datetime(2026, 5, 28, 12, 30, 0)
            update_job_status(database, "job-1", "running", pid=4321, start_time=start_time)
            update_job_status(database, "job-1", "finished", end_time=end_time, return_code=0)

            loaded = get_job(database, "job-1")
            all_jobs = list_jobs(database)
            static_jobs = list_jobs(database, calculation_type="static")
            finished_jobs = list_jobs(database, status="finished")

            self.assertIsNotNone(loaded)
            self.assertEqual(loaded.status, "finished")
            self.assertEqual(loaded.pid, 4321)
            self.assertEqual(loaded.start_time, start_time)
            self.assertEqual(loaded.end_time, end_time)
            self.assertEqual(loaded.return_code, 0)
            self.assertEqual(len(all_jobs), 1)
            self.assertEqual(len(static_jobs), 1)
            self.assertEqual(len(finished_jobs), 1)

    def test_parse_and_save_job_metrics_uses_existing_parser(self) -> None:
        with TemporaryDirectory() as tmp:
            workspace = Path(tmp) / "workspace"
            init_db(workspace).close()
            database = db_path(workspace)
            run_dir = workspace / "jobs" / "job-1" / "run"
            run_dir.mkdir(parents=True)
            (run_dir / "OUTCAR").write_text(
                """
                  free  energy   TOTEN  =       -10.000000 eV
                LOOP:  cpu time   1.00: real time   2.0
                  free  energy   TOTEN  =       -11.500000 eV
                LOOP:  cpu time   2.00: real time   4.0
                reached required accuracy - stopping structural energy minimisation
                """,
                encoding="utf-8",
            )
            (run_dir / "OSZICAR").write_text(
                """
                 1 F= -.10000000E+02 E0= -.90000000E+01 d E =0
                 2 F= -1.15000000E+01 E0= -1.10000000E+01 d E =0
                """,
                encoding="utf-8",
            )
            create_job(database, job_id="job-1", calculation_type="static", run_dir=run_dir)

            metrics = parse_and_save_job_metrics(database, "job-1", run_dir)
            saved = get_job_metrics(database, "job-1")

            self.assertEqual(metrics.toten_ev, -11.5)
            self.assertIsNotNone(saved)
            self.assertEqual(saved.toten_ev, -11.5)
            self.assertEqual(saved.loop_avg_seconds, 3.0)
            self.assertEqual(saved.loop_count, 2)
            self.assertTrue(saved.ionic_converged)
            self.assertIsNone(saved.electronic_converged)
            self.assertEqual(saved.oszicar_steps, (-10.0, -11.5))
            self.assertEqual(saved.errors, ())
            self.assertEqual(saved.energy_source, "OUTCAR")
            self.assertEqual(saved.energy_label, "final TOTEN")


if __name__ == "__main__":
    unittest.main()
