from __future__ import annotations

import sqlite3
import sys
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from vasp_mvp.db import create_task, db_path, get_metrics, init_db, list_tasks, save_metrics, update_status


class DbTest(unittest.TestCase):
    def test_init_db_uses_workspace_vasp_mvp_db(self) -> None:
        with TemporaryDirectory() as tmp:
            workspace = Path(tmp) / "workspace"
            conn = init_db(workspace)

            self.assertEqual(db_path(workspace), workspace / "vasp_mvp.db")
            self.assertTrue((workspace / "vasp_mvp.db").exists())
            tables = {
                row["name"]
                for row in conn.execute("SELECT name FROM sqlite_master WHERE type = 'table'").fetchall()
            }
            self.assertIn("tasks", tables)
            self.assertIn("metrics", tables)

    def test_create_update_list_and_save_metrics(self) -> None:
        with TemporaryDirectory() as tmp:
            workspace = Path(tmp) / "workspace"
            task_root = workspace / "project-a" / "task-1"
            conn = init_db(workspace)

            create_task(
                conn,
                task_id="task-1",
                project="project-a",
                task_type="relax",
                status="ready",
                task_root=task_root,
            )
            update_status(conn, "task-1", "running", pid=1234)
            save_metrics(conn, "task-1", toten=-12.34, loop_avg=5.6, converged=True)

            tasks = list_tasks(conn)
            metrics = get_metrics(conn, "task-1")

            self.assertEqual(len(tasks), 1)
            self.assertEqual(tasks[0].task_id, "task-1")
            self.assertEqual(tasks[0].project, "project-a")
            self.assertEqual(tasks[0].task_type, "relax")
            self.assertEqual(tasks[0].status, "running")
            self.assertEqual(tasks[0].task_root, task_root)
            self.assertEqual(tasks[0].pid, 1234)
            self.assertIsInstance(tasks[0].task_root, Path)
            self.assertIsNotNone(metrics)
            self.assertEqual(metrics.toten_ev, -12.34)
            self.assertEqual(metrics.loop_avg_seconds, 5.6)
            self.assertTrue(metrics.ionic_converged)


if __name__ == "__main__":
    unittest.main()
