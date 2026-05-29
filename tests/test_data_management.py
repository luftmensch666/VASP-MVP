from __future__ import annotations

import sqlite3
import sys
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from vasp_mvp.data_management import (
    delete_input_set,
    delete_legacy_task,
    delete_workflow,
    ensure_path_inside_workspace,
    factory_reset,
    preview_factory_reset,
    update_input_set_metadata,
    update_workflow_metadata,
)
from vasp_mvp.db import create_task, db_path, init_db, save_metrics
from vasp_mvp.input_sets import bind_input_set_to_task, create_input_set
from vasp_mvp.jobs import create_job, get_job, save_job_metrics
from vasp_mvp.models import Metrics
from vasp_mvp.workflows import bind_job_to_workflow, create_workflow, get_workflow


class DataManagementTest(unittest.TestCase):
    def test_delete_workflow_removes_owned_jobs_metrics_and_files(self) -> None:
        with TemporaryDirectory() as tmp:
            workspace, database = _init_workspace(tmp)
            _create_workflow_with_three_jobs(database, workspace)

            result = delete_workflow(database, "wf-1", workspace=workspace)

            self.assertTrue(result.ok, result.errors)
            self.assertIsNone(get_workflow(database, "wf-1"))
            self.assertIsNone(get_job(database, "job-clean"))
            self.assertFalse((workspace / "workflows" / "wf-1").exists())
            with sqlite3.connect(database) as conn:
                self.assertEqual(conn.execute("SELECT COUNT(*) FROM workflow_jobs").fetchone()[0], 0)
                self.assertEqual(conn.execute("SELECT COUNT(*) FROM job_metrics").fetchone()[0], 0)

    def test_delete_workflow_rejects_running_job_and_keeps_data(self) -> None:
        with TemporaryDirectory() as tmp:
            workspace, database = _init_workspace(tmp)
            _create_workflow_with_three_jobs(database, workspace, clean_status="running")

            result = delete_workflow(database, "wf-1", workspace=workspace)

            self.assertFalse(result.ok)
            self.assertIn("running", "\n".join(result.errors))
            self.assertIsNotNone(get_workflow(database, "wf-1"))
            self.assertTrue((workspace / "workflows" / "wf-1").exists())

    def test_delete_input_set_removes_unused_record_and_directory(self) -> None:
        with TemporaryDirectory() as tmp:
            workspace, database = _init_workspace(tmp)
            conn = sqlite3.connect(database)
            conn.row_factory = sqlite3.Row
            _create_input_set(conn, workspace, "is-unused")
            conn.close()

            result = delete_input_set(database, "is-unused", workspace=workspace)

            self.assertTrue(result.ok, result.errors)
            self.assertFalse((workspace / "input_sets" / "is-unused").exists())
            with sqlite3.connect(database) as conn:
                count = conn.execute("SELECT COUNT(*) FROM input_sets WHERE input_set_id = 'is-unused'").fetchone()[0]
            self.assertEqual(count, 0)

    def test_delete_input_set_rejects_job_and_legacy_task_references(self) -> None:
        with TemporaryDirectory() as tmp:
            workspace, database = _init_workspace(tmp)
            conn = sqlite3.connect(database)
            conn.row_factory = sqlite3.Row
            _create_input_set(conn, workspace, "is-used")
            create_task(
                conn,
                task_id="legacy-1",
                project="default",
                task_type="static",
                task_root=workspace / "legacy" / "legacy-1",
                status="committed",
            )
            bind_input_set_to_task(conn, "legacy-1", "primary", "is-used")
            conn.close()
            create_job(
                database,
                job_id="job-uses-input",
                calculation_type="static",
                run_dir=workspace / "jobs" / "job-uses-input" / "run",
                input_set_id="is-used",
            )

            result = delete_input_set(database, "is-used", workspace=workspace)

            self.assertFalse(result.ok)
            joined = "\n".join(result.errors)
            self.assertIn("job-uses-input", joined)
            self.assertIn("legacy-1", joined)
            self.assertTrue((workspace / "input_sets" / "is-used").exists())

    def test_delete_legacy_task_removes_metrics_and_files(self) -> None:
        with TemporaryDirectory() as tmp:
            workspace, database = _init_workspace(tmp)
            conn = sqlite3.connect(database)
            conn.row_factory = sqlite3.Row
            task_root = workspace / "legacy" / "legacy-1"
            task_root.mkdir(parents=True)
            create_task(
                conn,
                task_id="legacy-1",
                project="default",
                task_type="static",
                task_root=task_root,
                status="committed",
            )
            save_metrics(conn, "legacy-1", toten=-1.0)
            conn.close()

            result = delete_legacy_task(database, "legacy-1", workspace=workspace)

            self.assertTrue(result.ok, result.errors)
            self.assertFalse(task_root.exists())
            with sqlite3.connect(database) as conn:
                self.assertEqual(conn.execute("SELECT COUNT(*) FROM tasks").fetchone()[0], 0)
                self.assertEqual(conn.execute("SELECT COUNT(*) FROM metrics").fetchone()[0], 0)

    def test_factory_reset_clears_business_tables_recreates_dirs_and_preserves_backup(self) -> None:
        with TemporaryDirectory() as tmp:
            workspace, database = _init_workspace(tmp)
            _create_workflow_with_three_jobs(database, workspace)
            conn = sqlite3.connect(database)
            conn.row_factory = sqlite3.Row
            _create_input_set(conn, workspace, "is-reset")
            create_task(
                conn,
                task_id="legacy-reset",
                project="default",
                task_type="static",
                task_root=workspace / "legacy" / "legacy-reset",
                status="committed",
            )
            conn.close()

            preview = preview_factory_reset(database, workspace=workspace)
            result = factory_reset(database, workspace=workspace)

            self.assertTrue(preview.backup_path)
            self.assertTrue(result.ok, result.errors)
            self.assertTrue(result.backup_path.exists())
            self.assertTrue((workspace / "backups").exists())
            self.assertTrue((workspace / "workflows").is_dir())
            self.assertTrue((workspace / "input_sets").is_dir())
            self.assertTrue((workspace / "jobs").is_dir())
            self.assertEqual(list((workspace / "workflows").iterdir()), [])
            self.assertEqual(list((workspace / "input_sets").iterdir()), [])
            with sqlite3.connect(database) as conn:
                for table in ("workflow_jobs", "job_metrics", "jobs", "workflows", "input_sets", "tasks", "metrics"):
                    self.assertEqual(conn.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0], 0, table)

    def test_factory_reset_rejects_live_or_running_job_without_deleting_files(self) -> None:
        with TemporaryDirectory() as tmp:
            workspace, database = _init_workspace(tmp)
            _create_workflow_with_three_jobs(database, workspace, clean_status="running")

            result = factory_reset(database, workspace=workspace)

            self.assertFalse(result.ok)
            self.assertFalse(result.backup_path.exists())
            self.assertTrue((workspace / "workflows" / "wf-1").exists())

    def test_path_safety_rejects_workspace_outside_path(self) -> None:
        with TemporaryDirectory() as tmp:
            workspace = Path(tmp) / "workspace"
            outside = Path(tmp) / "outside"
            workspace.mkdir()
            outside.mkdir()

            with self.assertRaisesRegex(ValueError, "outside workspace"):
                ensure_path_inside_workspace(outside, workspace)

    def test_edit_metadata_updates_name_notes_and_updated_at(self) -> None:
        with TemporaryDirectory() as tmp:
            workspace, database = _init_workspace(tmp)
            conn = sqlite3.connect(database)
            conn.row_factory = sqlite3.Row
            _create_input_set(conn, workspace, "is-edit")
            conn.close()
            create_workflow(
                database,
                workflow_id="wf-edit",
                workflow_type="adsorption",
                name="old workflow",
                root_dir=workspace / "workflows" / "wf-edit",
                notes="old",
            )
            with sqlite3.connect(database) as conn:
                conn.execute("UPDATE workflows SET updated_at = '2000-01-01T00:00:00' WHERE workflow_id = 'wf-edit'")
                conn.execute("UPDATE input_sets SET updated_at = '2000-01-01T00:00:00' WHERE input_set_id = 'is-edit'")
                conn.commit()

            update_workflow_metadata(database, "wf-edit", name="new workflow", notes="new notes")
            update_input_set_metadata(database, "is-edit", name="new input", notes="input notes")

            with sqlite3.connect(database) as conn:
                conn.row_factory = sqlite3.Row
                workflow = conn.execute("SELECT * FROM workflows WHERE workflow_id = 'wf-edit'").fetchone()
                input_set = conn.execute("SELECT * FROM input_sets WHERE input_set_id = 'is-edit'").fetchone()
            self.assertEqual(workflow["name"], "new workflow")
            self.assertEqual(workflow["notes"], "new notes")
            self.assertNotEqual(workflow["updated_at"], "2000-01-01T00:00:00")
            self.assertEqual(input_set["name"], "new input")
            self.assertEqual(input_set["notes"], "input notes")
            self.assertNotEqual(input_set["updated_at"], "2000-01-01T00:00:00")

    def test_delete_workflow_checks_real_process_state_not_only_database_status(self) -> None:
        with TemporaryDirectory() as tmp:
            workspace, database = _init_workspace(tmp)
            _create_workflow_with_three_jobs(database, workspace)

            with patch("vasp_mvp.data_management.find_processes_by_run_dir", return_value=[object()]):
                result = delete_workflow(database, "wf-1", workspace=workspace)

            self.assertFalse(result.ok)
            self.assertIn("process", "\n".join(result.errors))
            self.assertTrue((workspace / "workflows" / "wf-1").exists())


def _init_workspace(tmp: str) -> tuple[Path, Path]:
    workspace = Path(tmp) / "workspace"
    init_db(workspace).close()
    for dirname in ("workflows", "input_sets", "jobs", "backups"):
        (workspace / dirname).mkdir(parents=True, exist_ok=True)
    return workspace, db_path(workspace)


def _create_workflow_with_three_jobs(
    database: Path,
    workspace: Path,
    *,
    clean_status: str = "committed",
) -> None:
    workflow_root = workspace / "workflows" / "wf-1"
    create_workflow(
        database,
        workflow_id="wf-1",
        workflow_type="adsorption",
        name="workflow",
        root_dir=workflow_root,
        status="committed",
    )
    roles = (
        ("clean_slab", "job-clean", "slab_static", clean_status, 1),
        ("molecule_ref", "job-molecule", "molecule_static", "committed", 2),
        ("adsorbed_system", "job-adsorbed", "adsorbed_static", "committed", 3),
    )
    for role, job_id, calculation_type, status, step_order in roles:
        run_dir = workflow_root / role / "run"
        run_dir.mkdir(parents=True, exist_ok=True)
        (run_dir / "marker.txt").write_text(job_id, encoding="utf-8")
        create_job(database, job_id=job_id, calculation_type=calculation_type, status=status, run_dir=run_dir)
        save_job_metrics(database, job_id, Metrics(toten_ev=-1.0))
        bind_job_to_workflow(database, workflow_id="wf-1", job_id=job_id, role=role, step_order=step_order)


def _create_input_set(conn, workspace: Path, input_set_id: str) -> None:
    root_dir = workspace / "input_sets" / input_set_id
    root_dir.mkdir(parents=True, exist_ok=True)
    for filename in ("INCAR", "POSCAR", "KPOINTS", "POTCAR"):
        (root_dir / filename).write_text(f"{filename}\n", encoding="utf-8")
    create_input_set(
        conn,
        input_set_id=input_set_id,
        name=input_set_id,
        source="manual",
        status="validated",
        usable_for_vasp=True,
        root_dir=root_dir,
        incar_path=root_dir / "INCAR",
        poscar_path=root_dir / "POSCAR",
        kpoints_path=root_dir / "KPOINTS",
        potcar_path=root_dir / "POTCAR",
    )


if __name__ == "__main__":
    unittest.main()
