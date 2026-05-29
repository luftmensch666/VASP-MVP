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
from vasp_mvp.data_management import update_workflow_metadata
from vasp_mvp.jobs import create_job
from vasp_mvp.workflows import (
    bind_job_to_workflow,
    create_workflow,
    get_workflow,
    get_workflow_with_jobs,
    list_jobs_for_workflow,
    list_workflow_jobs,
    list_workflows,
    update_workflow_status,
)


class WorkflowsTest(unittest.TestCase):
    def test_create_list_update_and_bind_workflow_jobs(self) -> None:
        with TemporaryDirectory() as tmp:
            workspace = Path(tmp) / "workspace"
            init_db(workspace).close()
            database = db_path(workspace)
            root_dir = workspace / "workflows" / "wf-1"

            workflow = create_workflow(
                database,
                workflow_id="wf-1",
                workflow_type="adsorption",
                name="CeO2 adsorption",
                root_dir=root_dir,
                status="draft",
                method_family="DFT",
                functional="PBE-D3",
                method_notes="inferred from INCAR",
                notes="model test",
            )

            self.assertEqual(workflow.workflow_id, "wf-1")
            self.assertEqual(workflow.workflow_type, "adsorption")
            self.assertEqual(workflow.root_dir, root_dir)
            self.assertEqual(workflow.method_family, "DFT")
            self.assertEqual(workflow.functional, "PBE-D3")
            self.assertEqual(workflow.method_notes, "inferred from INCAR")

            update_workflow_status(database, "wf-1", "committed", notes="ready")
            loaded = get_workflow(database, "wf-1")
            all_workflows = list_workflows(database)
            adsorption_workflows = list_workflows(database, workflow_type="adsorption")
            committed_workflows = list_workflows(database, status="committed")

            self.assertIsNotNone(loaded)
            self.assertEqual(loaded.status, "committed")
            self.assertEqual(loaded.notes, "ready")
            self.assertEqual(len(all_workflows), 1)
            self.assertEqual(len(adsorption_workflows), 1)
            self.assertEqual(len(committed_workflows), 1)

            create_job(
                database,
                job_id="job-clean",
                calculation_type="slab_static",
                run_dir=root_dir / "clean_slab" / "run",
            )
            create_job(
                database,
                job_id="job-mol",
                calculation_type="molecule_static",
                run_dir=root_dir / "molecule_ref" / "run",
            )

            clean_binding = bind_job_to_workflow(
                database,
                workflow_id="wf-1",
                job_id="job-clean",
                role="clean_slab",
                step_order=1,
                required=True,
            )
            duplicate_binding = bind_job_to_workflow(
                database,
                workflow_id="wf-1",
                job_id="job-clean",
                role="clean_slab",
                step_order=1,
                required=True,
            )
            bind_job_to_workflow(
                database,
                workflow_id="wf-1",
                job_id="job-mol",
                role="molecule_ref",
                step_order=2,
                required=True,
            )

            bindings = list_workflow_jobs(database, "wf-1")
            joined = list_jobs_for_workflow(database, "wf-1")
            workflow_with_jobs = get_workflow_with_jobs(database, "wf-1")

            self.assertEqual(clean_binding.workflow_job_id, duplicate_binding.workflow_job_id)
            self.assertEqual(len(bindings), 2)
            self.assertEqual([binding.role for binding in bindings], ["clean_slab", "molecule_ref"])
            self.assertEqual([binding.step_order for binding in bindings], [1, 2])
            self.assertEqual(len(joined), 2)
            self.assertEqual(joined[0][0].role, "clean_slab")
            self.assertEqual(joined[0][1].job_id, "job-clean")
            self.assertIsNotNone(workflow_with_jobs)
            self.assertEqual(workflow_with_jobs[0].workflow_id, "wf-1")
            self.assertEqual(len(workflow_with_jobs[1]), 2)

    def test_workflow_name_required_and_normalized_duplicate_checks(self) -> None:
        with TemporaryDirectory() as tmp:
            workspace = Path(tmp) / "workspace"
            init_db(workspace).close()
            database = db_path(workspace)

            create_workflow(
                database,
                workflow_id="wf-1",
                workflow_type="adsorption",
                name=" NH3 Adsorption ",
                root_dir=workspace / "workflows" / "wf-1",
            )

            with self.assertRaisesRegex(ValueError, "name_duplicate"):
                create_workflow(
                    database,
                    workflow_id="wf-2",
                    workflow_type="adsorption",
                    name="nh3 adsorption",
                    root_dir=workspace / "workflows" / "wf-2",
                )
            create_workflow(
                database,
                workflow_id="wf-2",
                workflow_type="adsorption",
                name="CO Adsorption",
                root_dir=workspace / "workflows" / "wf-2",
            )
            with self.assertRaisesRegex(ValueError, "name_duplicate"):
                update_workflow_metadata(database, "wf-2", name=" nh3 adsorption ")
            with self.assertRaisesRegex(ValueError, "name_required"):
                create_workflow(
                    database,
                    workflow_id="wf-empty",
                    workflow_type="adsorption",
                    name="  ",
                    root_dir=workspace / "workflows" / "wf-empty",
                )

    def test_workflow_notes_edit_does_not_overwrite_method_notes(self) -> None:
        with TemporaryDirectory() as tmp:
            workspace = Path(tmp) / "workspace"
            init_db(workspace).close()
            database = db_path(workspace)
            create_workflow(
                database,
                workflow_id="wf-notes",
                workflow_type="adsorption",
                name="notes workflow",
                root_dir=workspace / "workflows" / "wf-notes",
                method_notes="PBE-D3 method description",
                notes="ordinary note",
            )

            update_workflow_metadata(database, "wf-notes", notes="edited ordinary note")
            loaded = get_workflow(database, "wf-notes")

            self.assertEqual(loaded.notes, "edited ordinary note")
            self.assertEqual(loaded.method_notes, "PBE-D3 method description")


if __name__ == "__main__":
    unittest.main()
