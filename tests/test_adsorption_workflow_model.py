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
from vasp_mvp.jobs import create_job
from vasp_mvp.workflows import bind_job_to_workflow, create_workflow, list_jobs_for_workflow


class AdsorptionWorkflowModelTest(unittest.TestCase):
    def test_adsorption_workflow_binds_three_reusable_jobs_with_ordered_roles(self) -> None:
        with TemporaryDirectory() as tmp:
            workspace = Path(tmp) / "workspace"
            init_db(workspace).close()
            database = db_path(workspace)
            workflow_root = workspace / "workflows" / "ads-1"

            create_workflow(
                database,
                workflow_id="ads-1",
                workflow_type="adsorption",
                name="adsorption model",
                root_dir=workflow_root,
                method_family="DFT",
                functional="PBE-D3",
                method_notes="inferred from INCAR",
            )
            clean_job = create_job(
                database,
                job_id="job-clean-slab",
                calculation_type="slab_static",
                run_dir=workflow_root / "clean_slab" / "run",
            )
            mol_job = create_job(
                database,
                job_id="job-molecule-ref",
                calculation_type="molecule_static",
                run_dir=workflow_root / "molecule_ref" / "run",
            )
            ads_job = create_job(
                database,
                job_id="job-adsorbed-system",
                calculation_type="adsorbed_static",
                run_dir=workflow_root / "adsorbed_system" / "run",
            )

            bind_job_to_workflow(
                database,
                workflow_id="ads-1",
                job_id=clean_job.job_id,
                role="clean_slab",
                step_order=1,
            )
            bind_job_to_workflow(
                database,
                workflow_id="ads-1",
                job_id=mol_job.job_id,
                role="molecule_ref",
                step_order=2,
            )
            bind_job_to_workflow(
                database,
                workflow_id="ads-1",
                job_id=ads_job.job_id,
                role="adsorbed_system",
                step_order=3,
            )

            workflow_jobs = list_jobs_for_workflow(database, "ads-1")
            roles = [binding.role for binding, _job in workflow_jobs]
            step_orders = [binding.step_order for binding, _job in workflow_jobs]
            jobs = [job for _binding, job in workflow_jobs]

            self.assertEqual(roles, ["clean_slab", "molecule_ref", "adsorbed_system"])
            self.assertEqual(step_orders, [1, 2, 3])
            self.assertEqual({job.calculation_type for job in jobs}, {"slab_static", "molecule_static", "adsorbed_static"})

            for job in jobs:
                self.assertFalse(hasattr(job, "workflow_id"))
                self.assertFalse(hasattr(job, "role"))


if __name__ == "__main__":
    unittest.main()
