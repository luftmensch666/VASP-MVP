from __future__ import annotations

import sys
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from vasp_mvp.adsorption_workflow import create_adsorption_workflow
from vasp_mvp.db import db_path, init_db
from vasp_mvp.input_sets import create_input_set
from vasp_mvp.workflows import get_workflow, list_jobs_for_workflow


class AdsorptionWorkflowServiceTest(unittest.TestCase):
    def test_create_adsorption_workflow_copies_inputs_and_binds_three_jobs(self) -> None:
        with TemporaryDirectory() as tmp:
            workspace = Path(tmp) / "workspace"
            conn = init_db(workspace)
            database = db_path(workspace)
            clean = _create_fake_input_set(conn, workspace, "is-clean")
            mol = _create_fake_input_set(conn, workspace, "is-mol")
            ads = _create_fake_input_set(conn, workspace, "is-ads")
            conn.close()
            workflow_root = workspace / "workflows" / "ads-wf"

            workflow = create_adsorption_workflow(
                database,
                workflow_id="ads-wf",
                name="adsorption workflow",
                root_dir=workflow_root,
                clean_slab_input_set_id=clean,
                molecule_ref_input_set_id=mol,
                adsorbed_input_set_id=ads,
                functional="PBE-D3",
                method_notes="unit test",
                mpi_ranks=24,
                vasp_bin="/opt/vasp/vasp_std",
            )

            loaded = get_workflow(database, "ads-wf")
            workflow_jobs = list_jobs_for_workflow(database, "ads-wf")
            roles = [binding.role for binding, _job in workflow_jobs]
            step_orders = [binding.step_order for binding, _job in workflow_jobs]
            jobs_by_role = {binding.role: job for binding, job in workflow_jobs}

            self.assertEqual(workflow.workflow_type, "adsorption")
            self.assertEqual(workflow.status, "committed")
            self.assertEqual(workflow.method_family, "DFT")
            self.assertEqual(workflow.functional, "PBE-D3")
            self.assertEqual(workflow.method_notes, "unit test")
            self.assertIsNotNone(loaded)
            self.assertEqual(len(workflow_jobs), 3)
            self.assertEqual(roles, ["clean_slab", "molecule_ref", "adsorbed_system"])
            self.assertEqual(step_orders, [1, 2, 3])
            self.assertEqual(jobs_by_role["clean_slab"].calculation_type, "slab_static")
            self.assertEqual(jobs_by_role["molecule_ref"].calculation_type, "molecule_static")
            self.assertEqual(jobs_by_role["adsorbed_system"].calculation_type, "adsorbed_static")
            self.assertEqual(jobs_by_role["clean_slab"].input_set_id, clean)
            self.assertEqual(jobs_by_role["molecule_ref"].input_set_id, mol)
            self.assertEqual(jobs_by_role["adsorbed_system"].input_set_id, ads)
            self.assertEqual(jobs_by_role["clean_slab"].mpi_ranks, 24)
            self.assertEqual(jobs_by_role["clean_slab"].vasp_bin, "/opt/vasp/vasp_std")

            for role, job in jobs_by_role.items():
                self.assertEqual(job.run_dir, workflow_root / role / "run")
                for filename in ("INCAR", "POSCAR", "KPOINTS", "POTCAR"):
                    self.assertTrue((job.run_dir / filename).exists(), f"{role} missing {filename}")
                self.assertFalse(hasattr(job, "workflow_id"))
                self.assertFalse(hasattr(job, "role"))

    def test_rejects_input_set_that_is_not_usable(self) -> None:
        with TemporaryDirectory() as tmp:
            workspace = Path(tmp) / "workspace"
            conn = init_db(workspace)
            database = db_path(workspace)
            clean = _create_fake_input_set(conn, workspace, "is-clean", usable_for_vasp=False)
            mol = _create_fake_input_set(conn, workspace, "is-mol")
            ads = _create_fake_input_set(conn, workspace, "is-ads")
            conn.close()

            with self.assertRaisesRegex(ValueError, "not usable"):
                create_adsorption_workflow(
                    database,
                    workflow_id="ads-wf",
                    name="bad adsorption workflow",
                    root_dir=workspace / "workflows" / "ads-wf",
                    clean_slab_input_set_id=clean,
                    molecule_ref_input_set_id=mol,
                    adsorbed_input_set_id=ads,
                )

            self.assertIsNone(get_workflow(database, "ads-wf"))

    def test_rejects_input_set_missing_potcar(self) -> None:
        with TemporaryDirectory() as tmp:
            workspace = Path(tmp) / "workspace"
            conn = init_db(workspace)
            database = db_path(workspace)
            clean = _create_fake_input_set(conn, workspace, "is-clean", missing_files=("POTCAR",))
            mol = _create_fake_input_set(conn, workspace, "is-mol")
            ads = _create_fake_input_set(conn, workspace, "is-ads")
            conn.close()

            with self.assertRaisesRegex(FileNotFoundError, "POTCAR"):
                create_adsorption_workflow(
                    database,
                    workflow_id="ads-wf",
                    name="missing POTCAR workflow",
                    root_dir=workspace / "workflows" / "ads-wf",
                    clean_slab_input_set_id=clean,
                    molecule_ref_input_set_id=mol,
                    adsorbed_input_set_id=ads,
                )

            self.assertIsNone(get_workflow(database, "ads-wf"))

    def test_rejects_input_set_with_nonexistent_input_path(self) -> None:
        with TemporaryDirectory() as tmp:
            workspace = Path(tmp) / "workspace"
            conn = init_db(workspace)
            database = db_path(workspace)
            clean = _create_fake_input_set(conn, workspace, "is-clean", missing_files=("INCAR",))
            mol = _create_fake_input_set(conn, workspace, "is-mol")
            ads = _create_fake_input_set(conn, workspace, "is-ads")
            conn.close()

            with self.assertRaisesRegex(FileNotFoundError, "INCAR"):
                create_adsorption_workflow(
                    database,
                    workflow_id="ads-wf",
                    name="missing INCAR workflow",
                    root_dir=workspace / "workflows" / "ads-wf",
                    clean_slab_input_set_id=clean,
                    molecule_ref_input_set_id=mol,
                    adsorbed_input_set_id=ads,
                )

            self.assertIsNone(get_workflow(database, "ads-wf"))


def _create_fake_input_set(
    conn,
    workspace: Path,
    input_set_id: str,
    *,
    usable_for_vasp: bool = True,
    missing_files: tuple[str, ...] = (),
) -> str:
    root_dir = workspace / "input_sets" / input_set_id
    root_dir.mkdir(parents=True)
    contents = {
        "INCAR": "SYSTEM = test\nENCUT = 520\n",
        "POSCAR": "test\n1.0\n1 0 0\n0 1 0\n0 0 1\nH\n1\nDirect\n0 0 0\n",
        "KPOINTS": "Gamma\n0\nGamma\n1 1 1\n0 0 0\n",
        "POTCAR": "TITEL  = PAW_PBE H 01Jan2001\n",
    }
    for filename, text in contents.items():
        if filename not in missing_files:
            (root_dir / filename).write_text(text, encoding="utf-8")
    create_input_set(
        conn,
        input_set_id=input_set_id,
        name=input_set_id,
        source="manual",
        status="validated" if usable_for_vasp else "invalid",
        usable_for_vasp=usable_for_vasp,
        root_dir=root_dir,
        incar_path=root_dir / "INCAR",
        poscar_path=root_dir / "POSCAR",
        kpoints_path=root_dir / "KPOINTS",
        potcar_path=root_dir / "POTCAR",
    )
    return input_set_id


if __name__ == "__main__":
    unittest.main()
