from __future__ import annotations

import json
import sys
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from vasp_mvp.adsorption_wizard import (
    RELAX_ROLES,
    WIZARD_STEPS,
    adopt_clean_poscar_candidate,
    adopt_candidate_poscar,
    artifact_path,
    default_wizard_state,
    list_clean_poscar_candidates,
    load_wizard_state,
    poscar_summary,
    require_relax_contcars,
    save_candidate_file,
    save_candidate_poscar,
    save_wizard_state,
    static_roles_for_final_eads,
    sync_clean_poscar_to_structure,
)
from vasp_mvp.db import db_path, init_db
from vasp_mvp.jobs import create_job
from vasp_mvp.workflows import bind_job_to_workflow, create_workflow, list_jobs_for_workflow


POSCAR_TEXT = """CeO2 slab
1.0
5.0 0.0 0.0
0.0 5.0 0.0
0.0 0.0 20.0
Ce O
1 2
Direct
0.0 0.0 0.0
0.5 0.5 0.5
0.25 0.25 0.25
"""


class AdsorptionWizardTest(unittest.TestCase):
    def test_state_paths_are_relative_and_candidate_adoption_updates_clean_artifact(self) -> None:
        with TemporaryDirectory() as tmp:
            root = Path(tmp) / "workflow"
            source = Path(tmp) / "POSCAR"
            source.write_text(POSCAR_TEXT, encoding="utf-8")
            state = default_wizard_state("wf_ads")

            state = save_candidate_poscar(root, state, "clean", source, "candidate_POSCAR")
            self.assertEqual(state["candidates"]["clean"], "artifacts/clean/candidates/candidate_POSCAR")
            self.assertFalse(Path(state["candidates"]["clean"]).is_absolute())

            state = adopt_candidate_poscar(root, state, "clean")
            self.assertEqual(state["artifacts"]["clean_poscar"], "artifacts/clean/POSCAR")
            self.assertFalse(Path(state["artifacts"]["clean_poscar"]).is_absolute())
            self.assertTrue((root / "artifacts" / "clean" / "POSCAR").exists())

            loaded = load_wizard_state(root, "wf_ads")
            self.assertEqual(loaded["steps"]["clean_structure"], "done")
            raw_state = json.loads((root / "workflow_state.json").read_text(encoding="utf-8"))
            self.assertFalse(str(root) in json.dumps(raw_state))
            self.assertEqual(raw_state["clean_poscar_source"]["candidate"], "artifacts/clean/candidates/candidate_POSCAR")
            self.assertEqual(raw_state["clean_poscar_source"]["step"], "unknown")

    def test_candidate_records_source_step_parameters_and_backup_on_adopt(self) -> None:
        with TemporaryDirectory() as tmp:
            root = Path(tmp) / "workflow"
            first = Path(tmp) / "POSCAR.first"
            second = Path(tmp) / "POSCAR.second"
            first.write_text(POSCAR_TEXT, encoding="utf-8")
            second.write_text(POSCAR_TEXT.replace("CeO2 slab", "CeO2 slab updated"), encoding="utf-8")
            state = default_wizard_state("wf_ads")

            state = save_candidate_file(
                root,
                state,
                role="clean",
                source_path=first,
                candidate_name="POSCAR_105.vasp",
                source_step="105",
                parameters={"cif_filename": "clean.cif", "element_order": "Ce O"},
            )
            state = adopt_clean_poscar_candidate(root, state)
            state = save_candidate_file(
                root,
                state,
                role="clean",
                source_path=second,
                candidate_name="POSCAR_REV.vasp",
                source_step="801",
                parameters={"direction_index": 3, "vacuum_thickness": 15.0},
            )
            state = adopt_clean_poscar_candidate(root, state)

            source_info = state["clean_poscar_source"]
            self.assertEqual(source_info["step"], "801")
            self.assertEqual(source_info["candidate"], "artifacts/clean/candidates/POSCAR_REV.vasp")
            self.assertEqual(source_info["parameters"]["direction_index"], 3)
            self.assertIn("previous_backup", source_info)
            self.assertFalse(Path(source_info["previous_backup"]).is_absolute())
            self.assertTrue((root / source_info["previous_backup"]).exists())
            self.assertGreaterEqual(len(list_clean_poscar_candidates(root)), 2)

    def test_sync_clean_poscar_to_structure_requires_adopted_clean_poscar(self) -> None:
        with TemporaryDirectory() as tmp:
            root = Path(tmp) / "workflow"
            source = Path(tmp) / "POSCAR"
            source.write_text(POSCAR_TEXT, encoding="utf-8")
            state = default_wizard_state("wf_ads")

            with self.assertRaises(FileNotFoundError):
                sync_clean_poscar_to_structure(root, state)

            state = save_candidate_poscar(root, state, "clean", source, "candidate_POSCAR")
            state = adopt_candidate_poscar(root, state, "clean")
            synced = sync_clean_poscar_to_structure(root, state)

            self.assertEqual(synced, root / "structure" / "POSCAR")
            self.assertTrue(synced.exists())

    def test_poscar_summary_reuses_vasp5_element_order_parser(self) -> None:
        with TemporaryDirectory() as tmp:
            path = Path(tmp) / "POSCAR"
            path.write_text(POSCAR_TEXT, encoding="utf-8")

            summary = poscar_summary(path)

            self.assertTrue(summary.exists)
            self.assertEqual(summary.element_order, ("Ce", "O"))
            self.assertEqual(summary.atom_counts, (1, 2))
            self.assertEqual(summary.total_atoms, 3)
            self.assertEqual(summary.cell_lengths, (5.0, 5.0, 20.0))

    def test_step_6_requires_relax_contcars(self) -> None:
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            for role in RELAX_ROLES[:2]:
                run_dir = root / role / "run"
                run_dir.mkdir(parents=True)
                (run_dir / "CONTCAR").write_text(POSCAR_TEXT, encoding="utf-8")

            ok, missing = require_relax_contcars(root)

            self.assertFalse(ok)
            self.assertEqual(missing, ("adsorbed_relax/run/CONTCAR",))

    def test_workflow_roles_support_relax_and_static_wizard_jobs(self) -> None:
        with TemporaryDirectory() as tmp:
            workspace = Path(tmp)
            conn = init_db(workspace)
            conn.close()
            database = db_path(workspace)
            workflow = create_workflow(
                database,
                workflow_id="wf_ads",
                workflow_type="adsorption",
                name="Adsorption Wizard",
                root_dir=workspace / "workflows" / "wf_ads",
                status="draft",
                method_family="DFT",
                functional="PBE-D3",
                method_notes='{"zh": "方法说明", "en": "Method notes"}',
                notes="ordinary notes",
            )
            roles = (
                "clean_relax",
                "molecule_relax",
                "adsorbed_relax",
                "clean_static",
                "molecule_static",
                "adsorbed_static",
            )
            for index, role in enumerate(roles, start=1):
                job = create_job(
                    database,
                    job_id=f"job_{role}",
                    calculation_type="relax" if role.endswith("relax") else "static",
                    run_dir=workspace / "workflows" / "wf_ads" / role / "run",
                )
                bind_job_to_workflow(
                    database,
                    workflow_id=workflow.workflow_id,
                    job_id=job.job_id,
                    role=role,
                    step_order=index,
                )

            records = list_jobs_for_workflow(database, workflow.workflow_id)
            observed_roles = [binding.role for binding, _job in records]
            observed_orders = [binding.step_order for binding, _job in records]

            self.assertEqual(observed_roles, list(roles))
            self.assertEqual(observed_orders, [1, 2, 3, 4, 5, 6])
            for _binding, job in records:
                self.assertFalse(hasattr(job, "workflow_id"))
                self.assertFalse(hasattr(job, "role"))

    def test_final_eads_roles_are_static_only(self) -> None:
        self.assertEqual(static_roles_for_final_eads(), ("clean_static", "molecule_static", "adsorbed_static"))
        self.assertNotIn("clean_relax", static_roles_for_final_eads())
        self.assertIn("calculate_eads", WIZARD_STEPS)

    def test_artifact_path_rejects_absolute_paths(self) -> None:
        with TemporaryDirectory() as tmp:
            with self.assertRaises(ValueError):
                artifact_path(Path(tmp), "/tmp/POSCAR")


if __name__ == "__main__":
    unittest.main()
