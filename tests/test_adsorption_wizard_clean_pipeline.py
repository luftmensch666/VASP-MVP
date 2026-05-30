from __future__ import annotations

import sys
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from vasp_mvp.adsorption_wizard import default_wizard_state
from vasp_mvp.adsorption_wizard_clean_pipeline import (
    can_run_clean_pipeline_step,
    can_skip_clean_pipeline_step,
    get_clean_pipeline_steps,
    initialize_clean_pipeline,
    mark_clean_pipeline_candidate_adopted,
    mark_clean_pipeline_step_done,
    mark_clean_pipeline_step_failed,
    mark_clean_pipeline_step_skipped,
    reset_clean_pipeline_for_source_type_change,
)


class AdsorptionWizardCleanPipelineTest(unittest.TestCase):
    def test_slab_flow_only_contains_105(self) -> None:
        steps = get_clean_pipeline_steps("slab")
        self.assertEqual([step.key for step in steps], ["cif_105"])
        self.assertTrue(steps[0].required)

    def test_porous_flow_contains_symmetry_and_cell_steps(self) -> None:
        steps = get_clean_pipeline_steps("porous_or_mof")
        self.assertEqual(
            [step.key for step in steps],
            ["cif_105", "symmetry_601", "conventional_603", "supercell_401"],
        )
        self.assertTrue(steps[0].required)
        self.assertFalse(all(step.required for step in steps[1:]))
        self.assertIsNone(steps[1].expected_output)
        self.assertEqual(steps[2].expected_output, "CONVCELL.vasp")
        self.assertNotIn("primitive_602", [step.key for step in steps])

    def test_bulk_flow_uses_non_required_placeholders_for_bulk_optimization(self) -> None:
        steps = get_clean_pipeline_steps("bulk_surface")
        self.assertEqual(
            [step.key for step in steps],
            ["cif_105", "bulk_input", "bulk_relax", "slab_803", "vacuum_801", "supercell_401", "fix_atoms_402", "fix_atoms_403"],
        )
        by_key = {step.key: step for step in steps}
        self.assertFalse(by_key["bulk_input"].required)
        self.assertTrue(by_key["bulk_input"].disabled)
        self.assertFalse(by_key["bulk_relax"].required)
        self.assertTrue(by_key["bulk_relax"].disabled)
        self.assertTrue(by_key["slab_803"].required)
        self.assertTrue(by_key["fix_atoms_403"].disabled)
        self.assertNotIn("symmetry_601", by_key)
        self.assertNotIn("conventional_603", by_key)

    def test_required_step_cannot_skip_and_optional_skip_unlocks_next_step(self) -> None:
        state = initialize_clean_pipeline(default_wizard_state("wf"), "porous_or_mof")
        self.assertFalse(can_skip_clean_pipeline_step(state, "cif_105"))
        with self.assertRaises(ValueError):
            mark_clean_pipeline_step_skipped(state, "cif_105")

        state = mark_clean_pipeline_step_done(state, "cif_105", candidate="artifacts/clean/candidates/POSCAR_105.vasp", adopted=True)
        self.assertTrue(can_skip_clean_pipeline_step(state, "symmetry_601"))
        state = mark_clean_pipeline_step_skipped(state, "symmetry_601")
        self.assertTrue(can_run_clean_pipeline_step(state, "conventional_603"))

    def test_candidate_must_be_adopted_before_next_step_can_run(self) -> None:
        state = initialize_clean_pipeline(default_wizard_state("wf"), "porous_or_mof")
        state = mark_clean_pipeline_step_done(state, "cif_105", candidate="artifacts/clean/candidates/POSCAR_105.vasp", adopted=False)
        self.assertFalse(can_run_clean_pipeline_step(state, "symmetry_601"))
        state = mark_clean_pipeline_candidate_adopted(state, "cif_105")
        self.assertTrue(can_run_clean_pipeline_step(state, "symmetry_601"))

    def test_pending_or_failed_previous_step_locks_next_step(self) -> None:
        state = initialize_clean_pipeline(default_wizard_state("wf"), "porous_or_mof")
        self.assertFalse(can_run_clean_pipeline_step(state, "symmetry_601"))
        state = mark_clean_pipeline_step_failed(state, "cif_105", errors=["failed"])
        self.assertFalse(can_run_clean_pipeline_step(state, "symmetry_601"))

    def test_bulk_803_requires_bulk_contcar(self) -> None:
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            state = initialize_clean_pipeline(default_wizard_state("wf"), "bulk_surface")
            state = mark_clean_pipeline_step_done(state, "cif_105", candidate="artifacts/clean/candidates/POSCAR_105.vasp", adopted=True)
            self.assertFalse(can_run_clean_pipeline_step(state, "slab_803", root))
            contcar = root / "jobs" / "bulk_relax" / "run" / "CONTCAR"
            contcar.parent.mkdir(parents=True)
            contcar.write_text("CONTCAR\n", encoding="utf-8")
            self.assertTrue(can_run_clean_pipeline_step(state, "slab_803", root))

    def test_source_type_change_resets_pipeline_without_removing_artifacts(self) -> None:
        state = initialize_clean_pipeline(default_wizard_state("wf"), "slab")
        state["artifacts"]["clean_poscar"] = "artifacts/clean/POSCAR"
        state["candidates"]["clean"] = "artifacts/clean/candidates/POSCAR_105.vasp"
        state = mark_clean_pipeline_step_done(state, "cif_105", candidate="artifacts/clean/candidates/POSCAR_105.vasp", adopted=True)

        reset = reset_clean_pipeline_for_source_type_change(state, "porous_or_mof")

        self.assertEqual(reset["clean_pipeline"]["source_type"], "porous_or_mof")
        self.assertEqual(reset["clean_pipeline"]["steps"][0]["status"], "pending")
        self.assertEqual(reset["artifacts"]["clean_poscar"], "artifacts/clean/POSCAR")
        self.assertEqual(reset["candidates"]["clean"], "artifacts/clean/candidates/POSCAR_105.vasp")

    def test_old_porous_pipeline_with_primitive_602_is_migrated_without_crash(self) -> None:
        state = initialize_clean_pipeline(default_wizard_state("wf"), "porous_or_mof")
        state["clean_pipeline"]["steps"].insert(
            2,
            {
                "key": "primitive_602",
                "status": "done",
                "required": False,
                "candidate": "artifacts/clean/candidates/PRIMCELL.vasp",
                "adopted": False,
            },
        )

        from vasp_mvp.adsorption_wizard_clean_pipeline import ensure_clean_pipeline

        migrated = ensure_clean_pipeline(state)
        self.assertEqual(
            [step["key"] for step in migrated["clean_pipeline"]["steps"]],
            ["cif_105", "symmetry_601", "conventional_603", "supercell_401"],
        )


if __name__ == "__main__":
    unittest.main()
