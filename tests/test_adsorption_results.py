from __future__ import annotations

import sys
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from vasp_mvp.adsorption_results import (
    calculate_adsorption_energy,
    is_outcar_final_toten,
    normalize_energy_label,
    normalize_energy_source,
    parse_adsorption_workflow_jobs,
)
from vasp_mvp.db import db_path, init_db
from vasp_mvp.i18n import load_translations
from vasp_mvp.jobs import create_job, save_job_metrics
from vasp_mvp.models import Metrics
from vasp_mvp.workflows import bind_job_to_workflow, create_workflow


class AdsorptionResultsTest(unittest.TestCase):
    def test_adsorption_warning_i18n_keys_exist(self) -> None:
        required = {
            "adsorption.warning.missing_metrics",
            "adsorption.warning.invalid_energy_source",
            "adsorption.warning.missing_outcar",
            "adsorption.warning.missing_toten",
            "adsorption.warning.non_static_calculation",
            "adsorption.warning.missing_required_energy",
        }
        self.assertFalse(required - set(load_translations("zh")))
        self.assertFalse(required - set(load_translations("en")))

    def test_energy_source_and_label_normalization(self) -> None:
        self.assertEqual(normalize_energy_source(" OUTCAR "), "outcar")
        self.assertEqual(normalize_energy_label("final TOTEN"), "final_toten")
        self.assertEqual(normalize_energy_label("final  TOTEN"), "final_toten")
        self.assertEqual(normalize_energy_label("final_toten"), "final_toten")
        self.assertEqual(normalize_energy_label("final-toten"), "final_toten")
        self.assertEqual(normalize_energy_label("final\tTOTEN"), "final_toten")
        self.assertEqual(normalize_energy_label("final\nTOTEN"), "final_toten")

    def test_outcar_final_toten_predicate_accepts_equivalent_forms(self) -> None:
        self.assertTrue(is_outcar_final_toten("OUTCAR", "final TOTEN"))
        self.assertTrue(is_outcar_final_toten(" outcar ", " final toten "))
        self.assertTrue(is_outcar_final_toten("Outcar", "final_toten"))
        self.assertTrue(is_outcar_final_toten("OUTCAR", "final-toten"))
        self.assertFalse(is_outcar_final_toten("OSZICAR", "final TOTEN"))

    def test_parse_and_calculate_adsorption_energy_from_final_outcar_toten(self) -> None:
        with TemporaryDirectory() as tmp:
            database, workflow_root = _create_adsorption_workflow(Path(tmp))
            _write_outputs(workflow_root, "clean_slab", -598.0)
            _write_outputs(workflow_root, "molecule_ref", -14.8)
            _write_outputs(workflow_root, "adsorbed_system", -612.5)

            parsed = parse_adsorption_workflow_jobs(database, "ads-1")
            result = calculate_adsorption_energy(database, "ads-1")

            self.assertIsNone(parsed["clean_slab"]["warning"])
            self.assertTrue(result.ready)
            self.assertAlmostEqual(result.e_ads or 0.0, 0.3)
            self.assertEqual(result.energy_source, "OUTCAR")
            self.assertEqual(result.energy_label, "final TOTEN")
            self.assertEqual(result.formula_symbolic, "E_ads = E_adsorbed_system - E_clean_slab - E_molecule_ref")
            self.assertIn("-612.500000", result.formula_numeric or "")

    def test_missing_molecule_ref_outcar_makes_result_not_ready(self) -> None:
        with TemporaryDirectory() as tmp:
            database, workflow_root = _create_adsorption_workflow(Path(tmp))
            _write_outputs(workflow_root, "clean_slab", -598.0)
            _write_outputs(workflow_root, "adsorbed_system", -612.5)

            parse_adsorption_workflow_jobs(database, "ads-1")
            result = calculate_adsorption_energy(database, "ads-1")

            self.assertFalse(result.ready)
            self.assertIsNone(result.e_ads)
            self.assertIn("molecule_ref", _warning_text(result))
            self.assertIn("missing_required_energy", _warning_codes(result))

    def test_outcar_without_toten_does_not_crash_and_is_not_ready(self) -> None:
        with TemporaryDirectory() as tmp:
            database, workflow_root = _create_adsorption_workflow(Path(tmp))
            _write_outputs(workflow_root, "clean_slab", -598.0)
            _write_outputs(workflow_root, "molecule_ref", -14.8)
            role_dir = workflow_root / "adsorbed_system" / "run"
            role_dir.mkdir(parents=True, exist_ok=True)
            (role_dir / "OUTCAR").write_text("LOOP:  cpu time   1.00: real time   2.0\n", encoding="utf-8")
            (role_dir / "OSZICAR").write_text(" 1 F= -.10000000E+02\n", encoding="utf-8")

            parse_adsorption_workflow_jobs(database, "ads-1")
            result = calculate_adsorption_energy(database, "ads-1")

            self.assertFalse(result.ready)
            self.assertIsNone(result.e_ads)
            self.assertIn("adsorbed_system", _warning_text(result))
            self.assertIn("final TOTEN was not parsed", _warning_text(result))
            self.assertNotIn("Energy source is not OUTCAR final TOTEN", _warning_text(result))

    def test_method_metadata_is_returned(self) -> None:
        with TemporaryDirectory() as tmp:
            database, workflow_root = _create_adsorption_workflow(Path(tmp), method_family="DFT", functional="PBE-D3")
            _write_outputs(workflow_root, "clean_slab", -598.0)
            _write_outputs(workflow_root, "molecule_ref", -14.8)
            _write_outputs(workflow_root, "adsorbed_system", -612.5)

            parse_adsorption_workflow_jobs(database, "ads-1")
            result = calculate_adsorption_energy(database, "ads-1")

            self.assertEqual(result.method_family, "DFT")
            self.assertEqual(result.functional, "PBE-D3")

    def test_non_outcar_final_toten_source_is_rejected(self) -> None:
        with TemporaryDirectory() as tmp:
            database, workflow_root = _create_adsorption_workflow(Path(tmp))
            _write_outputs(workflow_root, "clean_slab", -598.0)
            _write_outputs(workflow_root, "molecule_ref", -14.8)
            _write_outputs(workflow_root, "adsorbed_system", -612.5)
            parse_adsorption_workflow_jobs(database, "ads-1")
            save_job_metrics(
                database,
                "job-molecule-ref",
                Metrics(toten_ev=-14.8),
                energy_source="manual",
                energy_label="manual energy",
            )

            result = calculate_adsorption_energy(database, "ads-1")

            self.assertFalse(result.ready)
            self.assertIsNone(result.e_ads)
            self.assertIn("molecule_ref", _warning_text(result))
            self.assertIn("source='manual'", _warning_text(result))
            self.assertIn("label='manual energy'", _warning_text(result))

    def test_outcar_final_toten_with_hidden_whitespace_is_accepted(self) -> None:
        with TemporaryDirectory() as tmp:
            database, workflow_root = _create_adsorption_workflow(Path(tmp))
            _write_outputs(workflow_root, "clean_slab", -598.0)
            _write_outputs(workflow_root, "molecule_ref", -14.8)
            _write_outputs(workflow_root, "adsorbed_system", -612.5)
            parse_adsorption_workflow_jobs(database, "ads-1")
            for job_id, energy in (
                ("job-clean-slab", -598.0),
                ("job-molecule-ref", -14.8),
                ("job-adsorbed-system", -612.5),
            ):
                save_job_metrics(
                    database,
                    job_id,
                    Metrics(toten_ev=energy),
                    energy_source="OUTCAR ",
                    energy_label="final TOTEN\n",
                )

            result = calculate_adsorption_energy(database, "ads-1")

            self.assertTrue(result.ready)
            self.assertAlmostEqual(result.e_ads or 0.0, 0.3)
            self.assertNotIn("energy source is not OUTCAR final TOTEN", _warning_text(result))

    def test_outcar_final_toten_source_with_none_energy_reports_missing_toten_not_invalid_source(self) -> None:
        with TemporaryDirectory() as tmp:
            database, workflow_root = _create_adsorption_workflow(Path(tmp))
            _write_outputs(workflow_root, "clean_slab", -598.0)
            _write_outputs(workflow_root, "molecule_ref", -14.8)
            _write_outputs(workflow_root, "adsorbed_system", -612.5)
            parse_adsorption_workflow_jobs(database, "ads-1")
            save_job_metrics(
                database,
                "job-clean-slab",
                Metrics(toten_ev=None),
                energy_source="OUTCAR",
                energy_label="final TOTEN",
            )

            result = calculate_adsorption_energy(database, "ads-1")
            clean_summary = next(item for item in result.role_summaries if item.role == "clean_slab")

            self.assertFalse(result.ready)
            self.assertIsNone(result.e_ads)
            self.assertIn("missing_toten", clean_summary.warning_types)
            self.assertIn("missing_toten", _warning_codes(result))
            self.assertIn("final TOTEN was not parsed", _warning_text(result))
            self.assertNotIn("Energy source is not OUTCAR final TOTEN", _warning_text(result))

    def test_missing_outcar_is_classified_separately_from_missing_toten(self) -> None:
        with TemporaryDirectory() as tmp:
            database, workflow_root = _create_adsorption_workflow(Path(tmp))
            _write_outputs(workflow_root, "clean_slab", -598.0)
            _write_outputs(workflow_root, "adsorbed_system", -612.5)
            parse_adsorption_workflow_jobs(database, "ads-1")

            result = calculate_adsorption_energy(database, "ads-1")
            molecule_summary = next(item for item in result.role_summaries if item.role == "molecule_ref")

            self.assertFalse(result.ready)
            self.assertIn("missing_outcar", molecule_summary.warning_types)
            self.assertIn("OUTCAR file is missing", _warning_text(result))
            self.assertNotIn("Energy source is not OUTCAR final TOTEN", _warning_text(result))

    def test_invalid_source_is_classified_as_invalid_energy_source(self) -> None:
        with TemporaryDirectory() as tmp:
            database, workflow_root = _create_adsorption_workflow(Path(tmp))
            _write_outputs(workflow_root, "clean_slab", -598.0)
            _write_outputs(workflow_root, "molecule_ref", -14.8)
            _write_outputs(workflow_root, "adsorbed_system", -612.5)
            parse_adsorption_workflow_jobs(database, "ads-1")
            save_job_metrics(
                database,
                "job-clean-slab",
                Metrics(toten_ev=-598.0),
                energy_source="OSZICAR",
                energy_label="final TOTEN",
            )

            result = calculate_adsorption_energy(database, "ads-1")
            clean_summary = next(item for item in result.role_summaries if item.role == "clean_slab")

            self.assertFalse(result.ready)
            self.assertIn("invalid_energy_source", clean_summary.warning_types)
            self.assertIn("invalid_energy_source", _warning_codes(result))
            self.assertIn("Energy source is not OUTCAR final TOTEN", _warning_text(result))

    def test_non_static_warning_does_not_hide_missing_toten_warning(self) -> None:
        with TemporaryDirectory() as tmp:
            database, workflow_root = _create_adsorption_workflow(Path(tmp), clean_type="relax")
            _write_outputs(workflow_root, "clean_slab", -598.0)
            _write_outputs(workflow_root, "molecule_ref", -14.8)
            _write_outputs(workflow_root, "adsorbed_system", -612.5)
            parse_adsorption_workflow_jobs(database, "ads-1")
            save_job_metrics(
                database,
                "job-clean-slab",
                Metrics(toten_ev=None),
                energy_source="OUTCAR",
                energy_label="final TOTEN",
            )

            result = calculate_adsorption_energy(database, "ads-1")
            clean_summary = next(item for item in result.role_summaries if item.role == "clean_slab")

            self.assertIn("missing_toten", clean_summary.warning_types)
            self.assertIn("non_static_calculation", clean_summary.warning_types)
            self.assertLess(clean_summary.warning_types.index("missing_toten"), clean_summary.warning_types.index("non_static_calculation"))

    def test_non_static_calculation_type_adds_warning_without_crashing(self) -> None:
        with TemporaryDirectory() as tmp:
            database, workflow_root = _create_adsorption_workflow(Path(tmp), clean_type="relax")
            _write_outputs(workflow_root, "clean_slab", -598.0)
            _write_outputs(workflow_root, "molecule_ref", -14.8)
            _write_outputs(workflow_root, "adsorbed_system", -612.5)

            parse_adsorption_workflow_jobs(database, "ads-1")
            result = calculate_adsorption_energy(database, "ads-1")

            self.assertTrue(result.ready)
            self.assertIn("clean_slab", _warning_text(result))
            self.assertIn("static single-point", _warning_text(result))

    def test_potcar_content_is_not_read_or_returned(self) -> None:
        with TemporaryDirectory() as tmp:
            database, workflow_root = _create_adsorption_workflow(Path(tmp))
            _write_outputs(workflow_root, "clean_slab", -598.0)
            _write_outputs(workflow_root, "molecule_ref", -14.8)
            _write_outputs(workflow_root, "adsorbed_system", -612.5)
            for role in ("clean_slab", "molecule_ref", "adsorbed_system"):
                (workflow_root / role / "run" / "POTCAR").write_text("SECRET_POTCAR_BODY_DO_NOT_READ\n", encoding="utf-8")

            parse_adsorption_workflow_jobs(database, "ads-1")
            result = calculate_adsorption_energy(database, "ads-1")

            self.assertTrue(result.ready)
            self.assertNotIn("SECRET_POTCAR_BODY_DO_NOT_READ", repr(result))


def _create_adsorption_workflow(
    tmp: Path,
    *,
    method_family: str = "DFT",
    functional: str = "PBE-D3",
    clean_type: str = "slab_static",
) -> tuple[Path, Path]:
    workspace = tmp / "workspace"
    init_db(workspace).close()
    database = db_path(workspace)
    workflow_root = workspace / "workflows" / "ads-1"
    create_workflow(
        database,
        workflow_id="ads-1",
        workflow_type="adsorption",
        name="adsorption test",
        root_dir=workflow_root,
        method_family=method_family,
        functional=functional,
        method_notes="unit test method",
    )
    role_specs = {
        "clean_slab": ("job-clean-slab", clean_type, 1),
        "molecule_ref": ("job-molecule-ref", "molecule_static", 2),
        "adsorbed_system": ("job-adsorbed-system", "adsorbed_static", 3),
    }
    for role, (job_id, calculation_type, step_order) in role_specs.items():
        run_dir = workflow_root / role / "run"
        run_dir.mkdir(parents=True, exist_ok=True)
        create_job(database, job_id=job_id, calculation_type=calculation_type, run_dir=run_dir)
        bind_job_to_workflow(
            database,
            workflow_id="ads-1",
            job_id=job_id,
            role=role,
            step_order=step_order,
        )
    return database, workflow_root


def _write_outputs(workflow_root: Path, role: str, final_toten: float) -> None:
    run_dir = workflow_root / role / "run"
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "OUTCAR").write_text(
        f"""
          free  energy   TOTEN  =       {final_toten + 0.5:.6f} eV
        LOOP:  cpu time   1.00: real time   2.0
          free  energy   TOTEN  =       {final_toten:.6f} eV
        LOOP:  cpu time   2.00: real time   4.0
        reached required accuracy - stopping structural energy minimisation
        """,
        encoding="utf-8",
    )
    (run_dir / "OSZICAR").write_text(
        f" 1 F= {final_toten + 0.5:.8E} E0= {final_toten + 0.5:.8E}\n"
        f" 2 F= {final_toten:.8E} E0= {final_toten:.8E}\n",
        encoding="utf-8",
    )


def _warning_text(result) -> str:
    return " ".join(warning.message for warning in result.warnings)


def _warning_codes(result) -> set[str]:
    return {warning.code for warning in result.warnings}


if __name__ == "__main__":
    unittest.main()
