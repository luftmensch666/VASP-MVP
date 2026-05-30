from __future__ import annotations

import json
import sys
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import Mock, patch

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from vasp_mvp.adsorption_wizard import default_wizard_state, save_wizard_state
from vasp_mvp.adsorption_wizard_relax import (
    check_relax_source_poscars,
    create_relax_jobs_from_inputs,
    generate_all_relax_input_packages,
    generate_relax_input_package,
)
from vasp_mvp.db import db_path, init_db
from vasp_mvp.workflows import create_workflow, list_jobs_for_workflow


POSCAR = """CeO2
1.0
5 0 0
0 5 0
0 0 20
Ce O
1 2
Direct
0 0 0
0.5 0.5 0.5
0.25 0.25 0.25
"""

POTCAR = """TITEL = PAW_PBE Ce 08Apr2002
VRHFIN =Ce: f1 d1 s2
   ENMAX = 273.000; ENMIN = 200.000
   ZVAL = 12.000
TITEL = PAW_PBE O 08Apr2002
VRHFIN =O: s2 p4
   ENMAX = 400.000; ENMIN = 300.000
   ZVAL = 6.000
"""


class AdsorptionWizardRelaxInputsTest(unittest.TestCase):
    def test_step4_not_ready_when_source_poscars_are_missing(self) -> None:
        with TemporaryDirectory() as tmp:
            root = Path(tmp) / "workflow"
            state = default_wizard_state("wf")
            ok, _sources, missing = check_relax_source_poscars(root, state)

            self.assertFalse(ok)
            self.assertEqual(set(missing), {"clean_relax", "molecule_relax", "adsorbed_relax"})

    def test_generate_all_relax_inputs_and_update_state_with_relative_paths(self) -> None:
        with TemporaryDirectory() as tmp:
            root, state = _prepared_workflow_root(Path(tmp))

            with patch("vasp_mvp.adsorption_wizard_relax.subprocess.run", side_effect=_fake_vaspkit_success):
                results = generate_all_relax_input_packages(
                    root,
                    state,
                    vaspkit_bin="vaspkit",
                    incar_key="SR",
                    kpoints_scheme=2,
                    kmesh_value=0.04,
                )

            self.assertTrue(all(result.ok for result in results))
            for role in ("clean_relax", "molecule_relax", "adsorbed_relax"):
                input_dir = root / "inputs" / role
                for filename in ("INCAR", "POSCAR", "KPOINTS", "POTCAR", "generation_result.json"):
                    self.assertTrue((input_dir / filename).exists(), f"{role}/{filename}")
            self.assertIn("Gamma-only", (root / "inputs" / "molecule_relax" / "KPOINTS").read_text(encoding="utf-8"))
            self.assertIn("Gamma", (root / "inputs" / "clean_relax" / "KPOINTS").read_text(encoding="utf-8"))

            raw_state = json.loads((root / "workflow_state.json").read_text(encoding="utf-8"))
            self.assertIn("relax_inputs", raw_state)
            self.assertEqual(raw_state["relax_inputs"]["clean_relax"]["input_dir"], "inputs/clean_relax")
            self.assertFalse(str(root) in json.dumps(raw_state))

    def test_failure_removes_staging_and_preserves_existing_inputs(self) -> None:
        with TemporaryDirectory() as tmp:
            root, state = _prepared_workflow_root(Path(tmp))
            old_dir = root / "inputs" / "clean_relax"
            old_dir.mkdir(parents=True)
            (old_dir / "INCAR").write_text("OLD INCAR\n", encoding="utf-8")

            with patch("vasp_mvp.adsorption_wizard_relax.subprocess.run", side_effect=_fake_vaspkit_fail_102):
                result = generate_relax_input_package(
                    root,
                    "clean_relax",
                    root / "artifacts" / "clean" / "POSCAR",
                    vaspkit_bin="vaspkit",
                    incar_key="SR",
                    kpoints_mode="2",
                    kmesh_value=0.04,
                )

            self.assertFalse(result.ok)
            self.assertEqual((old_dir / "INCAR").read_text(encoding="utf-8"), "OLD INCAR\n")
            self.assertFalse(any(path.name.startswith(".staging_") for path in old_dir.iterdir()))
            self.assertTrue((root / "inputs" / "clean_relax_generation_result.json").exists())
            self.assertIn("===== VASPKIT 102 KPOINTS =====", (root / "logs" / "relax_input_clean_relax.out").read_text(encoding="utf-8"))

    def test_create_relax_jobs_from_inputs_is_idempotent(self) -> None:
        with TemporaryDirectory() as tmp:
            workspace = Path(tmp)
            root, state = _prepared_workflow_root(workspace)
            conn = init_db(workspace)
            conn.close()
            database = db_path(workspace)
            workflow = create_workflow(
                database,
                workflow_id="wf",
                workflow_type="adsorption",
                name="wf relax",
                status="draft",
                root_dir=root,
            )
            with patch("vasp_mvp.adsorption_wizard_relax.subprocess.run", side_effect=_fake_vaspkit_success):
                generate_all_relax_input_packages(root, state, vaspkit_bin="vaspkit")

            first = create_relax_jobs_from_inputs(database, workflow.workflow_id, root, mpi_ranks=20, vasp_bin="vasp_std")
            second = create_relax_jobs_from_inputs(database, workflow.workflow_id, root, mpi_ranks=20, vasp_bin="vasp_std")

            self.assertEqual([item["status"] for item in first], ["created", "created", "created"])
            self.assertEqual([item["status"] for item in second], ["exists", "exists", "exists"])
            records = list_jobs_for_workflow(database, workflow.workflow_id)
            self.assertEqual(len(records), 3)
            self.assertEqual([binding.role for binding, _job in records], ["clean_relax", "molecule_relax", "adsorbed_relax"])
            self.assertEqual([binding.step_order for binding, _job in records], [1, 2, 3])
            for _binding, job in records:
                for filename in ("INCAR", "POSCAR", "KPOINTS", "POTCAR"):
                    self.assertTrue((job.run_dir / filename).exists())


def _prepared_workflow_root(base: Path) -> tuple[Path, dict]:
    root = base / "workflow"
    artifacts = {
        "clean": "clean_poscar",
        "adsorbed": "adsorbed_poscar",
        "molecule": "molecule_poscar",
    }
    state = default_wizard_state("wf")
    for directory, key in artifacts.items():
        path = root / "artifacts" / directory / "POSCAR"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(POSCAR, encoding="utf-8")
        state["artifacts"][key] = f"artifacts/{directory}/POSCAR"
    save_wizard_state(root, state)
    return root, state


def _fake_vaspkit_success(*args, **kwargs):
    cwd = Path(kwargs["cwd"])
    command = kwargs["input"].splitlines()[0].strip()
    if command == "101":
        (cwd / "INCAR").write_text("SYSTEM = SR\nENCUT = 520\nISPIN = 1\nNSW = 60\n", encoding="utf-8")
    elif command == "102":
        (cwd / "KPOINTS").write_text("Automatic mesh\n0\nGamma\n3 3 1\n0 0 0\n", encoding="utf-8")
    elif command == "103":
        (cwd / "POTCAR").write_text(POTCAR, encoding="utf-8")
    completed = Mock()
    completed.returncode = 0
    completed.stdout = f"ok {command}\n"
    completed.stderr = ""
    return completed


def _fake_vaspkit_fail_102(*args, **kwargs):
    cwd = Path(kwargs["cwd"])
    command = kwargs["input"].splitlines()[0].strip()
    completed = Mock()
    completed.stdout = f"stdout {command}\n"
    completed.stderr = f"stderr {command}\n"
    if command == "101":
        (cwd / "INCAR").write_text("SYSTEM = SR\nENCUT = 520\n", encoding="utf-8")
        completed.returncode = 0
    elif command == "102":
        completed.returncode = 2
    else:
        completed.returncode = 0
    return completed


if __name__ == "__main__":
    unittest.main()
