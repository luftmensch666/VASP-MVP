from __future__ import annotations

import json
import sys
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from vasp_mvp.vaspkit_runner import VaspkitRequest, generate_vasp_inputs_with_vaspkit, summarize_potcar


class Completed:
    def __init__(self, returncode: int = 0, stdout: str = "ok", stderr: str = "") -> None:
        self.returncode = returncode
        self.stdout = stdout
        self.stderr = stderr


class VaspkitRunnerFullGenerationTest(unittest.TestCase):
    def _request(self, tmp: str, input_set_id: str = "is-test") -> VaspkitRequest:
        workspace = Path(tmp) / "workspace"
        draft_dir = workspace / "input_sets" / input_set_id
        draft_dir.mkdir(parents=True)
        cif = Path(tmp) / "CePO4.cif"
        cif.write_text("data_test\n", encoding="utf-8")
        return VaspkitRequest(
            vaspkit_bin="vaspkit",
            workspace=workspace,
            draft_dir=draft_dir,
            input_set_id=input_set_id,
            uploaded_cif_path=cif,
        )

    def test_app_and_config_do_not_expose_generation_mode_choice(self) -> None:
        app_text = (ROOT / "app.py").read_text(encoding="utf-8")
        options = json.loads((ROOT / "config" / "vaspkit_options.json").read_text(encoding="utf-8"))

        self.assertNotIn("VASPKIT_GENERATION_MODES", app_text)
        self.assertNotIn("vaspkit.generation_mode.label", app_text)
        configured_keys = [
            option["key"]
            for section in options["sections"].values()
            for option in section.get("options", [])
        ]
        self.assertNotIn("generation_mode", configured_keys)

    def test_dry_run_creates_mock_files_without_real_potcar_or_rollback(self) -> None:
        with TemporaryDirectory() as tmp:
            request = self._request(tmp, "is-dry-run")

            result = generate_vasp_inputs_with_vaspkit(request, dry_run=True)

            self.assertTrue(result.ok)
            self.assertEqual(result.input_set_status, "dry_run")
            self.assertFalse(result.usable_for_vasp)
            self.assertTrue((request.draft_dir / "POSCAR").exists())
            self.assertTrue((request.draft_dir / "INCAR").exists())
            self.assertTrue((request.draft_dir / "KPOINTS").exists())
            self.assertTrue((request.draft_dir / "POTCAR.placeholder").exists())
            self.assertFalse((request.draft_dir / "POTCAR").exists())
            self.assertFalse((request.draft_dir / "generation_failure.json").exists())

    def test_full_generation_success_requires_all_four_files_and_marks_usable(self) -> None:
        with TemporaryDirectory() as tmp:
            request = self._request(tmp, "is-success")

            def fake_run(*args, **kwargs):
                text = kwargs["input"]
                if "\n105\n" in text:
                    (request.draft_dir / "POSCAR").write_text("POSCAR\n", encoding="utf-8")
                elif "\n101\n" in text:
                    (request.draft_dir / "INCAR").write_text("SYSTEM = test\n", encoding="utf-8")
                elif "\n102\n" in text:
                    (request.draft_dir / "KPOINTS").write_text("Gamma\n", encoding="utf-8")
                elif "\n103\n" in text:
                    (request.draft_dir / "POTCAR").write_text(
                        "TITEL  = PAW_PBE Ce\nlicensed body is not exposed\n",
                        encoding="utf-8",
                    )
                return Completed()

            with patch("vasp_mvp.vaspkit_runner.check_vaspkit_available", return_value=True):
                with patch("vasp_mvp.vaspkit_runner.subprocess.run", side_effect=fake_run):
                    result = generate_vasp_inputs_with_vaspkit(request, dry_run=False)

            self.assertTrue(result.ok)
            self.assertEqual(result.input_set_status, "generated")
            self.assertTrue(result.usable_for_vasp)
            for filename in ("POSCAR", "INCAR", "KPOINTS", "POTCAR"):
                self.assertTrue((request.draft_dir / filename).exists())

    def test_incar_failure_rolls_back_core_files_and_writes_failure_report(self) -> None:
        with TemporaryDirectory() as tmp:
            request = self._request(tmp, "is-failed")

            def fake_run(*args, **kwargs):
                text = kwargs["input"]
                if "\n105\n" in text:
                    (request.draft_dir / "POSCAR").write_text("POSCAR\n", encoding="utf-8")
                    return Completed(stdout="poscar ok")
                if "\n101\n" in text:
                    return Completed(returncode=2, stdout="incar failed", stderr="bad incar")
                self.fail("VASPKIT should stop after INCAR failure")

            with patch("vasp_mvp.vaspkit_runner.check_vaspkit_available", return_value=True):
                with patch("vasp_mvp.vaspkit_runner.subprocess.run", side_effect=fake_run):
                    result = generate_vasp_inputs_with_vaspkit(request, dry_run=False)

            self.assertFalse(result.ok)
            self.assertEqual(result.input_set_status, "invalid")
            self.assertFalse(result.usable_for_vasp)
            for filename in ("POSCAR", "INCAR", "KPOINTS", "POTCAR"):
                self.assertFalse((request.draft_dir / filename).exists())
            self.assertTrue((request.draft_dir / "vaspkit.out").exists())
            self.assertTrue((request.draft_dir / "vaspkit.err").exists())
            failure = json.loads((request.draft_dir / "generation_failure.json").read_text(encoding="utf-8"))
            self.assertIn("INCAR generation failed at VASPKIT 101", failure["failed_step"])
            self.assertIn("INCAR", failure["missing_files"])
            self.assertEqual(failure["cwd"], str(request.draft_dir))
            self.assertIn("vaspkit.out", failure["vaspkit_out"])
            self.assertIn("vaspkit.err", failure["vaspkit_err"])
            self.assertIn("POSCAR", failure["existing_files"])
            error_text = " ".join(result.errors)
            self.assertIn("INCAR generation failed at VASPKIT 101", error_text)
            self.assertIn("cwd:", error_text)
            self.assertIn("vaspkit.out:", error_text)

    def test_potcar_summary_does_not_expose_full_potcar_body(self) -> None:
        with TemporaryDirectory() as tmp:
            potcar = Path(tmp) / "POTCAR"
            potcar.write_text(
                "TITEL  = PAW_PBE Ce\nSECRET_LICENSED_POTCAR_BODY\n",
                encoding="utf-8",
            )

            summary = summarize_potcar(potcar)
            serialized = json.dumps(summary)

            self.assertTrue(summary["exists"])
            self.assertIn("TITEL  = PAW_PBE Ce", summary["titel_lines"])
            self.assertNotIn("SECRET_LICENSED_POTCAR_BODY", serialized)


if __name__ == "__main__":
    unittest.main()
