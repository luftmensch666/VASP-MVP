from __future__ import annotations

import sys
import unittest
import json
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from vasp_mvp.vaspkit_runner import (
    VaspkitRequest,
    build_vaspkit_inputs,
    generate_vasp_inputs_with_vaspkit,
    run_vaspkit_interactive,
    summarize_potcar,
)


class Completed:
    stdout = "ok"
    stderr = ""
    returncode = 0


class VaspkitRunnerTest(unittest.TestCase):
    def test_build_full_input_sequence(self) -> None:
        request = VaspkitRequest(
            vaspkit_bin="vaspkit",
            draft_dir=Path("/tmp/draft"),
            uploaded_cif_path=Path("/tmp/draft/CePO4.cif"),
            element_order_mode="custom",
            custom_element_order="O Ce P",
            incar_key_parameters=["SR"],
            kmesh_scheme="1",
            kmesh_resolved_value=0.04,
            potcar_mode="103",
        )

        self.assertEqual(
            build_vaspkit_inputs(request),
            ["1", "105", "CePO4.cif", "O Ce P", "1", "101", "SR", "1", "102", "1", "0.04", "1", "103"],
        )

    def test_custom_incar_string_takes_priority(self) -> None:
        request = VaspkitRequest(
            vaspkit_bin="vaspkit",
            draft_dir=Path("/tmp/draft"),
            uploaded_cif_path=Path("/tmp/draft/CePO4.cif"),
            incar_key_parameters=["SR"],
            incar_custom_key_string="STH6D3",
        )

        self.assertEqual(
            build_vaspkit_inputs(request),
            ["1", "105", "CePO4.cif", "", "1", "101", "STH6D3", "1", "102", "1", "0.04", "1", "103"],
        )

    def test_potcar_mode_104_returns_not_implemented_result(self) -> None:
        with TemporaryDirectory() as tmp:
            request = VaspkitRequest(
                vaspkit_bin="vaspkit",
                draft_dir=Path(tmp),
                uploaded_cif_path=Path(tmp) / "CePO4.cif",
                potcar_mode="104",
            )
            result = generate_vasp_inputs_with_vaspkit(request, dry_run=True)

            self.assertFalse(result.ok)
            self.assertTrue(any("not implemented" in item.lower() for item in result.errors))
            self.assertTrue((Path(tmp) / "vaspkit_result.json").exists())

    def test_dry_run_generates_fake_files_without_real_potcar(self) -> None:
        with TemporaryDirectory() as tmp:
            cif = Path(tmp) / "CePO4.cif"
            cif.write_text("data_test\n", encoding="utf-8")
            request = VaspkitRequest(
                vaspkit_bin="vaspkit",
                draft_dir=Path(tmp) / "draft",
                uploaded_cif_path=cif,
                potcar_mode="103",
            )

            result = generate_vasp_inputs_with_vaspkit(request, dry_run=True)

            self.assertTrue(result.ok)
            self.assertTrue(result.dry_run)
            self.assertTrue((request.draft_dir / "POSCAR").exists())
            self.assertTrue((request.draft_dir / "INCAR").exists())
            self.assertTrue((request.draft_dir / "KPOINTS").exists())
            self.assertTrue((request.draft_dir / "POTCAR.placeholder").exists())
            self.assertFalse((request.draft_dir / "POTCAR").exists())
            self.assertTrue((request.draft_dir / "potcar_summary.json").exists())
            self.assertTrue((request.draft_dir / "vaspkit_request.json").exists())
            self.assertTrue((request.draft_dir / "vaspkit_result.json").exists())
            self.assertTrue((request.draft_dir / "input_set.json").exists())
            self.assertTrue((request.draft_dir / "validation.json").exists())
            self.assertTrue((request.draft_dir / "file_hashes.json").exists())
            self.assertEqual(result.input_set_status, "dry_run")
            self.assertFalse(result.usable_for_vasp)

            input_set = json.loads((request.draft_dir / "input_set.json").read_text(encoding="utf-8"))
            validation = json.loads((request.draft_dir / "validation.json").read_text(encoding="utf-8"))
            hashes = json.loads((request.draft_dir / "file_hashes.json").read_text(encoding="utf-8"))

            self.assertEqual(input_set["status"], "dry_run")
            self.assertFalse(input_set["usable_for_vasp"])
            self.assertEqual(validation["status"], "dry_run")
            self.assertIsNotNone(hashes["INCAR"]["sha256"])
            self.assertIsNone(hashes["POTCAR"]["sha256"])

    def test_summarize_potcar_reads_only_titel_and_size(self) -> None:
        with TemporaryDirectory() as tmp:
            potcar = Path(tmp) / "POTCAR"
            potcar.write_text("TITEL  = PAW_PBE H\nfull body should not be exposed\n", encoding="utf-8")

            summary = summarize_potcar(potcar)

            self.assertTrue(summary["exists"])
            self.assertEqual(summary["titel_lines"], ["TITEL  = PAW_PBE H"])
            self.assertEqual(summary["size_bytes"], potcar.stat().st_size)
            self.assertEqual(summary["potential_order"], ["H"])
            self.assertEqual(summary["element_order"], ["H"])
            self.assertIsNotNone(summary["sha256"])

    def test_real_generation_marks_complete_input_set_usable(self) -> None:
        with TemporaryDirectory() as tmp:
            draft_dir = Path(tmp) / "input_sets" / "input-set"
            cif = draft_dir / "CePO4.cif"
            draft_dir.mkdir(parents=True)
            cif.write_text("data_test\n", encoding="utf-8")

            def fake_run(*args, **kwargs):
                (draft_dir / "POSCAR").write_text("POSCAR\n", encoding="utf-8")
                (draft_dir / "INCAR").write_text("SYSTEM = test\n", encoding="utf-8")
                (draft_dir / "KPOINTS").write_text("Gamma\n", encoding="utf-8")
                (draft_dir / "POTCAR").write_text("TITEL  = PAW_PBE Ce\n", encoding="utf-8")
                return Completed()

            request = VaspkitRequest(
                vaspkit_bin="vaspkit",
                draft_dir=draft_dir,
                input_set_id="is-real",
                uploaded_cif_path=cif,
                workspace=Path(tmp),
            )
            with patch("vasp_mvp.vaspkit_runner.check_vaspkit_available", return_value=True):
                with patch("vasp_mvp.vaspkit_runner.subprocess.run", side_effect=fake_run):
                    result = generate_vasp_inputs_with_vaspkit(request, dry_run=False)

            self.assertTrue(result.ok)
            self.assertEqual(result.input_set_id, "is-real")
            self.assertEqual(result.input_set_status, "generated")
            self.assertTrue(result.usable_for_vasp)
            validation = json.loads((draft_dir / "validation.json").read_text(encoding="utf-8"))
            hashes = json.loads((draft_dir / "file_hashes.json").read_text(encoding="utf-8"))
            self.assertEqual(validation["status"], "generated")
            self.assertIsNotNone(hashes["POTCAR"]["sha256"])

    def test_real_generation_missing_potcar_marks_input_set_invalid(self) -> None:
        with TemporaryDirectory() as tmp:
            draft_dir = Path(tmp) / "input_sets" / "input-set"
            cif = draft_dir / "CePO4.cif"
            draft_dir.mkdir(parents=True)
            cif.write_text("data_test\n", encoding="utf-8")

            def fake_run(*args, **kwargs):
                (draft_dir / "POSCAR").write_text("POSCAR\n", encoding="utf-8")
                (draft_dir / "INCAR").write_text("SYSTEM = test\n", encoding="utf-8")
                (draft_dir / "KPOINTS").write_text("Gamma\n", encoding="utf-8")
                return Completed()

            request = VaspkitRequest(
                vaspkit_bin="vaspkit",
                draft_dir=draft_dir,
                input_set_id="is-invalid",
                uploaded_cif_path=cif,
                workspace=Path(tmp),
            )
            with patch("vasp_mvp.vaspkit_runner.check_vaspkit_available", return_value=True):
                with patch("vasp_mvp.vaspkit_runner.subprocess.run", side_effect=fake_run):
                    result = generate_vasp_inputs_with_vaspkit(request, dry_run=False)

            self.assertFalse(result.ok)
            self.assertEqual(result.input_set_status, "invalid")
            self.assertFalse(result.usable_for_vasp)
            self.assertIn("POTCAR", " ".join(result.errors))

    def test_run_vaspkit_interactive_uses_argument_list_and_shell_false(self) -> None:
        with TemporaryDirectory() as tmp:
            cwd = Path(tmp)
            with patch("vasp_mvp.vaspkit_runner.subprocess.run", return_value=Completed()) as run:
                result = run_vaspkit_interactive("vaspkit", ["1", "101", "SR"], cwd)

            args, kwargs = run.call_args
            self.assertEqual(args[0], ["vaspkit"])
            self.assertEqual(kwargs["input"], "1\n101\nSR\n")
            self.assertEqual(kwargs["cwd"], cwd)
            self.assertIs(kwargs["shell"], False)
            self.assertTrue((cwd / "vaspkit.out").exists())
            self.assertTrue(result.ok)


if __name__ == "__main__":
    unittest.main()
