from __future__ import annotations

import sys
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import Mock, patch

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from vasp_mvp.vaspkit_structure_editor import (
    build_105_inputs,
    build_601_inputs,
    build_602_inputs,
    build_603_inputs,
    build_401_inputs,
    build_801_inputs,
    build_803_inputs,
    build_402_atom_indices_inputs,
    build_402_z_range_inputs,
    expected_401_output_name,
    expected_803_output_name,
    run_vaspkit_structure_step,
    validate_atom_indices,
    validate_direction_index,
    validate_hkl,
    validate_layer_text,
    validate_positive_float,
    validate_positive_int,
    validate_repeat_abc,
    validate_z_range,
)


class VaspkitStructureEditorTest(unittest.TestCase):
    def test_build_inputs_for_measured_structure_steps(self) -> None:
        self.assertEqual(build_105_inputs("sample.cif", "O Ce"), ["105", "sample.cif", "O Ce"])
        self.assertEqual(build_601_inputs(), ["601"])
        self.assertEqual(build_602_inputs(), ["602"])
        self.assertEqual(build_603_inputs(), ["603"])
        self.assertEqual(build_801_inputs(3, 18.5), ["801", "3", "18.5"])
        self.assertEqual(build_803_inputs(1, 1, 0, "1-3", 0.0, 15), ["803", "1 1 0", "1-3", "0", "15"])
        self.assertEqual(build_401_inputs(1, 2, 3), ["401", "1", "1 2 3"])
        self.assertEqual(build_402_atom_indices_inputs("1 2 5-7", 3), ["402", "1", "1", "1 2 5-7", "3"])
        self.assertEqual(build_402_atom_indices_inputs("H O all", 3), ["402", "1", "1", "H O all", "3"])
        self.assertEqual(build_402_z_range_inputs(0.0, 0.35, 3), ["402", "1", "3", "0 0.35", "3"])
        self.assertEqual(expected_803_output_name(1, 1, 0), "SLAB110.vasp")
        self.assertEqual(expected_401_output_name(1, 2, 3), "SC123.vasp")

    def test_invalid_parameters_are_rejected_before_vaspkit(self) -> None:
        for bad_direction in (0, 4, "q"):
            with self.assertRaises(ValueError):
                validate_direction_index(bad_direction)
        with self.assertRaises(ValueError):
            validate_positive_float(0)
        with self.assertRaises(ValueError):
            validate_positive_int("-1")
        with self.assertRaises(ValueError):
            validate_repeat_abc(1, 0, 1)
        with self.assertRaises(ValueError):
            validate_hkl("a", 1, 0)
        with self.assertRaises(ValueError):
            validate_hkl(0, 0, 0)
        self.assertEqual(validate_layer_text("1,2,3"), "1,2,3")
        self.assertEqual(validate_layer_text("1-3"), "1-3")
        with self.assertRaises(ValueError):
            validate_layer_text("abc")
        with self.assertRaises(ValueError):
            validate_layer_text("一")
        with self.assertRaises(ValueError):
            validate_atom_indices("")
        with self.assertRaises(ValueError):
            validate_atom_indices("0")
        with self.assertRaises(ValueError):
            validate_atom_indices("5-2")
        with self.assertRaises(ValueError):
            validate_z_range(0.5, 0.1)

    def test_run_structure_step_uses_argument_list_and_shell_false_for_105(self) -> None:
        with TemporaryDirectory() as tmp:
            workflow_root = Path(tmp)
            cwd = workflow_root / "structure"
            cwd.mkdir()
            completed = Mock()
            completed.returncode = 0
            completed.stdout = "ok"
            completed.stderr = ""

            def fake_run(*args, **kwargs):
                (cwd / "POSCAR").write_text("POSCAR\n", encoding="utf-8")
                return completed

            with patch("vasp_mvp.vaspkit_structure_editor.subprocess.run", side_effect=fake_run) as run:
                result = run_vaspkit_structure_step(
                    "vaspkit",
                    build_105_inputs("sample.cif"),
                    cwd,
                    "105",
                    "POSCAR",
                )

            self.assertTrue(result.ok)
            self.assertEqual(run.call_args.kwargs["shell"], False)
            self.assertEqual(run.call_args.args[0], ["vaspkit"])
            self.assertTrue((workflow_root / "logs" / "structure_105.out").exists())
            self.assertTrue((workflow_root / "logs" / "structure_105.err").exists())

    def test_failed_structure_step_returns_errors_and_logs(self) -> None:
        with TemporaryDirectory() as tmp:
            workflow_root = Path(tmp)
            cwd = workflow_root / "structure"
            cwd.mkdir()
            completed = Mock()
            completed.returncode = 2
            completed.stdout = "failed stdout"
            completed.stderr = "failed stderr"

            with patch("vasp_mvp.vaspkit_structure_editor.subprocess.run", return_value=completed):
                result = run_vaspkit_structure_step(
                    "vaspkit",
                    build_801_inputs(3, 15),
                    cwd,
                    "801",
                    "POSCAR_REV.vasp",
                )

            self.assertFalse(result.ok)
            self.assertIn("VASPKIT exited with return code 2", result.errors)
            self.assertIn("Expected output file was not generated", "; ".join(result.errors))
            self.assertEqual((workflow_root / "logs" / "structure_801.out").read_text(encoding="utf-8"), "failed stdout")

    def test_stdout_only_601_does_not_require_output_file(self) -> None:
        with TemporaryDirectory() as tmp:
            workflow_root = Path(tmp)
            cwd = workflow_root / "structure"
            cwd.mkdir()
            completed = Mock()
            completed.returncode = 0
            completed.stdout = "symmetry summary"
            completed.stderr = ""

            with patch("vasp_mvp.vaspkit_structure_editor.subprocess.run", return_value=completed):
                result = run_vaspkit_structure_step(
                    "vaspkit",
                    build_601_inputs(),
                    cwd,
                    "601",
                    None,
                )

            self.assertTrue(result.ok)
            self.assertIsNone(result.output_path)
            self.assertEqual((workflow_root / "logs" / "structure_601.out").read_text(encoding="utf-8"), "symmetry summary")

    def test_602_and_603_expected_outputs_are_checked(self) -> None:
        with TemporaryDirectory() as tmp:
            workflow_root = Path(tmp)
            cwd = workflow_root / "structure"
            cwd.mkdir()
            completed = Mock()
            completed.returncode = 0
            completed.stdout = "ok"
            completed.stderr = ""

            def fake_run_602(*args, **kwargs):
                (cwd / "PRIMCELL.vasp").write_text("POSCAR\n", encoding="utf-8")
                return completed

            with patch("vasp_mvp.vaspkit_structure_editor.subprocess.run", side_effect=fake_run_602):
                result_602 = run_vaspkit_structure_step("vaspkit", build_602_inputs(), cwd, "602", "PRIMCELL.vasp")
            self.assertTrue(result_602.ok)
            self.assertEqual(result_602.output_path, cwd / "PRIMCELL.vasp")

            def fake_run_603(*args, **kwargs):
                (cwd / "CONVCELL.vasp").write_text("POSCAR\n", encoding="utf-8")
                return completed

            with patch("vasp_mvp.vaspkit_structure_editor.subprocess.run", side_effect=fake_run_603):
                result_603 = run_vaspkit_structure_step("vaspkit", build_603_inputs(), cwd, "603", "CONVCELL.vasp")
            self.assertTrue(result_603.ok)
            self.assertEqual(result_603.output_path, cwd / "CONVCELL.vasp")

    def test_invalid_parameters_are_rejected_without_subprocess_call(self) -> None:
        with patch("vasp_mvp.vaspkit_structure_editor.subprocess.run") as run:
            with self.assertRaises(ValueError):
                build_803_inputs(1, 0, 0, "bad layer", 0.0, 15)
            with self.assertRaises(ValueError):
                build_401_inputs(1, 0, 1)
            with self.assertRaises(ValueError):
                build_402_atom_indices_inputs("bad_input", 3)

        run.assert_not_called()


if __name__ == "__main__":
    unittest.main()
