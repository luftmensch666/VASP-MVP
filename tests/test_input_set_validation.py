from __future__ import annotations

import sys
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from vasp_mvp.db import init_db
from vasp_mvp.input_sets import create_input_set, get_input_set, rename_input_set, update_input_set_status
from vasp_mvp.input_set_validation import validate_input_set


class InputSetValidationTest(unittest.TestCase):
    def test_poscar_potcar_order_mismatch_is_error(self) -> None:
        with TemporaryDirectory() as tmp:
            input_set, _conn = _create_valid_input_set(Path(tmp), potcar_order=("O", "Ce"))

            result = validate_input_set(input_set)

            self.assertFalse(result.ok)
            self.assertIn("poscar_potcar_order_mismatch", [issue.code for issue in result.errors])

    def test_poscar_vasp4_without_element_line_warns_without_crashing(self) -> None:
        with TemporaryDirectory() as tmp:
            input_set, _conn = _create_valid_input_set(Path(tmp), vasp4_poscar=True)

            result = validate_input_set(input_set)

            self.assertTrue(result.ok)
            self.assertIn("poscar_order_unreadable", [issue.code for issue in result.warnings])

    def test_encut_below_enmax_warns_but_does_not_block_usage(self) -> None:
        with TemporaryDirectory() as tmp:
            input_set, _conn = _create_valid_input_set(Path(tmp), encut=300)

            result = validate_input_set(input_set)

            self.assertTrue(result.ok)
            self.assertIn("encut_below_enmax", [issue.code for issue in result.warnings])

    def test_encut_recommended_ok_is_info(self) -> None:
        with TemporaryDirectory() as tmp:
            input_set, _conn = _create_valid_input_set(Path(tmp), encut=600)

            result = validate_input_set(input_set)

            self.assertTrue(result.ok)
            self.assertIn("encut_recommended_ok", [issue.code for issue in result.infos])

    def test_warning_does_not_block_validate_status_but_error_does(self) -> None:
        with TemporaryDirectory() as tmp:
            workspace = Path(tmp) / "workspace"
            input_set, conn = _create_valid_input_set(tmp_path=Path(tmp), workspace=workspace, encut=300)
            warning_result = validate_input_set(input_set)
            update_input_set_status(conn, input_set.input_set_id, "validated" if warning_result.usable_for_vasp else "invalid", usable_for_vasp=warning_result.usable_for_vasp)
            self.assertTrue(get_input_set(conn, input_set.input_set_id).usable_for_vasp)

            (input_set.root_dir / "POTCAR").write_text("", encoding="utf-8")
            error_result = validate_input_set(input_set)
            update_input_set_status(conn, input_set.input_set_id, "validated" if error_result.usable_for_vasp else "invalid", usable_for_vasp=error_result.usable_for_vasp)
            self.assertFalse(get_input_set(conn, input_set.input_set_id).usable_for_vasp)

    def test_name_required_and_normalized_duplicate_checks_on_create_and_edit(self) -> None:
        with TemporaryDirectory() as tmp:
            workspace = Path(tmp) / "workspace"
            root_dir = workspace / "input_sets" / "is-1"
            conn = init_db(workspace)
            _write_input_files(root_dir)
            create_input_set(
                conn,
                input_set_id="is-1",
                name=" Clean Slab ",
                source="manual",
                status="validated",
                usable_for_vasp=True,
                root_dir=root_dir,
                incar_path=root_dir / "INCAR",
                poscar_path=root_dir / "POSCAR",
                kpoints_path=root_dir / "KPOINTS",
                potcar_path=root_dir / "POTCAR",
            )
            root_dir_2 = workspace / "input_sets" / "is-2"
            _write_input_files(root_dir_2)
            with self.assertRaisesRegex(ValueError, "name_duplicate"):
                create_input_set(
                    conn,
                    input_set_id="is-2",
                    name="clean slab",
                    source="manual",
                    status="validated",
                    usable_for_vasp=True,
                    root_dir=root_dir_2,
                    incar_path=root_dir_2 / "INCAR",
                    poscar_path=root_dir_2 / "POSCAR",
                    kpoints_path=root_dir_2 / "KPOINTS",
                    potcar_path=root_dir_2 / "POTCAR",
                )
            create_input_set(
                conn,
                input_set_id="is-2",
                name="Molecule Ref",
                source="manual",
                status="validated",
                usable_for_vasp=True,
                root_dir=root_dir_2,
                incar_path=root_dir_2 / "INCAR",
                poscar_path=root_dir_2 / "POSCAR",
                kpoints_path=root_dir_2 / "KPOINTS",
                potcar_path=root_dir_2 / "POTCAR",
            )
            with self.assertRaisesRegex(ValueError, "name_duplicate"):
                rename_input_set(conn, "is-2", " clean slab ")
            with self.assertRaisesRegex(ValueError, "name_required"):
                rename_input_set(conn, "is-2", "   ")


def _create_valid_input_set(
    tmp_path: Path,
    *,
    workspace: Path | None = None,
    encut: int = 600,
    potcar_order: tuple[str, ...] = ("Ce", "O"),
    vasp4_poscar: bool = False,
):
    workspace = workspace or (tmp_path / "workspace")
    root_dir = workspace / "input_sets" / "is-1"
    _write_input_files(root_dir, encut=encut, potcar_order=potcar_order, vasp4_poscar=vasp4_poscar)
    conn = init_db(workspace)
    input_set = create_input_set(
        conn,
        input_set_id="is-1",
        name="validation set",
        source="manual",
        status="validated",
        usable_for_vasp=True,
        root_dir=root_dir,
        incar_path=root_dir / "INCAR",
        poscar_path=root_dir / "POSCAR",
        kpoints_path=root_dir / "KPOINTS",
        potcar_path=root_dir / "POTCAR",
    )
    return input_set, conn


def _write_input_files(
    root_dir: Path,
    *,
    encut: int = 600,
    potcar_order: tuple[str, ...] = ("Ce", "O"),
    vasp4_poscar: bool = False,
) -> None:
    root_dir.mkdir(parents=True, exist_ok=True)
    (root_dir / "INCAR").write_text(f"SYSTEM = test\nENCUT = {encut}\nNSW = 0\n", encoding="utf-8")
    if vasp4_poscar:
        (root_dir / "POSCAR").write_text(
            "test\n1.0\n1 0 0\n0 1 0\n0 0 1\n1 1\nDirect\n0 0 0\n0.5 0.5 0.5\n",
            encoding="utf-8",
        )
    else:
        (root_dir / "POSCAR").write_text(
            "test\n1.0\n1 0 0\n0 1 0\n0 0 1\nCe O\n1 1\nDirect\n0 0 0\n0.5 0.5 0.5\n",
            encoding="utf-8",
        )
    (root_dir / "KPOINTS").write_text("Gamma\n0\nGamma\n1 1 1\n0 0 0\n", encoding="utf-8")
    potcar_text = ""
    for element in potcar_order:
        label = "Ce_sv" if element == "Ce" else element
        enmax = 400 if element == "O" else 350
        zval = 6 if element == "O" else 12
        potcar_text += (
            f"TITEL  = PAW_PBE {label} 08Apr2002\n"
            f"VRHFIN ={element}: test\n"
            f"ENMAX = {enmax}.000; ENMIN = 250.000\n"
            f"ZVAL = {zval}.000\n"
        )
    (root_dir / "POTCAR").write_text(potcar_text, encoding="utf-8")


if __name__ == "__main__":
    unittest.main()
