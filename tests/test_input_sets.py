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
from vasp_mvp.input_sets import (
    bind_input_set_to_task,
    build_input_file_hashes,
    create_input_set,
    get_input_set,
    list_input_sets,
    list_usable_input_sets,
    list_task_input_sets,
    rename_input_set,
    save_editable_input_file,
    update_input_set_status,
)


class InputSetsTest(unittest.TestCase):
    def test_create_list_get_and_update_input_set(self) -> None:
        with TemporaryDirectory() as tmp:
            workspace = Path(tmp) / "workspace"
            root_dir = workspace / "input_sets" / "is-1"
            conn = init_db(workspace)

            created = create_input_set(
                conn,
                input_set_id="is-1",
                name="CeO2 dry run",
                source="vaspkit",
                status="dry_run",
                usable_for_vasp=False,
                root_dir=root_dir,
                incar_path=root_dir / "INCAR",
                poscar_path=root_dir / "POSCAR",
                kpoints_path=root_dir / "KPOINTS",
                potcar_path=root_dir / "POTCAR",
                notes="dry-run placeholder only",
            )

            self.assertEqual(created.input_set_id, "is-1")
            self.assertEqual(created.source, "vaspkit")
            self.assertEqual(created.status, "dry_run")
            self.assertFalse(created.usable_for_vasp)
            self.assertEqual(created.root_dir, root_dir)
            self.assertIsInstance(created.incar_path, Path)

            update_input_set_status(
                conn,
                "is-1",
                "validated",
                usable_for_vasp=True,
                notes="validated for local run",
            )

            loaded = get_input_set(conn, "is-1")
            listed = list_input_sets(conn)

            self.assertIsNotNone(loaded)
            self.assertEqual(loaded.status, "validated")
            self.assertTrue(loaded.usable_for_vasp)
            self.assertEqual(loaded.notes, "validated for local run")
            self.assertEqual(len(listed), 1)
            self.assertEqual(listed[0].input_set_id, "is-1")

            rename_input_set(conn, "is-1", "CeO2 validated")
            renamed = get_input_set(conn, "is-1")
            self.assertIsNotNone(renamed)
            self.assertEqual(renamed.name, "CeO2 validated")

    def test_list_usable_input_sets_uses_db_path_short_connection(self) -> None:
        with TemporaryDirectory() as tmp:
            workspace = Path(tmp) / "workspace"
            usable_root = workspace / "input_sets" / "usable"
            dry_run_root = workspace / "input_sets" / "dry-run"
            conn = init_db(workspace)

            create_input_set(
                conn,
                input_set_id="usable",
                name="usable",
                source="manual",
                status="validated",
                usable_for_vasp=True,
                root_dir=usable_root,
                incar_path=usable_root / "INCAR",
                poscar_path=usable_root / "POSCAR",
                kpoints_path=usable_root / "KPOINTS",
                potcar_path=usable_root / "POTCAR",
            )
            create_input_set(
                conn,
                input_set_id="dry-run",
                name="dry-run",
                source="vaspkit",
                status="dry_run",
                usable_for_vasp=False,
                root_dir=dry_run_root,
                incar_path=dry_run_root / "INCAR",
                poscar_path=dry_run_root / "POSCAR",
                kpoints_path=dry_run_root / "KPOINTS",
                potcar_path=dry_run_root / "POTCAR",
            )
            conn.close()

            usable_sets = list_usable_input_sets(workspace / "vasp_mvp.db")

            self.assertEqual(len(usable_sets), 1)
            self.assertEqual(usable_sets[0].input_set_id, "usable")
            self.assertTrue(usable_sets[0].usable_for_vasp)

    def test_save_editable_file_creates_backup_hashes_and_history(self) -> None:
        with TemporaryDirectory() as tmp:
            workspace = Path(tmp) / "workspace"
            root_dir = workspace / "input_sets" / "is-1"
            root_dir.mkdir(parents=True)
            (root_dir / "INCAR").write_text("SYSTEM = old\n", encoding="utf-8")
            (root_dir / "POSCAR").write_text("POSCAR\n", encoding="utf-8")
            (root_dir / "KPOINTS").write_text("Gamma\n", encoding="utf-8")
            conn = init_db(workspace)
            input_set = create_input_set(
                conn,
                input_set_id="is-1",
                name="editable",
                source="manual",
                status="generated",
                usable_for_vasp=True,
                root_dir=root_dir,
                incar_path=root_dir / "INCAR",
                poscar_path=root_dir / "POSCAR",
                kpoints_path=root_dir / "KPOINTS",
                potcar_path=root_dir / "POTCAR",
            )

            result = save_editable_input_file(input_set, "INCAR", "SYSTEM = new\n", user_action="unit_test")

            self.assertEqual((root_dir / "INCAR").read_text(encoding="utf-8"), "SYSTEM = new\n")
            self.assertIsNotNone(result["old_hash"])
            self.assertIsNotNone(result["new_hash"])
            self.assertNotEqual(result["old_hash"], result["new_hash"])
            self.assertIsNotNone(result["backup_path"])
            self.assertTrue(result["backup_path"].exists())
            self.assertTrue((root_dir / "file_hashes.json").exists())
            self.assertTrue((root_dir / "edit_history.jsonl").exists())
            self.assertIn("unit_test", (root_dir / "edit_history.jsonl").read_text(encoding="utf-8"))
            self.assertIsNotNone(build_input_file_hashes(input_set)["INCAR"]["sha256"])

            with self.assertRaises(ValueError):
                save_editable_input_file(input_set, "POTCAR", "forbidden\n")

    def test_bind_input_set_to_task_role(self) -> None:
        with TemporaryDirectory() as tmp:
            workspace = Path(tmp) / "workspace"
            root_dir = workspace / "input_sets" / "is-1"
            conn = init_db(workspace)

            create_input_set(
                conn,
                input_set_id="is-1",
                name="adsorbed static",
                source="manual",
                status="generated",
                usable_for_vasp=True,
                root_dir=root_dir,
                incar_path=root_dir / "INCAR",
                poscar_path=root_dir / "POSCAR",
                kpoints_path=root_dir / "KPOINTS",
                potcar_path=root_dir / "POTCAR",
            )

            binding = bind_input_set_to_task(conn, "task-ads", "adsorbed", "is-1")
            bindings = list_task_input_sets(conn, "task-ads")

            self.assertEqual(binding.task_id, "task-ads")
            self.assertEqual(binding.role, "adsorbed")
            self.assertEqual(binding.input_set_id, "is-1")
            self.assertEqual(len(bindings), 1)
            self.assertEqual(bindings[0].role, "adsorbed")
            self.assertEqual(bindings[0].input_set_id, "is-1")

    def test_reject_unknown_status_and_role(self) -> None:
        with TemporaryDirectory() as tmp:
            workspace = Path(tmp) / "workspace"
            root_dir = workspace / "input_sets" / "is-1"
            conn = init_db(workspace)

            with self.assertRaises(ValueError):
                create_input_set(
                    conn,
                    input_set_id="is-1",
                    name="bad",
                    source="vaspkit",
                    status="ready",
                    usable_for_vasp=False,
                    root_dir=root_dir,
                    incar_path=root_dir / "INCAR",
                    poscar_path=root_dir / "POSCAR",
                    kpoints_path=root_dir / "KPOINTS",
                    potcar_path=root_dir / "POTCAR",
                )

            with self.assertRaises(ValueError):
                bind_input_set_to_task(conn, "task-1", "unknown", "is-1")


if __name__ == "__main__":
    unittest.main()
