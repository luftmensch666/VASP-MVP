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
    create_input_set,
    get_input_set,
    list_input_sets,
    list_task_input_sets,
    rename_input_set,
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
