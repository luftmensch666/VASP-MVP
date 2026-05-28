from __future__ import annotations

import sys
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from vasp_mvp.db import create_task, init_db, list_tasks
from vasp_mvp.models import AppConfig, PotcarConfig, StructureInfo, TaskRequest
from vasp_mvp.renderers import build_draft
from vasp_mvp.runner import mpirun_args, run_dir, start_task_record, start_vasp, validate_run_inputs, write_confirmed_task


class FakeProcess:
    pid = 4321


class RunnerTest(unittest.TestCase):
    def make_draft(self, root: Path):
        potroot = root / "potpaw_PBE"
        (potroot / "Fe_pv").mkdir(parents=True)
        (potroot / "Fe_pv" / "POTCAR").write_text("fake potcar\n", encoding="utf-8")
        vasp_bin = root / "vasp_std"
        vasp_bin.write_text("#!/bin/sh\n", encoding="utf-8")
        vasp_bin.chmod(0o755)

        config = AppConfig(
            vasp_bin=vasp_bin,
            potpaw_pbe=potroot,
            workspace=root / "workspace",
            default_mpi_ranks=20,
            allowed_mpi_ranks=(20, 24),
        )
        potcars = PotcarConfig("PBE", {"Fe": "Fe_pv"})
        structure = StructureInfo(
            source_name="POSCAR",
            elements=("Fe",),
            counts=(1,),
            poscar_text=(
                "Fe\n1.0\n1 0 0\n0 1 0\n0 0 1\nFe\n1\nDirect\n0 0 0\n"
            ),
            is_periodic=True,
        )
        request = TaskRequest(
            task_id="task-1",
            task_type="relax",
            structure=structure,
            mpi_ranks=20,
            kpoints=(1, 1, 1),
        )
        return config, potcars, build_draft(config, potcars, request)

    def test_write_confirmed_task_writes_inputs_to_run_dir(self) -> None:
        with TemporaryDirectory() as tmp:
            config, potcars, draft = self.make_draft(Path(tmp))
            conn = init_db(config.workspace)

            task_root = write_confirmed_task(config, potcars, draft, conn)
            task_run = run_dir(task_root)

            self.assertEqual(task_root, config.workspace / "task-1")
            self.assertTrue((task_run / "POSCAR").exists())
            self.assertTrue((task_run / "INCAR").exists())
            self.assertTrue((task_run / "KPOINTS").exists())
            self.assertTrue((task_run / "POTCAR").exists())
            self.assertEqual(list_tasks(conn)[0].task_root, task_root)

    def test_dry_run_writes_fake_output_and_does_not_start_process(self) -> None:
        with TemporaryDirectory() as tmp:
            config, potcars, draft = self.make_draft(Path(tmp))
            conn = init_db(config.workspace)
            task_root = write_confirmed_task(config, potcars, draft, conn)

            with patch("vasp_mvp.runner.subprocess.Popen") as popen:
                result = start_vasp(config, draft, conn, dry_run=True)

            self.assertIsNone(result)
            popen.assert_not_called()
            text = (run_dir(task_root) / "vasp.out").read_text(encoding="utf-8")
            self.assertIn("DRY RUN", text)
            self.assertIn("mpirun -np 20 --map-by core --bind-to core", text)

    def test_start_vasp_uses_safe_popen_options(self) -> None:
        with TemporaryDirectory() as tmp:
            config, potcars, draft = self.make_draft(Path(tmp))
            conn = init_db(config.workspace)
            task_root = write_confirmed_task(config, potcars, draft, conn)
            task_run = run_dir(task_root)

            with patch("vasp_mvp.runner.subprocess.Popen", return_value=FakeProcess()) as popen:
                process = start_vasp(config, draft, conn)

            self.assertEqual(process.pid, 4321)
            args, kwargs = popen.call_args
            self.assertEqual(
                args[0],
                ["mpirun", "-np", "20", "--map-by", "core", "--bind-to", "core", str(config.vasp_bin.resolve())],
            )
            self.assertEqual(kwargs["cwd"], task_run)
            self.assertIs(kwargs["shell"], False)
            self.assertIs(kwargs["start_new_session"], True)
            self.assertEqual(kwargs["stderr"], -2)
            self.assertEqual(kwargs["env"]["OMP_NUM_THREADS"], "1")
            self.assertEqual(kwargs["env"]["OPENBLAS_NUM_THREADS"], "1")
            self.assertEqual(kwargs["env"]["OMP_STACKSIZE"], "512m")
            self.assertTrue((task_run / "vasp.out").exists())

    def test_mpirun_args_are_always_a_list(self) -> None:
        self.assertEqual(
            mpirun_args(20, Path("/opt/vasp/vasp_std")),
            ["mpirun", "-np", "20", "--map-by", "core", "--bind-to", "core", "/opt/vasp/vasp_std"],
        )

    def test_start_task_record_checks_all_run_inputs(self) -> None:
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            config, _, _ = self.make_draft(root)
            conn = init_db(config.workspace)
            task_root = config.workspace / "task-from-input-set"
            task_run = run_dir(task_root)
            task_run.mkdir(parents=True)
            for filename in ("INCAR", "POSCAR", "KPOINTS"):
                (task_run / filename).write_text(f"{filename}\n", encoding="utf-8")
            create_task(
                conn,
                task_id="task-from-input-set",
                project="default",
                task_type="static",
                task_root=task_root,
                status="committed",
            )
            task = list_tasks(conn)[0]

            self.assertEqual(validate_run_inputs(task_root), ["POTCAR"])
            with self.assertRaises(FileNotFoundError):
                start_task_record(config, task, conn, dry_run=True)

            (task_run / "POTCAR").write_text("TITEL = PAW_PBE Fe\n", encoding="utf-8")
            self.assertEqual(validate_run_inputs(task_root), [])
            result = start_task_record(config, task, conn, dry_run=True)
            self.assertIsNone(result)
            self.assertTrue((task_run / "vasp.out").exists())


if __name__ == "__main__":
    unittest.main()
