from __future__ import annotations

import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from vasp_mvp.models import AppConfig, PotcarConfig, StructureInfo, TaskRequest
from vasp_mvp.renderers import build_draft, render_potcar_command, render_run_sh
from vasp_mvp.rules import default_incar, default_kpoints


def request(task_type: str = "relax", ranks: int = 20) -> TaskRequest:
    return TaskRequest(
        task_id="demo",
        task_type=task_type,
        structure=StructureInfo(
            source_name="POSCAR",
            elements=("Fe", "O"),
            counts=(1, 1),
            poscar_text="POSCAR",
            is_periodic=True,
        ),
        mpi_ranks=ranks,
        kpoints=default_kpoints(task_type),
    )


class RulesRenderersTest(unittest.TestCase):
    def setUp(self) -> None:
        self.config = AppConfig(
            vasp_bin=Path("/opt/vasp/bin/vasp_std"),
            potpaw_pbe=Path("/opt/potpaw_PBE"),
            workspace=Path("/tmp/workspace"),
            default_mpi_ranks=20,
            allowed_mpi_ranks=(20, 24),
        )
        self.potcars = PotcarConfig("PBE", {"Fe": "Fe_pv", "O": "O"})

    def test_supported_task_rules_generate_incar_and_kpoints(self) -> None:
        for task_type in ("relax", "static", "molecule", "adsorption"):
            req = request(task_type)
            incar = default_incar(req)
            self.assertIn("ENCUT", incar)
            self.assertIn("EDIFF", incar)
            self.assertEqual(len(default_kpoints(task_type)), 3)

        self.assertEqual(default_incar(request("static"))["NSW"], "0")
        self.assertEqual(default_kpoints("molecule"), (1, 1, 1))
        self.assertEqual(default_kpoints("adsorption"), (3, 3, 1))

    def test_draft_texts_are_rendered_without_executing_commands(self) -> None:
        draft = build_draft(self.config, self.potcars, request("relax", ranks=20))

        self.assertIn("ENCUT = 520", draft.incar_text)
        self.assertIn("Gamma", draft.kpoints_text)
        self.assertIn("cat /opt/potpaw_PBE/Fe_pv/POTCAR /opt/potpaw_PBE/O/POTCAR > POTCAR", draft.potcar_command)
        self.assertIn("export OMP_NUM_THREADS=1", draft.run_sh_text)
        self.assertIn("export OPENBLAS_NUM_THREADS=1", draft.run_sh_text)
        self.assertIn("mpirun -np 20 /opt/vasp/bin/vasp_std", draft.run_sh_text)

    def test_run_sh_accepts_only_20_or_24_ranks(self) -> None:
        self.assertIn("mpirun -np 24", render_run_sh(self.config, request(ranks=24)))
        with self.assertRaisesRegex(ValueError, "20 or 24"):
            render_run_sh(self.config, request(ranks=16))

    def test_potcar_command_requires_mapping(self) -> None:
        with self.assertRaisesRegex(ValueError, "No POTCAR mapping"):
            render_potcar_command(self.config, self.potcars, ("Fe", "C"))


if __name__ == "__main__":
    unittest.main()
