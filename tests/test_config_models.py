from __future__ import annotations

import sys
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from vasp_mvp.config import load_app_config, load_configs, load_potcar_config
from vasp_mvp.models import Metrics, StructureInfo, TaskDraft, TaskRequest


class ConfigModelsTest(unittest.TestCase):
    def test_load_config_files(self) -> None:
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            defaults = root / "defaults.json"
            potcar_map = root / "potcar_map.json"
            defaults.write_text(
                """
                {
                  "vasp_bin": "/opt/vasp/vasp_std",
                  "potpaw_pbe": "/opt/potpaw_PBE",
                  "workspace": "workspace",
                  "default_mpi_ranks": 4,
                  "allowed_mpi_ranks": [4, 8],
                  "omp_num_threads": 1,
                  "openblas_num_threads": 1
                }
                """,
                encoding="utf-8",
            )
            potcar_map.write_text(
                """
                {
                  "family": "PBE",
                  "elements": {"Fe": "Fe_pv", "O": "O"}
                }
                """,
                encoding="utf-8",
            )

            app_config = load_app_config(defaults)
            potcar_config = load_potcar_config(potcar_map)
            both = load_configs(defaults, potcar_map)

            self.assertEqual(app_config.default_mpi_ranks, 4)
            self.assertEqual(app_config.allowed_mpi_ranks, (4, 8))
            self.assertTrue(app_config.workspace.is_absolute())
            self.assertEqual(potcar_config.elements["Fe"], "Fe_pv")
            self.assertEqual(both, (app_config, potcar_config))

    def test_models_are_plain_dataclasses(self) -> None:
        structure = StructureInfo(
            source_name="POSCAR",
            elements=("Fe",),
            counts=(1,),
            poscar_text="POSCAR text",
            is_periodic=True,
        )
        request = TaskRequest(
            task_id="demo",
            task_type="relax",
            structure=structure,
            mpi_ranks=4,
            kpoints=(1, 1, 1),
        )
        draft = TaskDraft(
            request=request,
            incar={"ENCUT": "520"},
            incar_text="ENCUT = 520\n",
            kpoints_text="Gamma\n",
            run_sh_text="mpirun -np 4 vasp_std\n",
            potcar_command="cat Fe/POTCAR > POTCAR",
        )
        metrics = Metrics(toten_ev=-1.23, loop_avg_seconds=2.5, ionic_converged=True)

        self.assertEqual(draft.request.task_id, "demo")
        self.assertEqual(metrics.toten_ev, -1.23)
        self.assertTrue(metrics.ionic_converged)


if __name__ == "__main__":
    unittest.main()
