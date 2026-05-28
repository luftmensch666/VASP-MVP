from __future__ import annotations

import sys
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from vasp_mvp.parser import parse_metrics, parse_oszicar, parse_outcar


class ParserTest(unittest.TestCase):
    def test_parse_outcar_uses_last_free_energy_toten_and_average_loop_time(self) -> None:
        with TemporaryDirectory() as tmp:
            outcar = Path(tmp) / "OUTCAR"
            outcar.write_text(
                """
                some unrelated TOTEN = 999.0
                  free  energy   TOTEN  =       -10.000000 eV
                LOOP:  cpu time   1.00: real time   2.0
                  free  energy   TOTEN  =       -11.500000 eV
                LOOP:  cpu time   2.00: real time   4.0
                reached required accuracy - stopping structural energy minimisation
                """,
                encoding="utf-8",
            )

            metrics = parse_outcar(outcar)

            self.assertEqual(metrics.toten_ev, -11.5)
            self.assertEqual(metrics.loop_avg_seconds, 3.0)
            self.assertEqual(metrics.loop_count, 2)
            self.assertTrue(metrics.ionic_converged)
            self.assertEqual(metrics.status, "parsed")

    def test_parse_oszicar_extracts_ionic_step_energies(self) -> None:
        with TemporaryDirectory() as tmp:
            oszicar = Path(tmp) / "OSZICAR"
            oszicar.write_text(
                """
                 1 F= -.10000000E+02 E0= -.90000000E+01 d E =0
                 2 F= -1.25000000E+01 E0= -1.20000000E+01 d E =0
                 3 E0= -13.5 d E =0
                """,
                encoding="utf-8",
            )

            self.assertEqual(parse_oszicar(oszicar), (-10.0, -12.5, -13.5))

    def test_missing_files_return_friendly_status_without_crashing(self) -> None:
        with TemporaryDirectory() as tmp:
            workdir = Path(tmp)

            outcar_metrics = parse_outcar(workdir / "OUTCAR")
            all_metrics = parse_metrics(workdir)

            self.assertIsNone(outcar_metrics.toten_ev)
            self.assertIsNone(outcar_metrics.loop_avg_seconds)
            self.assertEqual(outcar_metrics.status, "OUTCAR not found")
            self.assertEqual(parse_oszicar(workdir / "OSZICAR"), ())
            self.assertIsNone(all_metrics.toten_ev)
            self.assertEqual(all_metrics.oszicar_steps, ())
            self.assertEqual(all_metrics.status, "OUTCAR not found")


if __name__ == "__main__":
    unittest.main()
