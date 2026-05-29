from __future__ import annotations

import sys
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from vasp_mvp.input_set_validation import parse_potcar_summary


class PotcarSummaryTest(unittest.TestCase):
    def test_vrhfin_element_order_is_preferred(self) -> None:
        with TemporaryDirectory() as tmp:
            path = Path(tmp) / "POTCAR"
            path.write_text(
                "TITEL  = PAW_PBE Ce_sv 08Apr2002\n"
                "VRHFIN =Ce: s2p6d1f1\n"
                "   ENMAX = 300.000; ENMIN = 250.000\n"
                "   ZVAL   = 12.000\n"
                "TITEL  = PAW_PBE O 08Apr2002\n"
                "VRHFIN =O: s2p4\n"
                "   ENMAX = 400.000; ENMIN = 300.000\n"
                "   ZVAL   = 6.000\n"
                "PRIVATE_PARAMETER_BLOCK_SHOULD_NOT_APPEAR\n",
                encoding="utf-8",
            )

            summary = parse_potcar_summary(path)
            data = summary.to_dict()

            self.assertEqual(summary.element_order, ("Ce", "O"))
            self.assertEqual(summary.potential_labels, ("Ce_sv", "O"))
            self.assertEqual(data["number_of_potentials"], 2)
            self.assertEqual(data["potentials"][0]["enmax"], 300.0)
            self.assertEqual(data["potentials"][1]["zval"], 6.0)
            self.assertNotIn("PRIVATE_PARAMETER_BLOCK_SHOULD_NOT_APPEAR", str(data))

    def test_titel_parses_ce_suffix_labels(self) -> None:
        with TemporaryDirectory() as tmp:
            path = Path(tmp) / "POTCAR"
            path.write_text(
                "TITEL  = PAW_PBE Ce_pv 08Apr2002\n"
                "   ENMAX = 350.000; ENMIN = 250.000\n"
                "TITEL  = PAW_PBE Ce_3 08Apr2002\n"
                "   ENMAX = 320.000; ENMIN = 250.000\n"
                "TITEL  = PAW_PBE O 08Apr2002\n"
                "   ENMAX = 400.000; ENMIN = 300.000\n",
                encoding="utf-8",
            )

            summary = parse_potcar_summary(path)

            self.assertEqual(summary.element_order, ("Ce", "Ce", "O"))
            self.assertEqual(summary.potential_labels, ("Ce_pv", "Ce_3", "O"))


if __name__ == "__main__":
    unittest.main()
