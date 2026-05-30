from __future__ import annotations

import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from vasp_mvp.vaspkit_step_docs import get_vaspkit_step_doc


class VaspkitStepDocsTest(unittest.TestCase):
    def test_required_step_docs_have_content(self) -> None:
        for step_key in ("cif_105", "symmetry_601", "conventional_603", "slab_803", "vacuum_801", "supercell_401", "fix_atoms_402", "101", "102", "103"):
            with self.subTest(step_key=step_key):
                zh = get_vaspkit_step_doc(step_key, "zh")
                en = get_vaspkit_step_doc(step_key, "en")
                self.assertTrue(zh.title)
                self.assertTrue(zh.purpose)
                self.assertTrue(zh.risks)
                self.assertTrue(en.title)
                self.assertTrue(en.purpose)
                self.assertTrue(en.risks)


if __name__ == "__main__":
    unittest.main()
