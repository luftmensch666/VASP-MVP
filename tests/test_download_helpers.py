from __future__ import annotations

import sys
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from vasp_mvp.downloads import get_downloadable_text_file


class DownloadHelpersTest(unittest.TestCase):
    def test_poscar_candidate_inside_workflow_can_download(self) -> None:
        with TemporaryDirectory() as tmp:
            root = Path(tmp) / "workflow"
            path = root / "artifacts" / "clean" / "candidates" / "POSCAR"
            path.parent.mkdir(parents=True)
            path.write_text("POSCAR\n", encoding="utf-8")

            self.assertEqual(get_downloadable_text_file(path, root), b"POSCAR\n")
            self.assertEqual(get_downloadable_text_file(Path("artifacts/clean/candidates/POSCAR"), root), b"POSCAR\n")

    def test_path_escape_is_rejected(self) -> None:
        with TemporaryDirectory() as tmp:
            root = Path(tmp) / "workflow"
            outside = Path(tmp) / "outside.txt"
            outside.write_text("secret", encoding="utf-8")

            with self.assertRaises(ValueError):
                get_downloadable_text_file(outside, root)
            with self.assertRaises(ValueError):
                get_downloadable_text_file(Path("../outside.txt"), root)

    def test_potcar_download_is_rejected_by_default(self) -> None:
        with TemporaryDirectory() as tmp:
            root = Path(tmp) / "workflow"
            path = root / "POTCAR"
            root.mkdir()
            path.write_text("POTCAR content\n", encoding="utf-8")

            with self.assertRaises(PermissionError):
                get_downloadable_text_file(path, root)

    def test_large_file_is_rejected(self) -> None:
        with TemporaryDirectory() as tmp:
            root = Path(tmp) / "workflow"
            path = root / "POSCAR"
            root.mkdir()
            path.write_text("0123456789", encoding="utf-8")

            with self.assertRaises(ValueError):
                get_downloadable_text_file(path, root, max_bytes=5)


if __name__ == "__main__":
    unittest.main()
