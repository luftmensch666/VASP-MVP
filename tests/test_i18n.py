from __future__ import annotations

import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from vasp_mvp.i18n import get_available_languages, load_translations, t


class I18nTest(unittest.TestCase):
    def test_available_languages(self) -> None:
        self.assertEqual(get_available_languages(), ["zh", "en"])

    def test_load_translations(self) -> None:
        self.assertEqual(load_translations("zh")["app.title"], "VASP 本地工作流 MVP")
        self.assertEqual(load_translations("en")["app.title"], "VASP Local Workflow MVP")

    def test_missing_key_is_safe(self) -> None:
        self.assertEqual(t("missing.key", "zh"), "[[missing:missing.key]]")

    def test_variable_formatting(self) -> None:
        self.assertEqual(t("task.pid", "zh", pid=1234), "进程 ID：1234")
        self.assertEqual(t("task.pid", "en", pid=1234), "PID: 1234")

    def test_unknown_language_falls_back_to_zh(self) -> None:
        self.assertEqual(t("app.title", "de"), "VASP 本地工作流 MVP")

    def test_language_files_have_same_keys(self) -> None:
        self.assertEqual(set(load_translations("zh")), set(load_translations("en")))


if __name__ == "__main__":
    unittest.main()
