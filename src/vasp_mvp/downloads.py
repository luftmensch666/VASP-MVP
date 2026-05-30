from __future__ import annotations

from pathlib import Path


def get_downloadable_text_file(
    path: Path,
    workflow_root: Path,
    *,
    max_bytes: int = 5_000_000,
    allow_potcar: bool = False,
) -> bytes:
    """安全读取 workflow_root 内的小型文本文件用于 UI 下载。

    默认拒绝 POTCAR，避免普通 UI 误把赝势全文暴露为可下载内容。
    """

    root = Path(workflow_root).resolve()
    target = (root / path).resolve() if not Path(path).is_absolute() else Path(path).resolve()
    if not _is_relative_to(target, root):
        raise ValueError("download path must be inside workflow root")
    if target.name == "POTCAR" and not allow_potcar:
        raise PermissionError("POTCAR download is not allowed by default")
    if not target.exists() or not target.is_file():
        raise FileNotFoundError(f"download file does not exist: {target}")
    size = target.stat().st_size
    if size > max_bytes:
        raise ValueError(f"download file is too large: {size} bytes")
    return target.read_bytes()


def _is_relative_to(path: Path, root: Path) -> bool:
    try:
        path.relative_to(root)
        return True
    except ValueError:
        return False
