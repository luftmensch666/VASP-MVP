from __future__ import annotations

import re
from pathlib import Path

from .models import AppConfig, PotcarConfig


SAFE_ID_RE = re.compile(r"[^A-Za-z0-9_.-]+")


class SecurityError(ValueError):
    pass


def safe_task_id(raw: str) -> str:
    value = SAFE_ID_RE.sub("-", raw.strip())[:80].strip(".-")
    if not value:
        raise SecurityError("Task id is empty after sanitization.")
    return value


def ensure_within(path: Path, root: Path, *, must_exist: bool = False) -> Path:
    candidate = path.expanduser()
    if must_exist:
        candidate = candidate.resolve(strict=True)
    else:
        existing_parent = candidate.parent.resolve(strict=True)
        candidate = existing_parent / candidate.name
    root_resolved = root.expanduser().resolve(strict=True)
    try:
        candidate.relative_to(root_resolved)
    except ValueError as exc:
        raise SecurityError(f"Path escapes allowed root: {candidate}") from exc
    return candidate


def task_dir(config: AppConfig, task_id: str) -> Path:
    config.workspace.mkdir(parents=True, exist_ok=True)
    return ensure_within(config.workspace / safe_task_id(task_id), config.workspace)


def validate_mpi_ranks(config: AppConfig, ranks: int) -> int:
    if ranks not in config.allowed_mpi_ranks:
        allowed = ", ".join(str(v) for v in config.allowed_mpi_ranks)
        raise SecurityError(f"MPI ranks must be one of: {allowed}")
    return ranks


def potcar_paths(config: AppConfig, potcars: PotcarConfig, elements: tuple[str, ...]) -> list[Path]:
    paths: list[Path] = []
    missing: list[str] = []
    for element in elements:
        subdir = potcars.elements.get(element)
        if not subdir:
            missing.append(element)
            continue
        candidate = config.potpaw_pbe / subdir / "POTCAR"
        if not candidate.exists():
            missing.append(element)
            continue
        paths.append(ensure_within(candidate, config.potpaw_pbe, must_exist=True))
    if missing:
        raise SecurityError(f"Missing POTCAR mapping or file for: {', '.join(missing)}")
    return paths


def missing_potcar_elements(config: AppConfig, potcars: PotcarConfig, elements: tuple[str, ...]) -> tuple[str, ...]:
    missing: list[str] = []
    for element in elements:
        subdir = potcars.elements.get(element)
        if not subdir or not (config.potpaw_pbe / subdir / "POTCAR").exists():
            missing.append(element)
    return tuple(missing)


def validate_vasp_bin(config: AppConfig) -> Path:
    if not config.vasp_bin.exists() or not config.vasp_bin.is_file():
        raise SecurityError(f"VASP binary does not exist: {config.vasp_bin}")
    return config.vasp_bin.resolve(strict=True)
