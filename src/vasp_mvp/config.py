from __future__ import annotations

import json
from pathlib import Path

from .models import AppConfig, PotcarConfig


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULTS_PATH = PROJECT_ROOT / "config" / "defaults.json"
POTCAR_MAP_PATH = PROJECT_ROOT / "config" / "potcar_map.json"


def _read_json(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as fh:
        return json.load(fh)


def _require_keys(data: dict, keys: tuple[str, ...], path: Path) -> None:
    missing = [key for key in keys if key not in data]
    if missing:
        raise KeyError(f"{path} is missing required keys: {', '.join(missing)}")


def load_app_config(path: Path = DEFAULTS_PATH) -> AppConfig:
    data = _read_json(path)
    _require_keys(
        data,
        ("vasp_bin", "potpaw_pbe", "workspace", "default_mpi_ranks", "allowed_mpi_ranks"),
        path,
    )
    workspace = Path(data["workspace"])
    if not workspace.is_absolute():
        workspace = PROJECT_ROOT / workspace

    return AppConfig(
        vasp_bin=Path(data["vasp_bin"]).expanduser(),
        potpaw_pbe=Path(data["potpaw_pbe"]).expanduser(),
        workspace=workspace,
        default_mpi_ranks=int(data["default_mpi_ranks"]),
        allowed_mpi_ranks=tuple(int(v) for v in data["allowed_mpi_ranks"]),
        omp_num_threads=int(data.get("omp_num_threads", 1)),
        openblas_num_threads=int(data.get("openblas_num_threads", 1)),
    )


def load_potcar_config(path: Path = POTCAR_MAP_PATH) -> PotcarConfig:
    data = _read_json(path)
    _require_keys(data, ("family", "elements"), path)
    return PotcarConfig(
        family=str(data.get("family", "PBE")),
        elements={str(k): str(v) for k, v in data.get("elements", {}).items()},
    )


def load_configs(
    defaults_path: Path = DEFAULTS_PATH,
    potcar_map_path: Path = POTCAR_MAP_PATH,
) -> tuple[AppConfig, PotcarConfig]:
    return load_app_config(defaults_path), load_potcar_config(potcar_map_path)


def self_check() -> None:
    app_config, potcar_config = load_configs()
    if app_config.default_mpi_ranks not in app_config.allowed_mpi_ranks:
        raise ValueError("default_mpi_ranks must be included in allowed_mpi_ranks")
    if not potcar_config.elements:
        raise ValueError("potcar_map.json must define at least one element")
