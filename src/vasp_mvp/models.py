from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Literal


TaskType = Literal["relax", "static", "molecule", "adsorption"]
TaskStatus = Literal["draft", "committed", "running", "finished", "failed", "stopped"]
InputSetSource = Literal["vaspkit", "manual", "imported"]
InputSetStatus = Literal["dry_run", "generated", "edited", "validated", "committed", "invalid"]
InputSetRole = Literal["primary", "adsorbed", "clean_slab", "molecule_ref"]


@dataclass(frozen=True)
class AppConfig:
    vasp_bin: Path
    potpaw_pbe: Path
    workspace: Path
    default_mpi_ranks: int
    allowed_mpi_ranks: tuple[int, ...]
    omp_num_threads: int = 1
    openblas_num_threads: int = 1


@dataclass(frozen=True)
class PotcarConfig:
    family: str
    elements: dict[str, str]


@dataclass(frozen=True)
class StructureInfo:
    source_name: str
    elements: tuple[str, ...]
    counts: tuple[int, ...]
    poscar_text: str
    is_periodic: bool


@dataclass(frozen=True)
class TaskRequest:
    task_id: str
    task_type: TaskType
    structure: StructureInfo
    mpi_ranks: int
    kpoints: tuple[int, int, int]
    incar_overrides: dict[str, str] = field(default_factory=dict)
    created_at: datetime = field(default_factory=datetime.utcnow)


@dataclass(frozen=True)
class TaskDraft:
    request: TaskRequest
    incar: dict[str, str]
    incar_text: str
    kpoints_text: str
    run_sh_text: str
    potcar_command: str
    missing_potcars: tuple[str, ...] = ()


@dataclass(frozen=True)
class Metrics:
    toten_ev: float | None = None
    loop_avg_seconds: float | None = None
    loop_count: int = 0
    electronic_converged: bool | None = None
    ionic_converged: bool | None = None
    oszicar_steps: tuple[float, ...] = ()
    status: str = "ok"
    errors: tuple[str, ...] = ()


@dataclass(frozen=True)
class TaskRecord:
    task_id: str
    project: str
    status: TaskStatus
    task_root: Path
    pid: int | None
    created_at: datetime
    updated_at: datetime
    task_type: TaskType
    start_time: datetime | None = None
    end_time: datetime | None = None
    return_code: int | None = None

    @property
    def path(self) -> Path:
        return self.task_root


@dataclass(frozen=True)
class InputSet:
    input_set_id: str
    name: str
    source: InputSetSource
    status: InputSetStatus
    usable_for_vasp: bool
    root_dir: Path
    incar_path: Path
    poscar_path: Path
    kpoints_path: Path
    potcar_path: Path
    created_at: datetime
    updated_at: datetime
    notes: str = ""


@dataclass(frozen=True)
class TaskInputSet:
    task_id: str
    role: InputSetRole
    input_set_id: str
    created_at: datetime
