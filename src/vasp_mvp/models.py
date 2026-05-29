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
CalculationType = Literal[
    "relax",
    "static",
    "molecule_relax",
    "molecule_static",
    "slab_static",
    "adsorbed_static",
    "dos",
    "bader",
    "neb",
    "unknown",
]
JobStatus = Literal["draft", "committed", "running", "finished", "failed", "stopped"]
WorkflowType = Literal["adsorption", "neb", "dos", "bader", "custom"]
WorkflowStatus = Literal["draft", "committed", "running", "finished", "failed", "stopped"]
WorkflowRole = Literal[
    "clean_slab",
    "molecule_ref",
    "adsorbed_system",
    "initial_state",
    "final_state",
    "neb_image",
    "dos",
    "bader",
    "primary",
]


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
    name: str | None = None
    notes: str | None = None

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


@dataclass(frozen=True)
class JobRecord:
    job_id: str
    calculation_type: CalculationType
    status: JobStatus
    run_dir: Path
    input_set_id: str | None
    pid: int | None
    created_at: datetime
    updated_at: datetime
    start_time: datetime | None = None
    end_time: datetime | None = None
    return_code: int | None = None
    mpi_ranks: int | None = None
    vasp_bin: str | None = None
    name: str | None = None
    notes: str | None = None


@dataclass(frozen=True)
class JobMetricsRecord:
    job_id: str
    toten_ev: float | None
    loop_avg_seconds: float | None
    loop_count: int
    ionic_converged: bool | None
    electronic_converged: bool | None
    oszicar_steps: tuple[float, ...]
    errors: tuple[str, ...]
    energy_source: str
    energy_label: str
    updated_at: datetime


@dataclass(frozen=True)
class WorkflowRecord:
    workflow_id: str
    workflow_type: WorkflowType
    name: str
    status: WorkflowStatus
    root_dir: Path
    method_family: str | None
    functional: str | None
    method_notes: str | None
    created_at: datetime
    updated_at: datetime
    notes: str = ""


@dataclass(frozen=True)
class WorkflowJobRecord:
    workflow_job_id: str
    workflow_id: str
    job_id: str
    role: WorkflowRole
    step_order: int | None
    required: bool
    created_at: datetime
    notes: str = ""
