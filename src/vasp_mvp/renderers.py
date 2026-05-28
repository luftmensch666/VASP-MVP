from __future__ import annotations

import shlex

from .models import AppConfig, PotcarConfig, TaskDraft, TaskRequest
from .rules import default_incar
from .security import missing_potcar_elements


RUN_SH_MPI_RANKS = (20, 24)


def render_incar(incar: dict[str, str]) -> str:
    return "\n".join(f"{key} = {value}" for key, value in incar.items()) + "\n"


def render_kpoints(grid: tuple[int, int, int]) -> str:
    return "\n".join(
        [
            "Automatic mesh",
            "0",
            "Gamma",
            f"{grid[0]} {grid[1]} {grid[2]}",
            "0 0 0",
            "",
        ]
    )


def render_run_sh(config: AppConfig, request: TaskRequest) -> str:
    if request.mpi_ranks not in RUN_SH_MPI_RANKS:
        raise ValueError("run.sh only supports mpirun ranks 20 or 24")
    return "\n".join(
        [
            "#!/usr/bin/env bash",
            "set -euo pipefail",
            "export OMP_NUM_THREADS=1",
            "export OPENBLAS_NUM_THREADS=1",
            f"mpirun -np {request.mpi_ranks} {shlex.quote(str(config.vasp_bin))}",
            "",
        ]
    )


def render_potcar_command(config: AppConfig, potcars: PotcarConfig, elements: tuple[str, ...]) -> str:
    inputs: list[str] = []
    for element in elements:
        subdir = potcars.elements.get(element)
        if subdir is None:
            raise ValueError(f"No POTCAR mapping for element: {element}")
        inputs.append(shlex.quote(str(config.potpaw_pbe / subdir / "POTCAR")))
    return "cat " + " ".join(inputs) + " > POTCAR"


def build_draft(config: AppConfig, potcars: PotcarConfig, request: TaskRequest) -> TaskDraft:
    incar = default_incar(request)
    return TaskDraft(
        request=request,
        incar=incar,
        incar_text=render_incar(incar),
        kpoints_text=render_kpoints(request.kpoints),
        run_sh_text=render_run_sh(config, request),
        potcar_command=render_potcar_command(config, potcars, request.structure.elements),
        missing_potcars=missing_potcar_elements(config, potcars, request.structure.elements),
    )
