from __future__ import annotations

from .models import TaskRequest


SUPPORTED_TASK_TYPES = ("relax", "static", "molecule", "adsorption")


def default_incar(request: TaskRequest) -> dict[str, str]:
    base = {
        "SYSTEM": request.task_id,
        "ENCUT": "520",
        "EDIFF": "1E-5",
        "PREC": "Accurate",
        "LREAL": "Auto",
        "ALGO": "Normal",
        "ISPIN": "2",
    }

    if request.task_type == "relax":
        base.update(
            {
                "IBRION": "2",
                "NSW": "100",
                "ISIF": "3",
                "EDIFFG": "-0.02",
                "ISMEAR": "1",
                "SIGMA": "0.2",
            }
        )
    elif request.task_type == "static":
        base.update({"IBRION": "-1", "NSW": "0", "ISMEAR": "1", "SIGMA": "0.2"})
    elif request.task_type == "molecule":
        base.update(
            {
                "IBRION": "2",
                "NSW": "100",
                "ISIF": "2",
                "EDIFFG": "-0.02",
                "ISMEAR": "0",
                "SIGMA": "0.05",
            }
        )
    elif request.task_type == "adsorption":
        base.update(
            {
                "IBRION": "2",
                "NSW": "150",
                "ISIF": "2",
                "EDIFFG": "-0.02",
                "ISMEAR": "1",
                "SIGMA": "0.2",
            }
        )
    else:
        raise ValueError(f"Unsupported task type: {request.task_type}")

    base.update({str(k).strip().upper(): str(v).strip() for k, v in request.incar_overrides.items() if str(k).strip()})
    return base


def default_kpoints(task_type: str) -> tuple[int, int, int]:
    if task_type not in SUPPORTED_TASK_TYPES:
        raise ValueError(f"Unsupported task type: {task_type}")
    if task_type == "molecule":
        return (1, 1, 1)
    return (3, 3, 1) if task_type == "adsorption" else (3, 3, 3)
