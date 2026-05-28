from __future__ import annotations

import re
from pathlib import Path

from .models import Metrics


NUMBER = r"[-+]?(?:\d+(?:\.\d*)?|\.\d+)(?:[Ee][-+]?\d+)?"
TOTEN_RE = re.compile(rf"free\s+energy\s+TOTEN\s*=\s*({NUMBER})", re.IGNORECASE)
LOOP_REAL_RE = re.compile(rf"LOOP:\s+.*?real time\s+({NUMBER})", re.IGNORECASE)
OSZICAR_ENERGY_RE = re.compile(rf"\b(?:F|E0|E)=\s*({NUMBER})")


def parse_outcar(path: Path) -> Metrics:
    if not path.exists():
        return Metrics(status="OUTCAR not found")
    text = path.read_text(encoding="utf-8", errors="replace")
    toten_values = [float(match.group(1)) for match in TOTEN_RE.finditer(text)]
    loop_times = [float(match.group(1)) for match in LOOP_REAL_RE.finditer(text)]
    lower = text.lower()
    ionic_converged = "reached required accuracy" in lower

    errors: list[str] = []
    for marker in ("zbrent: fatal", "error", "very serious problems"):
        if marker in lower:
            errors.append(marker)

    return Metrics(
        toten_ev=toten_values[-1] if toten_values else None,
        loop_avg_seconds=sum(loop_times) / len(loop_times) if loop_times else None,
        loop_count=len(loop_times),
        electronic_converged=None,
        ionic_converged=ionic_converged if "reached required accuracy" in lower else None,
        status="parsed" if toten_values or loop_times or ionic_converged else "OUTCAR parsed, no metrics found",
        errors=tuple(errors),
    )


def parse_oszicar(path: Path) -> tuple[float, ...]:
    if not path.exists():
        return ()
    values: list[float] = []
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        match = OSZICAR_ENERGY_RE.search(line)
        if match:
            values.append(float(match.group(1)))
    return tuple(values)


def parse_metrics(workdir: Path) -> Metrics:
    outcar_metrics = parse_outcar(workdir / "OUTCAR")
    return Metrics(
        toten_ev=outcar_metrics.toten_ev,
        loop_avg_seconds=outcar_metrics.loop_avg_seconds,
        loop_count=outcar_metrics.loop_count,
        electronic_converged=outcar_metrics.electronic_converged,
        ionic_converged=outcar_metrics.ionic_converged,
        oszicar_steps=parse_oszicar(workdir / "OSZICAR"),
        status=outcar_metrics.status,
        errors=outcar_metrics.errors,
    )
