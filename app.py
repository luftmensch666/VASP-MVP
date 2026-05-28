from __future__ import annotations

import sys
import time
from datetime import datetime
from pathlib import Path

import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parent
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from vasp_mvp.adsorption import calculate_raw_adsorption_energy
from vasp_mvp.config import load_app_config, load_potcar_config
from vasp_mvp.db import connect, list_tasks
from vasp_mvp.models import TaskDraft, TaskRecord, TaskRequest
from vasp_mvp.parser import parse_metrics
from vasp_mvp.renderers import build_draft
from vasp_mvp.runner import run_dir, start_vasp, tail_file, write_confirmed_task
from vasp_mvp.rules import default_kpoints
from vasp_mvp.security import safe_task_id
from vasp_mvp.structure_io import read_structure_upload


TASK_TYPES = ("relax", "static", "molecule", "adsorption")
RUN_RANKS = (20, 24)


@st.cache_resource
def resources():
    config = load_app_config()
    return config, load_potcar_config(), connect(config.workspace)


def parse_incar_overrides(text: str) -> dict[str, str]:
    overrides: dict[str, str] = {}
    for line in text.splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        if "=" not in line:
            raise ValueError(f"Invalid INCAR override: {line}")
        key, value = line.split("=", 1)
        overrides[key.strip().upper()] = value.strip()
    return overrides


def parse_kpoints(text: str) -> tuple[int, int, int]:
    values = tuple(int(value) for value in text.split())
    if len(values) != 3:
        raise ValueError("KPOINTS grid must contain exactly three integers.")
    if any(value < 1 for value in values):
        raise ValueError("KPOINTS grid values must be positive integers.")
    return values


def sidebar_new_task(config, potcars) -> tuple[TaskDraft | None, bool]:
    st.sidebar.header("New Task")
    uploaded = st.sidebar.file_uploader("Structure file", help="CIF, POSCAR, CONTCAR, or .vasp")
    task_type = st.sidebar.selectbox("Task type", TASK_TYPES)
    task_id = st.sidebar.text_input("Task id", f"{task_type}-{datetime.utcnow():%Y%m%d-%H%M%S}")

    allowed_ranks = tuple(rank for rank in config.allowed_mpi_ranks if rank in RUN_RANKS) or RUN_RANKS
    default_rank = 20 if 20 in allowed_ranks else allowed_ranks[0]
    ranks = st.sidebar.selectbox("MPI ranks", allowed_ranks, index=allowed_ranks.index(default_rank))

    default_grid = default_kpoints(task_type)
    kpoints = st.sidebar.text_input("KPOINTS", " ".join(str(value) for value in default_grid))
    overrides = st.sidebar.text_area("INCAR overrides", placeholder="ENCUT = 520")
    dry_run = st.sidebar.checkbox("dry_run", value=True)

    if st.sidebar.button("Generate draft", disabled=uploaded is None, type="primary"):
        request = TaskRequest(
            task_id=safe_task_id(task_id),
            task_type=task_type,
            structure=read_structure_upload(uploaded.name, uploaded.getvalue()),
            mpi_ranks=int(ranks),
            kpoints=parse_kpoints(kpoints),
            incar_overrides=parse_incar_overrides(overrides),
        )
        st.session_state["draft"] = build_draft(config, potcars, request)
        st.session_state["confirmed_task_id"] = None

    return st.session_state.get("draft"), dry_run


def show_draft_preview(draft: TaskDraft) -> None:
    st.subheader("Draft Preview")
    st.caption(
        f"task_id={draft.request.task_id} | task_type={draft.request.task_type} | "
        f"species={', '.join(draft.request.structure.elements)}"
    )
    if draft.missing_potcars:
        st.warning("Missing POTCAR mapping or file: " + ", ".join(draft.missing_potcars))

    tabs = st.tabs(["POSCAR", "INCAR", "KPOINTS", "POTCAR command", "run.sh"])
    tabs[0].code(draft.request.structure.poscar_text, language="text")
    tabs[1].code(draft.incar_text, language="text")
    tabs[2].code(draft.kpoints_text, language="text")
    tabs[3].code(draft.potcar_command, language="bash")
    tabs[4].code(draft.run_sh_text, language="bash")


def draft_actions(config, potcars, conn, draft: TaskDraft, dry_run: bool) -> None:
    confirm_disabled = bool(draft.missing_potcars)
    if st.button("确认写入", disabled=confirm_disabled):
        task_root = write_confirmed_task(config, potcars, draft, conn)
        st.session_state["confirmed_task_id"] = draft.request.task_id
        st.success(f"Created run directory: {run_dir(task_root)}")

    confirmed = st.session_state.get("confirmed_task_id") == draft.request.task_id
    if st.button("启动 VASP", disabled=not confirmed):
        process = start_vasp(config, draft, conn, dry_run=dry_run)
        if process is None:
            st.success("dry_run complete; fake run/vasp.out written.")
        else:
            st.success(f"Started VASP PID {process.pid}")


def selected_task(records: list[TaskRecord]) -> TaskRecord | None:
    if not records:
        return None
    return st.selectbox(
        "Task",
        records,
        format_func=lambda task: f"{task.task_id} | {task.task_type} | {task.status}",
    )


def show_monitor(task: TaskRecord) -> None:
    st.subheader("Run Monitor")
    task_run = run_dir(task.task_root)
    st.caption(str(task_run))

    auto_refresh = st.checkbox("Auto refresh every 3 seconds", value=False)
    log_tabs = st.tabs(["vasp.out", "OSZICAR"])
    log_tabs[0].code(tail_file(task_run / "vasp.out"), language="text")
    log_tabs[1].code(tail_file(task_run / "OSZICAR"), language="text")
    if auto_refresh:
        time.sleep(3)
        st.rerun()


def show_results(task: TaskRecord) -> None:
    st.subheader("Results")
    metrics = parse_metrics(run_dir(task.task_root))
    cols = st.columns(4)
    cols[0].metric("TOTEN (eV)", "None" if metrics.toten_ev is None else f"{metrics.toten_ev:.8f}")
    cols[1].metric(
        "LOOP avg (s)",
        "None" if metrics.loop_avg_seconds is None else f"{metrics.loop_avg_seconds:.3f}",
    )
    cols[2].metric("Converged", "None" if metrics.ionic_converged is None else str(metrics.ionic_converged))
    cols[3].metric("Status", metrics.status)

    if metrics.oszicar_steps:
        st.line_chart(pd.DataFrame({"energy": metrics.oszicar_steps}))

    if task.task_type == "adsorption":
        show_adsorption_table(default_ads=metrics.toten_ev)


def show_adsorption_table(default_ads: float | None = None) -> None:
    st.subheader("E_ads")
    cols = st.columns(3)
    ads = cols[0].number_input("ads_static TOTEN", value=default_ads, format="%.8f")
    slab = cols[1].number_input("slab_static TOTEN", value=None, format="%.8f")
    mol = cols[2].number_input("mol_static TOTEN", value=None, format="%.8f")

    result = calculate_raw_adsorption_energy(ads, slab, mol)
    table = pd.DataFrame(
        [
            {"term": "E_ads_static", "value_eV": ads},
            {"term": "E_slab_static", "value_eV": slab},
            {"term": "E_mol_static", "value_eV": mol},
            {"term": "E_ads_raw", "value_eV": result.energy_ev},
        ]
    )
    st.dataframe(table, use_container_width=True, hide_index=True)
    if result.ok:
        st.caption(result.correction)
    else:
        st.warning(result.message)


def main() -> None:
    st.set_page_config(page_title="VASP MVP", layout="wide")
    config, potcars, conn = resources()

    st.title("VASP Local Workflow MVP")
    try:
        draft, dry_run = sidebar_new_task(config, potcars)
    except Exception as exc:
        st.sidebar.error(str(exc))
        draft, dry_run = st.session_state.get("draft"), True

    if draft:
        show_draft_preview(draft)
        draft_actions(config, potcars, conn, draft, dry_run)
    else:
        st.info("Upload a structure in the left sidebar and generate a draft.")

    st.divider()
    task = selected_task(list_tasks(conn))
    if task:
        left, right = st.columns([1, 1])
        with left:
            show_monitor(task)
        with right:
            show_results(task)


if __name__ == "__main__":
    main()
