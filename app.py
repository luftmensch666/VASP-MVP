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
from vasp_mvp.db import connect, list_tasks, update_task_status
from vasp_mvp.i18n import t
from vasp_mvp.models import TaskDraft, TaskRecord, TaskRequest
from vasp_mvp.parser import parse_metrics
from vasp_mvp.renderers import build_draft
from vasp_mvp.runner import run_dir, start_vasp, stop_task, tail_file, write_confirmed_task
from vasp_mvp.rules import default_kpoints
from vasp_mvp.security import safe_task_id
from vasp_mvp.structure_io import read_structure_upload


TASK_TYPES = ("relax", "static", "molecule", "adsorption")
RUN_RANKS = (20, 24)


@st.cache_resource
def resources():
    config = load_app_config()
    return config, load_potcar_config(), connect(config.workspace)


def current_lang() -> str:
    return st.session_state.get("lang", "zh")


def tr(key: str, **kwargs) -> str:
    return t(key, current_lang(), **kwargs)


def task_type_label(task_type: str) -> str:
    return tr(f"task_type.{task_type}")


def status_label(status: str) -> str:
    return tr(f"status.{status}")


def metric_status_label(status: str) -> str:
    mapped = tr(f"metric_status.{status}")
    return status if mapped.startswith("[[missing:") else mapped


def bool_label(value: bool | None) -> str:
    if value is None:
        return tr("value.none")
    return tr("value.true") if value else tr("value.false")


def none_text(value) -> str:
    return tr("value.none") if value is None else str(value)


def parse_incar_overrides(text: str) -> dict[str, str]:
    overrides: dict[str, str] = {}
    for line in text.splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        if "=" not in line:
            raise ValueError(tr("error.invalid_incar_override", line=line))
        key, value = line.split("=", 1)
        overrides[key.strip().upper()] = value.strip()
    return overrides


def parse_kpoints(text: str) -> tuple[int, int, int]:
    values = tuple(int(value) for value in text.split())
    if len(values) != 3:
        raise ValueError(tr("error.invalid_kpoints_count"))
    if any(value < 1 for value in values):
        raise ValueError(tr("error.invalid_kpoints_positive"))
    return values


def language_selector() -> None:
    if "lang" not in st.session_state:
        st.session_state["lang"] = "zh"
    selected = st.sidebar.selectbox(
        t("language.label", current_lang()),
        ["zh", "en"],
        index=["zh", "en"].index(current_lang()),
        format_func=lambda code: t(f"language.{code}", current_lang()),
    )
    st.session_state["lang"] = selected


def sidebar_new_task(config, potcars) -> tuple[TaskDraft | None, bool]:
    st.sidebar.header(tr("sidebar.new_task"))
    uploaded = st.sidebar.file_uploader(
        tr("sidebar.structure_file"),
        help=tr("sidebar.structure_file_help"),
    )
    task_type = st.sidebar.selectbox(
        tr("sidebar.task_type"),
        TASK_TYPES,
        format_func=task_type_label,
    )
    task_id = st.sidebar.text_input(
        tr("sidebar.task_id"),
        f"{task_type}-{datetime.utcnow():%Y%m%d-%H%M%S}",
    )

    allowed_ranks = tuple(rank for rank in config.allowed_mpi_ranks if rank in RUN_RANKS) or RUN_RANKS
    default_rank = 20 if 20 in allowed_ranks else allowed_ranks[0]
    ranks = st.sidebar.selectbox(
        tr("sidebar.mpi_ranks"),
        allowed_ranks,
        index=allowed_ranks.index(default_rank),
    )

    default_grid = default_kpoints(task_type)
    kpoints = st.sidebar.text_input(
        tr("sidebar.kpoints"),
        " ".join(str(value) for value in default_grid),
    )
    overrides = st.sidebar.text_area(
        tr("sidebar.incar_overrides"),
        placeholder=tr("placeholder.incar_overrides"),
    )
    dry_run = st.sidebar.checkbox(tr("sidebar.dry_run"), value=True)

    if st.sidebar.button(tr("button.generate_draft"), disabled=uploaded is None, type="primary"):
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
    st.subheader(tr("draft.preview"))
    st.caption(
        tr(
            "draft.caption",
            task_id=draft.request.task_id,
            task_type=task_type_label(draft.request.task_type),
            species=", ".join(draft.request.structure.elements),
        )
    )
    if draft.missing_potcars:
        st.warning(tr("warning.missing_potcar", elements=", ".join(draft.missing_potcars)))

    tabs = st.tabs(
        [
            tr("tabs.poscar"),
            tr("tabs.incar"),
            tr("tabs.kpoints"),
            tr("tabs.potcar_command"),
            tr("tabs.run_sh"),
        ]
    )
    tabs[0].code(draft.request.structure.poscar_text, language="text")
    tabs[1].code(draft.incar_text, language="text")
    tabs[2].code(draft.kpoints_text, language="text")
    tabs[3].code(draft.potcar_command, language="bash")
    tabs[4].code(draft.run_sh_text, language="bash")


def draft_actions(config, potcars, conn, draft: TaskDraft, dry_run: bool) -> None:
    confirm_disabled = bool(draft.missing_potcars)
    if st.button(tr("button.commit_write"), disabled=confirm_disabled):
        task_root = write_confirmed_task(config, potcars, draft, conn)
        st.session_state["confirmed_task_id"] = draft.request.task_id
        st.success(tr("success.created_run_dir", path=run_dir(task_root)))

    confirmed = st.session_state.get("confirmed_task_id") == draft.request.task_id
    if st.button(tr("button.start_vasp"), disabled=not confirmed):
        process = start_vasp(config, draft, conn, dry_run=dry_run)
        if process is None:
            st.success(tr("success.dry_run_complete"))
        else:
            st.success(tr("success.started_vasp_pid", pid=process.pid))


def selected_task(records: list[TaskRecord]) -> TaskRecord | None:
    if not records:
        return None
    show_task_list(records)
    return st.selectbox(
        tr("task.select"),
        records,
        format_func=lambda task: f"{task.task_id} | {task_type_label(task.task_type)} | {status_label(task.status)}",
    )


def show_task_list(records: list[TaskRecord]) -> None:
    st.subheader(tr("task.list"))
    st.dataframe(
        pd.DataFrame(
            [
                {
                    tr("table.task_id"): task.task_id,
                    tr("table.project"): task.project,
                    tr("table.type"): task_type_label(task.task_type),
                    tr("table.status"): status_label(task.status),
                    tr("table.pid"): task.pid,
                    tr("table.return_code"): task.return_code,
                    tr("table.start_time"): task.start_time,
                    tr("table.end_time"): task.end_time,
                    tr("table.task_root"): str(task.task_root),
                }
                for task in records
            ]
        ),
        use_container_width=True,
        hide_index=True,
    )


def show_running_cards(records: list[TaskRecord]) -> None:
    running = [task for task in records if task.status == "running"]
    if not running:
        return
    st.subheader(tr("task.running"))
    for task in running:
        with st.container(border=True):
            cols = st.columns(4)
            cols[0].metric(tr("metric.task"), task.task_id)
            cols[1].metric(tr("metric.pid"), none_text(task.pid))
            cols[2].metric(tr("metric.type"), task_type_label(task.task_type))
            started = None if task.start_time is None else task.start_time.isoformat(sep=" ")
            cols[3].metric(tr("metric.started"), none_text(started))


def show_stop_control(conn, task: TaskRecord) -> None:
    if task.status != "running" or task.pid is None:
        return
    st.subheader(tr("stop.title"))
    confirm_key = f"confirm_stop_{task.task_id}"
    st.checkbox(tr("stop.confirm"), key=confirm_key)
    if st.button(tr("button.stop_task"), disabled=not st.session_state.get(confirm_key, False)):
        try:
            stop_task(task.pid)
            update_task_status(conn, task.task_id, "stopped", end_time=datetime.utcnow())
            st.success(tr("success.stop_signal_sent"))
        except ProcessLookupError as exc:
            update_task_status(conn, task.task_id, "failed", end_time=datetime.utcnow(), return_code=-1)
            st.error(str(exc))
        except Exception as exc:
            st.error(str(exc))


def show_monitor(task: TaskRecord) -> None:
    st.subheader(tr("monitor.title"))
    task_run = run_dir(task.task_root)
    st.caption(str(task_run))

    auto_refresh = st.checkbox(tr("monitor.auto_refresh"), value=False)
    log_tabs = st.tabs([tr("tabs.vasp_out"), tr("tabs.oszicar")])
    log_tabs[0].code(tail_file(task_run / "vasp.out"), language="text")
    log_tabs[1].code(tail_file(task_run / "OSZICAR"), language="text")
    if auto_refresh:
        time.sleep(3)
        st.rerun()


def show_results(task: TaskRecord) -> None:
    st.subheader(tr("results.title"))
    metrics = parse_metrics(run_dir(task.task_root))
    cols = st.columns(4)
    cols[0].metric(tr("metric.toten"), tr("value.none") if metrics.toten_ev is None else f"{metrics.toten_ev:.8f}")
    cols[1].metric(
        tr("metric.loop_avg"),
        tr("value.none") if metrics.loop_avg_seconds is None else f"{metrics.loop_avg_seconds:.3f}",
    )
    cols[2].metric(tr("metric.converged"), bool_label(metrics.ionic_converged))
    cols[3].metric(tr("metric.status"), metric_status_label(metrics.status))

    if metrics.oszicar_steps:
        st.line_chart(pd.DataFrame({tr("chart.energy"): metrics.oszicar_steps}))

    if task.task_type == "adsorption":
        show_adsorption_table(default_ads=metrics.toten_ev)


def show_adsorption_table(default_ads: float | None = None) -> None:
    st.subheader(tr("adsorption.title"))
    cols = st.columns(3)
    ads = cols[0].number_input(tr("adsorption.ads_static"), value=default_ads, format="%.8f")
    slab = cols[1].number_input(tr("adsorption.slab_static"), value=None, format="%.8f")
    mol = cols[2].number_input(tr("adsorption.mol_static"), value=None, format="%.8f")

    result = calculate_raw_adsorption_energy(ads, slab, mol)
    table = pd.DataFrame(
        [
            {tr("table.term"): tr("adsorption.term.ads_static"), tr("table.value_ev"): ads},
            {tr("table.term"): tr("adsorption.term.slab_static"), tr("table.value_ev"): slab},
            {tr("table.term"): tr("adsorption.term.mol_static"), tr("table.value_ev"): mol},
            {tr("table.term"): tr("adsorption.term.ads_raw"), tr("table.value_ev"): result.energy_ev},
        ]
    )
    st.dataframe(table, use_container_width=True, hide_index=True)
    if result.ok:
        st.caption(tr("adsorption.raw_only"))
    else:
        missing = []
        if ads is None:
            missing.append(tr("adsorption.ads_static"))
        if slab is None:
            missing.append(tr("adsorption.slab_static"))
        if mol is None:
            missing.append(tr("adsorption.mol_static"))
        st.warning(tr("warning.adsorption_missing", fields=", ".join(missing)))


def main() -> None:
    st.set_page_config(page_title=t("app.page_title", "zh"), layout="wide")
    language_selector()
    config, potcars, conn = resources()

    st.title(tr("app.title"))
    try:
        draft, dry_run = sidebar_new_task(config, potcars)
    except Exception as exc:
        st.sidebar.error(str(exc))
        draft, dry_run = st.session_state.get("draft"), True

    if draft:
        show_draft_preview(draft)
        draft_actions(config, potcars, conn, draft, dry_run)
    else:
        st.info(tr("info.upload_generate"))

    st.divider()
    records = list_tasks(conn)
    task = selected_task(records)
    if task:
        show_running_cards(records)
        left, right = st.columns([1, 1])
        with left:
            show_monitor(task)
            show_stop_control(conn, task)
        with right:
            show_results(task)


if __name__ == "__main__":
    main()
