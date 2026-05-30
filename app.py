from __future__ import annotations

import json
import re
import shutil
import sqlite3
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
from vasp_mvp.adsorption_results import calculate_adsorption_energy, parse_adsorption_workflow_jobs
from vasp_mvp.adsorption_visualization import (
    MISSING_VALUE,
    build_adsorption_summary_rows,
    build_job_metrics_table,
    build_loop_time_summary,
    build_oszicar_steps_table,
    build_total_energy_chart_data,
    format_energy as format_visual_energy,
)
from vasp_mvp.adsorption_wizard import (
    WIZARD_STEPS,
    adopt_clean_poscar_candidate,
    artifact_path,
    can_use_structure_edit_step,
    load_wizard_state,
    poscar_summary,
    require_relax_contcars,
    save_candidate_file,
    save_wizard_state,
    step_status_rows,
    sync_clean_poscar_to_structure,
)
from vasp_mvp.adsorption_wizard_relax import (
    RELAX_STEP_ORDER,
    check_relax_source_poscars,
    check_relax_package_consistency,
    create_relax_jobs_from_inputs,
    generate_all_relax_input_packages,
)
from vasp_mvp.adsorption_workflow import create_adsorption_workflow
from vasp_mvp.config import load_app_config, load_potcar_config
from vasp_mvp.data_management import (
    delete_input_set,
    delete_legacy_task,
    delete_workflow,
    factory_reset,
    preview_delete_input_set,
    preview_delete_legacy_task,
    preview_delete_workflow,
    preview_factory_reset,
    update_input_set_metadata,
    update_job_metadata,
    update_legacy_task_metadata,
    update_workflow_metadata,
)
from vasp_mvp.db import connect, create_task, db_path as workspace_db_path, list_tasks, update_task_status
from vasp_mvp.i18n import t
from vasp_mvp.input_sets import (
    EDITABLE_INPUT_FILES,
    bind_input_set_to_task,
    build_input_file_hashes,
    create_input_set,
    input_set_file_paths,
    list_input_sets,
    list_usable_input_sets,
    rename_input_set,
    save_editable_input_file,
    update_input_set_status,
)
from vasp_mvp.input_set_validation import parse_potcar_summary, validate_input_set
from vasp_mvp.method_advisor import MethodDescription, generate_method_description
from vasp_mvp.models import InputSet, TaskDraft, TaskRecord, TaskRequest
from vasp_mvp.jobs import get_job_metrics, list_jobs
from vasp_mvp.parser import parse_metrics
from vasp_mvp.renderers import build_draft
from vasp_mvp.runner import run_dir, start_task_record, start_vasp, stop_task, tail_file, validate_run_inputs, write_confirmed_task
from vasp_mvp.rules import default_kpoints
from vasp_mvp.security import safe_task_id, task_dir
from vasp_mvp.structure_io import read_structure_upload
from vasp_mvp.vaspkit_options import get_vaspkit_section, validate_vaspkit_values
from vasp_mvp.vaspkit_runner import VaspkitRequest, VaspkitResult, generate_vasp_inputs_with_vaspkit, sha256_file, summarize_potcar
from vasp_mvp.vaspkit_structure_editor import (
    build_105_inputs,
    build_401_inputs,
    build_801_inputs,
    build_803_inputs,
    build_402_atom_indices_inputs,
    build_402_z_range_inputs,
    expected_401_output_name,
    expected_803_output_name,
    run_vaspkit_structure_step,
)
from vasp_mvp.workflow_runner import (
    get_workflow_job_process_state,
    get_workflow_job_log_paths,
    refresh_workflow_job_status,
    start_workflow_job,
    stop_workflow_job,
    tail_workflow_job_file,
)
from vasp_mvp.workflows import create_workflow, list_jobs_for_workflow, list_workflows


TASK_TYPES = ("relax", "static", "molecule", "adsorption")
RUN_RANKS = (20, 24)
VASPKIT_COMMON_INCAR_KEYS = ("SR", "ST", "BD", "PU", "D3", "H6")
INPUT_SET_FILTERS = ("all", "usable", "dry_run", "invalid", "edited")
CORE_INPUT_FILES = ("INCAR", "POSCAR", "KPOINTS", "POTCAR")
INPUT_SET_TASK_MODULES = (
    "single_atom_catalysis",
    "molecule_optimization",
    "slab_optimization",
    "static_single_point",
)
INPUT_SET_TASK_TYPE_MAP = {
    "single_atom_catalysis": "static",
    "molecule_optimization": "molecule",
    "slab_optimization": "relax",
    "static_single_point": "static",
}
ADSORPTION_INPUT_ROLES = ("adsorbed", "clean_slab", "molecule_ref")
ADSORPTION_WORKFLOW_ROLES = ("clean_slab", "molecule_ref", "adsorbed_system")
ADSORPTION_METHOD_FAMILIES = ("DFT", "Hybrid DFT", "DFT+U", "Other")
ADSORPTION_FUNCTIONALS = ("PBE", "PBE-D3", "HSE06", "PBE+U", "Other")
NAVIGATION_PAGES = (
    "dashboard",
    "vaspkit",
    "input_sets",
    "adsorption",
    "single_atom",
    "molecule_optimization",
    "jobs_logs",
    "data_management",
    "settings",
)


@st.cache_resource
def resources():
    config = load_app_config()
    return config, load_potcar_config()


def current_lang() -> str:
    return st.session_state.get("lang", "zh")


def tr(key: str, **kwargs) -> str:
    return t(key, current_lang(), **kwargs)


def task_type_label(task_type: str) -> str:
    return tr(f"task_type.{task_type}")


def status_label(status: str) -> str:
    return tr(f"status.{status}")


def input_set_status_label(status: str) -> str:
    return tr(f"input_set.status.{status}")


def input_set_source_label(source: str) -> str:
    return tr(f"input_set.source.{source}")


def metric_status_label(status: str) -> str:
    mapped = tr(f"metric_status.{status}")
    return status if mapped.startswith("[[missing:") else mapped


def bool_label(value: bool | None) -> str:
    if value is None:
        return tr("value.none")
    return tr("value.true") if value else tr("value.false")


def input_set_filter_label(value: str) -> str:
    return tr(f"input_set.filter.{value}")


def input_set_task_module_label(value: str) -> str:
    return tr(f"input_set.task_module.{value}")


def input_set_role_label(value: str) -> str:
    return tr(f"input_set.role.{value}")


def workflow_role_label(value: str) -> str:
    return tr(f"workflow.role.{value}")


def workflow_status_label(value: str) -> str:
    return tr(f"workflow.status.{value}")


def calculation_type_label(value: str) -> str:
    return tr(f"calculation_type.{value}")


def adsorption_choice_label(prefix: str, value: str) -> str:
    key = value.lower().replace("+", "_plus_").replace(" ", "_")
    return tr(f"{prefix}.{key}")


def none_text(value) -> str:
    return tr("value.none") if value is None else str(value)


def format_energy(value) -> str:
    return tr("value.none") if value is None else f"{float(value):.6f} eV"


def format_seconds(value) -> str:
    return tr("value.none") if value is None else f"{float(value):.3f} s"


def format_adsorption_warning(warning) -> str:
    key = f"adsorption.warning.{warning.code}"
    message = tr(key)
    if message.startswith("[[missing:"):
        message = warning.message
    role = workflow_role_label(warning.role) if warning.role else None
    prefix = f"{role}: " if role else ""
    details = warning.details or {}
    debug_parts = []
    if "source" in details:
        debug_parts.append(f"source={details['source']!r}")
    if "label" in details:
        debug_parts.append(f"label={details['label']!r}")
    if "calculation_type" in details:
        debug_parts.append(f"calculation_type={details['calculation_type']!r}")
    suffix = f" ({', '.join(debug_parts)})" if debug_parts else ""
    return prefix + message + suffix


def format_adsorption_warning_codes(role: str, codes: tuple[str, ...]) -> str:
    messages = []
    for code in codes:
        key = f"adsorption.warning.{code}"
        message = tr(key)
        messages.append(message if not message.startswith("[[missing:") else code)
    return "; ".join(messages)


def vaspkit_option(section: str, key: str) -> dict:
    for option in get_vaspkit_section(section)["options"]:
        if option["key"] == key:
            return option
    raise KeyError(key)


def vaspkit_choice_label(option: dict, value: str) -> str:
    label_key = option.get("choice_label_keys", {}).get(value)
    return tr(label_key) if label_key else value


def input_set_name_exists(conn, name: str, exclude_input_set_id: str | None = None) -> bool:
    target = name.strip().lower()
    rows = conn.execute("SELECT input_set_id, name FROM input_sets").fetchall()
    return any(
        (exclude_input_set_id is None or row["input_set_id"] != exclude_input_set_id)
        and (row["name"] or "").strip().lower() == target
        for row in rows
    )


def workflow_name_exists(db_file: Path, name: str, exclude_workflow_id: str | None = None) -> bool:
    target = name.strip().lower()
    with sqlite3.connect(Path(db_file)) as conn:
        conn.row_factory = sqlite3.Row
        rows = conn.execute("SELECT workflow_id, name FROM workflows").fetchall()
    return any(
        (exclude_workflow_id is None or row["workflow_id"] != exclude_workflow_id)
        and (row["name"] or "").strip().lower() == target
        for row in rows
    )


def vaspkit_task_root(config, task_id: str) -> Path:
    return task_dir(config, safe_task_id(task_id))


def vaspkit_draft_dir(config, task_id: str) -> Path:
    return vaspkit_task_root(config, task_id) / "draft"


def vaspkit_input_set_dir(config, input_set_id: str) -> Path:
    return Path(config.workspace) / "input_sets" / safe_task_id(input_set_id)


def new_vaspkit_input_set_id(task_id: str) -> str:
    return safe_task_id(f"{task_id}-{datetime.utcnow():%Y%m%d-%H%M%S-%f}")


def new_adsorption_workflow_id(adsorbate_name: str) -> str:
    slug = re.sub(r"[^a-z0-9_]+", "_", adsorbate_name.strip().lower()).strip("_")
    if not slug:
        slug = "adsorbate"
    return safe_task_id(f"ads_{datetime.utcnow():%Y%m%d_%H%M%S}_{slug}")[:80]


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


def navigation_selector() -> str:
    st.sidebar.header(tr("sidebar.navigation"))
    selected = st.sidebar.radio(
        tr("sidebar.page"),
        NAVIGATION_PAGES,
        format_func=lambda page: tr(f"nav.{page}"),
        label_visibility="collapsed",
    )
    st.sidebar.caption(tr("sidebar.system_status"))
    return selected


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


def show_task_start_control(config, conn, task: TaskRecord) -> None:
    if task.status != "committed":
        return
    st.subheader(tr("task.start_committed"))
    missing = validate_run_inputs(task.task_root)
    if missing:
        st.error(tr("error.task_run_inputs_missing", files=", ".join(missing)))
        return
    dry_run = st.checkbox(tr("sidebar.dry_run"), value=True, key=f"start_dry_run_{task.task_id}")
    ranks = st.selectbox(
        tr("sidebar.mpi_ranks"),
        config.allowed_mpi_ranks,
        index=tuple(config.allowed_mpi_ranks).index(config.default_mpi_ranks)
        if config.default_mpi_ranks in config.allowed_mpi_ranks
        else 0,
        key=f"start_ranks_{task.task_id}",
    )
    if st.button(tr("button.start_vasp"), key=f"start_committed_{task.task_id}"):
        process = start_task_record(config, task, conn, ranks=int(ranks), dry_run=dry_run)
        if process is None:
            st.success(tr("success.dry_run_complete"))
        else:
            st.success(tr("success.started_vasp_pid", pid=process.pid))
        st.rerun()


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


def register_vaspkit_input_set(conn, result: VaspkitResult) -> None:
    if result.input_set_path is None or not result.input_set_path.exists():
        return
    data = json.loads(result.input_set_path.read_text(encoding="utf-8"))
    create_input_set(
        conn,
        input_set_id=data["input_set_id"],
        name=data["name"],
        source=data["source"],
        status=data["status"],
        usable_for_vasp=bool(data["usable_for_vasp"]),
        root_dir=Path(data["root_dir"]),
        incar_path=Path(data["incar_path"]),
        poscar_path=Path(data["poscar_path"]),
        kpoints_path=Path(data["kpoints_path"]),
        potcar_path=Path(data["potcar_path"]),
        notes=data.get("notes", ""),
    )


def commit_vaspkit_draft(
    conn,
    request: VaspkitRequest,
    task_type: str,
    task_id: str,
    task_root: Path,
) -> list[str]:
    warnings: list[str] = []
    source_dir = request.draft_dir
    target_dir = run_dir(task_root)
    target_dir.mkdir(parents=True, exist_ok=True)
    for filename in ("POSCAR", "INCAR", "KPOINTS", "POTCAR"):
        source = source_dir / filename
        if source.exists():
            shutil.copy2(source, target_dir / filename)
        else:
            warnings.append(tr("warning.vaspkit_missing_commit_file", filename=filename))
    placeholder = source_dir / "POTCAR.placeholder"
    if placeholder.exists() and not (source_dir / "POTCAR").exists():
        shutil.copy2(placeholder, target_dir / "POTCAR.placeholder")
    create_task(
        conn,
        task_id=task_id,
        project="default",
        task_type=task_type,
        task_root=task_root,
        status="committed",
    )
    return warnings


def show_generated_file_status(result: VaspkitResult) -> None:
    st.subheader(tr("vaspkit.generated_file_status"))
    rows = []
    for filename in ("POSCAR", "INCAR", "KPOINTS", "POTCAR", "POTCAR.placeholder"):
        path = result.draft_dir / filename
        rows.append(
            {
                tr("table.file"): filename,
                tr("table.exists"): bool_label(path.exists()),
                tr("table.size_bytes"): path.stat().st_size if path.exists() else 0,
            }
        )
    st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)


def show_generated_file_preview(result: VaspkitResult) -> None:
    st.subheader(tr("vaspkit.preview_generated_files"))
    tabs = st.tabs([tr("tabs.poscar"), tr("tabs.incar"), tr("tabs.kpoints"), tr("tabs.potcar")])
    for tab, filename in zip(tabs[:3], ("POSCAR", "INCAR", "KPOINTS")):
        path = result.draft_dir / filename
        with tab:
            if path.exists():
                st.code(path.read_text(encoding="utf-8", errors="replace"), language="text")
            else:
                st.info(tr("info.file_not_generated", filename=filename))
    with tabs[3]:
        summary = result.potcar_summary or {}
        show_potcar_summary_table(summary)
        if not summary.get("exists"):
            st.warning(tr("warning.potcar_not_generated"))
        st.caption(tr("vaspkit.potcar_no_full_preview"))


def show_potcar_summary_table(summary: dict) -> None:
    st.markdown(f"**{tr('input_set.potcar_summary.title')}**")
    st.write(
        {
            tr("table.exists"): bool_label(summary.get("exists")),
            tr("table.size_bytes"): summary.get("size_bytes", 0),
            tr("table.sha256"): summary.get("sha256", ""),
            tr("input_set.potcar_summary.number_of_potentials"): summary.get("number_of_potentials", 0),
            tr("table.element_order"): summary.get("element_order", []),
            tr("input_set.potcar_summary.potential"): summary.get("potential_labels", []),
        }
    )
    potentials = summary.get("potentials") or []
    if potentials:
        st.dataframe(
            pd.DataFrame(
                [
                    {
                        tr("input_set.potcar_summary.element"): item.get("element"),
                        tr("input_set.potcar_summary.potential"): item.get("potential_label"),
                        tr("input_set.potcar_summary.family"): item.get("paw_family"),
                        tr("input_set.potcar_summary.vrhfin"): item.get("vrhfin"),
                        tr("input_set.potcar_summary.titel"): item.get("titel"),
                        tr("input_set.potcar_summary.enmax"): item.get("enmax"),
                        tr("input_set.potcar_summary.enmin"): item.get("enmin"),
                        tr("input_set.potcar_summary.zval"): item.get("zval"),
                    }
                    for item in potentials
                ]
            ),
            use_container_width=True,
            hide_index=True,
        )


def show_vaspkit_input_generator(config, conn) -> None:
    st.subheader(tr("vaspkit.generator.title"))
    input_set_name = st.text_input(
        tr("data_management.name"),
        value="",
        placeholder=tr("input_set.name_placeholder"),
        help=tr("input_set.auto_id"),
        key="vaspkit_input_set_name",
    )
    input_set_notes = st.text_area(
        tr("input_set.notes"),
        value="",
        key="vaspkit_input_set_notes",
    )
    vaspkit_bin = st.text_input(
        tr("vaspkit.bin.label"),
        "vaspkit",
        help=tr("vaspkit.bin.help"),
    )
    vaspkit_dry_run = st.checkbox(tr("sidebar.dry_run"), value=True, key="vaspkit_dry_run")

    uploaded_cif_option = vaspkit_option("poscar", "uploaded_cif")
    uploaded_cif = st.file_uploader(
        tr(uploaded_cif_option["label_key"]),
        type=["cif"],
        help=tr(uploaded_cif_option["help_key"]),
        key="vaspkit_uploaded_cif",
    )
    if uploaded_cif is not None:
        st.info(tr("vaspkit.uploaded_file_summary", filename=uploaded_cif.name, size=uploaded_cif.size))
    st.info(tr("vaspkit.full_input_set_only"))

    st.subheader(tr(get_vaspkit_section("poscar")["label_key"]))
    st.caption(tr(get_vaspkit_section("poscar")["help_key"]))
    element_order_mode_option = vaspkit_option("poscar", "element_order_mode")
    element_order_mode = st.selectbox(
        tr(element_order_mode_option["label_key"]),
        element_order_mode_option["choices"],
        index=element_order_mode_option["choices"].index(element_order_mode_option["default"]),
        help=tr(element_order_mode_option["help_key"]),
        format_func=lambda value: vaspkit_choice_label(element_order_mode_option, value),
    )
    custom_order_option = vaspkit_option("poscar", "custom_element_order")
    custom_element_order = st.text_input(
        tr(custom_order_option["label_key"]),
        value=custom_order_option["default"],
        help=tr(custom_order_option["help_key"]),
        placeholder=tr("vaspkit.placeholder.custom_element_order"),
        disabled=element_order_mode != "custom",
    )

    st.subheader(tr(get_vaspkit_section("incar")["label_key"]))
    st.caption(tr(get_vaspkit_section("incar")["help_key"]))
    incar_keys_option = vaspkit_option("incar", "incar_key_parameters")
    common_incar_key = st.selectbox(
        tr("vaspkit.common_incar_key.label"),
        VASPKIT_COMMON_INCAR_KEYS,
        help=tr("vaspkit.common_incar_key.help"),
        format_func=lambda value: vaspkit_choice_label(incar_keys_option, value),
    )
    custom_incar_option = vaspkit_option("incar", "incar_custom_key_string")
    incar_custom_key_string = st.text_input(
        tr(custom_incar_option["label_key"]),
        value=custom_incar_option["default"],
        help=tr(custom_incar_option["help_key"]),
        placeholder=tr("vaspkit.placeholder.incar_custom_key_string"),
    )

    st.subheader(tr(get_vaspkit_section("kpoints")["label_key"]))
    st.caption(tr(get_vaspkit_section("kpoints")["help_key"]))
    kmesh_scheme_option = vaspkit_option("kpoints", "kmesh_scheme")
    kmesh_scheme = st.selectbox(
        tr(kmesh_scheme_option["label_key"]),
        kmesh_scheme_option["choices"],
        index=kmesh_scheme_option["choices"].index(kmesh_scheme_option["default"]),
        help=tr(kmesh_scheme_option["help_key"]),
        format_func=lambda value: vaspkit_choice_label(kmesh_scheme_option, value),
    )
    accuracy_preset_option = vaspkit_option("kpoints", "accuracy_preset")
    accuracy_preset = st.selectbox(
        tr(accuracy_preset_option["label_key"]),
        accuracy_preset_option["choices"],
        index=accuracy_preset_option["choices"].index(accuracy_preset_option["default"]),
        help=tr(accuracy_preset_option["help_key"]),
        format_func=lambda value: vaspkit_choice_label(accuracy_preset_option, value),
    )
    kmesh_value_option = vaspkit_option("kpoints", "kmesh_resolved_value")
    kmesh_resolved_value = st.number_input(
        tr(kmesh_value_option["label_key"]),
        min_value=0.0,
        value=float(kmesh_value_option["default"]),
        step=0.01,
        format="%.3f",
        help=tr(kmesh_value_option["help_key"]),
    )

    st.subheader(tr(get_vaspkit_section("potcar")["label_key"]))
    st.caption(tr(get_vaspkit_section("potcar")["help_key"]))
    potcar_mode = "103"
    st.info(tr("vaspkit.potcar_default_103_only"))
    potcar_policy_option = vaspkit_option("potcar", "existing_potcar_policy")
    existing_potcar_policy = st.selectbox(
        tr(potcar_policy_option["label_key"]),
        potcar_policy_option["choices"],
        index=potcar_policy_option["choices"].index(potcar_policy_option["default"]),
        help=tr(potcar_policy_option["help_key"]),
        format_func=lambda value: vaspkit_choice_label(potcar_policy_option, value),
    )

    request = {
        "poscar": {
            "uploaded_cif": None
            if uploaded_cif is None
            else {"filename": uploaded_cif.name, "size": uploaded_cif.size},
            "element_order_mode": element_order_mode,
            "custom_element_order": custom_element_order,
        },
        "incar": {
            "incar_key_parameters": [common_incar_key],
            "incar_custom_key_string": incar_custom_key_string,
        },
        "kpoints": {
            "kmesh_scheme": kmesh_scheme,
            "accuracy_preset": accuracy_preset,
            "kmesh_resolved_value": kmesh_resolved_value,
        },
        "potcar": {
            "potcar_mode": potcar_mode,
            "existing_potcar_policy": existing_potcar_policy,
        },
    }

    if st.button(tr("button.generate_vasp_input_set"), type="primary"):
        normalized_name = input_set_name.strip()
        if not normalized_name:
            st.error(tr("input_set.name_required"))
            return
        if input_set_name_exists(conn, normalized_name):
            st.error(tr("input_set.name_duplicate"))
            return
        input_set_id = new_vaspkit_input_set_id(normalized_name)
        draft_dir = vaspkit_input_set_dir(config, input_set_id)
        draft_dir.mkdir(parents=True, exist_ok=True)
        uploaded_cif_path = None
        if uploaded_cif is not None:
            uploaded_name = Path(uploaded_cif.name).name
            uploaded_cif_path = draft_dir / uploaded_name
            uploaded_cif_path.write_bytes(uploaded_cif.getvalue())

        validation_errors = []
        validation_errors.extend(
            validate_vaspkit_values(
                "poscar",
                {
                    "uploaded_cif": request["poscar"]["uploaded_cif"] or uploaded_cif_path,
                    "element_order_mode": element_order_mode,
                    "custom_element_order": custom_element_order,
                },
            )
        )
        validation_errors.extend(validate_vaspkit_values("incar", request["incar"]))
        validation_errors.extend(validate_vaspkit_values("kpoints", request["kpoints"]))
        validation_errors.extend(validate_vaspkit_values("potcar", request["potcar"]))
        if validation_errors:
            st.error(tr("error.vaspkit_validation", errors="; ".join(validation_errors)))
        else:
            vaspkit_request = VaspkitRequest(
                vaspkit_bin=vaspkit_bin,
                draft_dir=draft_dir,
                input_set_id=input_set_id,
                input_set_name=normalized_name,
                input_set_notes=input_set_notes.strip(),
                uploaded_cif_path=uploaded_cif_path,
                workspace=Path(config.workspace),
                element_order_mode=element_order_mode,
                custom_element_order=custom_element_order,
                incar_key_parameters=[common_incar_key],
                incar_custom_key_string=incar_custom_key_string,
                kmesh_scheme=kmesh_scheme,
                kmesh_resolved_value=float(kmesh_resolved_value),
                potcar_mode=potcar_mode,
                existing_potcar_policy=existing_potcar_policy,
            )
            result = generate_vasp_inputs_with_vaspkit(vaspkit_request, dry_run=vaspkit_dry_run)
            register_vaspkit_input_set(conn, result)
            st.session_state["vaspkit_request"] = vaspkit_request
            st.session_state["vaspkit_result"] = result
            if result.ok:
                st.success(tr("success.vaspkit_draft_saved", path=result.request_path))
                if result.input_set_id:
                    st.success(tr("success.input_set_saved", input_set_id=result.input_set_id, path=result.draft_dir))
            else:
                st.error(tr("error.vaspkit_generation_failed", errors="; ".join(result.errors)))

    result = st.session_state.get("vaspkit_result")
    request_obj = st.session_state.get("vaspkit_request")
    if result:
        st.subheader(tr("vaspkit.output_logs"))
        logs = st.tabs([tr("tabs.vaspkit_out"), tr("tabs.vaspkit_err")])
        logs[0].code(tail_file(result.stdout_path), language="text")
        err_text = tail_file(result.stderr_path)
        if err_text:
            logs[1].code(err_text, language="text")
        else:
            logs[1].info(tr("info.no_stderr"))
        show_generated_file_status(result)
        show_generated_file_preview(result)
        if result.dry_run or result.input_set_status == "dry_run":
            st.warning(tr("warning.dry_run_input_set_not_usable"))
        st.subheader(tr("vaspkit.json_preview"))
        if result.request_path.exists():
            st.json(result.request_path.read_text(encoding="utf-8"))
        if result.warnings:
            st.warning("; ".join(result.warnings))
        if result.errors:
            st.error("; ".join(result.errors))
        st.info(tr("vaspkit.input_set_saved_no_legacy_task"))


def show_input_sets_page(config, conn) -> None:
    st.subheader(tr("input_set.title"))
    records = list_input_sets(conn)
    selected_filter = st.selectbox(
        tr("input_set.filter.label"),
        INPUT_SET_FILTERS,
        format_func=input_set_filter_label,
    )
    filtered = [record for record in records if input_set_matches_filter(record, selected_filter)]
    if not filtered:
        st.info(tr("input_set.no_records"))
        return

    show_input_set_table(filtered)
    selected = st.selectbox(
        tr("input_set.select"),
        filtered,
        format_func=lambda item: f"{item.input_set_id} | {item.name} | {input_set_status_label(item.status)}",
    )
    show_input_set_detail(config, conn, selected)


def input_set_matches_filter(record: InputSet, selected_filter: str) -> bool:
    if selected_filter == "all":
        return True
    if selected_filter == "usable":
        return record.usable_for_vasp
    return record.status == selected_filter


def show_input_set_table(records: list[InputSet]) -> None:
    st.dataframe(
        pd.DataFrame(
            [
                {
                    tr("table.input_set_id"): item.input_set_id,
                    tr("table.name"): item.name,
                    tr("table.source"): input_set_source_label(item.source),
                    tr("table.status"): input_set_status_label(item.status),
                    tr("table.usable_for_vasp"): bool_label(item.usable_for_vasp),
                    tr("table.created_at"): item.created_at,
                    tr("table.updated_at"): item.updated_at,
                    tr("table.root_dir"): str(item.root_dir),
                }
                for item in records
            ]
        ),
        use_container_width=True,
        hide_index=True,
    )


def show_input_set_detail(config, conn, input_set: InputSet) -> None:
    st.subheader(tr("input_set.details"))
    cols = st.columns(4)
    cols[0].metric(tr("table.input_set_id"), input_set.input_set_id)
    cols[1].metric(tr("table.status"), input_set_status_label(input_set.status))
    cols[2].metric(tr("table.usable_for_vasp"), bool_label(input_set.usable_for_vasp))
    cols[3].metric(tr("table.source"), input_set_source_label(input_set.source))
    st.caption(str(input_set.root_dir))
    if input_set.status == "dry_run":
        st.warning(tr("warning.dry_run_input_set_not_usable"))

    show_input_set_actions(conn, input_set)
    st.divider()
    show_input_set_file_summary(input_set)
    show_input_file_editor(conn, input_set)

    tabs = st.tabs(
        [
            tr("input_set.vaspkit_request"),
            tr("input_set.vaspkit_result"),
            tr("input_set.validation_summary"),
        ]
    )
    show_json_file(tabs[0], input_set.root_dir / "vaspkit_request.json")
    show_json_file(tabs[1], input_set.root_dir / "vaspkit_result.json")
    show_json_file(tabs[2], input_set.root_dir / "validation.json")


def show_input_file_editor(conn, input_set: InputSet) -> None:
    st.subheader(tr("input_set.editor.title"))
    tabs = st.tabs([tr("tabs.incar"), tr("tabs.poscar"), tr("tabs.kpoints"), tr("tabs.potcar")])
    paths = input_set_file_paths(input_set)
    for tab, filename in zip(tabs[:3], EDITABLE_INPUT_FILES):
        with tab:
            path = paths[filename]
            current_text = path.read_text(encoding="utf-8", errors="replace") if path.exists() else ""
            edited_text = st.text_area(
                tr("input_set.editor.file_content", filename=filename),
                value=current_text,
                height=320,
                key=f"editor_{input_set.input_set_id}_{filename}",
            )
            if st.button(tr("button.save_file"), key=f"save_{input_set.input_set_id}_{filename}"):
                result = save_editable_input_file(
                    input_set,
                    filename,
                    edited_text,
                    user_action="ui_save",
                )
                update_input_set_status(conn, input_set.input_set_id, "edited", usable_for_vasp=False)
                backup = result["backup_path"]
                if backup is None:
                    st.success(tr("success.input_file_saved_no_backup", filename=filename))
                else:
                    st.success(tr("success.input_file_saved", filename=filename, backup=backup))
                st.rerun()
    with tabs[3]:
        show_potcar_summary_table(parse_potcar_summary(input_set.potcar_path).to_dict())
        st.caption(tr("vaspkit.potcar_no_full_preview"))
        st.button(
            tr("button.regenerate_potcar"),
            disabled=True,
            help=tr("input_set.regenerate_potcar.help"),
        )


def show_input_set_actions(conn, input_set: InputSet) -> None:
    st.subheader(tr("input_set.actions"))
    current_validation = validate_input_set(input_set)
    rename_key = f"rename_{input_set.input_set_id}"
    new_name = st.text_input(
        tr("input_set.rename.label"),
        value=input_set.name,
        key=rename_key,
    )
    action_cols = st.columns(4)
    if action_cols[0].button(tr("button.rename_input_set"), key=f"rename_btn_{input_set.input_set_id}"):
        try:
            rename_input_set(conn, input_set.input_set_id, new_name)
            st.success(tr("success.input_set_renamed"))
            st.rerun()
        except ValueError as exc:
            st.error(tr(str(exc)) if not str(exc).startswith("[[") else str(exc))
    if action_cols[1].button(tr("button.validate_input_set"), key=f"validate_btn_{input_set.input_set_id}"):
        validation = validate_input_set(input_set)
        next_status = "dry_run" if input_set.status == "dry_run" else ("validated" if validation.usable_for_vasp else "invalid")
        next_usable = False if input_set.status == "dry_run" else bool(validation.usable_for_vasp)
        update_input_set_status(
            conn,
            input_set.input_set_id,
            next_status,
            usable_for_vasp=next_usable,
        )
        write_json_file(input_set.root_dir / "validation.json", validation.to_dict())
        write_json_file(input_set.root_dir / "file_hashes.json", build_input_file_hashes(input_set))
        show_validation_result(validation)
        if validation.usable_for_vasp:
            st.success(tr("success.input_set_validated"))
        elif input_set.status == "dry_run" and not validation.errors:
            st.warning(tr("warning.dry_run_input_set_not_usable"))
        else:
            st.warning(tr("warning.input_set_validation_failed", errors="; ".join(validation_issue_text(item) for item in validation.errors)))
        st.rerun()
    can_mark_usable = current_validation.usable_for_vasp and input_set.status != "dry_run"
    if action_cols[2].button(
        tr("button.mark_usable"),
        key=f"usable_btn_{input_set.input_set_id}",
        disabled=not can_mark_usable,
    ):
        update_input_set_status(conn, input_set.input_set_id, "validated", usable_for_vasp=True)
        st.success(tr("success.input_set_marked_usable"))
        st.rerun()
    if action_cols[3].button(tr("button.mark_not_usable"), key=f"not_usable_btn_{input_set.input_set_id}"):
        update_input_set_status(conn, input_set.input_set_id, "invalid", usable_for_vasp=False)
        st.success(tr("success.input_set_marked_not_usable"))
        st.rerun()

    show_validation_result(current_validation)


def show_validation_result(validation) -> None:
    st.markdown(f"**{tr('input_set.validation.title')}**")
    if validation.errors:
        st.error("\n".join(validation_issue_text(item) for item in validation.errors))
    if validation.warnings:
        st.warning("\n".join(validation_issue_text(item) for item in validation.warnings))
    if validation.infos:
        st.info("\n".join(validation_issue_text(item) for item in validation.infos))


def validation_issue_text(issue) -> str:
    message = tr(issue.message_key, **issue.details)
    return message if not message.startswith("[[missing:") else issue.code
def show_create_task_from_input_set(config, conn, input_set: InputSet) -> None:
    st.subheader(tr("input_set.create_task.title"))
    module = st.selectbox(
        tr("input_set.create_task.module"),
        INPUT_SET_TASK_MODULES,
        format_func=input_set_task_module_label,
        key=f"task_module_{input_set.input_set_id}",
    )
    task_id = st.text_input(
        tr("sidebar.task_id"),
        f"{module}-{datetime.utcnow():%Y%m%d-%H%M%S}",
        key=f"task_from_input_set_{input_set.input_set_id}",
    )
    if input_set.status == "dry_run":
        st.error(tr("error.input_set_dry_run_not_usable"))
    elif not input_set.usable_for_vasp:
        st.error(tr("error.input_set_not_usable"))
    missing = missing_input_set_files(input_set)
    if missing:
        st.error(tr("error.input_set_files_missing", files=", ".join(missing)))
    disabled = not input_set.usable_for_vasp or bool(missing)
    if st.button(tr("button.use_input_set_create_task"), disabled=disabled, key=f"create_task_{input_set.input_set_id}"):
        safe_id = safe_task_id(task_id)
        task_type = INPUT_SET_TASK_TYPE_MAP[module]
        task_root = task_dir(config, safe_id)
        task_run = run_dir(task_root)
        if task_run.exists() and any((task_run / filename).exists() for filename in CORE_INPUT_FILES):
            st.error(tr("error.task_run_already_exists", path=task_run))
            return
        task_run.mkdir(parents=True, exist_ok=True)
        copy_input_set_to_run(input_set, task_run)
        task_metadata = {
            "task_id": safe_id,
            "task_type": task_type,
            "module": module,
            "created_from": "input_set",
            "input_set_id": input_set.input_set_id,
            "created_at": datetime.utcnow().isoformat(timespec="seconds"),
        }
        binding_metadata = {
            "task_id": safe_id,
            "role": "primary",
            "input_set_id": input_set.input_set_id,
            "created_at": datetime.utcnow().isoformat(timespec="seconds"),
        }
        write_json_file(task_run / "task.json", task_metadata)
        write_json_file(task_run / "input_set_binding.json", binding_metadata)
        create_task(
            conn,
            task_id=safe_id,
            project="default",
            task_type=task_type,
            task_root=task_root,
            status="committed",
        )
        bind_input_set_to_task(conn, safe_id, "primary", input_set.input_set_id)
        st.success(tr("success.task_created_from_input_set", task_id=safe_id, path=task_run))
        st.rerun()


def missing_input_set_files(input_set: InputSet) -> list[str]:
    missing: list[str] = []
    for filename, path in input_set_file_paths(input_set).items():
        if not path.exists() or path.stat().st_size == 0:
            missing.append(filename)
    return missing


def copy_input_set_to_run(input_set: InputSet, task_run: Path) -> None:
    for filename, source in input_set_file_paths(input_set).items():
        shutil.copy2(source, task_run / filename)


def show_dashboard_page(db_file: Path) -> None:
    st.subheader(tr("dashboard.title"))
    cols = st.columns(3)
    cols[0].metric(tr("dashboard.input_sets"), count_table_rows(db_file, "input_sets"))
    cols[1].metric(tr("dashboard.workflows"), count_table_rows(db_file, "workflows"))
    cols[2].metric(tr("dashboard.jobs"), count_table_rows(db_file, "jobs"))
    workflows = list_workflows(db_file)[:5]
    if workflows:
        st.subheader(tr("dashboard.recent_workflows"))
        st.dataframe(
            pd.DataFrame(
                [
                    {
                        tr("table.workflow_id"): workflow.workflow_id,
                        tr("table.name"): workflow.name,
                        tr("table.status"): workflow_status_label(workflow.status),
                        tr("table.functional"): workflow.functional,
                        tr("table.updated_at"): workflow.updated_at,
                    }
                    for workflow in workflows
                ]
            ),
            use_container_width=True,
            hide_index=True,
        )
    else:
        st.info(tr("dashboard.no_workflows"))


def count_table_rows(db_file: Path, table: str) -> int:
    allowed = {"input_sets", "workflows", "jobs"}
    if table not in allowed:
        raise ValueError(f"Unsupported table for dashboard count: {table}")
    with sqlite3.connect(Path(db_file)) as conn:
        row = conn.execute(f"SELECT COUNT(*) FROM {table}").fetchone()
    return int(row[0])


def show_module_placeholder_page(title_key: str) -> None:
    st.subheader(tr(title_key))
    st.info(tr("module.placeholder"))


def show_settings_page() -> None:
    show_module_placeholder_page("settings.title")


def show_jobs_logs_page(db_file: Path) -> None:
    st.subheader(tr("jobs_logs.title"))
    st.info(tr("jobs_logs.legacy_notice"))
    conn = connect_to_legacy_tasks(db_file)
    try:
        records = list_tasks(conn)
    finally:
        conn.close()
    task = selected_task(records)
    if task:
        show_running_cards(records)
        show_monitor(task)


def connect_to_legacy_tasks(db_file: Path):
    conn = sqlite3.connect(Path(db_file))
    conn.row_factory = sqlite3.Row
    return conn


def show_data_management_page(config, db_file: Path) -> None:
    st.subheader(tr("data_management.title"))
    st.warning(tr("data_management.safety_notice"))
    sections = st.tabs(
        [
            tr("data_management.workflows"),
            tr("data_management.input_sets"),
            tr("data_management.legacy_tasks"),
            tr("data_management.factory_reset"),
        ]
    )
    with sections[0]:
        show_workflow_data_management(config, db_file)
    with sections[1]:
        show_input_set_data_management(config, db_file)
    with sections[2]:
        show_legacy_task_data_management(config, db_file)
    with sections[3]:
        show_factory_reset_section(config, db_file)


def show_workflow_data_management(config, db_file: Path) -> None:
    workflows = list_workflows(db_file)
    st.markdown(f"**{tr('data_management.workflows')}**")
    if workflows:
        st.dataframe(
            pd.DataFrame(
                [
                    {
                        tr("table.workflow_id"): workflow.workflow_id,
                        tr("table.name"): workflow.name,
                        tr("table.status"): workflow_status_label(workflow.status),
                        tr("table.method_family"): workflow.method_family,
                        tr("table.functional"): workflow.functional,
                        tr("table.created_at"): workflow.created_at,
                        tr("table.updated_at"): workflow.updated_at,
                        tr("table.root_dir"): str(workflow.root_dir),
                    }
                    for workflow in workflows
                ]
            ),
            use_container_width=True,
            hide_index=True,
        )
    else:
        st.info(tr("data_management.no_records"))

    selected = st.selectbox(
        tr("data_management.select_workflow"),
        workflows,
        format_func=lambda workflow: f"{workflow.name} | {workflow.workflow_id}",
        disabled=not workflows,
        key="dm_select_workflow",
    )
    if selected:
        show_metadata_editor(
            "workflow",
            selected.workflow_id,
            selected.name,
            selected.notes,
            lambda name, notes: update_workflow_metadata(db_file, selected.workflow_id, name=name, notes=notes),
        )
        show_delete_preview_and_action(
            entity_type="workflow",
            entity_id=selected.workflow_id,
            preview_fn=lambda: preview_delete_workflow(db_file, selected.workflow_id, workspace=config.workspace),
            delete_fn=lambda: delete_workflow(db_file, selected.workflow_id, workspace=config.workspace),
        )

    st.markdown(f"**{tr('data_management.jobs')}**")
    jobs = list_jobs(db_file)
    if jobs:
        st.dataframe(
            pd.DataFrame(
                [
                    {
                        tr("table.job_id"): job.job_id,
                        tr("table.name"): job.name,
                        tr("table.calculation_type"): calculation_type_label(job.calculation_type),
                        tr("table.status"): status_label(job.status),
                        tr("table.input_set_id"): job.input_set_id,
                        tr("table.run_dir"): str(job.run_dir),
                        tr("table.updated_at"): job.updated_at,
                    }
                    for job in jobs
                ]
            ),
            use_container_width=True,
            hide_index=True,
        )
        selected_job = st.selectbox(
            tr("data_management.select_job"),
            jobs,
            format_func=lambda job: f"{job.name or job.job_id} | {job.job_id}",
            key="dm_select_job",
        )
        show_metadata_editor(
            "job",
            selected_job.job_id,
            selected_job.name or "",
            selected_job.notes or "",
            lambda name, notes: update_job_metadata(db_file, selected_job.job_id, name=name, notes=notes),
        )
    else:
        st.info(tr("data_management.no_jobs"))


def show_input_set_data_management(config, db_file: Path) -> None:
    conn = connect_to_legacy_tasks(db_file)
    try:
        input_sets = list_input_sets(conn)
    finally:
        conn.close()
    if input_sets:
        st.dataframe(
            pd.DataFrame(
                [
                    {
                        tr("table.input_set_id"): item.input_set_id,
                        tr("table.name"): item.name,
                        tr("table.status"): input_set_status_label(item.status),
                        tr("table.usable_for_vasp"): bool_label(item.usable_for_vasp),
                        tr("table.created_at"): item.created_at,
                        tr("table.updated_at"): item.updated_at,
                        tr("table.root_dir"): str(item.root_dir),
                    }
                    for item in input_sets
                ]
            ),
            use_container_width=True,
            hide_index=True,
        )
    else:
        st.info(tr("data_management.no_records"))
    selected = st.selectbox(
        tr("data_management.select_input_set"),
        input_sets,
        format_func=lambda item: f"{item.name} | {item.input_set_id}",
        disabled=not input_sets,
        key="dm_select_input_set",
    )
    if selected:
        show_metadata_editor(
            "input_set",
            selected.input_set_id,
            selected.name,
            selected.notes,
            lambda name, notes: update_input_set_metadata(db_file, selected.input_set_id, name=name, notes=notes),
        )
        show_delete_preview_and_action(
            entity_type="input_set",
            entity_id=selected.input_set_id,
            preview_fn=lambda: preview_delete_input_set(db_file, selected.input_set_id, workspace=config.workspace),
            delete_fn=lambda: delete_input_set(db_file, selected.input_set_id, workspace=config.workspace),
        )


def show_legacy_task_data_management(config, db_file: Path) -> None:
    conn = connect_to_legacy_tasks(db_file)
    try:
        tasks = list_tasks(conn)
    finally:
        conn.close()
    if tasks:
        st.dataframe(
            pd.DataFrame(
                [
                    {
                        tr("table.task_id"): task.task_id,
                        tr("table.name"): task.name,
                        tr("table.status"): status_label(task.status),
                        tr("table.pid"): task.pid,
                        tr("table.task_root"): str(task.task_root),
                        tr("table.updated_at"): task.updated_at,
                    }
                    for task in tasks
                ]
            ),
            use_container_width=True,
            hide_index=True,
        )
    else:
        st.info(tr("data_management.no_records"))
    selected = st.selectbox(
        tr("data_management.select_legacy_task"),
        tasks,
        format_func=lambda task: f"{task.name or task.task_id} | {task.task_id}",
        disabled=not tasks,
        key="dm_select_legacy_task",
    )
    if selected:
        show_metadata_editor(
            "legacy_task",
            selected.task_id,
            selected.name or selected.project,
            selected.notes or "",
            lambda name, notes: update_legacy_task_metadata(db_file, selected.task_id, name=name, notes=notes),
        )
        show_delete_preview_and_action(
            entity_type="legacy_task",
            entity_id=selected.task_id,
            preview_fn=lambda: preview_delete_legacy_task(db_file, selected.task_id, workspace=config.workspace),
            delete_fn=lambda: delete_legacy_task(db_file, selected.task_id, workspace=config.workspace),
        )


def show_metadata_editor(entity_type: str, entity_id: str, name: str | None, notes: str | None, save_fn) -> None:
    with st.expander(tr("data_management.edit_metadata"), expanded=False):
        st.caption(tr("data_management.id_locked", entity_id=entity_id))
        name_value = st.text_input(tr("data_management.name"), value=name or "", key=f"dm_name_{entity_type}_{entity_id}")
        notes_value = st.text_area(tr("data_management.notes"), value=notes or "", key=f"dm_notes_{entity_type}_{entity_id}")
        if st.button(tr("data_management.save"), key=f"dm_save_{entity_type}_{entity_id}"):
            try:
                save_fn(name_value, notes_value)
                st.success(tr("data_management.save_success"))
                st.rerun()
            except Exception as exc:
                st.error(tr("data_management.save_failed", error=str(exc)))


def show_delete_preview_and_action(entity_type: str, entity_id: str, preview_fn, delete_fn) -> None:
    with st.expander(tr("data_management.delete_confirm"), expanded=False):
        preview = preview_fn()
        show_delete_result(preview, title_key="data_management.delete_preview")
        confirm = st.checkbox(
            tr("data_management.delete_checkbox", entity_id=entity_id),
            key=f"dm_confirm_delete_{entity_type}_{entity_id}",
        )
        typed = st.text_input(
            tr("data_management.delete_type_id", entity_id=entity_id),
            key=f"dm_delete_text_{entity_type}_{entity_id}",
        )
        if st.button(
            tr("data_management.delete_selected"),
            disabled=not confirm or typed != entity_id,
            key=f"dm_delete_{entity_type}_{entity_id}",
        ):
            try:
                result = delete_fn()
                show_delete_result(result, title_key="data_management.delete_result")
                if result.ok:
                    st.success(tr("data_management.delete_success"))
                    st.rerun()
                else:
                    st.error(tr("data_management.delete_failed"))
            except Exception as exc:
                st.error(tr("data_management.delete_failed_with_error", error=str(exc)))


def show_factory_reset_section(config, db_file: Path) -> None:
    st.warning(tr("data_management.factory_reset_warning"))
    preview = preview_factory_reset(db_file, workspace=config.workspace)
    show_delete_result(preview, title_key="data_management.factory_reset_preview")
    if preview.backup_path:
        st.info(tr("data_management.backup_path", path=preview.backup_path))
    confirm_text = st.text_input(tr("data_management.factory_reset_confirm_text"), key="dm_factory_reset_text")
    confirmed = st.checkbox(tr("data_management.factory_reset_checkbox"), key="dm_factory_reset_checkbox")
    if st.button(
        tr("data_management.factory_reset_button"),
        disabled=confirm_text != "CLEAR ALL" or not confirmed,
        key="dm_factory_reset_button",
    ):
        try:
            result = factory_reset(db_file, workspace=config.workspace)
            show_delete_result(result, title_key="data_management.factory_reset_result")
            if result.backup_path:
                st.info(tr("data_management.backup_path", path=result.backup_path))
            if result.ok:
                st.success(tr("data_management.factory_reset_success"))
                st.rerun()
            else:
                st.error(tr("data_management.factory_reset_failed"))
        except Exception as exc:
            st.error(tr("data_management.factory_reset_failed_with_error", error=str(exc)))


def show_delete_result(result, title_key: str) -> None:
    st.markdown(f"**{tr(title_key)}**")
    if result.deleted_db_records:
        st.write(tr("data_management.deleted_db_records"))
        st.code("\n".join(result.deleted_db_records), language="text")
    if result.deleted_paths:
        st.write(tr("data_management.deleted_paths"))
        st.code("\n".join(result.deleted_paths), language="text")
    if result.skipped_paths:
        st.write(tr("data_management.skipped_paths"))
        st.code("\n".join(result.skipped_paths), language="text")
    if result.warnings:
        st.warning("\n".join(result.warnings))
    if result.errors:
        st.error("\n".join(result.errors))

def show_adsorption_workflow_page(config, db_file: Path) -> None:
    show_adsorption_wizard_page(config, db_file)


def show_adsorption_wizard_page(config, db_file: Path) -> None:
    st.subheader(tr("adsorption_wizard.title"))
    show_create_adsorption_wizard_form(config, db_file)
    st.divider()
    workflows = list_workflows(db_file, workflow_type="adsorption")
    if not workflows:
        st.info(tr("adsorption.workflow.no_records"))
        return
    selected = st.selectbox(
        tr("adsorption.workflow.select"),
        workflows,
        format_func=lambda workflow: f"{workflow.name} | {workflow.workflow_id} | {workflow_status_label(workflow.status)}",
        key="adsorption_wizard_selected_workflow",
    )
    st.session_state["selected_adsorption_workflow_id"] = selected.workflow_id
    show_adsorption_wizard_detail(config, db_file, selected)


def show_create_adsorption_wizard_form(config, db_file: Path) -> None:
    with st.expander(tr("adsorption_wizard.create_workflow"), expanded=False):
        workflow_name = st.text_input(tr("adsorption.workflow.name"), key="wizard_workflow_name")
        adsorbate_name = st.text_input(
            tr("adsorption.workflow.adsorbate_name"),
            placeholder=tr("adsorption.workflow.adsorbate_placeholder"),
            key="wizard_adsorbate_name",
        )
        method_family = st.selectbox(
            tr("adsorption.workflow.method_family"),
            ADSORPTION_METHOD_FAMILIES,
            format_func=lambda value: adsorption_choice_label("adsorption.method_family", value),
            key="wizard_method_family",
        )
        functional = st.selectbox(
            tr("adsorption.workflow.functional"),
            ADSORPTION_FUNCTIONALS,
            format_func=lambda value: adsorption_choice_label("adsorption.functional", value),
            key="wizard_functional",
        )
        description = generate_method_description(
            method_family=method_family,
            functional=functional,
            system_type="adsorption",
            adsorbate_name=adsorbate_name,
        )
        st.markdown(f"**{tr('method_advisor.generated_description')}**")
        st.info(method_description_text(description))
        notes_key = "wizard_method_notes"
        if notes_key not in st.session_state:
            st.session_state[notes_key] = method_description_text(description)
        if st.button(tr("method_advisor.regenerate"), key="wizard_regenerate_method_notes"):
            st.session_state[notes_key] = method_description_text(description)
            st.rerun()
        method_notes = st.text_area(tr("adsorption.workflow.method_notes"), key=notes_key)
        notes = st.text_area(tr("adsorption.workflow.notes"), key="wizard_notes")
        if st.button(tr("button.create_adsorption_workflow"), type="primary", key="wizard_create_workflow"):
            normalized_name = workflow_name.strip()
            if not normalized_name:
                st.error(tr("workflow.name_required"))
                return
            if workflow_name_exists(db_file, normalized_name):
                st.error(tr("workflow.name_duplicate"))
                return
            workflow_id = new_adsorption_workflow_id(adsorbate_name or normalized_name)
            root_dir = Path(config.workspace) / "workflows" / workflow_id
            try:
                method_notes_payload = json.dumps(
                    {
                        "zh": method_notes if current_lang() == "zh" else description.zh,
                        "en": method_notes if current_lang() == "en" else description.en,
                    },
                    ensure_ascii=False,
                )
                workflow = create_workflow(
                    db_file,
                    workflow_id=workflow_id,
                    workflow_type="adsorption",
                    name=normalized_name,
                    root_dir=root_dir,
                    status="draft",
                    method_family=method_family,
                    functional=functional,
                    method_notes=method_notes_payload,
                    notes=notes.strip(),
                )
                state = load_wizard_state(workflow.root_dir, workflow.workflow_id)
                save_wizard_state(workflow.root_dir, state)
            except Exception as exc:
                st.error(tr("error.adsorption_workflow_create_failed", error=str(exc)))
                return
            st.session_state["selected_adsorption_workflow_id"] = workflow.workflow_id
            st.success(tr("success.adsorption_workflow_created", workflow_id=workflow.workflow_id))
            st.rerun()


def show_adsorption_wizard_detail(config, db_file: Path, workflow) -> None:
    state = load_wizard_state(workflow.root_dir, workflow.workflow_id)
    st.markdown(f"**{workflow.name}**")
    st.caption(f"{workflow.workflow_id} | {workflow.root_dir}")
    show_wizard_status_table(workflow.root_dir, state)
    step = st.radio(
        tr("adsorption_wizard.step_navigation"),
        WIZARD_STEPS,
        format_func=lambda value: tr(f"adsorption_wizard.step.{value}"),
        horizontal=True,
        key=f"wizard_step_{workflow.workflow_id}",
    )
    if step == "clean_structure":
        show_wizard_step_clean_structure(config, workflow, state)
    elif step == "adsorbed_structure":
        show_wizard_step_adsorbed_structure(workflow, state)
    elif step == "molecule_reference":
        show_wizard_step_molecule_reference(workflow, state)
    elif step == "generate_relax_inputs":
        show_wizard_step_generate_relax_inputs(config, db_file, workflow, state)
    elif step == "run_relax_jobs":
        show_wizard_step_run_relax_jobs(db_file, workflow)
    elif step == "generate_static_inputs":
        ok, missing = require_relax_contcars(workflow.root_dir)
        if ok:
            st.info(tr("adsorption_wizard.step_not_implemented_10a"))
        else:
            st.warning(tr("adsorption_wizard.contcar_missing", files=", ".join(missing)))
    else:
        st.info(tr("adsorption_wizard.step_not_implemented_10a"))


def show_wizard_status_table(workflow_root: Path, state: dict) -> None:
    st.dataframe(
        pd.DataFrame(
            [
                {
                    tr("adsorption_wizard.step_number"): row["step_number"],
                    tr("adsorption_wizard.step_name"): tr(f"adsorption_wizard.step.{row['step']}"),
                    tr("table.status"): tr(f"adsorption_wizard.status.{row['status']}"),
                    tr("adsorption_wizard.required_artifact"): row["required_artifact"],
                    tr("table.exists"): bool_label(row["artifact_exists"]),
                }
                for row in step_status_rows(workflow_root, state)
            ]
        ),
        use_container_width=True,
        hide_index=True,
    )


def show_wizard_step_clean_structure(config, workflow, state: dict) -> None:
    st.markdown(f"**{tr('adsorption_wizard.step.clean_structure')}**")
    source_type = st.selectbox(
        tr("adsorption_wizard.source_type.label"),
        ("already_slab", "porous_or_mof", "bulk_surface"),
        format_func=lambda value: tr(f"adsorption_wizard.source_type.{value}"),
        key=f"source_type_{workflow.workflow_id}",
    )
    state["source_model_type"] = source_type
    save_wizard_state(workflow.root_dir, state)
    st.info(tr(f"adsorption_wizard.source_type.{source_type}.help"))
    vaspkit_bin = st.text_input(tr("vaspkit.bin.label"), "vaspkit", key=f"wizard_vaspkit_bin_{workflow.workflow_id}")
    show_current_clean_poscar(workflow.root_dir, state)
    show_clean_poscar_candidate(workflow.root_dir, state)

    tabs = st.tabs(
        [
            tr("adsorption_wizard.clean.upload_poscar"),
            tr("adsorption_wizard.clean.cif_105"),
            tr("adsorption_wizard.clean.optional_steps"),
        ]
    )
    with tabs[0]:
        uploaded = st.file_uploader(
            tr("adsorption_wizard.clean.upload_poscar"),
            key=f"clean_poscar_upload_{workflow.workflow_id}",
        )
        if uploaded is not None and st.button(tr("adsorption_wizard.save_candidate"), key=f"save_clean_upload_{workflow.workflow_id}"):
            tmp = workflow.root_dir / "artifacts" / "clean" / "candidates" / Path(uploaded.name).name
            tmp.parent.mkdir(parents=True, exist_ok=True)
            tmp.write_bytes(uploaded.getvalue())
            state = save_candidate_file(
                workflow.root_dir,
                state,
                role="clean",
                source_path=tmp,
                candidate_name="uploaded_clean_POSCAR",
                source_step="upload",
                parameters={"filename": Path(uploaded.name).name},
            )
            st.success(tr("adsorption_wizard.candidate_saved"))
            st.rerun()
    with tabs[1]:
        element_order = st.text_input(tr("adsorption_wizard.element_order_optional"), key=f"wizard_105_elements_{workflow.workflow_id}")
        cif = st.file_uploader(tr("adsorption_wizard.clean.upload_cif"), type=["cif"], key=f"wizard_cif_{workflow.workflow_id}")
        if st.button(tr("adsorption_wizard.structure_editor.run_105"), disabled=cif is None, key=f"wizard_run_105_{workflow.workflow_id}"):
            try:
                workdir = workflow.root_dir / "structure"
                workdir.mkdir(parents=True, exist_ok=True)
                cif_path = workdir / Path(cif.name).name
                cif_path.write_bytes(cif.getvalue())
                result = run_vaspkit_structure_step(
                    vaspkit_bin,
                    build_105_inputs(cif_path.name, element_order),
                    workdir,
                    "105",
                    "POSCAR",
                )
                if not result.ok:
                    show_structure_edit_failure(result)
                else:
                    save_candidate_file(
                        workflow.root_dir,
                        state,
                        role="clean",
                        source_path=result.output_path,
                        candidate_name="POSCAR_105.vasp",
                        source_step="105",
                        parameters={"cif_filename": cif_path.name, "element_order": element_order.strip()},
                    )
                    st.success(tr("adsorption_wizard.structure_editor.candidate_created"))
                    st.rerun()
            except Exception as exc:
                st.error(tr("adsorption_wizard.structure_editor.invalid_parameter", error=str(exc)))
    with tabs[2]:
        show_structure_edit_forms(workflow, state, vaspkit_bin)


def show_current_clean_poscar(workflow_root: Path, state: dict) -> None:
    clean_rel = state.get("artifacts", {}).get("clean_poscar")
    if not clean_rel:
        st.warning(tr("adsorption_wizard.clean_poscar_missing"))
        return
    show_poscar_summary_block(tr("adsorption_wizard.current_clean_poscar"), artifact_path(workflow_root, clean_rel), workflow_root)


def show_clean_poscar_candidate(workflow_root: Path, state: dict) -> None:
    candidate_rel = state.get("candidates", {}).get("clean")
    if not candidate_rel:
        return
    candidate = artifact_path(workflow_root, candidate_rel)
    show_poscar_summary_block(tr("adsorption_wizard.current_candidate"), candidate, workflow_root)
    if st.button(tr("adsorption_wizard.structure_editor.adopt_candidate"), key=f"adopt_clean_{candidate_rel}"):
        adopt_clean_poscar_candidate(workflow_root, state)
        st.success(tr("adsorption_wizard.structure_editor.adopt_success"))
        st.rerun()


def show_structure_edit_forms(workflow, state: dict, vaspkit_bin: str) -> None:
    enabled = can_use_structure_edit_step(workflow.root_dir, state)
    if not enabled:
        st.warning(tr("adsorption_wizard.structure_editor.no_clean_poscar"))
    st.caption(tr("adsorption_wizard.structure_edit_real_call_10b"))
    with st.expander(tr("adsorption_wizard.vacuum_801")):
        direction = st.selectbox(tr("adsorption_wizard.direction_index"), (1, 2, 3), key=f"801_dir_{workflow.workflow_id}")
        vacuum = st.number_input(tr("adsorption_wizard.vacuum_thickness"), min_value=0.001, value=15.0, key=f"801_vac_{workflow.workflow_id}")
        if st.button(tr("adsorption_wizard.structure_editor.run_801"), disabled=not enabled, key=f"801_run_{workflow.workflow_id}"):
            try:
                result = run_clean_structure_step(
                    workflow,
                    state,
                    vaspkit_bin,
                    "801",
                    build_801_inputs(direction, vacuum),
                    "POSCAR_REV.vasp",
                    "POSCAR_REV.vasp",
                    {"direction_index": direction, "vacuum_thickness": vacuum},
                )
                show_structure_edit_result(result)
            except Exception as exc:
                st.error(tr("adsorption_wizard.structure_editor.invalid_parameter", error=str(exc)))
    with st.expander(tr("adsorption_wizard.slab_803")):
        st.warning(tr("adsorption_wizard.structure_editor.slab_warning"))
        h = st.number_input(tr("adsorption_wizard.h_index"), value=1, step=1, key=f"803_h_{workflow.workflow_id}")
        k = st.number_input(tr("adsorption_wizard.k_index"), value=1, step=1, key=f"803_k_{workflow.workflow_id}")
        l = st.number_input(tr("adsorption_wizard.l_index"), value=0, step=1, key=f"803_l_{workflow.workflow_id}")
        layer = st.text_input(tr("adsorption_wizard.layer_text"), placeholder=tr("adsorption_wizard.layer_placeholder"), key=f"803_layer_{workflow.workflow_id}")
        shift = st.number_input(tr("adsorption_wizard.shift_value"), value=0.0, key=f"803_shift_{workflow.workflow_id}")
        vacuum = st.number_input(tr("adsorption_wizard.vacuum_thickness"), min_value=0.001, value=15.0, key=f"803_vac_{workflow.workflow_id}")
        if st.button(tr("adsorption_wizard.structure_editor.run_803"), disabled=not enabled, key=f"803_run_{workflow.workflow_id}"):
            try:
                output_name = expected_803_output_name(h, k, l)
                result = run_clean_structure_step(
                    workflow,
                    state,
                    vaspkit_bin,
                    "803",
                    build_803_inputs(h, k, l, layer, shift, vacuum),
                    output_name,
                    output_name,
                    {"h": h, "k": k, "l": l, "layer_text": layer, "shift_value": shift, "vacuum_thickness": vacuum},
                )
                show_structure_edit_result(result)
            except Exception as exc:
                st.error(tr("adsorption_wizard.structure_editor.invalid_parameter", error=str(exc)))
    with st.expander(tr("adsorption_wizard.supercell_401")):
        a = st.number_input(tr("adsorption_wizard.repeat_a"), min_value=1, value=1, step=1, key=f"401_a_{workflow.workflow_id}")
        b = st.number_input(tr("adsorption_wizard.repeat_b"), min_value=1, value=1, step=1, key=f"401_b_{workflow.workflow_id}")
        c = st.number_input(tr("adsorption_wizard.repeat_c"), min_value=1, value=1, step=1, key=f"401_c_{workflow.workflow_id}")
        if st.button(tr("adsorption_wizard.structure_editor.run_401"), disabled=not enabled, key=f"401_run_{workflow.workflow_id}"):
            try:
                output_name = expected_401_output_name(a, b, c)
                result = run_clean_structure_step(
                    workflow,
                    state,
                    vaspkit_bin,
                    "401",
                    build_401_inputs(a, b, c),
                    output_name,
                    output_name,
                    {"repeat_a": a, "repeat_b": b, "repeat_c": c},
                )
                show_structure_edit_result(result)
            except Exception as exc:
                st.error(tr("adsorption_wizard.structure_editor.invalid_parameter", error=str(exc)))
    with st.expander(tr("adsorption_wizard.fix_402")):
        st.warning(tr("adsorption_wizard.structure_editor.fix_atoms_warning"))
        st.caption(tr("adsorption_wizard.structure_editor.experimental"))
        mode = st.radio(tr("adsorption_wizard.fix_mode"), ("atom_indices", "z_range"), format_func=lambda value: tr(f"adsorption_wizard.fix_mode.{value}"), key=f"402_mode_{workflow.workflow_id}")
        direction = st.selectbox(tr("adsorption_wizard.direction_index"), (1, 2, 3), key=f"402_dir_{workflow.workflow_id}")
        if mode == "atom_indices":
            indices = st.text_input(tr("adsorption_wizard.atom_indices"), placeholder=tr("adsorption_wizard.atom_indices_placeholder"), key=f"402_indices_{workflow.workflow_id}")
            if st.button(tr("adsorption_wizard.structure_editor.run_402"), disabled=not enabled, key=f"402_indices_run_{workflow.workflow_id}"):
                try:
                    result = run_clean_structure_step(
                        workflow,
                        state,
                        vaspkit_bin,
                        "402",
                        build_402_atom_indices_inputs(indices, direction),
                        "POSCAR_FIX",
                        "POSCAR_FIX",
                        {"mode": "atom_indices", "atom_indices": indices, "direction_index": direction},
                    )
                    show_structure_edit_result(result)
                except Exception as exc:
                    st.error(tr("adsorption_wizard.structure_editor.invalid_parameter", error=str(exc)))
        else:
            z_min = st.number_input(tr("adsorption_wizard.z_min"), value=0.0, key=f"402_zmin_{workflow.workflow_id}")
            z_max = st.number_input(tr("adsorption_wizard.z_max"), value=0.3, key=f"402_zmax_{workflow.workflow_id}")
            if st.button(tr("adsorption_wizard.structure_editor.run_402"), disabled=not enabled, key=f"402_z_run_{workflow.workflow_id}"):
                try:
                    result = run_clean_structure_step(
                        workflow,
                        state,
                        vaspkit_bin,
                        "402",
                        build_402_z_range_inputs(z_min, z_max, direction),
                        "POSCAR_FIX",
                        "POSCAR_FIX",
                        {"mode": "z_range", "z_min": z_min, "z_max": z_max, "direction_index": direction},
                    )
                    show_structure_edit_result(result)
                except Exception as exc:
                    st.error(tr("adsorption_wizard.structure_editor.invalid_parameter", error=str(exc)))


def run_clean_structure_step(
    workflow,
    state: dict,
    vaspkit_bin: str,
    step_name: str,
    inputs: list[str],
    expected_output: str,
    candidate_name: str,
    parameters: dict,
):
    sync_clean_poscar_to_structure(workflow.root_dir, state)
    structure_dir = workflow.root_dir / "structure"
    result = run_vaspkit_structure_step(
        vaspkit_bin,
        inputs,
        structure_dir,
        step_name,
        expected_output,
    )
    if result.ok and result.output_path is not None:
        save_candidate_file(
            workflow.root_dir,
            state,
            role="clean",
            source_path=result.output_path,
            candidate_name=candidate_name,
            source_step=step_name,
            parameters=parameters,
        )
    return result


def show_structure_edit_result(result) -> None:
    if result.ok:
        st.success(tr("adsorption_wizard.structure_editor.candidate_created"))
        if result.output_path is not None:
            st.info(tr("adsorption_wizard.expected_output", filename=result.output_path.name))
        st.rerun()
    else:
        show_structure_edit_failure(result)


def show_structure_edit_failure(result) -> None:
    st.error(tr("adsorption_wizard.structure_editor.vaspkit_failed", step=result.step_name))
    if result.errors:
        st.error("; ".join(result.errors))
    st.info(
        tr(
            "adsorption_wizard.structure_editor.check_log",
            stdout=result.stdout_path,
            stderr=result.stderr_path,
        )
    )


def show_wizard_step_adsorbed_structure(workflow, state: dict) -> None:
    st.markdown(f"**{tr('adsorption_wizard.step.adsorbed_structure')}**")
    with st.expander(tr("adsorption_wizard.tutorial.vesta_adsorbed")):
        st.write(tr("adsorption_wizard.tutorial.vesta_adsorbed.body"))
    uploaded = st.file_uploader(tr("adsorption_wizard.upload_adsorbed_poscar"), key=f"ads_poscar_{workflow.workflow_id}")
    if uploaded is not None and st.button(tr("button.save"), key=f"save_ads_{workflow.workflow_id}"):
        path = workflow.root_dir / "artifacts" / "adsorbed" / "POSCAR"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(uploaded.getvalue())
        state["artifacts"]["adsorbed_poscar"] = "artifacts/adsorbed/POSCAR"
        save_wizard_state(workflow.root_dir, state)
        st.success(tr("adsorption_wizard.artifact_saved"))
        st.rerun()
    rel = state.get("artifacts", {}).get("adsorbed_poscar")
    if rel:
        show_poscar_summary_block(tr("adsorption_wizard.step.adsorbed_structure"), artifact_path(workflow.root_dir, rel), workflow.root_dir)


def show_wizard_step_molecule_reference(workflow, state: dict) -> None:
    st.markdown(f"**{tr('adsorption_wizard.step.molecule_reference')}**")
    with st.expander(tr("adsorption_wizard.tutorial.molecule_poscar")):
        st.write(tr("adsorption_wizard.tutorial.molecule_poscar.body"))
    uploaded = st.file_uploader(tr("adsorption_wizard.upload_molecule_poscar"), key=f"mol_poscar_{workflow.workflow_id}")
    if uploaded is not None and st.button(tr("button.save"), key=f"save_mol_{workflow.workflow_id}"):
        path = workflow.root_dir / "artifacts" / "molecule" / "POSCAR"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(uploaded.getvalue())
        state["artifacts"]["molecule_poscar"] = "artifacts/molecule/POSCAR"
        save_wizard_state(workflow.root_dir, state)
        st.success(tr("adsorption_wizard.artifact_saved"))
        st.rerun()
    rel = state.get("artifacts", {}).get("molecule_poscar")
    if rel:
        show_poscar_summary_block(tr("adsorption_wizard.step.molecule_reference"), artifact_path(workflow.root_dir, rel), workflow.root_dir)


def show_wizard_step_generate_relax_inputs(config, db_file: Path, workflow, state: dict) -> None:
    st.markdown(f"**{tr('adsorption_wizard.relax_inputs.title')}**")
    ready, sources, missing = check_relax_source_poscars(workflow.root_dir, state)
    source_rows = []
    for role in RELAX_ROLES:
        rel_path = sources.get(role, "")
        source_rows.append(
            {
                tr("table.role"): tr(f"adsorption_wizard.relax_inputs.{role}"),
                tr("table.path"): rel_path,
                tr("table.exists"): bool_label(bool(rel_path) and artifact_path(workflow.root_dir, rel_path).exists()),
            }
        )
    st.dataframe(pd.DataFrame(source_rows), use_container_width=True, hide_index=True)
    if not ready:
        st.warning(tr("adsorption_wizard.relax_inputs.source_poscar_missing", roles=", ".join(missing)))
        return

    st.info(tr("adsorption_wizard.relax_inputs.gamma_only_molecule"))
    vaspkit_bin = st.text_input(tr("vaspkit.bin.label"), "vaspkit", key=f"relax_vaspkit_bin_{workflow.workflow_id}")
    incar_key = st.text_input(tr("vaspkit.option.incar_custom_key_string.label"), value="SR", key=f"relax_incar_key_{workflow.workflow_id}")
    scheme = st.selectbox(
        tr("vaspkit.option.kmesh_scheme.label"),
        (1, 2, 3),
        format_func=lambda value: tr(f"vaspkit.choice.kmesh_scheme.{value}"),
        key=f"relax_kmesh_scheme_{workflow.workflow_id}",
    )
    kmesh_value = st.number_input(tr("vaspkit.option.kmesh_resolved_value.label"), min_value=0.001, value=0.04, format="%.3f", key=f"relax_kmesh_value_{workflow.workflow_id}")

    if st.button(tr("adsorption_wizard.relax_inputs.generate"), type="primary", key=f"generate_relax_inputs_{workflow.workflow_id}"):
        try:
            results = generate_all_relax_input_packages(
                workflow.root_dir,
                state,
                vaspkit_bin=vaspkit_bin,
                incar_key=incar_key,
                kpoints_scheme=scheme,
                kmesh_value=kmesh_value,
            )
            for result in results:
                if result.ok:
                    st.success(tr("adsorption_wizard.relax_inputs.generation_success", role=tr(f"adsorption_wizard.relax_inputs.{result.role}")))
                else:
                    st.error(tr("adsorption_wizard.relax_inputs.generation_failed", role=tr(f"adsorption_wizard.relax_inputs.{result.role}"), errors="; ".join(result.errors)))
            consistency_warnings = check_relax_package_consistency(workflow.root_dir)
            if consistency_warnings:
                st.warning("\n".join(consistency_warnings))
            st.rerun()
        except Exception as exc:
            st.error(tr("adsorption_wizard.relax_inputs.generation_failed", role="all", errors=str(exc)))

    state = load_wizard_state(workflow.root_dir, workflow.workflow_id)
    relax_inputs = state.get("relax_inputs", {})
    if relax_inputs:
        st.markdown(f"**{tr('adsorption_wizard.relax_inputs.validation_summary')}**")
        st.dataframe(
            pd.DataFrame(
                [
                    {
                        tr("table.role"): tr(f"adsorption_wizard.relax_inputs.{role}"),
                        tr("table.status"): data.get("status", ""),
                        tr("table.path"): data.get("input_dir", ""),
                        tr("table.warning"): "; ".join(data.get("warnings", [])),
                        tr("error.vaspkit_validation"): "; ".join(data.get("errors", [])),
                    }
                    for role, data in relax_inputs.items()
                ]
            ),
            use_container_width=True,
            hide_index=True,
        )

    if st.button(tr("adsorption_wizard.relax_inputs.create_jobs"), key=f"create_relax_jobs_{workflow.workflow_id}"):
        results = create_relax_jobs_from_inputs(
            db_file,
            workflow.workflow_id,
            workflow.root_dir,
            mpi_ranks=config.default_mpi_ranks,
            vasp_bin=config.vasp_bin,
        )
        for result in results:
            if result["status"] == "created":
                st.success(tr("adsorption_wizard.relax_jobs.created", role=tr(f"adsorption_wizard.relax_inputs.{result['role']}"), job_id=result["job_id"]))
            elif result["status"] == "exists":
                st.info(tr("adsorption_wizard.relax_jobs.already_exists", role=tr(f"adsorption_wizard.relax_inputs.{result['role']}")))
            else:
                st.error(tr("adsorption_wizard.relax_jobs.create_failed", role=tr(f"adsorption_wizard.relax_inputs.{result['role']}"), errors=", ".join(result.get("errors", []))))
        st.rerun()


def show_wizard_step_run_relax_jobs(db_file: Path, workflow) -> None:
    st.markdown(f"**{tr('adsorption_wizard.relax_jobs.title')}**")
    workflow_jobs = list_jobs_for_workflow(db_file, workflow.workflow_id)
    jobs_by_role = {binding.role: (binding, job) for binding, job in workflow_jobs}
    for role in RELAX_ROLES:
        st.markdown(f"**{tr(f'adsorption_wizard.relax_inputs.{role}')}**")
        if role not in jobs_by_role:
            st.warning(tr("adsorption_wizard.relax_jobs.no_relax_job"))
            continue
        binding, job = jobs_by_role[role]
        with st.container(border=True):
            process_state = get_workflow_job_process_state(db_file, job.job_id)
            contcar = Path(job.run_dir) / "CONTCAR"
            st.write(
                {
                    tr("table.role"): tr(f"adsorption_wizard.relax_inputs.{binding.role}"),
                    tr("table.step_order"): binding.step_order,
                    tr("table.job_id"): job.job_id,
                    tr("table.calculation_type"): calculation_type_label(job.calculation_type),
                    tr("table.status"): status_label(job.status),
                    tr("workflow_job.pid"): none_text(job.pid),
                    tr("workflow_job.process_alive"): bool_label(process_state["process_alive"]),
                    tr("table.run_dir"): str(job.run_dir),
                    tr("adsorption_wizard.relax_jobs.contcar_exists"): bool_label(contcar.exists() and contcar.stat().st_size > 0),
                }
            )
            show_workflow_job_controls(db_file, job.job_id)
            show_workflow_job_logs(db_file, job.job_id)


def show_poscar_summary_block(title: str, path: Path, workflow_root: Path) -> None:
    summary = poscar_summary(path)
    st.markdown(f"**{title}**")
    st.write(
        {
            tr("table.path"): summary.to_dict(workflow_root)["path"],
            tr("table.exists"): bool_label(summary.exists),
            tr("table.size_bytes"): summary.size_bytes,
            tr("table.element_order"): list(summary.element_order),
            tr("adsorption_wizard.atom_counts"): list(summary.atom_counts),
            tr("adsorption_wizard.total_atoms"): summary.total_atoms,
            tr("adsorption_wizard.cell_lengths"): summary.cell_lengths,
        }
    )
    if summary.warnings:
        st.warning("; ".join(summary.warnings))


def method_description_text(description: MethodDescription) -> str:
    return description.zh if current_lang() == "zh" else description.en


def workflow_method_notes_for_lang(raw: str | None) -> str:
    if not raw:
        return ""
    try:
        data = json.loads(raw)
    except json.JSONDecodeError:
        return raw
    if isinstance(data, dict):
        return str(data.get(current_lang()) or data.get("en") or data.get("zh") or "")
    return raw


def show_legacy_adsorption_workflow_page(config, db_file: Path) -> None:
    show_create_adsorption_workflow_form(config, db_file)
    st.divider()
    show_adsorption_workflow_list(db_file)
    st.divider()
    selected_workflow_id = st.session_state.get("selected_adsorption_workflow_id")
    if selected_workflow_id:
        show_adsorption_workflow_detail(db_file, selected_workflow_id)


def show_create_adsorption_workflow_form(config, db_file: Path) -> None:
    st.subheader(tr("adsorption.workflow.create.title"))
    usable_sets = list_usable_input_sets(db_file)
    if not usable_sets:
        st.warning(tr("warning.adsorption_no_usable_input_sets"))
        return
    if "adsorption_method_notes_text" not in st.session_state:
        st.session_state["adsorption_method_notes_text"] = method_description_text(generate_method_description(
            method_family="DFT",
            functional="PBE",
            system_type="adsorption",
        ))

    with st.form("create_adsorption_workflow_form"):
        workflow_name = st.text_input(
            tr("adsorption.workflow.name"),
            value="",
            placeholder=tr("adsorption.workflow.default_name"),
        )
        adsorbate_name = st.text_input(
            tr("adsorption.workflow.adsorbate_name"),
            value="",
            placeholder=tr("adsorption.workflow.adsorbate_placeholder"),
        )
        method_family = st.selectbox(
            tr("adsorption.workflow.method_family"),
            ADSORPTION_METHOD_FAMILIES,
            format_func=lambda value: adsorption_choice_label("adsorption.method_family", value),
            key="adsorption_method_family",
        )
        functional = st.selectbox(
            tr("adsorption.workflow.functional"),
            ADSORPTION_FUNCTIONALS,
            format_func=lambda value: adsorption_choice_label("adsorption.functional", value),
            key="adsorption_functional",
        )
        st.caption(tr("method_advisor.generated_description"))
        method_notes = st.text_area(tr("adsorption.workflow.method_notes"), key="adsorption_method_notes_text")
        selections = {
            "clean_slab": st.selectbox(
                tr("adsorption.workflow.input_set.clean_slab"),
                usable_sets,
                format_func=format_input_set_choice,
            ),
            "molecule_ref": st.selectbox(
                tr("adsorption.workflow.input_set.molecule_ref"),
                usable_sets,
                format_func=format_input_set_choice,
            ),
            "adsorbed_system": st.selectbox(
                tr("adsorption.workflow.input_set.adsorbed_system"),
                usable_sets,
                format_func=format_input_set_choice,
            ),
        }
        notes = st.text_area(tr("adsorption.workflow.notes"))
        regenerate_method = st.form_submit_button(tr("method_advisor.regenerate"))
        submitted = st.form_submit_button(tr("button.create_adsorption_workflow"), type="primary")

    if regenerate_method:
        st.session_state["adsorption_method_notes_text"] = method_description_text(generate_method_description(
            method_family=method_family,
            functional=functional,
            system_type="adsorption",
            adsorbate_name=adsorbate_name,
        ))
        st.rerun()

    if submitted:
        normalized_workflow_name = workflow_name.strip()
        if not normalized_workflow_name:
            st.error(tr("workflow.name_required"))
            return
        if workflow_name_exists(db_file, normalized_workflow_name):
            st.error(tr("workflow.name_duplicate"))
            return
        workflow_id = new_adsorption_workflow_id(adsorbate_name or workflow_name)
        root_dir = Path(config.workspace) / "workflows" / workflow_id
        try:
            workflow = create_adsorption_workflow(
                db_file,
                workflow_id=workflow_id,
                name=normalized_workflow_name,
                root_dir=root_dir,
                clean_slab_input_set_id=selections["clean_slab"].input_set_id,
                molecule_ref_input_set_id=selections["molecule_ref"].input_set_id,
                adsorbed_input_set_id=selections["adsorbed_system"].input_set_id,
                method_family=method_family,
                functional=functional,
                method_notes=method_notes.strip() or None,
                mpi_ranks=config.default_mpi_ranks,
                vasp_bin=config.vasp_bin,
                notes=notes.strip(),
            )
        except Exception as exc:
            st.error(tr("error.adsorption_workflow_create_failed", error=str(exc)))
            return
        st.session_state["selected_adsorption_workflow_id"] = workflow.workflow_id
        st.success(tr("success.adsorption_workflow_created", workflow_id=workflow.workflow_id))
        show_adsorption_workflow_detail(db_file, workflow.workflow_id)


def format_input_set_choice(input_set: InputSet) -> str:
    return f"{input_set.name} | {input_set.input_set_id} | {tr('table.status')}={input_set_status_label(input_set.status)}"


def show_adsorption_workflow_list(db_file: Path) -> None:
    st.subheader(tr("adsorption.workflow.list.title"))
    workflows = list_workflows(db_file, workflow_type="adsorption")
    if not workflows:
        st.info(tr("adsorption.workflow.no_records"))
        return
    st.dataframe(
        pd.DataFrame(
            [
                {
                    tr("table.workflow_id"): workflow.workflow_id,
                    tr("table.name"): workflow.name,
                    tr("table.status"): workflow_status_label(workflow.status),
                    tr("table.method_family"): workflow.method_family,
                    tr("table.functional"): workflow.functional,
                    tr("table.root_dir"): str(workflow.root_dir),
                    tr("table.created_at"): workflow.created_at,
                    tr("table.updated_at"): workflow.updated_at,
                }
                for workflow in workflows
            ]
        ),
        use_container_width=True,
        hide_index=True,
    )
    selected = st.selectbox(
        tr("adsorption.workflow.select"),
        workflows,
        format_func=lambda workflow: f"{workflow.name} | {workflow.workflow_id} | {workflow_status_label(workflow.status)}",
    )
    st.session_state["selected_adsorption_workflow_id"] = selected.workflow_id


def show_adsorption_workflow_detail(db_file: Path, workflow_id: str) -> None:
    workflow_jobs = list_jobs_for_workflow(db_file, workflow_id)
    workflows = [workflow for workflow in list_workflows(db_file, workflow_type="adsorption") if workflow.workflow_id == workflow_id]
    if not workflows:
        st.warning(tr("warning.adsorption_workflow_not_found", workflow_id=workflow_id))
        return
    workflow = workflows[0]
    st.subheader(tr("adsorption.workflow.detail.title"))
    st.write(
        {
            tr("table.workflow_id"): workflow.workflow_id,
            tr("table.name"): workflow.name,
            tr("table.status"): workflow_status_label(workflow.status),
            tr("table.root_dir"): str(workflow.root_dir),
            tr("table.method_family"): workflow.method_family,
            tr("table.functional"): workflow.functional,
            tr("table.method_notes"): workflow_method_notes_for_lang(workflow.method_notes),
            tr("table.notes"): workflow.notes,
        }
    )
    jobs_by_role = {binding.role: (binding, job) for binding, job in workflow_jobs}
    cols = st.columns(3)
    for col, role in zip(cols, ADSORPTION_WORKFLOW_ROLES):
        with col:
            st.markdown(f"**{workflow_role_label(role)}**")
            if role not in jobs_by_role:
                st.warning(tr("warning.adsorption_workflow_role_missing", role=workflow_role_label(role)))
                continue
            binding, job = jobs_by_role[role]
            with st.container(border=True):
                process_state = get_workflow_job_process_state(db_file, job.job_id)
                st.write(
                    {
                        tr("table.role"): workflow_role_label(binding.role),
                        tr("table.step_order"): binding.step_order,
                        tr("table.required"): bool_label(binding.required),
                        tr("table.job_id"): job.job_id,
                        tr("table.calculation_type"): calculation_type_label(job.calculation_type),
                        tr("table.status"): status_label(job.status),
                        tr("workflow_job.pid"): none_text(job.pid),
                        tr("workflow_job.process_alive"): bool_label(process_state["process_alive"]),
                        tr("table.input_set_id"): job.input_set_id,
                        tr("table.run_dir"): str(job.run_dir),
                        tr("table.start_time"): none_text(job.start_time),
                        tr("table.end_time"): none_text(job.end_time),
                        tr("table.created_at"): job.created_at,
                        tr("table.updated_at"): job.updated_at,
                    }
                )
                show_workflow_job_metrics(db_file, job.job_id)
                show_workflow_job_controls(db_file, job.job_id)
                show_workflow_job_logs(db_file, job.job_id)
    show_adsorption_result_section(db_file, workflow_id)


def show_workflow_job_metrics(db_file: Path, job_id: str) -> None:
    metrics = get_job_metrics(db_file, job_id)
    paths = get_workflow_job_log_paths(db_file, job_id)
    st.write(
        {
            tr("adsorption.result.outcar_exists"): bool_label(paths["OUTCAR"]["exists"]),
            tr("adsorption.result.final_toten"): format_energy(metrics.toten_ev if metrics else None),
            tr("adsorption.result.loop_avg"): format_seconds(metrics.loop_avg_seconds if metrics else None),
            tr("adsorption.result.converged"): bool_label(metrics.ionic_converged if metrics else None),
        }
    )


def show_adsorption_result_section(db_file: Path, workflow_id: str) -> None:
    st.divider()
    st.subheader(tr("adsorption.visualization.title"))
    st.caption(tr("adsorption.result.method_explanation"))
    if st.button(tr("button.parse_adsorption_results"), key=f"parse_adsorption_results_{workflow_id}"):
        try:
            parsed = parse_adsorption_workflow_jobs(db_file, workflow_id)
            warnings = [item.get("warning") for item in parsed.values() if item.get("warning")]
            if warnings:
                st.warning("; ".join(warnings))
            else:
                st.success(tr("success.adsorption_results_parsed"))
            st.rerun()
        except Exception as exc:
            st.error(tr("error.adsorption_results_parse_failed", error=str(exc)))
            return

    try:
        result = calculate_adsorption_energy(db_file, workflow_id)
    except Exception as exc:
        st.error(tr("error.adsorption_energy_failed", error=str(exc)))
        return

    workflow_jobs = list_jobs_for_workflow(db_file, workflow_id)
    jobs_by_role = {binding.role: job for binding, job in workflow_jobs}
    metrics_by_job_id = {
        job.job_id: metrics
        for _binding, job in workflow_jobs
        if (metrics := get_job_metrics(db_file, job.job_id)) is not None
    }

    st.markdown(f"**{tr('adsorption.result_summary')}**")
    if result.warnings:
        st.warning("\n".join(format_adsorption_warning(warning) for warning in result.warnings))
    if not result.ready:
        st.info(tr("adsorption.not_ready"))
    else:
        summary_cols = st.columns(4)
        summary_cols[0].metric(tr("adsorption.final_eads"), format_visual_energy(result.e_ads))
        summary_cols[1].metric("E_clean_slab", format_visual_energy(result.e_clean_slab))
        summary_cols[2].metric("E_molecule_ref", format_visual_energy(result.e_molecule_ref))
        summary_cols[3].metric("E_adsorbed_system", format_visual_energy(result.e_adsorbed_system))
        summary_rows = build_adsorption_summary_rows(result)
        st.dataframe(
            pd.DataFrame(
                [
                    {
                        tr("table.term"): row["field"],
                        tr("table.value"): row["value"],
                    }
                    for row in summary_rows
                ]
            ),
            use_container_width=True,
            hide_index=True,
        )

    st.markdown(f"**{tr('adsorption.formula.symbolic')}**")
    st.code(result.formula_symbolic, language="text")
    st.markdown(f"**{tr('adsorption.formula.physical_meaning')}**")
    st.write(tr("adsorption.result.physical_meaning_text"))
    if result.ready:
        st.markdown(f"**{tr('adsorption.formula.numeric_substitution')}**")
        st.code(result.formula_numeric or "", language="text")
        st.caption(tr("adsorption.energy_interpretation"))

    st.markdown(f"**{tr('adsorption.job_metrics_table')}**")
    st.dataframe(
        pd.DataFrame(
            [
                {
                    tr("table.role"): workflow_role_label(row["role"]),
                    tr("table.job_id"): row["job_id"],
                    tr("table.calculation_type"): calculation_type_label(row["calculation_type"]) if row["calculation_type"] != MISSING_VALUE else row["calculation_type"],
                    tr("table.status"): status_label(row["status"]) if row["status"] != MISSING_VALUE else row["status"],
                    tr("adsorption.result.outcar_exists"): bool_label(row["outcar_exists"]),
                    tr("adsorption.result.final_toten"): row["final_toten"],
                    tr("adsorption.result.loop_avg"): row["loop_avg"],
                    tr("adsorption.loop_count"): row["loop_count"],
                    tr("adsorption.ionic_converged"): bool_label(row["ionic_converged"]),
                    tr("adsorption.electronic_converged"): bool_label(row["electronic_converged"]),
                    tr("table.warning"): format_adsorption_warning_codes(row["role"], row["warning_codes"]),
                }
                for row in build_job_metrics_table(result, jobs_by_role)
            ]
        ),
        use_container_width=True,
        hide_index=True,
    )

    chart_rows = build_total_energy_chart_data(result)
    if chart_rows:
        st.markdown(f"**{tr('adsorption.energy_bar_chart')}**")
        chart_df = pd.DataFrame(
            [
                {
                    tr("table.role"): workflow_role_label(row["role"]),
                    tr("table.energy_ev"): row["energy_ev"],
                }
                for row in chart_rows
            ]
        )
        st.bar_chart(chart_df.set_index(tr("table.role")))

    st.markdown(f"**{tr('adsorption.oszicar_steps')}**")
    oszicar_rows = build_oszicar_steps_table(result, metrics_by_job_id)
    if oszicar_rows:
        oszicar_df = pd.DataFrame(
            [
                {
                    tr("table.role"): workflow_role_label(row["role"]),
                    tr("table.job_id"): row["job_id"],
                    tr("adsorption.ionic_step"): row["ionic_step"],
                    tr("table.energy_ev"): row["energy_ev"],
                }
                for row in oszicar_rows
            ]
        )
        st.dataframe(oszicar_df, use_container_width=True, hide_index=True)
        chart_data = oszicar_df.pivot(
            index=tr("adsorption.ionic_step"),
            columns=tr("table.role"),
            values=tr("table.energy_ev"),
        )
        st.line_chart(chart_data)
    else:
        st.info(tr("adsorption.oszicar_steps_empty"))

    st.markdown(f"**{tr('adsorption.loop_time_summary')}**")
    st.dataframe(
        pd.DataFrame(
            [
                {
                    tr("table.role"): workflow_role_label(row["role"]),
                    tr("table.job_id"): row["job_id"],
                    tr("adsorption.result.loop_avg"): row["loop_avg"],
                    tr("adsorption.loop_count"): row["loop_count"],
                }
                for row in build_loop_time_summary(result)
            ]
        ),
        use_container_width=True,
        hide_index=True,
    )


def show_workflow_job_controls(db_file: Path, job_id: str) -> None:
    st.caption(tr("workflow_job.monitoring"))
    action_cols = st.columns(4)
    if action_cols[0].button(tr("workflow_job.start_dry_run"), key=f"wf_start_dry_{job_id}"):
        try:
            updated = start_workflow_job(db_file, job_id=job_id, dry_run=True)
            st.success(tr("workflow_job.dry_run_finished", job_id=updated.job_id))
            st.rerun()
        except Exception as exc:
            st.error(str(exc))

    confirm_key = f"wf_real_confirm_{job_id}"
    real_confirmed = st.checkbox(tr("workflow_job.real_vasp_confirm"), key=confirm_key)
    if not real_confirmed:
        st.warning(tr("workflow_job.real_vasp_not_confirmed"))
    if action_cols[1].button(
        tr("workflow_job.start_real_vasp"),
        key=f"wf_start_real_{job_id}",
        disabled=not real_confirmed,
    ):
        try:
            updated = start_workflow_job(db_file, job_id=job_id, dry_run=False)
            st.success(tr("success.workflow_job_started", job_id=updated.job_id))
            st.rerun()
        except Exception as exc:
            st.error(str(exc))

    if action_cols[2].button(tr("workflow_job.stop_job"), key=f"wf_stop_{job_id}"):
        try:
            result = stop_workflow_job(db_file, job_id=job_id)
            if result.message_key == "workflow_job.stop_success":
                st.success(tr(result.message_key))
            else:
                st.info(tr(result.message_key))
            for warning in result.warnings:
                st.warning(tr(warning))
            st.rerun()
        except Exception as exc:
            st.error(tr("workflow_job.stop_failed", error=str(exc)))

    if action_cols[3].button(tr("workflow_job.refresh_button"), key=f"wf_refresh_{job_id}"):
        try:
            result = refresh_workflow_job_status(db_file, job_id)
            for warning in result.warnings:
                st.warning(tr(warning))
            st.info(tr("workflow_job.refresh", job_id=result.job_id, status=status_label(result.status)))
            st.rerun()
        except Exception as exc:
            st.error(str(exc))


def show_workflow_job_logs(db_file: Path, job_id: str) -> None:
    paths = get_workflow_job_log_paths(db_file, job_id)
    outcar_key = "workflow_job.outcar_exists" if paths["OUTCAR"]["exists"] else "workflow_job.outcar_missing"
    st.write({tr("workflow_job.outcar_exists"): tr(outcar_key)})
    with st.expander(tr("workflow_job.vasp_out_tail"), expanded=False):
        try:
            st.code(tail_workflow_job_file(db_file, job_id, "vasp.out"), language="text")
        except Exception as exc:
            st.error(str(exc))
    with st.expander(tr("workflow_job.oszicar_tail"), expanded=False):
        try:
            st.code(tail_workflow_job_file(db_file, job_id, "OSZICAR"), language="text")
        except Exception as exc:
            st.error(str(exc))


def show_adsorption_input_sets_page(config, conn) -> None:
    st.subheader(tr("adsorption.input_sets.title"))
    usable_sets = [record for record in list_input_sets(conn) if record.usable_for_vasp]
    if not usable_sets:
        st.info(tr("adsorption.input_sets.no_usable"))
        return

    task_id = st.text_input(
        tr("sidebar.task_id"),
        f"adsorption-{datetime.utcnow():%Y%m%d-%H%M%S}",
        key="adsorption_input_set_task_id",
    )
    selections: dict[str, InputSet] = {}
    for role in ADSORPTION_INPUT_ROLES:
        selections[role] = st.selectbox(
            tr(f"adsorption.input_set.{role}"),
            usable_sets,
            format_func=lambda item: f"{item.input_set_id} | {item.name}",
            key=f"adsorption_input_set_{role}",
        )

    warnings = adsorption_input_set_warnings(
        selections["adsorbed"],
        selections["clean_slab"],
        selections["molecule_ref"],
    )
    for warning in warnings:
        st.warning(warning)

    if st.button(tr("button.create_adsorption_task"), type="primary"):
        safe_id = safe_task_id(task_id)
        task_root = task_dir(config, safe_id)
        existing_inputs = [
            role
            for role in ADSORPTION_INPUT_ROLES
            if any(((task_root / role / "run") / filename).exists() for filename in CORE_INPUT_FILES)
        ]
        if existing_inputs:
            st.error(tr("error.adsorption_role_run_exists", roles=", ".join(existing_inputs), path=task_root))
            return
        create_adsorption_task_from_input_sets(conn, safe_id, task_root, selections, warnings)
        st.success(tr("success.adsorption_task_created", task_id=safe_id, path=task_root))
        st.rerun()


def create_adsorption_task_from_input_sets(
    conn,
    task_id: str,
    task_root: Path,
    selections: dict[str, InputSet],
    warnings: list[str],
) -> None:
    task_root.mkdir(parents=True, exist_ok=True)
    binding_metadata: list[dict] = []
    for role, input_set in selections.items():
        role_run = task_root / role / "run"
        role_run.mkdir(parents=True, exist_ok=True)
        copy_input_set_to_run(input_set, role_run)
        role_binding = {
            "task_id": task_id,
            "role": role,
            "input_set_id": input_set.input_set_id,
            "input_set_name": input_set.name,
            "run_dir": str(role_run),
            "created_at": datetime.utcnow().isoformat(timespec="seconds"),
        }
        write_json_file(role_run / "input_set_binding.json", role_binding)
        binding_metadata.append(role_binding)

    task_metadata = {
        "task_id": task_id,
        "task_type": "adsorption",
        "created_from": "adsorption_input_sets",
        "formula": "E_ads = E_adsorbed - E_clean_slab - E_molecule_ref",
        "roles": binding_metadata,
        "warnings": warnings,
        "created_at": datetime.utcnow().isoformat(timespec="seconds"),
    }
    write_json_file(task_root / "task.json", task_metadata)
    write_json_file(task_root / "input_set_bindings.json", {"bindings": binding_metadata})
    create_task(
        conn,
        task_id=task_id,
        project="default",
        task_type="adsorption",
        task_root=task_root,
        status="committed",
    )
    for role, input_set in selections.items():
        bind_input_set_to_task(conn, task_id, role, input_set.input_set_id)


def adsorption_input_set_warnings(adsorbed: InputSet, clean_slab: InputSet, molecule_ref: InputSet) -> list[str]:
    warnings: list[str] = []
    if len({adsorbed.input_set_id, clean_slab.input_set_id, molecule_ref.input_set_id}) < 3:
        warnings.append(tr("warning.adsorption_duplicate_input_sets"))

    ads_encut = parse_incar_value(adsorbed.incar_path, "ENCUT")
    slab_encut = parse_incar_value(clean_slab.incar_path, "ENCUT")
    if ads_encut and slab_encut and ads_encut != slab_encut:
        warnings.append(tr("warning.adsorption_encut_mismatch", adsorbed=ads_encut, clean_slab=slab_encut))
    elif not ads_encut or not slab_encut:
        warnings.append(tr("warning.adsorption_encut_missing"))

    ads_kpoints = normalized_text(adsorbed.kpoints_path)
    slab_kpoints = normalized_text(clean_slab.kpoints_path)
    if ads_kpoints and slab_kpoints and ads_kpoints != slab_kpoints:
        warnings.append(tr("warning.adsorption_kpoints_mismatch"))

    for role, input_set in (
        ("adsorbed", adsorbed),
        ("clean_slab", clean_slab),
        ("molecule_ref", molecule_ref),
    ):
        warnings.extend(potcar_order_warnings(role, input_set))

    if not is_gamma_only_kpoints(molecule_ref.kpoints_path):
        warnings.append(tr("warning.adsorption_molecule_not_gamma"))
    return warnings


def potcar_order_warnings(role: str, input_set: InputSet) -> list[str]:
    poscar_species = extract_poscar_species(input_set.poscar_path) if input_set.poscar_path.exists() else []
    potcar_species = summarize_potcar(input_set.potcar_path).get("element_order", []) if input_set.potcar_path.exists() else []
    if poscar_species and potcar_species and poscar_species != potcar_species:
        return [
            tr(
                "warning.adsorption_potcar_order_mismatch",
                role=input_set_role_label(role),
                poscar=", ".join(poscar_species),
                potcar=", ".join(potcar_species),
            )
        ]
    return []


def parse_incar_value(path: Path, key: str) -> str | None:
    if not path.exists():
        return None
    target = key.upper()
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        clean = line.split("#", 1)[0].split("!", 1)[0].strip()
        if "=" not in clean:
            continue
        left, right = clean.split("=", 1)
        if left.strip().upper() == target:
            return right.strip()
    return None


def normalized_text(path: Path) -> tuple[str, ...]:
    if not path.exists():
        return ()
    return tuple(line.strip() for line in path.read_text(encoding="utf-8", errors="replace").splitlines() if line.strip())


def is_gamma_only_kpoints(path: Path) -> bool:
    lines = normalized_text(path)
    text = " ".join(lines).lower()
    return "gamma" in text and any(line.replace("\t", " ").split()[:3] == ["1", "1", "1"] for line in lines)


def show_input_set_file_summary(input_set: InputSet) -> None:
    st.subheader(tr("input_set.files"))
    file_hashes = read_json_file(input_set.root_dir / "file_hashes.json") or {}
    rows = []
    for filename, path in input_set_file_paths(input_set).items():
        hash_info = file_hashes.get(filename, {})
        exists = path.exists()
        rows.append(
            {
                tr("table.file"): filename,
                tr("table.exists"): bool_label(exists),
                tr("table.size_bytes"): path.stat().st_size if exists else 0,
                tr("table.sha256"): hash_info.get("sha256") or (sha256_file(path) if exists else None),
            }
        )
    st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)

    potcar_summary = summarize_potcar(input_set.potcar_path)
    st.subheader(tr("tabs.potcar"))
    st.write(
        {
            tr("table.exists"): bool_label(potcar_summary.get("exists")),
            tr("table.size_bytes"): potcar_summary.get("size_bytes", 0),
            tr("table.sha256"): potcar_summary.get("sha256"),
            tr("table.titel_lines"): potcar_summary.get("titel_lines", []),
        }
    )
    st.caption(tr("vaspkit.potcar_no_full_preview"))


def show_json_file(tab, path: Path) -> None:
    with tab:
        data = read_json_file(path)
        if data is None:
            st.info(tr("info.file_not_generated", filename=path.name))
        else:
            st.json(data)


def validate_input_set_on_disk(input_set: InputSet) -> dict:
    missing: list[str] = []
    empty: list[str] = []
    poscar_species: list[str] = []
    potcar_species: list[str] = []
    for filename, path in input_set_file_paths(input_set).items():
        if not path.exists():
            missing.append(filename)
        elif path.stat().st_size == 0:
            empty.append(filename)
    errors: list[str] = []
    if missing:
        errors.append(tr("input_set.validation.missing_files", files=", ".join(missing)))
    if empty:
        errors.append(tr("input_set.validation.empty_files", files=", ".join(empty)))
    if input_set.poscar_path.exists() and input_set.poscar_path.stat().st_size > 0:
        poscar_species = extract_poscar_species(input_set.poscar_path)
        if not poscar_species:
            errors.append(tr("input_set.validation.poscar_species_missing"))
    if input_set.potcar_path.exists() and input_set.potcar_path.stat().st_size > 0:
        potcar_summary = summarize_potcar(input_set.potcar_path)
        potcar_species = list(potcar_summary.get("element_order", []))
        if not potcar_species:
            errors.append(tr("input_set.validation.potcar_titel_missing"))
    if poscar_species and potcar_species and poscar_species != potcar_species:
        errors.append(
            tr(
                "input_set.validation.species_mismatch",
                poscar=", ".join(poscar_species),
                potcar=", ".join(potcar_species),
            )
        )
    if input_set.status == "dry_run":
        return {
            "status": "dry_run",
            "usable_for_vasp": False,
            "required_files": list(CORE_INPUT_FILES),
            "missing_files": missing,
            "empty_files": empty,
            "poscar_species": poscar_species,
            "potcar_species": potcar_species,
            "warnings": [tr("warning.dry_run_input_set_not_usable")],
            "errors": errors,
            "validated_at": datetime.utcnow().isoformat(timespec="seconds"),
        }
    return {
        "status": "validated" if not errors else "invalid",
        "usable_for_vasp": not errors,
        "required_files": list(CORE_INPUT_FILES),
        "missing_files": missing,
        "empty_files": empty,
        "poscar_species": poscar_species,
        "potcar_species": potcar_species,
        "warnings": [],
        "errors": errors,
        "validated_at": datetime.utcnow().isoformat(timespec="seconds"),
    }


def extract_poscar_species(path: Path) -> list[str]:
    lines = path.read_text(encoding="utf-8", errors="replace").splitlines()
    if len(lines) < 7:
        return []
    species = lines[5].split()
    if not species:
        return []
    # VASP5 POSCAR 的第 6 行应为元素符号；如果这里是计数行，则说明不是可安全校验的 VASP5 格式。
    if any(not token[:1].isalpha() for token in species):
        return []
    return species


def read_json_file(path: Path):
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return {"error": tr("error.invalid_json_file", filename=path.name)}


def write_json_file(path: Path, data: dict) -> None:
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def show_workflow_page(config, potcars, conn) -> None:
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
            show_task_start_control(config, conn, task)
            show_stop_control(conn, task)
        with right:
            show_results(task)


def main() -> None:
    st.set_page_config(page_title=t("app.page_title", "zh"), layout="wide")
    language_selector()
    config, potcars = resources()
    init_conn = connect(config.workspace)
    init_conn.close()
    db_file = workspace_db_path(config.workspace)
    selected_page = navigation_selector()

    st.title(tr("app.title"))
    if selected_page == "dashboard":
        show_dashboard_page(db_file)
    elif selected_page == "vaspkit":
        conn = connect(config.workspace)
        try:
            show_vaspkit_input_generator(config, conn)
        finally:
            conn.close()
    elif selected_page == "input_sets":
        conn = connect(config.workspace)
        try:
            show_input_sets_page(config, conn)
        finally:
            conn.close()
    elif selected_page == "adsorption":
        show_adsorption_workflow_page(config, db_file)
    elif selected_page == "single_atom":
        show_module_placeholder_page("nav.single_atom")
    elif selected_page == "molecule_optimization":
        show_module_placeholder_page("nav.molecule_optimization")
    elif selected_page == "jobs_logs":
        show_jobs_logs_page(db_file)
    elif selected_page == "data_management":
        show_data_management_page(config, db_file)
    elif selected_page == "settings":
        show_settings_page()


if __name__ == "__main__":
    main()
