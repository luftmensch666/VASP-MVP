from __future__ import annotations

import shutil
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
from vasp_mvp.db import connect, create_task, list_tasks, update_task_status
from vasp_mvp.i18n import t
from vasp_mvp.models import TaskDraft, TaskRecord, TaskRequest
from vasp_mvp.parser import parse_metrics
from vasp_mvp.renderers import build_draft
from vasp_mvp.runner import run_dir, start_vasp, stop_task, tail_file, write_confirmed_task
from vasp_mvp.rules import default_kpoints
from vasp_mvp.security import safe_task_id, task_dir
from vasp_mvp.structure_io import read_structure_upload
from vasp_mvp.vaspkit_options import get_vaspkit_section, validate_vaspkit_values
from vasp_mvp.vaspkit_runner import VaspkitRequest, VaspkitResult, generate_vasp_inputs_with_vaspkit


TASK_TYPES = ("relax", "static", "molecule", "adsorption")
RUN_RANKS = (20, 24)
VASPKIT_GENERATION_MODES = (
    "cif_to_poscar",
    "full",
    "incar_only",
    "kpoints_only",
    "potcar_only",
)
VASPKIT_COMMON_INCAR_KEYS = ("SR", "ST", "BD", "PU", "D3", "H6")


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


def metric_status_label(status: str) -> str:
    mapped = tr(f"metric_status.{status}")
    return status if mapped.startswith("[[missing:") else mapped


def bool_label(value: bool | None) -> str:
    if value is None:
        return tr("value.none")
    return tr("value.true") if value else tr("value.false")


def none_text(value) -> str:
    return tr("value.none") if value is None else str(value)


def vaspkit_option(section: str, key: str) -> dict:
    for option in get_vaspkit_section(section)["options"]:
        if option["key"] == key:
            return option
    raise KeyError(key)


def vaspkit_choice_label(option: dict, value: str) -> str:
    label_key = option.get("choice_label_keys", {}).get(value)
    return tr(label_key) if label_key else value


def vaspkit_task_root(config, task_id: str) -> Path:
    return task_dir(config, safe_task_id(task_id))


def vaspkit_draft_dir(config, task_id: str) -> Path:
    return vaspkit_task_root(config, task_id) / "draft"


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


def commit_vaspkit_draft(conn, request: VaspkitRequest, task_type: str) -> list[str]:
    warnings: list[str] = []
    source_dir = request.draft_dir
    target_dir = run_dir(request.draft_dir.parent)
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
        task_id=request.draft_dir.parent.name,
        project="default",
        task_type=task_type,
        task_root=request.draft_dir.parent,
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
        st.write(
            {
                tr("table.exists"): bool_label(summary.get("exists")),
                tr("table.size_bytes"): summary.get("size_bytes", 0),
                tr("table.path"): summary.get("path", ""),
                tr("table.titel_lines"): summary.get("titel_lines", []),
            }
        )
        if not summary.get("exists"):
            st.warning(tr("warning.potcar_not_generated"))
        st.caption(tr("vaspkit.potcar_no_full_preview"))


def show_vaspkit_input_generator(config, conn) -> None:
    st.subheader(tr("vaspkit.generator.title"))
    task_id = st.text_input(
        tr("sidebar.task_id"),
        f"vaspkit-{datetime.utcnow():%Y%m%d-%H%M%S}",
        key="vaspkit_task_id",
    )
    task_type = st.selectbox(
        tr("sidebar.task_type"),
        TASK_TYPES,
        format_func=task_type_label,
        key="vaspkit_task_type",
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

    generation_mode = st.selectbox(
        tr("vaspkit.generation_mode.label"),
        VASPKIT_GENERATION_MODES,
        help=tr("vaspkit.generation_mode.help"),
        format_func=lambda value: tr(f"vaspkit.generation_mode.{value}"),
    )

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
    potcar_mode_option = vaspkit_option("potcar", "potcar_mode")
    potcar_mode = st.selectbox(
        tr(potcar_mode_option["label_key"]),
        potcar_mode_option["choices"],
        index=potcar_mode_option["choices"].index(potcar_mode_option["default"]),
        help=tr(potcar_mode_option["help_key"]),
        format_func=lambda value: vaspkit_choice_label(potcar_mode_option, value),
    )
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

    if st.button(tr("button.generate_vaspkit_draft"), type="primary"):
        safe_id = safe_task_id(task_id)
        draft_dir = vaspkit_draft_dir(config, safe_id)
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
                uploaded_cif_path=uploaded_cif_path,
                generation_mode=generation_mode,
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
            st.session_state["vaspkit_request"] = vaspkit_request
            st.session_state["vaspkit_result"] = result
            st.session_state["vaspkit_commit_task_type"] = task_type
            if result.ok:
                st.success(tr("success.vaspkit_draft_saved", path=result.request_path))
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
        st.subheader(tr("vaspkit.json_preview"))
        if result.request_path.exists():
            st.json(result.request_path.read_text(encoding="utf-8"))
        if result.warnings:
            st.warning("; ".join(result.warnings))
        if result.errors:
            st.error("; ".join(result.errors))
        if st.button(tr("button.confirm_commit_vaspkit"), disabled=not result.ok or request_obj is None):
            warnings = commit_vaspkit_draft(conn, request_obj, st.session_state.get("vaspkit_commit_task_type", "static"))
            if warnings:
                st.warning("; ".join(warnings))
            st.success(tr("success.vaspkit_committed", path=run_dir(request_obj.draft_dir.parent)))


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
            show_stop_control(conn, task)
        with right:
            show_results(task)


def main() -> None:
    st.set_page_config(page_title=t("app.page_title", "zh"), layout="wide")
    language_selector()
    config, potcars = resources()
    conn = connect(config.workspace)

    st.title(tr("app.title"))
    workflow_tab, vaspkit_tab = st.tabs([tr("tabs.workflow"), tr("tabs.vaspkit_generator")])
    with workflow_tab:
        show_workflow_page(config, potcars, conn)
    with vaspkit_tab:
        show_vaspkit_input_generator(config, conn)


if __name__ == "__main__":
    main()
