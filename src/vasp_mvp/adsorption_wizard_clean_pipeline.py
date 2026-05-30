from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from .adsorption_wizard import artifact_path


CLEAN_SOURCE_TYPES = ("slab", "porous_or_mof", "bulk_surface")
CLEAN_STEP_STATUS = ("pending", "done", "skipped", "failed")


@dataclass(frozen=True)
class CleanPipelineStep:
    key: str
    label_key: str
    required: bool
    vaspkit_task: str | None = None
    expected_output: str | None = None
    produces_candidate: bool = False
    description_key: str = ""
    disabled: bool = False

    def to_state(self) -> dict:
        """生成可写入 workflow_state.json 的轻量步骤状态。"""

        return {
            "key": self.key,
            "label_key": self.label_key,
            "status": "pending",
            "required": self.required,
            "candidate": None,
            "adopted": False,
            "vaspkit_task": self.vaspkit_task,
            "expected_output": self.expected_output,
            "produces_candidate": self.produces_candidate,
            "description_key": self.description_key,
            "disabled": self.disabled,
            "parameters": {},
        }


def get_clean_pipeline_steps(source_type: str) -> list[CleanPipelineStep]:
    """按 CIF 来源类型返回 Step 1 的分支流程定义。

    本阶段 bulk 优化采用方案 B：只显示后续阶段提示，不创建 bulk input/job；
    因此 bulk_input/bulk_relax 不是 required，避免流程在 105 后被 placeholder 卡死。
    """

    normalized = normalize_source_type(source_type)
    if normalized == "slab":
        return [
            CleanPipelineStep(
                "cif_105",
                "adsorption_wizard.clean_pipeline.step_105",
                True,
                vaspkit_task="105",
                expected_output="POSCAR",
                produces_candidate=True,
                description_key="adsorption_wizard.clean_pipeline.step_105.description",
            )
        ]
    if normalized == "porous_or_mof":
        return [
            CleanPipelineStep("cif_105", "adsorption_wizard.clean_pipeline.step_105", True, "105", "POSCAR", True, "adsorption_wizard.clean_pipeline.step_105.description"),
            CleanPipelineStep("symmetry_601", "adsorption_wizard.clean_pipeline.step_601", False, "601", None, False, "adsorption_wizard.clean_pipeline.step_601.description"),
            CleanPipelineStep("conventional_603", "adsorption_wizard.clean_pipeline.step_603", False, "603", "CONVCELL.vasp", True, "adsorption_wizard.clean_pipeline.step_603.description"),
            CleanPipelineStep("supercell_401", "adsorption_wizard.clean_pipeline.step_401", False, "401", "SCabc.vasp", True, "adsorption_wizard.clean_pipeline.step_401.description"),
        ]
    if normalized == "bulk_surface":
        return [
            CleanPipelineStep("cif_105", "adsorption_wizard.clean_pipeline.step_105", True, "105", "POSCAR", True, "adsorption_wizard.clean_pipeline.step_105.description"),
            CleanPipelineStep("bulk_input", "adsorption_wizard.clean_pipeline.step_bulk_input", False, None, None, False, "adsorption_wizard.clean_pipeline.step_bulk_input.description", disabled=True),
            CleanPipelineStep("bulk_relax", "adsorption_wizard.clean_pipeline.step_bulk_relax", False, None, None, False, "adsorption_wizard.clean_pipeline.step_bulk_relax.description", disabled=True),
            CleanPipelineStep("slab_803", "adsorption_wizard.clean_pipeline.step_803", True, "803", "SLAB<hkl>.vasp", True, "adsorption_wizard.clean_pipeline.step_803.description"),
            CleanPipelineStep("vacuum_801", "adsorption_wizard.clean_pipeline.step_801", False, "801", "POSCAR_REV.vasp", True, "adsorption_wizard.clean_pipeline.step_801.description"),
            CleanPipelineStep("supercell_401", "adsorption_wizard.clean_pipeline.step_401", False, "401", "SCabc.vasp", True, "adsorption_wizard.clean_pipeline.step_401.description"),
            CleanPipelineStep("fix_atoms_402", "adsorption_wizard.clean_pipeline.step_402", False, "402", "POSCAR_FIX", True, "adsorption_wizard.clean_pipeline.step_402.description"),
            CleanPipelineStep("fix_atoms_403", "adsorption_wizard.clean_pipeline.step_403", False, None, None, False, "adsorption_wizard.clean_pipeline.step_403.description", disabled=True),
        ]
    raise ValueError(f"Unsupported clean structure source type: {source_type}")


def normalize_source_type(source_type: str | None) -> str:
    """兼容旧 state 中的 already_slab，新的主流程统一使用 slab。"""

    if source_type == "already_slab":
        return "slab"
    if source_type in CLEAN_SOURCE_TYPES:
        return str(source_type)
    return "slab"


def initialize_clean_pipeline(state: dict, source_type: str) -> dict:
    state = dict(state)
    normalized = normalize_source_type(source_type)
    state["source_model_type"] = normalized
    state["clean_pipeline"] = {
        "source_type": normalized,
        "steps": [step.to_state() for step in get_clean_pipeline_steps(normalized)],
    }
    return state


def ensure_clean_pipeline(state: dict) -> dict:
    source_type = normalize_source_type(state.get("source_model_type"))
    pipeline = state.get("clean_pipeline")
    if not pipeline or normalize_source_type(pipeline.get("source_type")) != source_type:
        return initialize_clean_pipeline(state, source_type)
    expected_keys = [step.key for step in get_clean_pipeline_steps(source_type)]
    current_steps = pipeline.get("steps", [])
    current_keys = [step.get("key") for step in current_steps]
    if current_keys != expected_keys:
        return _migrate_clean_pipeline(state, source_type, current_steps)
    return state


def reset_clean_pipeline_for_source_type_change(state: dict, source_type: str) -> dict:
    """切换来源类型时只重置 pipeline 状态，不删除任何已有结构文件。"""

    return initialize_clean_pipeline(state, source_type)


def clean_pipeline_rows(state: dict) -> list[dict]:
    pipeline = ensure_clean_pipeline(state).get("clean_pipeline", {})
    return list(pipeline.get("steps", []))


def get_clean_pipeline_step(state: dict, step_key: str) -> dict | None:
    for step in clean_pipeline_rows(state):
        if step.get("key") == step_key:
            return step
    return None


def can_skip_clean_pipeline_step(state: dict, step_key: str) -> bool:
    step = get_clean_pipeline_step(state, step_key)
    return bool(step and not step.get("required") and step.get("status") == "pending" and can_run_clean_pipeline_step(state, step_key))


def can_run_clean_pipeline_step(state: dict, step_key: str, workflow_root: Path | None = None) -> bool:
    """判断某个步骤是否解锁。

    规则：
    - 前序 required/optional 步骤必须 done 或 skipped；
    - 前序 failed 会锁住后续步骤；
    - candidate 必须 adopt 后才允许下一步；
    - bulk 的 803 额外要求 bulk_relax/run/CONTCAR 存在，本阶段不会自动生成。
    """

    steps = clean_pipeline_rows(state)
    for index, step in enumerate(steps):
        if step.get("key") != step_key:
            continue
        if step.get("disabled"):
            return False
        if step.get("status") in {"done", "skipped"}:
            return False
        for previous in steps[:index]:
            if previous.get("disabled"):
                continue
            status = previous.get("status")
            if status == "failed":
                return False
            if previous.get("produces_candidate") and status == "done" and not previous.get("adopted"):
                return False
            if status not in {"done", "skipped"}:
                return False
        if step_key == "slab_803" and workflow_root is not None:
            return bulk_contcar_exists(workflow_root)
        return True
    return False


def mark_clean_pipeline_step_done(
    state: dict,
    step_key: str,
    *,
    candidate: str | None = None,
    adopted: bool = False,
    parameters: dict | None = None,
    logs: dict | None = None,
) -> dict:
    return _update_step(
        state,
        step_key,
        status="done",
        candidate=candidate,
        adopted=adopted,
        parameters=parameters,
        logs=logs,
    )


def mark_clean_pipeline_step_skipped(state: dict, step_key: str) -> dict:
    step = get_clean_pipeline_step(state, step_key)
    if not step:
        raise ValueError(f"Unknown clean pipeline step: {step_key}")
    if step.get("required"):
        raise ValueError("required clean pipeline steps cannot be skipped")
    if not can_skip_clean_pipeline_step(state, step_key):
        raise ValueError(f"Clean pipeline step is not skippable now: {step_key}")
    return _update_step(state, step_key, status="skipped")


def mark_clean_pipeline_step_failed(
    state: dict,
    step_key: str,
    *,
    errors: list[str] | None = None,
    parameters: dict | None = None,
    logs: dict | None = None,
) -> dict:
    return _update_step(state, step_key, status="failed", errors=errors, parameters=parameters, logs=logs)


def mark_clean_pipeline_candidate_adopted(state: dict, step_key: str) -> dict:
    return _update_step(state, step_key, status="done", adopted=True)


def get_next_enabled_clean_step(state: dict, workflow_root: Path | None = None) -> str | None:
    for step in clean_pipeline_rows(state):
        if can_run_clean_pipeline_step(state, step.get("key", ""), workflow_root):
            return step.get("key")
    return None


def bulk_contcar_exists(workflow_root: Path) -> bool:
    path = Path(workflow_root) / "jobs" / "bulk_relax" / "run" / "CONTCAR"
    return path.exists() and path.stat().st_size > 0


def candidate_for_step(state: dict, step_key: str) -> str | None:
    step = get_clean_pipeline_step(state, step_key)
    if not step:
        return None
    candidate = step.get("candidate")
    return str(candidate) if candidate else None


def adopted_clean_poscar_exists(workflow_root: Path, state: dict) -> bool:
    rel_path = state.get("artifacts", {}).get("clean_poscar")
    if not rel_path:
        return False
    path = artifact_path(workflow_root, rel_path)
    return path.exists() and path.stat().st_size > 0


def _update_step(
    state: dict,
    step_key: str,
    *,
    status: str,
    candidate: str | None = None,
    adopted: bool | None = None,
    parameters: dict | None = None,
    logs: dict | None = None,
    errors: list[str] | None = None,
) -> dict:
    if status not in CLEAN_STEP_STATUS:
        raise ValueError(f"Unsupported clean pipeline step status: {status}")
    state = ensure_clean_pipeline(dict(state))
    steps = []
    found = False
    for step in state["clean_pipeline"]["steps"]:
        item = dict(step)
        if item.get("key") == step_key:
            found = True
            item["status"] = status
            if candidate is not None:
                item["candidate"] = candidate
            if adopted is not None:
                item["adopted"] = adopted
            if parameters is not None:
                item["parameters"] = parameters
            if logs is not None:
                item["logs"] = logs
            if errors is not None:
                item["errors"] = errors
        steps.append(item)
    if not found:
        raise ValueError(f"Unknown clean pipeline step: {step_key}")
    state["clean_pipeline"]["steps"] = steps
    return state


def _migrate_clean_pipeline(state: dict, source_type: str, old_steps: list[dict]) -> dict:
    """兼容旧 workflow_state。

    例如 10D-2 从 porous_or_mof 主流程移除了 primitive_602。迁移时只保留
    新流程中仍存在的 step key，旧的隐藏/废弃步骤不会继续显示，也不会导致 UI 崩溃。
    已有 artifacts、candidates、logs 文件都不删除。
    """

    migrated = initialize_clean_pipeline(state, source_type)
    old_by_key = {step.get("key"): step for step in old_steps}
    merged_steps: list[dict] = []
    for step in migrated["clean_pipeline"]["steps"]:
        old = old_by_key.get(step.get("key"))
        if old:
            merged = dict(step)
            for field in ("status", "candidate", "adopted", "parameters", "logs", "errors"):
                if field in old:
                    merged[field] = old[field]
            merged_steps.append(merged)
        else:
            merged_steps.append(step)
    migrated["clean_pipeline"]["steps"] = merged_steps
    return migrated
