from __future__ import annotations

import json
import shutil
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path

from .input_set_validation import parse_poscar_element_order


WIZARD_STEPS = (
    "clean_structure",
    "adsorbed_structure",
    "molecule_reference",
    "generate_relax_inputs",
    "run_relax_jobs",
    "generate_static_inputs",
    "run_static_jobs",
    "calculate_eads",
)

# Adsorption Wizard 的 relax 阶段 role 元数据集中定义在这里。
# UI、输入文件生成和 job 创建都从同一处引用，避免 app.py 或后端模块各自
# 硬编码一份 role 列表，导致 Step 4/5 运行时出现未定义或顺序不一致。
RELAX_ROLES = ("clean_relax", "molecule_relax", "adsorbed_relax")
RELAX_ROLE_LABELS = {
    "clean_relax": "Clean relax",
    "molecule_relax": "Molecule relax",
    "adsorbed_relax": "Adsorbed relax",
}
RELAX_STEP_ORDER = {
    "clean_relax": 1,
    "molecule_relax": 2,
    "adsorbed_relax": 3,
}
RELAX_CALCULATION_TYPES = {
    "clean_relax": "relax",
    "molecule_relax": "molecule_relax",
    "adsorbed_relax": "relax",
}
STATIC_EADS_ROLES = ("clean_static", "molecule_static", "adsorbed_static")
STEP_ARTIFACTS = {
    "clean_structure": "artifacts/clean/POSCAR",
    "adsorbed_structure": "artifacts/adsorbed/POSCAR",
    "molecule_reference": "artifacts/molecule/POSCAR",
}


@dataclass(frozen=True)
class PoscarSummary:
    exists: bool
    path: Path
    size_bytes: int = 0
    element_order: tuple[str, ...] = ()
    atom_counts: tuple[int, ...] = ()
    total_atoms: int = 0
    cell_lengths: tuple[float, float, float] | None = None
    warnings: tuple[str, ...] = ()

    def to_dict(self, workflow_root: Path | None = None) -> dict:
        path_value = _relative_to_root(self.path, workflow_root) if workflow_root else str(self.path)
        return {
            "exists": self.exists,
            "path": path_value,
            "size_bytes": self.size_bytes,
            "element_order": list(self.element_order),
            "atom_counts": list(self.atom_counts),
            "total_atoms": self.total_atoms,
            "cell_lengths": None if self.cell_lengths is None else list(self.cell_lengths),
            "warnings": list(self.warnings),
        }


def default_wizard_state(workflow_id: str) -> dict:
    return {
        "workflow_id": workflow_id,
        "current_step": "clean_structure",
        "source_model_type": "already_slab",
        "artifacts": {},
        "candidates": {},
        "candidate_sources": {},
        "steps": {step: "pending" for step in WIZARD_STEPS},
        "updated_at": _now(),
    }


def workflow_state_path(workflow_root: Path) -> Path:
    return Path(workflow_root) / "workflow_state.json"


def load_wizard_state(workflow_root: Path, workflow_id: str) -> dict:
    path = workflow_state_path(workflow_root)
    if not path.exists():
        return default_wizard_state(workflow_id)
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        data = default_wizard_state(workflow_id)
        data["warnings"] = ["workflow_state_json_invalid"]
    return _normalize_state(data, workflow_id)


def save_wizard_state(workflow_root: Path, state: dict) -> None:
    state = _normalize_state(state, state.get("workflow_id", ""))
    state["updated_at"] = _now()
    path = workflow_state_path(workflow_root)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(state, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def artifact_path(workflow_root: Path, relative_path: str) -> Path:
    """用 workflow_root 拼接相对路径，避免 state 中保存绝对路径。"""

    if Path(relative_path).is_absolute():
        raise ValueError("wizard artifact paths must be relative")
    return Path(workflow_root) / relative_path


def save_candidate_file(
    workflow_root: Path,
    state: dict,
    *,
    role: str,
    source_path: Path,
    candidate_name: str,
    source_step: str,
    parameters: dict | None = None,
) -> dict:
    """把 VASPKIT 输出或用户上传文件复制为候选结构文件。

    候选文件统一放在 artifacts/<role>/candidates/，state 中只保存相对路径。
    """

    target_rel = f"artifacts/{role}/candidates/{Path(candidate_name).name}"
    target = artifact_path(workflow_root, target_rel)
    target.parent.mkdir(parents=True, exist_ok=True)
    if Path(source_path).resolve() != target.resolve():
        shutil.copy2(source_path, target)
    state = dict(state)
    candidates = dict(state.get("candidates", {}))
    candidates[role] = target_rel
    state["candidates"] = candidates
    candidate_sources = dict(state.get("candidate_sources", {}))
    candidate_sources[role] = {
        "step": source_step,
        "candidate": target_rel,
        "parameters": parameters or {},
    }
    state["candidate_sources"] = candidate_sources
    _update_step_from_artifacts(state)
    save_wizard_state(workflow_root, state)
    return state


def save_candidate_poscar(workflow_root: Path, state: dict, role: str, source_path: Path, candidate_name: str) -> dict:
    return save_candidate_file(
        workflow_root,
        state,
        role=role,
        source_path=source_path,
        candidate_name=candidate_name,
        source_step="unknown",
        parameters={},
    )


def list_clean_poscar_candidates(workflow_root: Path) -> list[dict]:
    candidates_dir = Path(workflow_root) / "artifacts" / "clean" / "candidates"
    if not candidates_dir.exists():
        return []
    rows: list[dict] = []
    for path in sorted(candidates_dir.iterdir(), key=lambda item: item.stat().st_mtime, reverse=True):
        if path.is_file() and path.stat().st_size > 0:
            rows.append(
                {
                    "name": path.name,
                    "relative_path": str(path.relative_to(workflow_root)),
                    "size_bytes": path.stat().st_size,
                    "updated_at": datetime.utcfromtimestamp(path.stat().st_mtime).isoformat(timespec="seconds"),
                }
            )
    return rows


def adopt_clean_poscar_candidate(workflow_root: Path, state: dict) -> dict:
    return adopt_candidate_poscar(workflow_root, state, "clean")


def adopt_candidate_poscar(workflow_root: Path, state: dict, role: str) -> dict:
    candidates = dict(state.get("candidates", {}))
    candidate_rel = candidates.get(role)
    if not candidate_rel:
        raise ValueError(f"No candidate POSCAR is available for role: {role}")
    source = artifact_path(workflow_root, candidate_rel)
    if not source.exists() or source.stat().st_size == 0:
        raise FileNotFoundError(f"Candidate POSCAR is missing or empty: {candidate_rel}")
    target_rel = f"artifacts/{role}/POSCAR"
    target = artifact_path(workflow_root, target_rel)
    target.parent.mkdir(parents=True, exist_ok=True)
    backup_rel = None
    if target.exists() and target.stat().st_size > 0:
        backup_dir = target.parent / "backups"
        backup_dir.mkdir(parents=True, exist_ok=True)
        backup = backup_dir / f"POSCAR.{datetime.utcnow():%Y%m%d_%H%M%S}.bak"
        shutil.copy2(target, backup)
        backup_rel = str(backup.relative_to(workflow_root))
    shutil.copy2(source, target)
    state = dict(state)
    artifacts = dict(state.get("artifacts", {}))
    artifacts[f"{role}_poscar"] = target_rel
    state["artifacts"] = artifacts
    if role == "clean":
        source_info = dict(state.get("candidate_sources", {}).get(role, {}))
        state["clean_poscar_source"] = {
            "step": source_info.get("step", "unknown"),
            "candidate": candidate_rel,
            "parameters": source_info.get("parameters", {}),
            "adopted_at": _now(),
        }
        if backup_rel:
            state["clean_poscar_source"]["previous_backup"] = backup_rel
    _update_step_from_artifacts(state)
    save_wizard_state(workflow_root, state)
    return state


def sync_clean_poscar_to_structure(workflow_root: Path, state: dict) -> Path:
    """把已采用的 clean POSCAR 同步到 structure/POSCAR。

    VASPKIT 801/803/401/402 都依赖当前工作目录存在 POSCAR，因此真实调用前
    必须先同步，而不是直接在 artifacts/clean/ 上操作。
    """

    clean_rel = state.get("artifacts", {}).get("clean_poscar")
    if not clean_rel:
        raise FileNotFoundError("No adopted clean POSCAR is available")
    source = artifact_path(workflow_root, clean_rel)
    if not source.exists() or source.stat().st_size == 0:
        raise FileNotFoundError(f"Adopted clean POSCAR is missing or empty: {clean_rel}")
    structure_dir = Path(workflow_root) / "structure"
    structure_dir.mkdir(parents=True, exist_ok=True)
    target = structure_dir / "POSCAR"
    shutil.copy2(source, target)
    return target


def poscar_summary(path: Path) -> PoscarSummary:
    path = Path(path)
    if not path.exists():
        return PoscarSummary(False, path, warnings=("poscar_missing",))
    if path.stat().st_size == 0:
        return PoscarSummary(True, path, size_bytes=0, warnings=("poscar_empty",))
    lines = path.read_text(encoding="utf-8", errors="replace").splitlines()
    warnings: list[str] = []
    element_order = parse_poscar_element_order(path)
    atom_counts = _parse_atom_counts(lines, bool(element_order))
    cell_lengths = _parse_cell_lengths(lines)
    if not element_order:
        warnings.append("poscar_element_order_unreadable")
    if not atom_counts:
        warnings.append("poscar_atom_counts_unreadable")
    return PoscarSummary(
        exists=True,
        path=path,
        size_bytes=path.stat().st_size,
        element_order=element_order,
        atom_counts=atom_counts,
        total_atoms=sum(atom_counts),
        cell_lengths=cell_lengths,
        warnings=tuple(warnings),
    )


def can_use_structure_edit_step(workflow_root: Path, state: dict) -> bool:
    clean_rel = state.get("artifacts", {}).get("clean_poscar")
    if not clean_rel:
        return False
    clean = artifact_path(workflow_root, clean_rel)
    return clean.exists() and clean.stat().st_size > 0


def step_status_rows(workflow_root: Path, state: dict) -> list[dict]:
    _update_step_from_artifacts(state)
    rows: list[dict] = []
    for index, step in enumerate(WIZARD_STEPS, start=1):
        artifact_rel = STEP_ARTIFACTS.get(step)
        artifact_exists = False
        if artifact_rel:
            artifact_exists = artifact_path(workflow_root, artifact_rel).exists()
        rows.append(
            {
                "step_number": index,
                "step": step,
                "status": state.get("steps", {}).get(step, "pending"),
                "required_artifact": artifact_rel or "",
                "artifact_exists": artifact_exists,
            }
        )
    return rows


def require_relax_contcars(workflow_root: Path) -> tuple[bool, tuple[str, ...]]:
    missing: list[str] = []
    for role in RELAX_ROLES:
        path = Path(workflow_root) / role / "run" / "CONTCAR"
        if not path.exists() or path.stat().st_size == 0:
            missing.append(str(path.relative_to(workflow_root)))
    return not missing, tuple(missing)


def static_roles_for_final_eads() -> tuple[str, str, str]:
    """返回最终吸附能允许使用的 static role。

    后续 Step 8 必须基于这三个 static role 计算：
    E_ads = E_adsorbed_static - E_clean_static - E_molecule_static。
    relax role 即使有 TOTEN，也不能用于最终吸附能。
    """

    return STATIC_EADS_ROLES


def _normalize_state(data: dict, workflow_id: str) -> dict:
    normalized = default_wizard_state(workflow_id)
    normalized.update(data)
    normalized["workflow_id"] = workflow_id or normalized.get("workflow_id", "")
    steps = dict(normalized.get("steps", {}))
    for step in WIZARD_STEPS:
        steps.setdefault(step, "pending")
    normalized["steps"] = steps
    normalized.setdefault("artifacts", {})
    normalized.setdefault("candidates", {})
    normalized.setdefault("candidate_sources", {})
    return normalized


def _update_step_from_artifacts(state: dict) -> None:
    artifacts = state.get("artifacts", {})
    steps = dict(state.get("steps", {}))
    steps["clean_structure"] = "done" if artifacts.get("clean_poscar") else "pending"
    steps["adsorbed_structure"] = "done" if artifacts.get("adsorbed_poscar") else "pending"
    steps["molecule_reference"] = "done" if artifacts.get("molecule_poscar") else "pending"
    state["steps"] = steps


def _parse_atom_counts(lines: list[str], has_element_line: bool) -> tuple[int, ...]:
    index = 6 if has_element_line else 5
    if len(lines) <= index:
        return ()
    tokens = lines[index].split()
    if not tokens or not all(token.isdigit() for token in tokens):
        return ()
    return tuple(int(token) for token in tokens)


def _parse_cell_lengths(lines: list[str]) -> tuple[float, float, float] | None:
    if len(lines) < 5:
        return None
    try:
        scale = float(lines[1].split()[0])
        vectors = []
        for line in lines[2:5]:
            values = [float(item) for item in line.split()[:3]]
            vectors.append((sum(value * value for value in values) ** 0.5) * scale)
        return tuple(vectors)  # type: ignore[return-value]
    except (ValueError, IndexError):
        return None


def _relative_to_root(path: Path, workflow_root: Path | None) -> str:
    if workflow_root is None:
        return str(path)
    try:
        return str(Path(path).relative_to(workflow_root))
    except ValueError:
        return str(path)


def _now() -> str:
    return datetime.utcnow().isoformat(timespec="seconds")
