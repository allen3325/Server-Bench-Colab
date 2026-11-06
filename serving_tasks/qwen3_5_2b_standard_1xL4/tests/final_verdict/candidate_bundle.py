"""Shared, bounded source-delivery policy; no GPU work or model imports.

Publication and development cold starts use exactly this file selection. Runtime
resources must either survive this policy or have an explicit startup recipe.
The report describes exclusions; it never pretends static analysis proves boot.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
from pathlib import Path

NEVER_CANDIDATE = (
    ".venv",
    ".git",
    "__pycache__",
    ".claude",
    ".review",
    "measure",
    "tests",
    "baselines",
    "knowledge",
    "MEMORY",
    "recall",
    "wait",
    "trace",
    "tools",
    "agentx",
    "microbench",
    "task_context",
    "clock",
    "BOUNDARY_CHECKS.md",
    ".triton_cache",
    ".uv-cache",
    "node_modules",
    ".feedback",
)
NOT_CANDIDATE_SUFFIXES = (
    ".pyc",
    ".pyo",
    ".pyd",
    ".o",
    ".obj",
    ".a",
    ".so",
    ".dylib",
    ".dll",
    ".lib",
    ".exe",
    ".d",
    ".gcda",
    ".gcno",
    ".cubin",
    ".fatbin",
    ".sass",
    ".safetensors",
    ".bin",
    ".pt",
    ".pth",
    ".ckpt",
    ".onnx",
    ".engine",
    ".gguf",
    ".npy",
    ".npz",
    ".h5",
    ".pkl",
    ".parquet",
    ".arrow",
    ".tar",
    ".gz",
    ".tgz",
    ".zip",
    ".whl",
    ".bz2",
    ".xz",
    ".zst",
    ".log",
    ".out",
    ".err",
    ".prof",
    ".nsys-rep",
    ".qdrep",
    ".ncu-rep",
    ".sqlite",
    ".db",
    ".core",
    ".swp",
    ".tmp",
    ".lock",
    ".png",
    ".jpg",
    ".jpeg",
    ".gif",
    ".pdf",
    ".svg",
)
HARNESS_OWNED = (
    "AGENT_TASK.md",
    "MAP.md",
    "MISSION.md",
    "TASK.md",
    "WORKSPACE.json",
    "DECISIONS.md",
    "FEEDBACK.md",
    "version",
    "checkpoint",
    "promote",
    "clock",
    "gpu-clock",
    "BOUNDARY_CHECKS.md",
    "DELIVERY_REPORT.json",
    "DELIVERY_POLICY.json",
    "full-test",
    "measurements",
)
RUN_EVIDENCE = (
    "identity.json",
    "journal.jsonl",
    "SUMMARY.json",
    "evolve_manifest.json",
    "attempt_state.json",
)
MAX_CANDIDATE_DEPTH = 4
MAX_CANDIDATE_BYTES = 8 * 1024 * 1024
_RUN_DIR = re.compile(
    r"^(run-\d+|[0-9a-f]{8}-\d{8}T\d{6}Z-[0-9a-f]{6})$")
RESOURCE_SUFFIXES = set(NOT_CANDIDATE_SUFFIXES[:37])


def policy() -> dict:
    return {
        "schema": "serving_evolution.delivery_policy/1",
        "excluded_names": list(NEVER_CANDIDATE),
        "excluded_suffixes": list(NOT_CANDIDATE_SUFFIXES),
        "harness_owned": list(HARNESS_OWNED),
        "maximum_depth": MAX_CANDIDATE_DEPTH,
        "maximum_file_bytes": MAX_CANDIDATE_BYTES,
        "hidden_entries": False,
        "symlinks": False,
        "run_directory_pattern": _RUN_DIR.pattern,
        "run_evidence_markers": list(RUN_EVIDENCE),
    }


def fingerprint(root: Path, files: list[str]) -> str:
    """Bind the delivered names, file bytes and modes before a cold start."""
    digests = {}
    for name in files:
        path = Path(root) / name
        if path.is_symlink() or not path.resolve().is_relative_to(Path(root).resolve()):
            raise ValueError(f"source path escapes delivery root: {name}")
        digests[name] = hashlib.sha256(
            str(path.stat().st_mode & 0o777).encode() + b"\0" + path.read_bytes()).hexdigest()
    return hashlib.sha256(json.dumps(digests, sort_keys=True).encode()).hexdigest()


def _relative(value: str) -> str:
    if not isinstance(value, str) or not value or "\\" in value:
        raise ValueError("resource paths must be nonempty relative POSIX paths")
    path = Path(value)
    if path.is_absolute() or ".." in path.parts or str(path) == ".":
        raise ValueError(f"resource escapes deployment root: {value!r}")
    return str(path)


def audit(root: Path) -> dict:
    root = Path(root)
    files, excluded = [], []
    for directory, dirs, names in os.walk(root, followlinks=False):
        base = Path(directory)
        depth = len(base.relative_to(root).parts)
        for name in sorted(dirs[:]):
            path = base / name
            reason = (
                "symlink"
                if path.is_symlink()
                else "excluded_directory"
                if name in NEVER_CANDIDATE
                else "hidden"
                if name.startswith(".")
                else "depth"
                if depth + 1 >= MAX_CANDIDATE_DEPTH
                else "run_evidence"
                if _RUN_DIR.match(name) or any((path / marker).exists() for marker in RUN_EVIDENCE)
                else ""
            )
            if reason:
                dirs.remove(name)
                excluded.append({"path": str(path.relative_to(root)) + "/", "reason": reason})
        dirs.sort()
        for name in sorted(names):
            path = base / name
            try:
                reason = (
                    "symlink"
                    if path.is_symlink()
                    else "not_regular"
                    if not path.is_file()
                    else "hidden"
                    if name.startswith(".")
                    else "harness_owned"
                    if name in HARNESS_OWNED
                    else "suffix"
                    if path.suffix.lower() in NOT_CANDIDATE_SUFFIXES
                    else "size"
                    if path.stat().st_size > MAX_CANDIDATE_BYTES
                    else ""
                )
            except OSError:
                reason = "unreadable"
            rel = str(path.relative_to(root))
            if reason:
                excluded.append({"path": rel, "reason": reason})
            else:
                files.append(rel)
    files.sort()
    errors, generated = [], {}
    manifest = root / "runtime_resources.json"
    if manifest.is_file() and not manifest.is_symlink():
        try:
            body = json.loads(manifest.read_text())
            if not isinstance(body, dict) or set(body) - {"required", "generated"}:
                raise ValueError("manifest accepts only required and generated")
            required = body.get("required", [])
            generated = body.get("generated", {})
            if not isinstance(required, list) or not isinstance(generated, dict):
                raise ValueError("required must be a list; generated maps output paths to recipe paths")
            generated = {_relative(k): _relative(v) for k, v in generated.items()}
            for name in required:
                rel = _relative(name)
                if rel not in files and rel not in generated:
                    errors.append(f"required runtime resource not delivered: {rel}")
            for output, recipe in generated.items():
                if recipe not in files:
                    errors.append(f"generator for {output} not delivered: {recipe}")
        except (OSError, ValueError, TypeError) as exc:
            errors.append(f"invalid runtime_resources.json: {exc}")
            generated = {}
    # A bounded conservative diagnostic, not a dependency proof: source literals
    # naming a dropped binary/data asset require a declared startup recipe.
    dropped = {
        Path(item["path"]).name: item["path"]
        for item in excluded
        if not item["path"].endswith("/")
        and Path(item["path"]).suffix.lower() in RESOURCE_SUFFIXES
        and item["path"] not in generated
    }
    for rel in files:
        source = root / rel
        if source.suffix not in {".py", ".sh", ".json", ".yaml", ".toml", ".cpp", ".cu"}:
            continue
        try:
            body = source.read_text(errors="replace")
        except OSError:
            continue
        for basename, missing in dropped.items():
            if basename in body:
                errors.append(
                    f"{rel} mentions excluded runtime asset {missing}; "
                    "declare a delivered resource or startup generation recipe"
                )
    return {
        "schema": "serving_evolution.delivery_report/1",
        "policy": policy(),
        "files": files,
        "excluded": excluded,
        "errors": sorted(set(errors)),
        "generated": generated,
        "static_only": True,
        "status": "failed" if errors else "passed",
        "scope": "static delivery audit only; cold start of this exact snapshot is still required",
    }


def check_generated(root: Path, report: dict) -> list[str]:
    """After readiness, every declared startup product must really exist."""
    return [
        name
        for name in report.get("generated", {})
        if not (root / name).is_file() or (root / name).is_symlink()
    ]
