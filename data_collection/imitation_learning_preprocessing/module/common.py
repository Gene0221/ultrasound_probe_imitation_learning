from __future__ import annotations

import json
from pathlib import Path
import re
from typing import Any, Iterable

import yaml


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_COLLECTION_ROOT = PROJECT_ROOT.parent
DEFAULT_CONFIG_PATH = PROJECT_ROOT / "config" / "preprocess_dataset.yaml"


def session_sort_key(path: Path, prefix: str) -> tuple[int, int | str]:
    suffix = path.name[len(prefix) :]
    if suffix.isdigit():
        return (0, int(suffix))
    match = re.search(r"(\d+)$", path.name)
    if match:
        return (1, int(match.group(1)))
    return (2, path.name)


def load_config(path: str | Path | None = None) -> dict[str, Any]:
    config_path = Path(path).resolve() if path else DEFAULT_CONFIG_PATH
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")
    payload = yaml.safe_load(config_path.read_text(encoding="utf-8")) or {}
    if not isinstance(payload, dict):
        raise ValueError(f"Config must be a YAML mapping: {config_path}")
    return payload


def resolve_path(value: str | Path | None, *, base: Path = PROJECT_ROOT) -> Path | None:
    if value in (None, ""):
        return None
    path = Path(str(value))
    if path.is_absolute():
        return path.resolve()
    return (base / path).resolve()


def config_get(config: dict[str, Any], section: str, key: str, default: Any = None) -> Any:
    payload = config.get(section, {})
    if isinstance(payload, dict) and key in payload:
        return payload[key]
    return default


def resolve_session_root(config: dict[str, Any], session_root_arg: str | None = None) -> Path:
    value = session_root_arg or config_get(config, "paths", "session_root") or config.get("session_root")
    if not value:
        raise ValueError("No session root provided. Set paths.session_root in config or pass --session-root.")
    root = resolve_path(value)
    if root is None:
        raise ValueError("Session root resolved to None.")
    return root


def resolve_session_dirs(
    *,
    config: dict[str, Any],
    session_arg: str | None,
    session_root_arg: str | None,
) -> list[Path]:
    if session_arg and session_root_arg:
        raise ValueError("--session and --session-root are mutually exclusive.")

    session_root = resolve_session_root(config, session_root_arg)
    if session_arg:
        session_path = Path(session_arg)
        if session_path.is_absolute():
            return [session_path.resolve(strict=True)]
        candidate = session_path.resolve()
        if candidate.exists():
            return [candidate]
        return [(session_root / session_arg).resolve(strict=True)]

    session_prefix = str(config_get(config, "paths", "session_prefix", "session_"))
    root = session_root.resolve(strict=True)
    sessions = sorted(
        (path for path in root.iterdir() if path.is_dir() and path.name.startswith(session_prefix)),
        key=lambda path: session_sort_key(path, session_prefix),
    )
    if not sessions:
        raise FileNotFoundError(f"No {session_prefix}* directories found under {root}")
    return sessions


def latest_child_dir(root: Path, prefixes: Iterable[str]) -> Path:
    prefixes_tuple = tuple(prefixes)
    candidates = sorted(
        [path for path in root.iterdir() if path.is_dir() and path.name.startswith(prefixes_tuple)],
        key=lambda path: path.stat().st_mtime,
    )
    if not candidates:
        prefix_text = ", ".join(prefixes_tuple)
        raise FileNotFoundError(f"No directory starting with one of [{prefix_text}] found under {root}")
    return candidates[-1]


def latest_matching_file(root: Path, patterns: Iterable[str]) -> Path:
    candidates: list[Path] = []
    for pattern in patterns:
        candidates.extend(path for path in root.glob(pattern) if path.is_file())
    unique_candidates = sorted(set(candidates), key=lambda path: path.stat().st_mtime)
    if not unique_candidates:
        pattern_text = ", ".join(patterns)
        raise FileNotFoundError(f"No file matching [{pattern_text}] found under {root}")
    return unique_candidates[-1]


def write_jsonl(path: Path, records: Iterable[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=True) + "\n")


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=True) + "\n", encoding="utf-8")


def output_path_for_session(
    *,
    session_dir: Path,
    output_dir_arg: str | None,
    output_subdir: str,
    file_name: str,
) -> Path:
    if output_dir_arg:
        output_dir = Path(output_dir_arg)
        if not output_dir.is_absolute():
            output_dir = (session_dir / output_dir).resolve()
        else:
            output_dir = output_dir.resolve()
    else:
        output_dir = session_dir / output_subdir
    return output_dir / file_name
