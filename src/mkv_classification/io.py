"""Small dependency-free helpers for tidy and atomic research outputs."""

from __future__ import annotations

import csv
import json
import os
import platform
from pathlib import Path
import subprocess
from typing import Any

import numpy as np


def read_rows(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def write_rows_atomic(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
        handle.flush()
        os.fsync(handle.fileno())
    temporary.replace(path)


def git_revision(repo_root: Path) -> str | None:
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"], cwd=repo_root, check=True,
            capture_output=True, text=True,
        )
        return result.stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def git_worktree_dirty(repo_root: Path) -> bool | None:
    try:
        result = subprocess.run(
            ["git", "status", "--porcelain", "--untracked-files=no"],
            cwd=repo_root, check=True, capture_output=True, text=True,
        )
        return bool(result.stdout.strip())
    except (OSError, subprocess.CalledProcessError):
        return None


def write_manifest(path: Path, *, config: dict[str, Any], elapsed_seconds: float) -> None:
    payload = {
        "config": config,
        "elapsed_seconds": elapsed_seconds,
        "python": platform.python_version(),
        "numpy": np.__version__,
        "git_revision": git_revision(path.parent.parent),
        "git_worktree_dirty": git_worktree_dirty(path.parent.parent),
    }
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def sample_summary(values: list[float]) -> dict[str, float]:
    array = np.asarray(values, dtype=float)
    mean = float(array.mean())
    standard_deviation = float(array.std(ddof=1)) if array.size > 1 else float("nan")
    standard_error = standard_deviation / np.sqrt(array.size) if array.size > 1 else float("nan")
    return {
        "mean": mean,
        "sample_standard_deviation": standard_deviation,
        "standard_error": standard_error,
        "ci95_lower": mean - 1.96 * standard_error,
        "ci95_upper": mean + 1.96 * standard_error,
    }
