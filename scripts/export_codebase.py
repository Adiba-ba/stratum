#!/usr/bin/env python
"""Export the entire STRATUM codebase into COMPLETE_CODEBASE.md.

The exporter walks the source directories ``stratum/``, ``scripts/`` and
``tests/``, collects every Python file, the repository configuration
files, and generated report templates, and compiles them into a single
Markdown reference at the repository root.

The output starts with a tree-style directory overview, a current pytest
status summary (the suite is executed by this script), and per-file
line counts, followed by every collected file as a ``## <path>`` section
inside a syntax-highlighted code fence.

Run from anywhere with the project interpreter::

    .venv/bin/python scripts/export_codebase.py

Binary caches (``__pycache__``, ``*.pyc``), virtual environments, VCS
directories, and large data files (``*.nc``, ``*.parquet``, ...) are
excluded. Files larger than :data:`MAX_FILE_BYTES` are skipped as data.
"""

from __future__ import annotations

import subprocess
import sys
import time
from pathlib import Path

#: Repository root: this file lives in ``<root>/scripts/``.
PROJECT_ROOT = Path(__file__).resolve().parents[1]

#: Name of the generated reference file at the repository root.
OUTPUT_NAME = "COMPLETE_CODEBASE.md"

#: Directories scanned recursively for source files.
SOURCE_DIRS: tuple[str, ...] = ("stratum", "scripts", "tests")

#: Directory names skipped anywhere in the tree.
EXCLUDED_DIR_NAMES: frozenset[str] = frozenset(
    {
        ".venv",
        "venv",
        "__pycache__",
        ".git",
        ".pytest_cache",
        ".mypy_cache",
        ".ruff_cache",
        ".ipynb_checkpoints",
    }
)

#: File suffixes skipped: binary caches, compiled artifacts, data files.
EXCLUDED_FILE_SUFFIXES: frozenset[str] = frozenset(
    {
        ".pyc",
        ".pyo",
        ".so",
        ".dylib",
        ".dll",
        ".pyd",
        ".npy",
        ".npz",
        ".joblib",
        ".pkl",
        ".nc",
        ".parquet",
        ".grib",
        ".grib2",
        ".h5",
        ".hdf5",
        ".zip",
        ".gz",
        ".tar",
        ".log",
    }
)

#: Configuration file names picked up from the repository root.
CONFIG_FILE_NAMES: frozenset[str] = frozenset(
    {
        "pyproject.toml",
        "setup.py",
        "setup.cfg",
        "pytest.ini",
        "tox.ini",
        "mypy.ini",
        "requirements.txt",
    }
)

#: Glob patterns for additional root-level configuration files.
CONFIG_GLOBS: tuple[str, ...] = ("requirements-*.txt", "*.toml", "*.cfg", "*.ini")

#: Glob patterns for report templates/artifacts collected from the root.
REPORT_GLOBS: tuple[str, ...] = ("*.html",)

#: Files above this size are treated as data and skipped.
MAX_FILE_BYTES = 1_000_000

#: Markdown fence language per file suffix.
LANGUAGE_BY_SUFFIX: dict[str, str] = {
    ".py": "python",
    ".toml": "toml",
    ".cfg": "ini",
    ".ini": "ini",
    ".txt": "text",
    ".json": "json",
    ".yml": "yaml",
    ".yaml": "yaml",
    ".html": "html",
    ".md": "markdown",
    ".sh": "bash",
}


def _is_excluded(path: Path) -> bool:
    """Return True when ``path`` is a cache, virtualenv, or data file."""
    if any(part in EXCLUDED_DIR_NAMES for part in path.parts):
        return True
    if path.suffix.lower() in EXCLUDED_FILE_SUFFIXES:
        return True
    return path.name == OUTPUT_NAME


def _collect_source_files() -> list[Path]:
    """Collect all Python files under the configured source directories."""
    collected: list[Path] = []
    for directory in SOURCE_DIRS:
        base = PROJECT_ROOT / directory
        if not base.is_dir():
            continue
        for candidate in base.rglob("*.py"):
            if not _is_excluded(candidate):
                collected.append(candidate)
    return collected


def _collect_root_files() -> list[Path]:
    """Collect root-level Python, configuration, and report files."""
    collected: list[Path] = []
    for candidate in PROJECT_ROOT.iterdir():
        if not candidate.is_file() or _is_excluded(candidate):
            continue
        name = candidate.name
        is_python = candidate.suffix == ".py"
        is_config = name in CONFIG_FILE_NAMES or any(
            candidate.match(pattern) for pattern in CONFIG_GLOBS
        )
        is_report = any(candidate.match(pattern) for pattern in REPORT_GLOBS)
        if is_python or is_config or is_report:
            collected.append(candidate)
    return collected


def _collect_files() -> list[Path]:
    """Collect every file that belongs in the reference, sorted by path."""
    files = {path.resolve() for path in _collect_source_files()}
    files.update(path.resolve() for path in _collect_root_files())
    return sorted(files, key=lambda path: path.relative_to(PROJECT_ROOT).as_posix())


def _run_pytest() -> tuple[int, list[str]]:
    """Run the test suite and return ``(exit_code, output_tail_lines)``."""
    result = subprocess.run(
        [sys.executable, "-m", "pytest", "tests/", "-q", "--tb=no"],
        cwd=PROJECT_ROOT,
        capture_output=True,
        text=True,
        check=False,
    )
    lines = [line.rstrip() for line in result.stdout.splitlines() if line.strip()]
    if not lines:
        lines = [line.rstrip() for line in result.stderr.splitlines() if line.strip()]
    return result.returncode, lines


def _summarize_pytest(lines: list[str]) -> str:
    """Extract the one-line pass/fail summary from the pytest tail."""
    for line in reversed(lines):
        lowered = line.lower()
        if "passed" in lowered or "failed" in lowered or "error" in lowered:
            return line.strip()
    return "no pytest summary available"


def _count_lines(path: Path) -> tuple[int, int]:
    """Return ``(total_lines, code_lines)`` for a text file.

    Code lines are non-blank lines whose first non-space character is not
    ``#``; docstring lines count as code.
    """
    try:
        text = path.read_text(encoding="utf-8", errors="replace")
    except OSError:
        return 0, 0
    total = 0
    code = 0
    for line in text.splitlines():
        total += 1
        stripped = line.strip()
        if stripped and not stripped.startswith("#"):
            code += 1
    return total, code


def _build_tree(relative_paths: list[str]) -> dict[str, object]:
    """Build a nested dictionary describing the collected file tree."""
    tree: dict[str, object] = {}
    for raw in relative_paths:
        node = tree
        parts = Path(raw).parts
        for part in parts[:-1]:
            node = node.setdefault(part, {})  # type: ignore[assignment]
        node.setdefault("__files__", []).append(parts[-1])  # type: ignore[union-attr]
    return tree


def _render_tree(node: dict[str, object], prefix: str = "") -> list[str]:
    """Render the tree dictionary with box-drawing connectors."""
    lines: list[str] = []
    directories = sorted(key for key in node if key != "__files__")
    files = sorted(node.get("__files__", []))  # type: ignore[arg-type]
    entries = [(name, "dir") for name in directories] + [
        (name, "file") for name in files
    ]
    for position, (name, kind) in enumerate(entries):
        last = position == len(entries) - 1
        connector = "└── " if last else "├── "
        suffix = "/" if kind == "dir" else ""
        lines.append(f"{prefix}{connector}{name}{suffix}")
        if kind == "dir":
            extension = "    " if last else "│   "
            lines.extend(_render_tree(node[name], prefix + extension))  # type: ignore[arg-type]
    return lines


def _fence_for(content: str, language: str) -> tuple[str, str]:
    """Return opening and closing fences that cannot collide with content."""
    fence = "```"
    while fence in content:
        fence += "`"
    return f"{fence}{language}", fence


def _file_group(relative_path: str) -> str:
    """Group label for the line-count summary."""
    if "/" not in relative_path:
        return "repository root"
    return relative_path.split("/", 1)[0]


def _render_markdown(
    files: list[Path],
    relative_paths: list[str],
    pytest_exit: int,
    pytest_lines: list[str],
    totals: dict[str, tuple[int, int]],
) -> str:
    """Compose the complete Markdown reference document."""
    now = time.strftime("%Y-%m-%d %H:%M:%S UTC", time.gmtime())
    summary = _summarize_pytest(pytest_lines)

    parts: list[str] = []
    parts.append("# Project STRATUM — Complete Codebase Reference\n")
    parts.append(
        f"Generated by `scripts/export_codebase.py` on {now}. "
        f"{len(files)} files exported. Data files (`*.nc`, `*.parquet`), "
        "binary caches, and virtual environments are excluded.\n"
    )

    parts.append("## Directory overview\n")
    parts.append("```text")
    parts.append("stratum-codebase/")
    tree = _build_tree(relative_paths)
    parts.extend(_render_tree(tree))
    parts.append("```\n")

    parts.append("## Test status\n")
    parts.append(
        f"`{sys.executable} -m pytest tests/ -q --tb=no` → **{summary}** "
        f"(exit code {pytest_exit}).\n"
    )
    parts.append("```text")
    parts.extend(pytest_lines[-5:] if pytest_lines else ["no pytest output"])
    parts.append("```\n")

    parts.append("## Line counts\n")
    parts.append("| File | Lines | Code |")
    parts.append("| ---- | ----: | ---: |")
    for relative in relative_paths:
        total, code = totals[relative]
        parts.append(f"| `{relative}` | {total} | {code} |")
    grand_total = sum(total for total, _ in totals.values())
    grand_code = sum(code for _, code in totals.values())
    parts.append(f"| **Total** | **{grand_total}** | **{grand_code}** |")
    parts.append("")

    parts.append("| Group | Files | Lines | Code |")
    parts.append("| ----- | ----: | ----: | ---: |")
    groups: dict[str, list[str]] = {}
    for relative in relative_paths:
        groups.setdefault(_file_group(relative), []).append(relative)
    for group in sorted(groups):
        members = groups[group]
        lines_total = sum(totals[member][0] for member in members)
        code_total = sum(totals[member][1] for member in members)
        parts.append(
            f"| {group} | {len(members)} | {lines_total} | {code_total} |"
        )
    parts.append("")

    parts.append("## File contents\n")
    for path, relative in zip(files, relative_paths):
        content = path.read_text(encoding="utf-8", errors="replace")
        if content and not content.endswith("\n"):
            content += "\n"
        language = LANGUAGE_BY_SUFFIX.get(path.suffix.lower(), "text")
        opening, closing = _fence_for(content, language)
        parts.append(f"## {relative}\n")
        parts.append(opening)
        parts.append(content.rstrip("\n"))
        parts.append(closing)
        parts.append("")

    return "\n".join(parts) + "\n"


def main() -> int:
    """Export the codebase and write ``COMPLETE_CODEBASE.md``."""
    files = _collect_files()
    relative_paths = [
        path.relative_to(PROJECT_ROOT).as_posix() for path in files
    ]

    skipped: list[str] = []
    kept_files: list[Path] = []
    kept_relatives: list[str] = []
    for path, relative in zip(files, relative_paths):
        if path.stat().st_size > MAX_FILE_BYTES:
            skipped.append(relative)
            continue
        kept_files.append(path)
        kept_relatives.append(relative)

    totals = {
        relative: _count_lines(path)
        for path, relative in zip(kept_files, kept_relatives)
    }

    print(f"Exporting {len(kept_files)} files into {OUTPUT_NAME} ...")
    pytest_exit, pytest_lines = _run_pytest()
    print(f"pytest exit code {pytest_exit}: {_summarize_pytest(pytest_lines)}")
    for relative in skipped:
        print(f"  skipped (size > {MAX_FILE_BYTES} bytes): {relative}")

    document = _render_markdown(
        kept_files, kept_relatives, pytest_exit, pytest_lines, totals
    )
    output_path = PROJECT_ROOT / OUTPUT_NAME
    output_path.write_text(document, encoding="utf-8")
    print(f"Report written: {output_path}")
    print(f"  files: {len(kept_files)} | size: {len(document)} chars")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
