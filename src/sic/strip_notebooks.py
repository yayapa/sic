"""Strip execution outputs from Jupyter notebooks and optionally push to remote."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path


def strip_notebook(path: Path) -> bool:
    """Clear code-cell outputs in ``path``. Returns True if the file changed."""
    nb = json.loads(path.read_text(encoding="utf-8"))
    changed = False
    for cell in nb.get("cells", []):
        if cell.get("cell_type") != "code":
            continue
        if cell.get("outputs") or cell.get("execution_count") is not None:
            cell["outputs"] = []
            cell["execution_count"] = None
            changed = True
    if changed:
        path.write_text(json.dumps(nb, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")
    return changed


def find_notebooks(paths: list[Path]) -> list[Path]:
    notebooks: list[Path] = []
    for path in paths:
        if path.is_file() and path.suffix == ".ipynb":
            notebooks.append(path)
        elif path.is_dir():
            notebooks.extend(sorted(path.rglob("*.ipynb")))
    return notebooks


def git_push(paths: list[Path], remote: str, branch: str | None, message: str) -> None:
    rel_paths = [str(p) for p in paths]
    subprocess.run(["git", "add", *rel_paths], check=True)
    status = subprocess.run(["git", "diff", "--cached", "--quiet"], check=False)
    if status.returncode != 0:
        subprocess.run(["git", "commit", "-m", message], check=True)
    push_args = ["git", "push", remote]
    if branch:
        push_args.append(f"HEAD:{branch}")
    subprocess.run(push_args, check=True)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Strip Jupyter notebook outputs and optionally push to a git remote.",
    )
    parser.add_argument(
        "paths",
        nargs="*",
        default=["examples"],
        help="Notebook files or directories to process (default: examples)",
    )
    parser.add_argument(
        "--push",
        action="store_true",
        help="Commit stripped notebooks and push to the remote",
    )
    parser.add_argument(
        "--remote",
        default="origin",
        help="Git remote name used with --push (default: origin)",
    )
    parser.add_argument(
        "--branch",
        default=None,
        help="Remote branch for --push (default: current branch)",
    )
    parser.add_argument(
        "-m",
        "--message",
        default="Strip notebook outputs",
        help="Commit message used with --push",
    )
    args = parser.parse_args(argv)

    notebooks = find_notebooks([Path(p) for p in args.paths])
    if not notebooks:
        print("No notebooks found.", file=sys.stderr)
        return 0 if args.paths else 1

    changed: list[Path] = []
    for nb_path in notebooks:
        if strip_notebook(nb_path):
            changed.append(nb_path)
            print(f"stripped {nb_path}")

    if not changed:
        print("No notebook outputs to remove.")
        return 0

    if args.push:
        git_push(changed, remote=args.remote, branch=args.branch, message=args.message)
        print(f"Pushed {len(changed)} notebook(s) to {args.remote}.")
        return 0

    if os.environ.get("PRE_COMMIT") == "1":
        subprocess.run(["git", "add", *[str(p) for p in changed]], check=False)
    return 0


if __name__ == "__main__":
    sys.exit(main())
