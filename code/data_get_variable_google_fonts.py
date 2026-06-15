from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
import time
import urllib.request
from pathlib import Path
from typing import Optional, Tuple


GOOGLE_FONTS_REPO = "https://github.com/google/fonts.git"
GITHUB_API_REPO = "https://api.github.com/repos/google/fonts"


def run(cmd: list[str], cwd: Optional[Path] = None) -> None:
    try:
        subprocess.run(cmd, cwd=str(cwd) if cwd else None, check=True)
    except FileNotFoundError:
        raise RuntimeError(f"Missing executable: {cmd[0]!r}. Install it and retry.")
    except subprocess.CalledProcessError as e:
        raise RuntimeError(f"Command failed ({e.returncode}): {' '.join(cmd)}") from e


def prompt_yes_no(msg: str, default: bool = False) -> bool:
    suffix = " [Y/n]: " if default else " [y/N]: "
    while True:
        ans = input(msg + suffix).strip().lower()
        if not ans:
            return default
        if ans in ("y", "yes"):
            return True
        if ans in ("n", "no"):
            return False
        print("Please answer y/n.")


def human_bytes(n: int) -> str:
    units = ["B", "KiB", "MiB", "GiB", "TiB", "PiB"]
    x = float(n)
    for u in units:
        if x < 1024.0 or u == units[-1]:
            if u == "B":
                return f"{int(x)} {u}"
            return f"{x:.2f} {u}"
        x /= 1024.0
    return f"{n} B"


def disk_free_bytes(path: Path) -> int:
    usage = shutil.disk_usage(str(path))
    return usage.free


def dir_size_bytes(path: Path) -> int:
    total = 0
    for p in path.rglob("*"):
        try:
            if p.is_file():
                total += p.stat().st_size
        except OSError:
            pass
    return total


def github_repo_size_kb(timeout: float = 15.0) -> Optional[int]:

    try:
        req = urllib.request.Request(
            GITHUB_API_REPO,
            headers={
                "Accept": "application/vnd.github+json",
                "User-Agent": "get-variable-fonts-script",
            },
        )
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            data = json.loads(resp.read().decode("utf-8"))
        size_kb = data.get("size", None)
        return int(size_kb) if isinstance(size_kb, int) else None
    except Exception:
        return None


def ensure_empty_or_confirm(path: Path) -> None:
    if path.exists():
        if any(path.iterdir()):
            ok = prompt_yes_no(
                f"Path exists and is not empty: {path}\nContinue anyway?",
                default=False,
            )
            if not ok:
                print("Aborting.")
                sys.exit(1)
    else:
        path.mkdir(parents=True, exist_ok=True)


def clone_google_fonts(repo_dir: Path, branch: str = "main") -> None:

    if repo_dir.exists() and (repo_dir / ".git").exists():
        print(f"Repo already exists at {repo_dir} (skipping clone).")
        return

    repo_dir.parent.mkdir(parents=True, exist_ok=True)

    print("Cloning full repo (simple clone)...")
    run(["git", "clone", GOOGLE_FONTS_REPO, str(repo_dir)])
    run(["git", "checkout", branch], cwd=repo_dir)


def is_variable_font_ttf(path: Path) -> bool:

    name = path.name
    return path.suffix.lower() == ".ttf" and ("[" in name and "]" in name)


def copy_variable_fonts(repo_dir: Path, out_dir: Path) -> Tuple[int, int]:

    ensure_empty_or_confirm(out_dir)

    count = 0
    total = 0
    for p in repo_dir.rglob("*.ttf"):
        if not is_variable_font_ttf(p):
            continue

        rel = p.relative_to(repo_dir)
        dst = out_dir / rel
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(p, dst)

        try:
            total += dst.stat().st_size
        except OSError:
            pass
        count += 1

    return count, total


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--base-dir",
        type=Path,
        default=Path.cwd(),
        help="Working directory (default: cwd)",
    )
    ap.add_argument(
        "--repo-dir",
        type=Path,
        default=None,
        help="Where to clone repo (default: <base>/google-fonts)",
    )
    ap.add_argument(
        "--out-dir",
        type=Path,
        default=None,
        help="Where to store variable fonts (default: <base>/google-fonts-variable)",
    )
    ap.add_argument(
        "--branch", type=str, default="main", help="Branch to checkout (default: main)"
    )
    args = ap.parse_args()

    base_dir = args.base_dir.resolve()
    repo_dir = (args.repo_dir or (base_dir / "google-fonts")).resolve()
    out_dir = (args.out_dir or (base_dir / "google-fonts-variable")).resolve()

    print(f"Base dir: {base_dir}")
    print(f"Repo dir: {repo_dir}")
    print(f"Out dir : {out_dir}")
    print()

    size_kb = github_repo_size_kb()
    if size_kb is not None:
        approx_bytes = size_kb * 1024
        print(
            f"Google Fonts repo size (GitHub API 'size'): ~{human_bytes(approx_bytes)} (approx; not exact clone size)"
        )
    else:
        print(
            "Could not fetch remote repo size from GitHub API (no internet / rate limit)."
        )

    free = disk_free_bytes(base_dir)
    print(f"Free space at {base_dir}: {human_bytes(free)}")

    if not prompt_yes_no("Proceed to download/clone Google Fonts?", default=False):
        print("Aborting.")
        return

    t0 = time.time()
    clone_google_fonts(repo_dir=repo_dir, branch=args.branch)
    t1 = time.time()
    print(f"Clone/checkout done in {t1 - t0:.1f}s")
    print()

    print("Extracting variable fonts (*[*].ttf) into output folder...")

    if out_dir.exists() and any(out_dir.iterdir()):
        if prompt_yes_no(
            f"Output folder is not empty: {out_dir}\nDelete it and recreate?",
            default=False,
        ):
            shutil.rmtree(out_dir)
        else:
            print("Aborting (to avoid mixing outputs).")
            return

    out_dir.mkdir(parents=True, exist_ok=True)

    n, copied_bytes = copy_variable_fonts(repo_dir, out_dir)
    print(f"Copied {n} variable-font TTF files, total {human_bytes(copied_bytes)}")
    print(f"Output: {out_dir}")
    print()

    repo_size = dir_size_bytes(repo_dir) if repo_dir.exists() else 0
    print(f"Current cloned repo size on disk: {human_bytes(repo_size)}")
    if prompt_yes_no(
        "Delete the cloned repo to save space (keep only filtered variable fonts)?",
        default=False,
    ):
        shutil.rmtree(repo_dir)
        print(f"Deleted {repo_dir}")
    else:
        print("Keeping cloned repo.")

    free2 = disk_free_bytes(base_dir)
    print(f"Free space now at {base_dir}: {human_bytes(free2)}")


if __name__ == "__main__":
    main()
